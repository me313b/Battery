"""FEA 1: enclosure lid under burst pressure (fast Kronecker assembly).

Kirchhoff plate as FD biharmonic: B4 = kron(Iy,D4x) + 2 kron(D2y,D2x)
+ kron(D4y,Ix) on interior nodes (w=0 edges; clamped/SS via D4 end rows).
Ribs = line bending stiffness EI*D4 along their row/column, smeared /dx.
Validated against Roark, then a grillage mass optimisation to test the
app's t_flat = b*sqrt(0.31 p/sigma) and 0.45 stiffened knock-down.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E, NU, RHO, SIG_A = 70e9, 0.33, 2700.0, 80e6
P, LX, LY, T_RIB = 0.5e5, 0.916, 0.889, 0.006

def D2m(n, d):
    return sp.diags([1, -2, 1], [-1, 0, 1], (n, n)) / d**2
def D4m(n, d, bc):
    m = sp.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], (n, n)).tolil()
    end = 7.0 if bc == "clamped" else 5.0
    m[0, 0] = end; m[-1, -1] = end
    return (m / d**4).tocsr()

class Lid:
    def __init__(self, a, b, bc="clamped", nx=61, ribs_x=(), ribs_y=()):
        self.a, self.b, self.bc = a, b, bc
        ny = max(int(round(nx * b / a)) | 1, 31)
        self.nx, self.ny = nx, ny
        self.dx, self.dy = a/(nx-1), b/(ny-1)
        nxi, nyi = nx-2, ny-2
        Ix, Iy = sp.identity(nxi), sp.identity(nyi)
        D2x, D2y = D2m(nxi, self.dx), D2m(nyi, self.dy)
        D4x, D4y = D4m(nxi, self.dx, bc), D4m(nyi, self.dy, bc)
        self.B4 = (sp.kron(Iy, D4x) + 2*sp.kron(D2y, D2x)
                   + sp.kron(D4y, Ix)).tocsr()
        Kr = sp.csr_matrix(self.B4.shape)
        self.rib_ii, self.rib_jj = [], []
        for x in ribs_x:                      # rib running in y at column i
            i = int(round(x/self.dx)) - 1
            self.rib_ii.append(i)
            e = sp.csr_matrix(([1.0], ([i], [i])), (nxi, nxi))
            Kr += sp.kron(D4y, e) / self.dx
        for y in ribs_y:
            j = int(round(y/self.dy)) - 1
            self.rib_jj.append(j)
            e = sp.csr_matrix(([1.0], ([j], [j])), (nyi, nyi))
            Kr += sp.kron(e, D4x) / self.dy
        self.Kr0 = Kr.tocsr()
        self.rhs = np.full(nxi*nyi, 1.0)

    def solve(self, t, q, h_r):
        D = E*t**3/(12*(1-NU**2))
        EI = E*T_RIB*h_r**3/12
        A = D*self.B4 + (EI*self.Kr0 if h_r > 0 else 0)
        w = spla.spsolve(A.tocsr(), q*self.rhs).reshape(self.ny-2, self.nx-2)
        W = np.zeros((self.ny, self.nx)); W[1:-1, 1:-1] = w
        wxx = np.zeros_like(W); wyy = np.zeros_like(W)
        wxx[:, 1:-1] = (W[:, 2:]-2*W[:, 1:-1]+W[:, :-2])/self.dx**2
        wyy[1:-1, :] = (W[2:, :]-2*W[1:-1, :]+W[:-2, :])/self.dy**2
        if self.bc == "clamped":
            wxx[:, 0] = 2*W[:, 1]/self.dx**2; wxx[:, -1] = 2*W[:, -2]/self.dx**2
            wyy[0, :] = 2*W[1, :]/self.dy**2; wyy[-1, :] = 2*W[-2, :]/self.dy**2
        Mx = -D*(wxx+NU*wyy); My = -D*(wyy+NU*wxx)
        sx, sy = 6*Mx/t**2, 6*My/t**2
        svm = np.sqrt(sx**2+sy**2-sx*sy).max()
        s_comp = max(np.abs(sx).max(), np.abs(sy).max())
        s_rib = 0.0
        for i in self.rib_ii:
            s_rib = max(s_rib, E*(h_r/2)*np.abs(wyy[:, i+1]).max())
        for j in self.rib_jj:
            s_rib = max(s_rib, E*(h_r/2)*np.abs(wxx[j+1, :]).max())
        return dict(w_max=float(np.abs(W).max()), s_plate=float(svm),
                    s_comp=float(s_comp), s_rib=float(s_rib))

def validate():
    print("="*72)
    print("VALIDATION vs Roark (square, uniform load; ref tables nu=0.3)")
    t, q = 0.010, 1e4
    D = E*t**3/(12*(1-NU**2))
    for bc, a_ref, b_ref, where in [("ss", 0.00406, 0.2874, "centre"),
                                    ("clamped", 0.00126, 0.3078, "mid-edge")]:
        L = Lid(1.0, 1.0, bc=bc, nx=97)
        r = L.solve(t, q, 0.0)
        al = r["w_max"]*D/q
        be = r["s_comp"]*t**2/q
        print(f"  {bc:8s} alpha={al:.5f} (ref {a_ref:.5f}, "
              f"{100*(al/a_ref-1):+.1f}%)  beta={be:.4f} "
              f"(ref {b_ref:.4f}, {100*(be/b_ref-1):+.1f}%) [{where}]")
    print("  (design study below uses von Mises, which is what the 80 MPa "
          "allowable should be compared against)")

def study():
    print("="*72)
    b = min(LX, LY)
    t_app = b*np.sqrt(0.31*P/SIG_A)
    m_app = LX*LY*0.45*t_app*RHO*1.18
    print(f"LID {LX*1000:.0f}x{LY*1000:.0f} mm, p={P/1e5:.1f} bar g, "
          f"sigma_a={SIG_A/1e6:.0f} MPa")
    print(f"  app: t_flat={t_app*1000:.1f} mm, t_eff={0.45*t_app*1000:.1f} mm, "
          f"lid mass proxy (x1.18 fasteners) = {m_app:.1f} kg")
    results = []
    for n in range(0, 5):
        xs = [LX*(k+1)/(n+1) for k in range(n)]
        ys = [LY*(k+1)/(n+1) for k in range(n)]
        L = Lid(LX, LY, "clamped", 61, xs, ys)
        best = None
        for h in ([0.0] if n == 0 else np.arange(0.02, 0.201, 0.02)):
            lo, hi = 0.0015, 0.016
            if L.solve(hi, P, h)["s_plate"] > SIG_A:
                continue
            for _ in range(9):
                mid = 0.5*(lo+hi)
                r = L.solve(mid, P, h)
                if r["s_plate"] > SIG_A: lo = mid
                else: hi = mid
            r = L.solve(hi, P, h)
            if r["s_rib"] > SIG_A:
                continue
            m = (LX*LY*hi + T_RIB*h*n*(LX+LY))*RHO
            if best is None or m < best[0]:
                best = (m, hi, h, r)
        if best:
            m, t, h, r = best
            results.append((n, t, h, m))
            print(f"  {n} ribs/dir: t={t*1000:4.1f} mm h_rib={h*1000:3.0f} mm "
                  f"mass={m:5.1f} kg t_eq={m/(LX*LY*RHO)*1000:4.1f} mm "
                  f"(sig pl/rib {r['s_plate']/1e6:3.0f}/{r['s_rib']/1e6:3.0f} MPa, "
                  f"w={r['w_max']*1000:.1f} mm)")
    flat, bst = results[0], min(results, key=lambda x: x[3])
    kd = bst[3]/flat[3]
    print(f"  -> flat t={flat[1]*1000:.1f} mm ({flat[3]:.1f} kg); best "
          f"{bst[0]} ribs/dir -> {bst[3]:.1f} kg; mass knock-down = {kd:.2f} "
          f"(app assumes 0.45)")
    return results, t_app, m_app

if __name__ == "__main__":
    validate()
    res, t_app, m_app = study()
    ns = [r[0] for r in res]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ns, [r[3] for r in res], "o-", color="#C0392B",
               label="grillage optimum (FE)")
    ax[0].axhline(m_app/1.18, ls="--", color="#555", label="app 0.45 knock-down")
    ax[0].set_xlabel("ribs per direction"); ax[0].set_ylabel("lid mass [kg]")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Lid mass, {P/1e5:.1f} bar g, sigma_a={SIG_A/1e6:.0f} MPa")
    ax[1].plot(ns, [r[1]*1000 for r in res], "s-", label="skin t [mm]")
    ax[1].plot(ns, [r[2]*1000 for r in res], "^-", label="rib h [mm]")
    ax[1].set_xlabel("ribs per direction"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("Optimal sizes")
    plt.tight_layout(); plt.savefig("lid_results.png", dpi=130)
    print("saved lid_results.png")
