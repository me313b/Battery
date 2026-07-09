"""FEA 2: axisymmetric cell conduction, anisotropic jellyroll.

Solves  k_r (1/r) d/dr(r dT/dr) + k_z d2T/dz2 + q''' = 0
on r in [r0, R], z in [0, H]; side face convective (h_s, T_oil), inner
mandrel adiabatic, ends adiabatic or convective (h_e).

Purpose: test the app's core resistance r_core = 1/(4 pi k_r H)
(solid cylinder, uniform generation, adiabatic ends) against:
  (a) the FD solve of exactly that case (verification),
  (b) the real mandrel hole (2 mm on a 21700),
  (c) wetted ends through the axial conductivity,
  (d) the 4680 geometry.
Validated against the exact solid- and hollow-cylinder solutions.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

T_OIL = 35.0

def cell_solve(R=0.0105, H=0.070, r0=0.002, kr=0.9, kz=30.0,
               Q=2.26, h_s=120.0, h_e=0.0, nr=61, nz=121):
    """Return field and summary. Q [W] total; h_e = 0 -> adiabatic ends."""
    r = np.linspace(r0, R, nr); z = np.linspace(0, H, nz)
    dr, dz = r[1]-r[0], z[1]-z[0]
    qv = Q / (np.pi*(R**2-r0**2)*H)
    N = nr*nz
    ind = lambda i, j: j*nr + i
    rows, cols, vals = [], [], []
    rhs = np.full(N, -qv)

    def add(p, q_, v): rows.append(p); cols.append(q_); vals.append(v)

    for j in range(nz):
        for i in range(nr):
            p = ind(i, j)
            # radial part with cylindrical metric: (1/r) d/dr (r k dT/dr)
            if 0 < i < nr-1:
                rp, rm = 0.5*(r[i]+r[i+1]), 0.5*(r[i]+r[i-1])
                add(p, ind(i+1, j), kr*rp/(r[i]*dr**2))
                add(p, ind(i-1, j), kr*rm/(r[i]*dr**2))
                add(p, p, -kr*(rp+rm)/(r[i]*dr**2))
            elif i == 0:                        # inner: adiabatic, half cell
                rp = 0.5*(r[0]+r[1])
                add(p, ind(1, j), 2*kr*rp/(r[0]*dr**2))
                add(p, p, -2*kr*rp/(r[0]*dr**2))
            else:                               # outer: convection, half cell
                rm = 0.5*(r[-1]+r[-2])
                add(p, ind(nr-2, j), 2*kr*rm/(r[-1]*dr**2))
                add(p, p, -2*kr*rm/(r[-1]*dr**2) - 2*h_s/dr)
                rhs[p] += -2*h_s*T_OIL/dr
            # axial part
            if 0 < j < nz-1:
                add(p, ind(i, j+1), kz/dz**2)
                add(p, ind(i, j-1), kz/dz**2)
                add(p, p, -2*kz/dz**2)
            else:
                jn = 1 if j == 0 else nz-2
                add(p, ind(i, jn), 2*kz/dz**2)
                add(p, p, -2*kz/dz**2)
                if h_e > 0:
                    add(p, p, -2*h_e/dz)
                    rhs[p] += -2*h_e*T_OIL/dz
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    T = spla.spsolve(A, rhs).reshape(nz, nr)
    T_surf = float(T[:, -1].mean())
    return dict(T=T, r=r, z=z, T_max=float(T.max()),
                dT_core=float(T.max()-T_surf), T_surf=T_surf)

def analytic_solid(Q, R, H, kr):     # adiabatic ends, uniform q'''
    return Q/(4*np.pi*kr*H)
def analytic_hollow(Q, R, r0, H, kr):
    qv = Q/(np.pi*(R**2-r0**2)*H)
    return qv/(4*kr)*(R**2-r0**2) - qv*r0**2/(2*kr)*np.log(R/r0)

def validate():
    print("="*72); print("VALIDATION vs exact 1D solutions (adiabatic ends)")
    Q, R, H, kr = 2.26, 0.0105, 0.070, 0.9
    r = cell_solve(R, H, r0=R*1e-3, kr=kr, Q=Q, h_s=120.0)
    ex = analytic_solid(Q, R, H, kr)
    print(f"  solid : FD dT_core = {r['dT_core']:.3f} K, exact = {ex:.3f} K "
          f"({100*(r['dT_core']/ex-1):+.2f}%)")
    print(f"          FD film rise = {r['T_surf']-T_OIL:.3f} K, exact "
          f"Q/(hA) = {Q/(120*2*np.pi*R*H):.3f} K")
    r2 = cell_solve(R, H, r0=0.002, kr=kr, Q=Q, h_s=120.0)
    ex2 = analytic_hollow(Q, R, 0.002, H, kr)
    print(f"  hollow: FD dT_core = {r2['dT_core']:.3f} K, exact = {ex2:.3f} K "
          f"({100*(r2['dT_core']/ex2-1):+.2f}%)")

def study():
    print("="*72); print("CASES (dT core-to-mean-surface, K)")
    rows = []
    cases = [
        ("21700 2C solid, adiab. ends (app's model)",
         dict(R=.0105, H=.070, r0=.0105e-2, kr=.9, kz=30, Q=2.26, h_s=120, h_e=0)),
        ("21700 2C + 2 mm mandrel hole",
         dict(R=.0105, H=.070, r0=.002, kr=.9, kz=30, Q=2.26, h_s=120, h_e=0)),
        ("21700 2C + mandrel + wetted ends (h=80)",
         dict(R=.0105, H=.070, r0=.002, kr=.9, kz=30, Q=2.26, h_s=120, h_e=80)),
        ("21700 4C + mandrel + wetted ends",
         dict(R=.0105, H=.070, r0=.002, kr=.9, kz=30, Q=9.05, h_s=140, h_e=80)),
        ("4680 2C solid, adiab. ends (app)",
         dict(R=.023, H=.080, r0=.023e-2, kr=.9, kz=30, Q=12.67, h_s=120, h_e=0)),
        ("4680 2C + 4 mm mandrel + wetted ends",
         dict(R=.023, H=.080, r0=.004, kr=.9, kz=30, Q=12.67, h_s=120, h_e=80)),
    ]
    app_ref = {"21700": analytic_solid(2.26, .0105, .070, .9),
               "21700_4C": analytic_solid(9.05, .0105, .070, .9),
               "4680": analytic_solid(12.67, .023, .080, .9)}
    for name, kw in cases:
        r = cell_solve(**kw)
        key = "4680" if "4680" in name else ("21700_4C" if "4C" in name else "21700")
        ref = app_ref[key]
        rows.append((name, r["dT_core"], ref))
        print(f"  {name:45s} FD {r['dT_core']:5.2f}  app {ref:5.2f} "
              f"({100*(r['dT_core']/ref-1):+.0f}%)")
    # field plot for the realistic 21700 case
    r = cell_solve(R=.0105, H=.070, r0=.002, kr=.9, kz=30, Q=2.26,
                   h_s=120, h_e=80)
    fig, ax = plt.subplots(figsize=(4.2, 6))
    c = ax.contourf(r["r"]*1000, r["z"]*1000, r["T"], 24, cmap="RdYlBu_r")
    fig.colorbar(c, label="T [degC]")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("21700 at 2C: mandrel + wetted ends\n(k_r=0.9, k_z=30)")
    plt.tight_layout(); plt.savefig("cell_results.png", dpi=130)
    print("saved cell_results.png")
    return rows

if __name__ == "__main__":
    validate()
    study()
