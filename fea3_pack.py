"""FEA 3: pack-section thermal network - is the worst-cell spread honest?

The app estimates best-to-worst cell spread as (water rise + stratification),
i.e. +/- (dT_water + dT_loop)/2 about the mean. This model resolves the
plan-direction physics behind the water-rise half:

  * oil discretised into Nb bins along the tube direction x,
  * every cell dumps Q_cell into its bin through the cell film,
  * each tube marches water from inlet to outlet, extracting per segment
    through the app's chain resistance (oil film + wall + water film),
  * adjacent bins exchange heat by thermosiphon-driven dispersion
    D_eff ~ u_ts * H_loop (swept x0.2 / x1 / x5),
  * end bins leak to ambient through the casing.

A pure conduction FE of the section would be dishonest (the oil moves), so
the dispersion term carries the convective mixing explicitly.

Validated in the mixing limit: D_eff -> large must recover a uniform oil
temperature and an energy balance closing to <0.1%.
Also runs the counterflow variant (alternate tube directions).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- defaults lifted from the app's smoke case (2C, DCIR-coupled) ----
Q_TOT = 2258.0          # W at 2C
N_CELLS = 1080
LX, LY, FILL_H = 0.916, 0.889, 0.125
N_TUBES, MDOT_TOT, CP_W = 16, 0.166, 4180.0
T_W_IN = 20.0
R_CHAIN = 5.7e-3        # K/W oil->water chain, whole HX (app: R_ot+R_wall+R_in)
R_CELLFILM_TOT = 2.9e-3 # K/W all cells in parallel (app R_b)
U_TS, H_LOOP = 4.0e-3, 0.06
RHO_O, CP_O = 900.0, 2000.0
H_EXT, A_END, T_AMB = 5.0, 0.9*0.135, 25.0
NB = 48

def solve(D_fac=1.0, counterflow=False, nb=NB, outer=60):
    dx = LX/nb
    q_bin = Q_TOT/nb
    UA_seg_tube = 1.0/(R_CHAIN*nb*N_TUBES)
    mdot_t = MDOT_TOT/N_TUBES
    eps = 1.0-np.exp(-UA_seg_tube/(mdot_t*CP_W))
    K_t = mdot_t*CP_W*eps                      # per tube per segment [W/K]
    D_eff = D_fac*U_TS*H_LOOP
    A_x = 0.5*LY*FILL_H
    G = RHO_O*CP_O*D_eff*A_x/dx
    dirs = [(-1 if (counterflow and k % 2) else 1) for k in range(N_TUBES)]
    T_o = np.full(nb, T_W_IN+8.0)
    Tw_in = {d: np.full(nb, T_W_IN) for d in set(dirs)}
    for _ in range(outer):
        # linear tridiagonal balance given current water-inlet profiles
        A = np.zeros((nb, nb)); rhs = np.full(nb, q_bin)
        for b in range(nb):
            diag = 0.0
            if b > 0:   A[b, b-1] -= G; diag += G
            if b < nb-1: A[b, b+1] -= G; diag += G
            n_dir = {d: dirs.count(d) for d in set(dirs)}
            for d, cnt in n_dir.items():
                diag += cnt*K_t
                rhs[b] += cnt*K_t*Tw_in[d][b]
            if b in (0, nb-1):
                diag += H_EXT*A_END
                rhs[b] += H_EXT*A_END*T_AMB
            A[b, b] += diag
        T_new = np.linalg.solve(A, rhs)
        if np.max(np.abs(T_new-T_o)) < 1e-9:
            T_o = T_new; break
        T_o = T_new
        # re-march water inlets against the new oil field
        for d in set(dirs):
            Tw = T_W_IN
            rng = range(nb) if d == 1 else range(nb-1, -1, -1)
            for b in rng:
                Tw_in[d][b] = Tw
                Tw += K_t*(T_o[b]-Tw)/(mdot_t*CP_W)
    # per-bin extraction for closure
    Q_w = np.zeros(nb)
    for d in dirs:
        for b in (range(nb) if d == 1 else range(nb-1, -1, -1)):
            Q_w[b] += K_t*(T_o[b]-Tw_in[d][b])
    dT_film = Q_TOT*R_CELLFILM_TOT
    T_c = T_o + dT_film
    Q_out = Q_w.sum() + H_EXT*A_END*((T_o[0]-T_AMB)+(T_o[-1]-T_AMB))
    return dict(x=(np.arange(nb)+0.5)*dx, T_o=T_o, T_c=T_c,
                spread=float(T_c.max()-T_c.min()),
                closure=float(Q_out/Q_TOT-1.0),
                T_w_out_rise=Q_TOT/(MDOT_TOT*CP_W))

def validate():
    print("="*72)
    print("VALIDATION: mixing limit and energy closure")
    r = solve(D_fac=500.0)
    print(f"  D_eff x500: oil spread = {r['T_o'].max()-r['T_o'].min():.3f} K "
          f"(-> ~0 expected), energy closure {100*r['closure']:+.2f}%")
    r1 = solve(D_fac=1.0)
    print(f"  D_eff x1  : energy closure {100*r1['closure']:+.2f}%")
    assert abs(r["closure"]) < 2e-3 and abs(r1["closure"]) < 2e-3
    assert r["T_o"].max()-r["T_o"].min() < 0.05

def study():
    print("="*72)
    print(f"PLAN-DIRECTION SPREAD (app's water-rise half = "
          f"{Q_TOT/(MDOT_TOT*CP_W):.2f} K; app adds stratification 0.9 K "
          f"-> total 4.2 K)")
    rows = []
    for cf in (False, True):
        for fac in (0.2, 1.0, 5.0):
            r = solve(D_fac=fac, counterflow=cf)
            rows.append((cf, fac, r))
            print(f"  {'counterflow' if cf else 'co-flow    '} D_eff x{fac:<3}: "
                  f"cell spread = {r['spread']:.2f} K  "
                  f"(oil {r['T_o'].min():.1f}..{r['T_o'].max():.1f} degC)")
    # figure: profiles + plan heatmap for the nominal case
    r = solve(D_fac=1.0)
    rc = solve(D_fac=1.0, counterflow=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(r["x"]*1000, r["T_c"], color="#C0392B", label="cells, co-flow")
    ax[0].plot(rc["x"]*1000, rc["T_c"], color="#2E7D52",
               label="cells, counterflow")
    ax[0].plot(r["x"]*1000, r["T_o"], "--", color="#E8A13A", label="oil, co-flow")
    ax[0].set_xlabel("x along tubes [mm]"); ax[0].set_ylabel("T [degC]")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Plan-direction temperature profiles at 2C")
    ny = 24
    plan = np.tile(r["T_c"], (ny, 1))
    im = ax[1].imshow(plan, origin="lower", aspect="auto", cmap="RdYlBu_r",
                      extent=[0, LX*1000, 0, LY*1000])
    fig.colorbar(im, ax=ax[1], label="cell T [degC]")
    ax[1].set_title("Cell temperature map (co-flow, D_eff x1)")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("y [mm]")
    plt.tight_layout(); plt.savefig("pack_spread.png", dpi=130)
    print("saved pack_spread.png")
    return rows

if __name__ == "__main__":
    validate()
    study()
