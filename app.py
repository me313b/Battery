# =====================================================================
#  Immersion Pack Lab
#  Static / stirred immersion-cooled 21700 battery pack with internal
#  water-cooled tube heat exchanger (ICDC architecture).
#
#  Physics anchored to:
#   [1] Wang, Zhao, Wang & Huang (2023) "Heat transfer characteristics and
#       influencing factors of immersion coupled direct cooling for battery
#       thermal management", J. Energy Storage 62, 106821.
#   [2] batterydesign.net, "Mercedes AMG HPB80" (Nov 2023) - production
#       reference for a dielectric-cooled 21700 pack.
#   [3] Coolant property table: coolant_comparison_reviewed.xlsx (mb, 2026).
#  Correlations: Churchill-Chu (vertical plate, horizontal cylinder),
#  Churchill-Bernstein (crossflow), Hausen (laminar entry), Gnielinski
#  (turbulent tube), Schmidt (annular fin efficiency).
#
#  Run:  streamlit run app.py       Smoke test:  SMOKE=1 python app.py
# =====================================================================

import os, math, contextlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

pio.templates["packlab"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, 'Segoe UI', Roboto, sans-serif",
              size=13, color="#334155"),
    title=dict(font=dict(size=15, color="#0F172A")),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#6366F1", "#06B6D4", "#F59E0B", "#EF4444", "#10B981",
              "#8B5CF6", "#64748B"],
    xaxis=dict(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=20, r=20, t=48, b=20)))
pio.templates.default = "packlab" 

G = 9.81
T_REF = 25.0          # deg C reference for tabulated properties
KELVIN = 273.15

# ------------------------------------------------------------------ #
#  Fluid properties                                                   #
# ------------------------------------------------------------------ #
FALLBACK_CSV = """name,family,k,rho,cp,nu_cSt,beta,B_visc,bp_C,flash_C,dielectric,bdv_kV,notes,review
Transformer oil,Mineral hydrocarbon,0.13,875,1900,9.8,0.00075,3200,280,150,True,50,Baseline mineral oil,
MIVOLT DF7,Dielectric ester,0.13,900,2000,7.0,0.00075,3200,250,170,True,50,Low-viscosity EV ester,
Novec 7100 (HFE-7100),Hydrofluoroether,0.069,1510,1180,0.38,0.0015,1500,61,,True,28,Fluorinated reference,
Deionized water,Water,0.6,997,4180,0.89,0.00026,1900,100,,False,,Thermal reference only,
"""

def _read_coolants() -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "coolants.csv")
    try:
        df = pd.read_csv(path)
    except Exception:
        from io import StringIO
        df = pd.read_csv(StringIO(FALLBACK_CSV))
    df["dielectric"] = df["dielectric"].astype(bool)
    for c in ["k", "rho", "cp", "nu_cSt", "beta", "B_visc", "bp_C", "flash_C"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fluid_dict(row) -> dict:
    return dict(name=row["name"], family=row["family"], k=float(row["k"]),
                rho=float(row["rho"]), cp=float(row["cp"]),
                nu25=float(row["nu_cSt"]) * 1e-6, beta=float(row["beta"]),
                B=float(row["B_visc"]),
                bp=row.get("bp_C", np.nan), flash=row.get("flash_C", np.nan),
                dielectric=bool(row["dielectric"]))

def nu_of_T(fl: dict, T_C: float) -> float:
    """Andrade-type viscosity-temperature law anchored at 25 °C.
    nu(T) = nu25 * exp(B*(1/T - 1/298.15)), T in K. B set per fluid family
    (mineral/ester ~3200 K fitted to transformer-oil data in [1] Table 3)."""
    T = max(T_C, -30.0) + KELVIN
    return fl["nu25"] * math.exp(fl["B"] * (1.0 / T - 1.0 / (T_REF + KELVIN)))

def film_props(fl: dict, T_film_C: float) -> dict:
    """Constant k, rho, cp from the table; nu evaluated at film temperature."""
    nu = nu_of_T(fl, T_film_C)
    alpha = fl["k"] / (fl["rho"] * fl["cp"])
    return dict(k=fl["k"], rho=fl["rho"], cp=fl["cp"], nu=nu,
                alpha=alpha, Pr=nu / alpha, beta=fl["beta"])

# Water-loop fluids (inside the tubes), properties near 20-25 °C
WATER_LOOP = {
    "Water": dict(rho=998.0, cp=4182.0, k=0.60, mu=1.0e-3),
    "Water-glycol 50/50": dict(rho=1070.0, cp=3300.0, k=0.37, mu=3.8e-3),
}

# ------------------------------------------------------------------ #
#  Convection correlations                                            #
# ------------------------------------------------------------------ #
def rayleigh(p: dict, dT: float, L: float) -> float:
    dT = max(abs(dT), 0.05)
    return G * p["beta"] * dT * L ** 3 / (p["nu"] * p["alpha"])

def nu_vertical_cc(Ra: float, Pr: float) -> float:
    """Churchill-Chu, vertical plate, all Ra (used for the cell wall,
    slightly conservative for a slender cylinder in oil)."""
    f = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    return (0.825 + 0.387 * Ra ** (1.0 / 6.0) / f) ** 2

def nu_horiz_cyl_cc(Ra: float, Pr: float) -> float:
    """Churchill-Chu, horizontal cylinder (the HX tubes)."""
    f = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    return (0.60 + 0.387 * Ra ** (1.0 / 6.0) / f) ** 2

def nu_crossflow_cb(Re: float, Pr: float) -> float:
    """Churchill-Bernstein, forced crossflow over a cylinder."""
    if Re < 1e-6:
        return 0.0
    a = 0.62 * Re ** 0.5 * Pr ** (1.0 / 3.0)
    b = (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25
    c = (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return 0.3 + a / b * c

def blend_mixed(h_nat: float, h_for: float) -> float:
    """Mixed convection blend Nu^3 = Nu_n^3 + Nu_f^3 (transverse flow)."""
    return (h_nat ** 3 + h_for ** 3) ** (1.0 / 3.0)

def gap_factor(gap_mm: float, expo: float = 0.6, floor: float = 0.35) -> float:
    """Confinement penalty on cell-side natural convection.
    Calibrated to [1] Fig. 9: full performance for gap >= 6 mm, degrading
    below (gap velocity fell 1.8 -> 0.5 mm/s from 8 -> 2 mm spacing,
    i.e. h roughly halved at 2 mm)."""
    if gap_mm >= 6.0:
        return 1.0
    return max(floor, (max(gap_mm, 0.3) / 6.0) ** expo)

def h_cell_side(fl, T_s, T_bulk, H_cell, D_cell, gap_mm, u_oil) -> dict:
    p = film_props(fl, 0.5 * (T_s + T_bulk))
    Ra = rayleigh(p, T_s - T_bulk, H_cell)
    Nun = nu_vertical_cc(Ra, p["Pr"]) * gap_factor(gap_mm)
    h_n = Nun * p["k"] / H_cell
    h_f = 0.0
    Re = u_oil * D_cell / p["nu"]
    if u_oil > 1e-6:
        h_f = nu_crossflow_cb(Re, p["Pr"]) * p["k"] / D_cell
    return dict(h=blend_mixed(h_n, h_f), h_nat=h_n, h_for=h_f,
                Ra=Ra, Re=Re, Pr=p["Pr"])

def h_tube_side(fl, T_bulk, T_wall, D_o, u_oil) -> dict:
    p = film_props(fl, 0.5 * (T_bulk + T_wall))
    Ra = rayleigh(p, T_bulk - T_wall, D_o)
    h_n = nu_horiz_cyl_cc(Ra, p["Pr"]) * p["k"] / D_o
    h_f = 0.0
    Re = u_oil * D_o / p["nu"]
    if u_oil > 1e-6:
        h_f = nu_crossflow_cb(Re, p["Pr"]) * p["k"] / D_o
    return dict(h=blend_mixed(h_n, h_f), h_nat=h_n, h_for=h_f,
                Ra=Ra, Re=Re, Pr=p["Pr"])

def h_water_inside(loop: dict, mdot_tube: float, d_i: float, L: float) -> dict:
    """Hausen (laminar, entry-corrected) / Gnielinski (turbulent) with a
    linear bridge across the 2300-3000 transition."""
    mu, k, cp = loop["mu"], loop["k"], loop["cp"]
    Pr = mu * cp / k
    Re = 4.0 * mdot_tube / (math.pi * mu * d_i) if mdot_tube > 0 else 0.0
    def nu_lam(Re_):
        gz = (d_i / L) * Re_ * Pr
        return 3.66 + 0.0668 * gz / (1.0 + 0.04 * gz ** (2.0 / 3.0))
    def nu_turb(Re_):
        f = (0.790 * math.log(Re_) - 1.64) ** -2
        return (f / 8.0) * (Re_ - 1000.0) * Pr / (
            1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
    if Re <= 0:
        Nu, regime = 3.66, "no flow"
    elif Re < 2300:
        Nu, regime = nu_lam(Re), "laminar"
    elif Re < 3000:
        w = (Re - 2300.0) / 700.0
        Nu = (1 - w) * nu_lam(2300) + w * nu_turb(3000)
        regime = "transitional"
    else:
        Nu, regime = nu_turb(Re), "turbulent"
    return dict(h=Nu * k / d_i, Re=Re, Pr=Pr, Nu=Nu, regime=regime)

# ------------------------------------------------------------------ #
#  Annular fins (Schmidt approximation)                               #
# ------------------------------------------------------------------ #
def fin_pack(d_o, H_f, t_f, p_f, k_fin, h_oil) -> dict:
    """Per metre of finned tube: bare area, fin area, Schmidt efficiency."""
    r1 = d_o / 2.0
    r2 = r1 + H_f
    r2c = r2 + t_f / 2.0
    n_per_m = 1.0 / p_f
    A_bare = math.pi * d_o * max(0.0, 1.0 - t_f / p_f)
    A_fin_each = 2.0 * math.pi * (r2c ** 2 - r1 ** 2)
    A_fin = n_per_m * A_fin_each
    m = math.sqrt(2.0 * max(h_oil, 1.0) / (k_fin * t_f))
    Lc = H_f + t_f / 2.0
    phi = 1.0 + 0.35 * math.log(r2c / r1)
    x = m * Lc * phi
    eta = math.tanh(x) / x if x > 1e-9 else 1.0
    A_eff = A_bare + eta * A_fin
    m_per_m = n_per_m * (math.pi * (r2 ** 2 - r1 ** 2) * t_f)  # fin metal volume/m
    return dict(A_eff_per_m=A_eff, A_bare_per_m=A_bare, A_fin_per_m=A_fin,
                eta=eta, area_gain=A_eff / (math.pi * d_o),
                fin_metal_vol_per_m=m_per_m, fin_gap=p_f - t_f)

# ------------------------------------------------------------------ #
#  Geometry and mass build-up                                         #
# ------------------------------------------------------------------ #
K_TUBE = {"Copper": 385.0, "Aluminium": 205.0, "Stainless steel": 16.0}
RHO_TUBE = {"Copper": 8940.0, "Aluminium": 2700.0, "Stainless steel": 7900.0}

def build_geometry(d: dict) -> dict:
    N = d["Ns"] * d["Np"]
    D, H, p = d["d_cell"], d["h_cell"], d["pitch"]
    gap = (p - D) * 1000.0                                   # mm
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)
    row_pitch = p * (math.sqrt(3) / 2 if d["arrangement"] == "Hexagonal" else 1.0)
    edge = d["edge_margin"]
    Lx = n_cols * p + 2 * edge
    Ly = (n_rows - 1) * row_pitch + p + 2 * edge
    Lz = d["bottom_gap"] + H + d["tube_zone"] + d["gas_gap"]
    fill_h = Lz - d["gas_gap"]

    # heat exchanger tubes (horizontal, in the tube zone above the cells)
    d_o, t_w = d["tube_od"], d["tube_wall"]
    d_i = max(d_o - 2 * t_w, 1e-3)
    L_tube = max(Lx - 2 * d["manifold_margin"], 0.05) * d["passes"]
    A_tube_bare = math.pi * d_o * L_tube * d["n_tubes"]
    A_tube_in = math.pi * d_i * L_tube * d["n_tubes"]

    # areas and volumes
    f_ends = d["end_fraction"]
    A_cells = N * (math.pi * D * H + f_ends * 2 * math.pi * D ** 2 / 4)
    V_box_fill = Lx * Ly * fill_h
    V_cells = N * math.pi * D ** 2 / 4 * H
    V_tubes = d["n_tubes"] * L_tube * math.pi * d_o ** 2 / 4
    A_box_ext = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)

    free_per_cell = max(p * row_pitch - math.pi * D ** 2 / 4, 1e-6)
    blk = 1.0 - d.get("holder_block", 0.0)           # cell-holder blockage
    A_flow = N * free_per_cell * blk                 # riser flow area, m²
    D_h = 4.0 * free_per_cell / (math.pi * D) * math.sqrt(max(blk, 0.05))

    return dict(N=N, gap_mm=gap, n_cols=n_cols, n_rows=n_rows,
                A_flow=A_flow, D_h=D_h, row_pitch=row_pitch,
                Lx=Lx, Ly=Ly, Lz=Lz, fill_h=fill_h,
                d_i=d_i, L_tube=L_tube, A_tube_bare=A_tube_bare,
                A_tube_in=A_tube_in, A_cells=A_cells,
                V_box_fill=V_box_fill, V_cells=V_cells, V_tubes=V_tubes,
                A_box_ext=A_box_ext)

def enclosure_calc(d, g):
    """Wall thickness of the largest flat panel as a stiffened plate under
    the burst-disc set pressure (dominates over oil static head), then mass
    from total surface area. sigma_allow and the stiffening knock-down are
    sliders; override entirely with struct_mass > 0."""
    p_des = max(d["p_des_bar"] * 1e5, 900.0 * G * g["fill_h"])
    b = min(g["Lx"], g["Ly"])
    t_flat = b * math.sqrt(0.31 * p_des / (d["sigma_MPa"] * 1e6))
    t_eff = max(t_flat * d["stiff"], 0.0015)
    m = g["A_box_ext"] * t_eff * 2700.0 * 1.18      # + fasteners, feedthroughs
    return dict(t_mm=t_eff * 1000, m=m, p_des_bar=p_des / 1e5)

def build_masses(d, g, fl, finres) -> dict:
    V_fins = finres["fin_metal_vol_per_m"] * g["L_tube"] * d["n_tubes"] if d["fins_on"] else 0.0
    V_oil = max(g["V_box_fill"] - g["V_cells"] - g["V_tubes"] - V_fins, 1e-4)
    m_oil = V_oil * fl["rho"]
    m_cells = g["N"] * d["m_cell"]
    rho_t = RHO_TUBE[d["tube_mat"]]
    m_tubes = rho_t * d["n_tubes"] * g["L_tube"] * math.pi / 4 * (d["tube_od"] ** 2 - g["d_i"] ** 2)
    m_fins = V_fins * (2700.0 if d["fin_mat"] == "Aluminium" else 8940.0)
    enc = enclosure_calc(d, g)
    m_struct = d["struct_mass"] if d["struct_mass"] > 0 else enc["m"]
    m_holders = g["N"] * d["m_holder_g"] / 1000.0
    m_bus = busbar_props(d, g)["m"]
    m_plates = plate_fin_area(d, g, 100.0)[2] if d.get("plate_on") else 0.0
    m_pack = m_cells + m_oil + m_tubes + m_fins + m_struct + m_holders + m_bus \
             + m_plates
    E_kwh = d["Ns"] * d["Np"] * d["v_nom"] * d["cap_Ah"] / 1000.0
    V_outer_L = (g["Lx"] + 2 * enc["t_mm"] / 1000) * (g["Ly"] + 2 * enc["t_mm"] / 1000) \
                * (g["Lz"] + 2 * enc["t_mm"] / 1000) * 1000
    return dict(V_oil_L=V_oil * 1000, m_oil=m_oil, m_cells=m_cells,
                m_tubes=m_tubes, m_fins=m_fins, m_struct=m_struct,
                m_holders=m_holders, m_bus=m_bus, m_plates=m_plates, enc=enc,
                m_pack=m_pack, E_kwh=E_kwh, V_outer_L=V_outer_L,
                whkg_pack=E_kwh * 1000 / m_pack,
                whkg_cells=E_kwh * 1000 / m_cells,
                whl_pack=E_kwh * 1000 / max(V_outer_L, 1.0),
                C_batt=m_cells * d["cp_cell"],
                C_oil=m_oil * fl["cp"])

# ------------------------------------------------------------------ #
#  Steady-state network solver                                        #
# ------------------------------------------------------------------ #
def solve_steady(d, g, fl, Q_total, T_amb, C_rate=None) -> dict:
    """Two-node (battery, oil) network with h(dT) fixed-point iteration.
    Chain: cells -> oil film -> bulk oil -> oil film on tubes (finned) ->
    tube wall -> water film -> water; parallel leak oil -> ambient.
    If C_rate is given, heat generation is recomputed from DCIR(T) each
    iteration; otherwise Q_total is a fixed heater power (benchmark mode).
    Oil-film h values carry the calibration factor d['cal_h']."""
    loop = WATER_LOOP[d["loop_fluid"]]
    mdot_tot = d["flow_lpm"] / 60.0 * loop["rho"] / 1000.0
    mdot_tube = mdot_tot / max(d["n_tubes"], 1)
    wat = h_water_inside(loop, mdot_tube, g["d_i"], g["L_tube"])
    R_in = 1.0 / max(wat["h"] * g["A_tube_in"], 1e-9)
    R_wall = math.log(d["tube_od"] / g["d_i"]) / (
        2 * math.pi * K_TUBE[d["tube_mat"]] * g["L_tube"] * d["n_tubes"])
    R_atm = 1.0 / max(d["h_ext"] * g["A_box_ext"], 1e-9)

    # initial guesses
    T_w_in = d["T_water_in"]
    T_il = T_w_in + 8.0
    T_b = T_il + 6.0
    T_wall = T_w_in + 2.0
    cell = tube = fin = None
    ch = d.get("cal_h", 1.0)
    u_ts, dT_loop = 0.0, 0.0
    for _ in range(60):
        if C_rate is not None:
            I = C_rate * d["cap_Ah"]
            Q_total = g["N"] * I * I * r_of_T(d, T_b) * 1e-3
            Q_total += (I * d["Np"]) ** 2 * busbar_props(d, g)["R"]
        u_ts, dT_loop = thermosiphon_u(d, g, fl, max(Q_total, 1.0),
                                       0.5 * (T_b + T_il))
        u_eff = max(d["u_oil"], u_ts)
        cell = h_cell_side(fl, T_b, T_il, d["h_cell"], d["d_cell"], g["gap_mm"], u_eff)
        tube0 = h_tube_side(fl, T_il, T_wall, d["tube_od"], u_eff)
        for hh in (cell, tube0):
            hh["h"] *= ch; hh["h_nat"] *= ch; hh["h_for"] *= ch
        if d["fins_on"]:
            fin = fin_pack(d["tube_od"], d["fin_h"], d["fin_t"], d["fin_p"],
                           205.0 if d["fin_mat"] == "Aluminium" else 385.0, tube0["h"])
            A_oilside = fin["A_eff_per_m"] * g["L_tube"] * d["n_tubes"]
            A_pl, eta_pl, _mpl = plate_fin_area(d, g, ch * tube0["h"])
            A_oilside += A_pl
        else:
            fin = dict(A_eff_per_m=math.pi * d["tube_od"], eta=1.0, area_gain=1.0,
                       fin_metal_vol_per_m=0.0, fin_gap=1.0, A_fin_per_m=0.0,
                       A_bare_per_m=math.pi * d["tube_od"])
            A_oilside = g["A_tube_bare"]
            A_pl, eta_pl, _mpl = plate_fin_area(d, g, ch * tube0["h"])
            A_oilside += A_pl
        R_b = 1.0 / max(cell["h"] * g["A_cells"], 1e-9)
        R_ot = 1.0 / max(tube0["h"] * A_oilside, 1e-9)
        R_chain = R_ot + R_wall + R_in

        # split heat between water chain and ambient leak
        dT_w_rise = Q_total / max(mdot_tot * loop["cp"], 1e-9)
        T_sink = T_w_in + 0.5 * min(dT_w_rise, 60.0)
        T_il_new = (Q_total + T_sink / R_chain + T_amb / R_atm) / (1.0 / R_chain + 1.0 / R_atm)
        Q_w = (T_il_new - T_sink) / R_chain
        dT_w_rise = max(Q_w, 0.0) / max(mdot_tot * loop["cp"], 1e-9)
        T_wall_new = T_sink + Q_w * (R_in + R_wall)
        T_b_new = T_il_new + Q_total * R_b
        # relax
        T_il += 0.6 * (T_il_new - T_il)
        T_b += 0.6 * (T_b_new - T_b)
        T_wall += 0.6 * (T_wall_new - T_wall)
        tube = tube0
    Q_w = (T_il - (T_w_in + 0.5 * Q_total / max(mdot_tot * loop["cp"], 1e-9))) / R_chain
    Q_atm = (T_il - T_amb) / R_atm
    Q_cell = Q_total / g["N"]
    T_core = T_b + Q_cell * r_core(d)
    dT_water = Q_total / max(mdot_tot * loop["cp"], 1e-9)
    if d.get("tube_plane") == "Interstitial (between rows)" or d.get("plate_on"):
        dT_loop *= 0.35                          # distributed sinks
    dT_pos = 0.5 * dT_water + 0.5 * dT_loop     # worst-position penalty
    return dict(T_b=T_b, T_il=T_il, T_wall=T_wall,
                T_core=T_core, dT_core=Q_cell * r_core(d),
                u_ts=u_ts, dT_loop=dT_loop, Q_eff=Q_total,
                T_worst=T_b + dT_pos, T_best=T_b - dT_pos, spread=2 * dT_pos,
                R_b=R_b, R_ot=R_ot, R_wall=R_wall, R_in=R_in, R_atm=R_atm,
                h_cell=cell["h"], h_cell_nat=cell["h_nat"], h_cell_for=cell["h_for"],
                Ra_cell=cell["Ra"], Re_cell=cell["Re"],
                h_tube=tube["h"], h_tube_nat=tube["h_nat"], h_tube_for=tube["h_for"],
                Ra_tube=tube["Ra"],
                h_water=wat["h"], Re_water=wat["Re"], water_regime=wat["regime"],
                A_oilside=A_oilside, A_plate=A_pl, eta_plate=eta_pl, fin=fin,
                dT_water=Q_total / max(mdot_tot * loop["cp"], 1e-9),
                mdot_tot=mdot_tot, Q_w=Q_w, Q_atm=Q_atm,
                gapf=gap_factor(g["gap_mm"]))

def r_of_T(d, T_cell_C: float) -> float:
    """Cell DCIR in mΩ at temperature T. Exponential fall with T
    (default -1.2 %/K around 25 °C) - the reason AMG run a 45 °C
    set point. Set k_dcir = 0 to disable the coupling."""
    return d["r_dc"] * math.exp(-d.get("k_dcir", 0.0) * (T_cell_C - 25.0))

def r_core(d) -> float:
    """Peak core-to-surface resistance of a cylindrical cell with uniform
    volumetric generation: dT = q''' R^2/(4 k_r) -> R_th = 1/(4 pi k_r H).
    Radial jellyroll k_r ~ 0.8-1.0 W/mK. Mean-to-surface is half this."""
    return 1.0 / (4.0 * math.pi * d.get("k_rad", 0.9) * d["h_cell"])

def thermosiphon_u(d, g, fl, Q: float, T_oil_C: float):
    """Self-circulation velocity in the cell gaps: buoyant head
    rho*beta*g*H_loop*dT_loop against laminar loop friction plus minor
    losses, with dT_loop = Q/(rho u A cp). Solved by bisection.
    H_loop = vertical offset between cell mid-height and the tube plane,
    so tube placement is a live design variable. Order-of-magnitude
    validation: Wang et al. measured 0.5-1.8 mm/s gap velocities."""
    if d.get("tube_plane", "Top of pack") == "Top of pack":
        H_loop = d["h_cell"] / 2 + d["tube_zone"] / 2
    elif d.get("tube_plane") == "Interstitial (between rows)":
        H_loop = 0.008
    elif d.get("tube_plane") == "Mid-height":
        H_loop = 0.01
    else:                                   # below the cells
        return 1e-5, Q / max(fl["rho"] * 1e-5 * g["A_flow"] * fl["cp"], 1e-9)
    p = film_props(fl, T_oil_C)
    A, Dh = g["A_flow"], g["D_h"]
    L_loop = 2.2 * g["fill_h"]
    K = d.get("K_loop", 5.0)
    def resid(u):
        drive = p["beta"] * G * H_loop * Q / (u * A * p["cp"])
        fric = 32.0 * p["rho"] * p["nu"] * L_loop * u / Dh ** 2 \
               + K * 0.5 * p["rho"] * u ** 2
        return drive - fric
    lo, hi = 1e-6, 0.08
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            lo = mid
        else:
            hi = mid
    u = lo
    dT_loop = Q / (p["rho"] * u * A * p["cp"])
    return u, min(dT_loop, 60.0)

def q_gen_per_cell(d, C_rate, T_cell_C: float = 25.0) -> float:
    I = C_rate * d["cap_Ah"]
    return I * I * r_of_T(d, T_cell_C) * 1e-3   # DCIR in mΩ

def max_continuous_C(d, g, fl, T_amb, T_limit) -> float:
    lo, hi = 0.05, 12.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        res = solve_steady(d, g, fl, 1.0, T_amb, C_rate=mid)
        T_check = res["T_core"] if d.get("limit_core", False) else res["T_b"]
        if T_check > T_limit:
            hi = mid
        else:
            lo = mid
    return lo

# ------------------------------------------------------------------ #
#  Transient solver                                                   #
# ------------------------------------------------------------------ #
def duty_profile(kind, dur, C1, t1, C2, t2, csv_tc=None):
    """Return times [s] and C-rate arrays. csv_tc = (t, C) from an upload."""
    t = np.arange(0.0, dur + 1e-9, 2.0)
    if kind == "CSV upload" and csv_tc is not None:
        t = np.arange(0.0, min(dur, csv_tc[0][-1]) + 1e-9, 2.0)
        C = np.interp(t, csv_tc[0], csv_tc[1])
    elif kind == "Constant C":
        C = np.full_like(t, C1)
    elif kind == "Fast charge then rest":
        C = np.where(t < t1, C1, 0.0)
    else:  # pulse train
        period = max(t1 + t2, 1.0)
        C = np.where((t % period) < t1, C1, C2)
    return t, C

def duty_from_csv(file, d) -> tuple:
    """Parse a duty CSV with columns t_s plus either C or P_kW."""
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    t = df[cols["t_s"]].to_numpy(dtype=float)
    if "c" in cols:
        C = df[cols["c"]].to_numpy(dtype=float)
    elif "p_kw" in cols:
        P = df[cols["p_kw"]].to_numpy(dtype=float) * 1000.0
        C = P / (d["Ns"] * d["Np"] * d["v_nom"] * d["cap_Ah"])
    else:
        raise ValueError("CSV needs columns t_s and C (or P_kW)")
    order = np.argsort(t)
    return np.abs(t[order]), np.abs(C[order])

def solve_transient(d, g, fl, masses, T_amb, t_arr, C_arr) -> dict:
    loop = WATER_LOOP[d["loop_fluid"]]
    mdot = d["flow_lpm"] / 60.0 * loop["rho"] / 1000.0
    C_b, C_il = masses["C_batt"], masses["C_oil"]
    T_b = np.zeros_like(t_arr); T_il = np.zeros_like(t_arr)
    T_b[0] = T_il[0] = d["T_start"]
    # freeze wall/water resistances from a representative solve, update oil films each step
    rep = solve_steady(d, g, fl, 1.0, T_amb, C_rate=max(C_arr.max(), 0.5))
    R_wall, R_in, R_atm = rep["R_wall"], rep["R_in"], rep["R_atm"]
    ch = d.get("cal_h", 1.0)
    T_core = np.zeros_like(t_arr); T_core[0] = d["T_start"]
    Q_tr = np.zeros_like(t_arr)
    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        Q = q_gen_per_cell(d, C_arr[i - 1], T_b[i - 1]) * g["N"]
        u_ts, _ = thermosiphon_u(d, g, fl, max(Q, 1.0), 0.5 * (T_b[i-1] + T_il[i-1]))
        u_eff = max(d["u_oil"], u_ts)
        cell = h_cell_side(fl, T_b[i-1], T_il[i-1], d["h_cell"], d["d_cell"], g["gap_mm"], u_eff)
        T_wall_est = T_il[i-1] - 0.6 * (T_il[i-1] - d["T_water_in"])
        tub = h_tube_side(fl, T_il[i-1], T_wall_est, d["tube_od"], u_eff)
        R_b = 1.0 / max(ch * cell["h"] * g["A_cells"], 1e-9)
        R_ot = 1.0 / max(ch * tub["h"] * rep["A_oilside"], 1e-9)
        Q_bi = (T_b[i-1] - T_il[i-1]) / R_b
        Q_w = (T_il[i-1] - (d["T_water_in"] + 0.5 * max(Q_bi, 0) / max(mdot * loop["cp"], 1e-9))) / (R_ot + R_wall + R_in)
        Q_a = (T_il[i-1] - T_amb) / R_atm
        T_b[i] = T_b[i-1] + dt * (Q - Q_bi) / C_b
        T_il[i] = T_il[i-1] + dt * (Q_bi - Q_w - Q_a) / C_il
        T_core[i] = T_b[i] + (Q / g["N"]) * r_core(d)   # quasi-steady radial
        Q_tr[i] = Q
    return dict(t=t_arr, T_b=T_b, T_il=T_il, T_core=T_core, Q=Q_tr)

# ------------------------------------------------------------------ #
#  Benchmark against Wang et al. (2023)                               #
# ------------------------------------------------------------------ #
def benchmark_wang() -> dict:
    """Rebuild the paper's rig with this app's correlations and compare with
    their measured/derived values (Table 6, Figs 5-8). Prismatic cells, so
    the vertical-plate correlation applies directly."""
    oil = dict(name="Transformer oil (paper Table 3)", family="mineral",
               k=0.13, rho=875.0, cp=1900.0, nu25=17.0e-6, beta=7.5e-4,
               B=3900.0, dielectric=True, bp=280, flash=150)
    A_cells = 6 * (2 * (0.148 * 0.097) + 2 * (0.148 * 0.027) + 2 * (0.097 * 0.027))
    N_t, d_o, d_i, L_t = 4, 0.006, 0.005, 0.222
    A_ot = math.pi * d_o * L_t * N_t          # 0.0167 m2, matches paper
    A_in = math.pi * d_i * L_t * N_t
    loop = dict(rho=1000.0, cp=4200.0, k=0.58, mu=1.3e-3)  # water ~8 °C
    mdot = 17.1e-6 * 1000.0
    wat = h_water_inside(loop, mdot / N_t, d_i, L_t)
    R_in = 1.0 / (wat["h"] * A_in)
    R_wall = math.log(d_o / d_i) / (2 * math.pi * 385.0 * L_t * N_t)
    R_atm_meas = 6.7                           # take the paper's measured value
    C_b, C_il = 7620.0, 9184.0
    Q = 16.4 * 6                               # W, model calibration Table 5 at 2C
    T_amb, T_w = 25.0, 5.0
    t = np.arange(0, 1801.0, 2.0)
    T_b = np.full_like(t, 24.5); T_il = np.full_like(t, 24.5)
    hb_last = ht_last = 0.0
    for i in range(1, len(t)):
        pf = film_props(oil, 0.5 * (T_b[i-1] + T_il[i-1]))
        Ra = rayleigh(pf, T_b[i-1] - T_il[i-1], 0.148)
        hb = nu_vertical_cc(Ra, pf["Pr"]) * pf["k"] / 0.148
        Twall = T_w + 3.0
        pt = film_props(oil, 0.5 * (T_il[i-1] + Twall))
        Rad = rayleigh(pt, T_il[i-1] - Twall, d_o)
        ht = nu_horiz_cyl_cc(Rad, pt["Pr"]) * pt["k"] / d_o
        R_b = 1.0 / (hb * A_cells); R_ot = 1.0 / (ht * A_ot)
        Q_bi = (T_b[i-1] - T_il[i-1]) / R_b
        Q_w = (T_il[i-1] - (T_w + 0.5 * max(Q_bi, 0) / (mdot * loop["cp"]))) / (R_ot + R_wall + R_in)
        Q_a = (T_il[i-1] - T_amb) / R_atm_meas
        T_b[i] = T_b[i-1] + 2.0 * (Q - Q_bi) / C_b
        T_il[i] = T_il[i-1] + 2.0 * (Q_bi - Q_w - Q_a) / C_il
        hb_last, ht_last = hb, ht
    rows = [
        ("h battery-to-oil  [W/m²·K]", f"{hb_last:.0f}", "~80-110 (Fig. 11b, measured)"),
        ("h oil-to-tube  [W/m²·K]", f"{ht_last:.0f}", "~170-270 (Fig. 11b, measured)"),
        ("R battery-oil  [K/W]", f"{1.0/(hb_last*A_cells):.3f}", "0.04 (Table 6)"),
        ("R oil-tube  [K/W]", f"{1.0/(ht_last*A_ot):.2f}", "0.30 (Table 6)"),
        ("R water film  [K/W]", f"{R_in:.3f}", "0.06 (Table 6)"),
        ("R tube wall  [K/W]", f"{R_wall:.5f}", "0.0005 (Table 6)"),
        ("T battery at 1800 s  [°C]", f"{T_b[-1]:.1f}", "~32.3 (Fig. 5/6, measured)"),
        ("T oil at 1800 s  [°C]", f"{T_il[-1]:.1f}", "~28.5 (Fig. 8b)"),
        ("dT battery-oil plateau [K]", f"{T_b[-1]-T_il[-1]:.1f}", "~3.2 (Fig. 8b)"),
    ]
    return dict(rows=rows, t=t, T_b=T_b, T_il=T_il)

# ------------------------------------------------------------------ #
#  Advice engine and sensitivity study                                #
# ------------------------------------------------------------------ #
def diagnose(d, g, fl, masses, res, T_limit, Q) -> list:
    msgs = []
    chain = {"cell-to-oil film": res["R_b"], "oil-to-tube film (finned)": res["R_ot"],
             "tube wall": res["R_wall"], "water film inside tubes": res["R_in"]}
    worst = max(chain, key=chain.get)
    tot = sum(chain.values())
    msgs.append(("info", f"**Bottleneck: {worst}** carries {100*chain[worst]/tot:.0f}% of the "
                 f"cell-to-water resistance ({chain[worst]*1000:.1f} mK/W of {tot*1000:.1f} mK/W)."))
    if worst == "cell-to-oil film":
        msgs.append(("do", "Cell film dominates: add gentle stirring (a few cm/s), widen the "
                     "cell gap towards 6 mm, or switch to a lower-viscosity dielectric. "
                     "Fins on the tubes will NOT help while this film dominates."))
    if worst == "oil-to-tube film (finned)":
        msgs.append(("do", "Tube-side film dominates: add/extend fins, add tubes, or stir. "
                     "This is area-starved, exactly as in Wang et al. (R = 0.3 K/W there)."))
    if worst == "water film inside tubes":
        msgs.append(("do", f"Water film dominates and flow is **{res['water_regime']}** "
                     f"(Re = {res['Re_water']:.0f}). Raise flow past Re 3000, or use more "
                     "smaller tubes in parallel; in laminar flow extra velocity does nothing."))
    msgs.append(("info", f"Predicted self-circulation (thermosiphon): "
                 f"**{res['u_ts']*1000:.1f} mm/s** in the cell gaps (Wang et al. measured "
                 f"0.5-1.8 mm/s), giving a top-to-bottom oil stratification of "
                 f"~{res['dT_loop']:.1f} K. Tube plane: {d.get('tube_plane','Top of pack')}."))
    if d.get("tube_plane") == "Below the cells":
        msgs.append(("bad", "Tubes below the cells: buoyancy stratifies stably, the "
                     "thermosiphon dies, and hot oil strands at the top. The model's "
                     "well-mixed-oil assumption is optimistic here - expect worse."))
    if res["spread"] > 5.0:
        msgs.append(("warn", f"Estimated best-to-worst cell spread ~{res['spread']:.1f} K "
                     "exceeds the 5 K uniformity criterion (water rise + stratification). "
                     "Raise water flow, stir, or split the water loop into counterflowing "
                     "halves."))
    if res["dT_core"] > 3.0:
        msgs.append(("info", f"Core runs ~{res['dT_core']:.1f} K above the can at this duty "
                     f"(k_r = {d.get('k_rad',0.9):.1f} W/mK). No coolant choice touches this "
                     "term; only lower current or tab/axial extraction do."))
    if res["dT_water"] > 5:
        msgs.append(("warn", f"Water heats by {res['dT_water']:.1f} K end to end (> 5 K): last "
                     "cells see a warmer sink. Raise flow or split the loop."))
    if g["gap_mm"] < 6:
        msgs.append(("warn", f"Cell gap {g['gap_mm']:.1f} mm < 6 mm: buoyant flow in the gaps "
                     f"is throttled (penalty factor {res['gapf']:.2f}, per Wang et al. Fig. 9)."))
    if res["T_b"] > T_limit:
        msgs.append(("bad", f"Steady cell temperature {res['T_b']:.1f} °C exceeds the "
                     f"{T_limit:.0f} °C limit at this duty. See sensitivity chart for the "
                     "cheapest fix."))
    if not fl["dielectric"]:
        msgs.append(("bad", f"**{fl['name']} is not a dielectric.** Fine as a thermal reference "
                     "in this model, unusable as an immersion fluid in a live pack."))
    if not math.isnan(fl.get("bp", float("nan"))) and res["T_b"] + 10 > fl["bp"]:
        msgs.append(("warn", f"Cell temperature is within 10 K of the fluid boiling point "
                     f"({fl['bp']:.0f} °C): you are entering two-phase territory (pressure "
                     "management needed)."))
    if not math.isnan(fl.get("flash", float("nan"))) and fl["flash"] < 120:
        msgs.append(("warn", f"Flash point {fl['flash']:.0f} °C is low for a lithium pack; "
                     "prefer > 150 °C (ester class)."))
    if d["fins_on"] and res["fin"]["fin_gap"] < 0.004 and d["u_oil"] < 0.005:
        msgs.append(("warn", "Fin gap < 4 mm with still oil: natural-convection boundary "
                     "layers will merge between fins and the Schmidt-efficiency estimate "
                     "becomes optimistic. Open the fin pitch or stir."))
    if masses["m_oil"] / masses["m_pack"] > 0.30:
        msgs.append(("info", f"Oil is {100*masses['m_oil']/masses['m_pack']:.0f}% of pack mass. "
                     "Reduce headspace/tube-zone height, or accept it as buffer thermal mass."))
    exp_L = fl["beta"] * masses["V_oil_L"] * (d.get("T_service_max", 60) - (-10))
    msgs.append(("info", f"Thermal expansion over -10 to {d.get('T_service_max',60):.0f} °C: "
                 f"~{exp_L:.1f} L. Size a bellows/bladder for this; do not leave a free air "
                 "headspace (moisture, tilt)."))
    msgs.append(("info", "Water-in-oil leak is the single-point failure: keep oil static "
                 "pressure above water pressure, or use double-walled tubes with leak "
                 "detection (transformer practice)."))
    return msgs

def sensitivity(d, g, fl, cool_df, T_amb, C_duty, T_limit) -> pd.DataFrame:
    """One-at-a-time perturbations; report change in steady cell temperature."""
    def run(dd, ff):
        gg = build_geometry(dd)
        return solve_steady(dd, gg, ff, 1.0, T_amb, C_rate=C_duty)["T_b"]
    base = run(d, fl)
    cases = []
    dd = dict(d); dd["n_tubes"] = max(1, int(round(d["n_tubes"] * 1.5))); cases.append(("Tubes +50%", run(dd, fl)))
    dd = dict(d)
    if d["fins_on"]:
        dd["fin_p"] = d["fin_p"] * 2 / 3; cases.append(("Fin area +50%", run(dd, fl)))
    else:
        dd["fins_on"] = True; cases.append(("Add fins (8 mm, 4 mm pitch)", run(dd, fl)))
    dd = dict(d); dd["u_oil"] = d["u_oil"] + 0.05; cases.append(("Stir oil +5 cm/s", run(dd, fl)))
    dd = dict(d); dd["flow_lpm"] = d["flow_lpm"] * 2; cases.append(("Water flow x2", run(dd, fl)))
    dd = dict(d); dd["T_water_in"] = d["T_water_in"] - 5; cases.append(("Water inlet -5 K", run(dd, fl)))
    dd = dict(d); dd["pitch"] = d["pitch"] + 0.002; cases.append(("Cell pitch +2 mm", run(dd, fl)))
    diel = cool_df[cool_df["dielectric"]].copy()
    best = diel.loc[diel["nu_cSt"].idxmin()]
    if best["name"] != fl["name"]:
        cases.append((f"Fluid: {best['name']}", run(d, fluid_dict(best))))
    rows = [dict(change=n, T_b=t, dT=t - base) for n, t in cases]
    return pd.DataFrame(rows).sort_values("dT"), base

# ------------------------------------------------------------------ #
#  UI helpers                                                         #
# ------------------------------------------------------------------ #
ACCENT, INK, PAPER = "#F59E0B", "#0F172A", "#F7F8FA"
BRAND_A, BRAND_B = "#6366F1", "#06B6D4"
CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, .stApp, [class*="css"] {{
      font-family: Inter, -apple-system, 'Segoe UI', Roboto, sans-serif; }}
  .stApp {{ background: {PAPER}; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.02em; font-weight: 700; }}
  /* hero */
  .hero {{ padding: 18px 22px; border-radius: 18px; margin-bottom: 6px;
      background: linear-gradient(120deg, {BRAND_A} 0%, {BRAND_B} 100%);
      color: white; box-shadow: 0 8px 24px rgba(99,102,241,.25); }}
  .hero h1 {{ color: white; margin: 0; font-size: 1.55rem; }}
  .hero p {{ margin: 4px 0 0 0; opacity: .85; font-size: .9rem; }}
  .chip {{ display:inline-block; padding: 3px 12px; border-radius: 999px;
      font-size: .78rem; font-weight: 600; margin-top: 8px;
      background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);}}
  .chip.bad {{ background:#FEE2E2; color:#B91C1C; border-color:#FCA5A5; }}
  .chip.ok  {{ background:#DCFCE7; color:#15803D; border-color:#86EFAC; }}
  /* KPI cards */
  .kpis {{ display:flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 4px 0; }}
  .kpi {{ flex:1 1 140px; background:#FFFFFF; border:1px solid #E7EAF0;
      border-radius: 14px; padding: 12px 14px;
      box-shadow: 0 1px 3px rgba(16,24,40,.05); }}
  .kpi .l {{ font-size:.68rem; font-weight:600; letter-spacing:.06em;
      text-transform: uppercase; color:#64748B; }}
  .kpi .v {{ font-size:1.45rem; font-weight:800; color:{INK};
      margin: 2px 0 0 0; line-height:1.15; }}
  .kpi .v.ok {{ color:#15803D; }} .kpi .v.bad {{ color:#B91C1C; }}
  .kpi .v.brand {{ background: linear-gradient(120deg,{BRAND_A},{BRAND_B});
      -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .kpi .s {{ font-size:.75rem; color:#64748B; margin-top:2px; }}
  /* tabs as pills */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none;
      flex-wrap: wrap; }}
  .stTabs [data-baseweb="tab"] {{ background:#FFFFFF; border:1px solid #E7EAF0;
      border-radius: 999px; padding: 6px 16px; color:#475569;
      font-weight:600; font-size:.86rem; }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(120deg,{BRAND_A},{BRAND_B}) !important;
      color: white !important; border-color: transparent !important; }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]
      {{ display:none; }}
  /* cards (bordered containers) */
  [data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF;
      border:1px solid #E7EAF0 !important; border-radius:16px !important;
      box-shadow: 0 1px 3px rgba(16,24,40,.05);
      padding: 6px 10px !important; }}
  /* sidebar */
  [data-testid="stSidebar"] {{ background:#FFFFFF;
      border-right:1px solid #E7EAF0; }}
  /* buttons */
  .stButton > button, .stDownloadButton > button {{ border-radius: 10px;
      border:1px solid #E7EAF0; font-weight:600; }}
  .stButton > button[kind="primary"] {{
      background: linear-gradient(120deg,{BRAND_A},{BRAND_B}); border:none; }}
  /* expanders */
  [data-testid="stExpander"] {{ border:1px solid #E7EAF0; border-radius:12px;
      background:#FFFFFF; }}
  /* progress */
  [data-testid="stProgress"] > div > div > div {{
      background: linear-gradient(90deg,{BRAND_A},{BRAND_B}); }}
  .small-note {{ color:#64748B; font-size:0.84rem; }}
  /* silk: transitions + hover lift */
  [data-testid="stVerticalBlockBorderWrapper"], .kpi, .stButton > button,
  .stTabs [data-baseweb="tab"] {{ transition: box-shadow .18s ease,
      transform .18s ease, background .18s ease; }}
  [data-testid="stVerticalBlockBorderWrapper"]:hover, .kpi:hover {{
      box-shadow: 0 6px 18px rgba(16,24,40,.10); transform: translateY(-1px); }}
  .stTabs [data-baseweb="tab"]:hover {{ background:#F1F5F9; }}
  ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
  ::-webkit-scrollbar-thumb {{ background:#CBD5E1; border-radius: 8px; }}
  /* sticky live design panel */
  .st-key-liveview {{ position: sticky; top: 3.4rem; z-index: 3; }}
</style>"""

PLOTCFG = dict(displaylogo=False,
               modeBarButtonsToAdd=["drawline", "drawrect", "eraseshape",
                                    "toggleSpikelines"])

def kpi_cards(items):
    """items: list of (label, value, sub, tone in '', 'ok', 'bad', 'brand')."""
    cells = "".join(
        f"<div class='kpi'><div class='l'>{l}</div>"
        f"<div class='v {t}'>{v}</div><div class='s'>{s}</div></div>"
        for l, v, s, t in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)

DEFAULTS = dict(
    Ns=108, Np=10, cap_Ah=5.0, v_nom=3.7, r_dc=25.0, m_cell=0.070, cp_cell=950.0,
    d_cell=0.021, h_cell=0.070, arrangement="Square", pitch=0.027, edge_margin=0.010,
    bottom_gap=0.005, tube_zone=0.035, gas_gap=0.010, end_fraction=0.0,
    coolant="MIVOLT DF7", u_oil=0.0,
    n_tubes=16, tube_od=0.010, tube_wall=0.001, tube_mat="Copper", passes=1,
    manifold_margin=0.020,
    fins_on=True, fin_h=0.008, fin_t=0.0005, fin_p=0.004, fin_mat="Aluminium",
    loop_fluid="Water", flow_lpm=10.0, T_water_in=20.0,
    duty="Constant C", C1=2.0, t1=900.0, C2=0.5, t2=600.0, duration=3600.0,
    T_start=25.0, T_limit=45.0, T_amb=25.0, h_ext=5.0, struct_mass=0.0,
    T_service_max=60.0,
    # v2 additions
    k_dcir=0.012, k_rad=0.9, limit_core=False, tube_plane="Top of pack",
    K_loop=5.0, cal_h=1.0,
    E_tr=55.0, frac_oil=0.6, zone_pitches=1.5, vent_L=5.0,
    # v3 additions
    fmt="21700", v_max=4.2, v_cut=0.05, chg_mult=1.10, entropic=True,
    soc0=0.90, soc_min=0.10, C_chg=1.0, dirn="Discharge", track_soc=False,
    cyc_rest=600.0, n_cyc=3,
    R_bus=0.0, bus_J=5.0, m_holder_g=8.0, holder_block=0.20,
    sigma_MPa=80.0, stiff=0.45, p_des_bar=0.5,
    circ="Thermosiphon only", plate_on=False, plate_t=0.0015,
    plate_mat="Aluminium", plate_contact=0.8, u_guided=0.05,
    veh_m=1900.0, CdA=0.62, Crr=0.009, eta_dt=0.92, eta_rg=0.65,
    P_rg=60.0, P_acc=500.0, cycle="WLTP Class 3b", repeat_cyc=True,
)

def resistance_chart(res):
    items = [("Cell-to-oil film", res["R_b"]), ("Oil-to-tube film (finned area)", res["R_ot"]),
             ("Tube wall", res["R_wall"]), ("Water film in tubes", res["R_in"])]
    tot = sum(v for _, v in items)
    worst = max(items, key=lambda x: x[1])[0]
    fig = go.Figure(go.Bar(
        y=[n for n, _ in items][::-1], x=[v * 1000 for _, v in items][::-1],
        orientation="h",
        marker_color=["#EF4444" if n == worst else "#6366F1" for n, _ in items][::-1],
        text=[f"{v*1000:.2f} mK/W  ({100*v/tot:.0f}%)" for _, v in items][::-1],
        textposition="outside"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title="Where the resistance lives (cell -> water chain)",
                      xaxis_title="Thermal resistance [mK/W]",
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig, worst

def waterfall_chart(res, Q, d):
    steps = [("Water inlet", d["T_water_in"], "absolute"),
             ("Water warm-up (mean)", 0.5 * res["dT_water"], "relative"),
             ("Water film", res["Q_w"] * res["R_in"], "relative"),
             ("Tube wall", res["Q_w"] * res["R_wall"], "relative"),
             ("Oil film on tubes", res["Q_w"] * res["R_ot"], "relative"),
             ("Cell-to-oil film", Q * res["R_b"], "relative"),
             ("Cell surface", None, "total")]
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], measure=[s[2] for s in steps],
        y=[s[1] if s[1] is not None else 0 for s in steps],
        connector=dict(line=dict(color="#94A3B8")),
        increasing=dict(marker=dict(color=ACCENT)),
        totals=dict(marker=dict(color=INK))))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                      title="Temperature waterfall at this duty [°C]",
                      yaxis_title="°C", plot_bgcolor="rgba(255,255,255,0)",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig

def chain_schematic(res, Q):
    boxes = [("CELL", "#D96C4F"), ("oil film", "#F3D9A4"), ("BULK OIL", ACCENT),
             ("oil film", "#F3D9A4"), ("TUBE+FINS", "#D97706"), ("wall", "#8C8C8C"),
             ("water film", "#BFD9EA"), ("WATER", "#0EA5E9")]
    drops = [None, Q * res["R_b"], None, res["Q_w"] * res["R_ot"], None,
             res["Q_w"] * res["R_wall"], res["Q_w"] * res["R_in"], None]
    fig = go.Figure()
    x = 0.0
    for (label, colr), dT in zip(boxes, drops):
        w = 1.4 if label.isupper() else 0.9
        fig.add_shape(type="rect", x0=x, x1=x + w, y0=0, y1=1,
                      fillcolor=colr, line=dict(color=INK, width=1))
        fig.add_annotation(x=x + w / 2, y=0.5, text=label, showarrow=False,
                           font=dict(size=12, color=INK))
        if dT is not None:
            fig.add_annotation(x=x + w / 2, y=1.18, text=f"dT = {dT:.1f} K",
                               showarrow=False, font=dict(size=11, color="#EF4444"))
        x += w + 0.12
    fig.add_annotation(x=x / 2, y=-0.28, showarrow=False,
                       text="Heat flows left to right. The films are where the kelvins are spent; "
                            "the bulk oil and copper are nearly free.",
                       font=dict(size=11, color="#64748B"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, range=[-0.5, 1.5])
    fig.update_layout(height=210, margin=dict(l=5, r=5, t=10, b=5),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ------------------------------------------------------------------ #
#  Main app                                                           #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  Parasitic power                                                    #
# ------------------------------------------------------------------ #
def water_pump_power(d, g, loop) -> dict:
    mdot_tot = d["flow_lpm"] / 60.0 * loop["rho"] / 1000.0
    mdot_tube = mdot_tot / max(d["n_tubes"], 1)
    A_i = math.pi * g["d_i"] ** 2 / 4
    v = mdot_tube / (loop["rho"] * A_i)
    Re = loop["rho"] * v * g["d_i"] / loop["mu"]
    f = 64.0 / max(Re, 1.0) if Re < 2300 else 0.316 * Re ** -0.25
    dp = (f * g["L_tube"] / g["d_i"] + 6.0) * 0.5 * loop["rho"] * v ** 2
    return dict(P=dp * (mdot_tot / loop["rho"]) / 0.35, dp=dp, v=v, Re=Re)

def stirrer_power(d, g, fl, u, T_oil=35.0) -> float:
    """Sealed circulator driving the whole free section at u through the
    cell bank: laminar bank friction + minor losses, 30% wire-to-fluid."""
    if u <= 1e-6:
        return 0.0
    p = film_props(fl, T_oil)
    dp = 32.0 * p["rho"] * p["nu"] * (2.2 * g["fill_h"]) * u / g["D_h"] ** 2 \
         + d.get("K_loop", 5.0) * 0.5 * p["rho"] * u ** 2
    return dp * (u * g["A_flow"]) / 0.30

def plate_fin_area(d, g, h_oil):
    """Effective wetted area of thin plates hung from the tubes between cell
    rows (both sides wetted, conduction only). Fin: eta = tanh(mL)/mL,
    m = sqrt(2h/(k t)); base contact to the tube derated by plate_contact."""
    if not d.get("plate_on"):
        return 0.0, 0.0, 0.0
    k = 205.0 if d.get("plate_mat", "Aluminium") == "Aluminium" else 380.0
    t = d.get("plate_t", 0.0015)
    L = d["h_cell"]
    m = math.sqrt(2.0 * max(h_oil, 5.0) / (k * t))
    eta = math.tanh(m * L) / max(m * L, 1e-9)
    n_pl = max(g["n_rows"] - 1, 1)
    length = max(g["Lx"] - 2 * d["manifold_margin"], 0.1)
    A_eff = n_pl * 2.0 * L * length * eta * d.get("plate_contact", 0.8)
    m_pl = n_pl * L * length * t * (2700.0 if k < 300 else 8940.0)
    return A_eff, eta, m_pl

def serpentine_pump(d, g, fl, u, T_oil=35.0):
    """Parallel row-channels guided by the plates: laminar slot flow,
    Dp = 12 mu L u / s^2 per channel, all channels manifolded in parallel.
    Returns pump electrical power, oil dT along a channel, and flow."""
    if u <= 1e-6:
        return dict(P=0.0, dT_path=0.0, mdot=0.0, dp=0.0)
    p = film_props(fl, T_oil)
    s = max((d["pitch"] - d["d_cell"] - d.get("plate_t", 0.0015)) / 2, 5e-4)
    Lch = max(g["Lx"] - 2 * d["manifold_margin"], 0.1)
    n_ch = max(g["n_rows"] - 1, 1) * 2            # both sides of each plate
    A_ch = s * d["h_cell"]
    dp = 12.0 * p["rho"] * p["nu"] * Lch * u / s ** 2 + 3.0 * 0.5 * p["rho"] * u ** 2
    mdot = p["rho"] * u * A_ch * n_ch
    P = dp * (mdot / p["rho"]) / 0.35
    return dict(P=P, dT_path=0.0 if mdot < 1e-9 else 0.0, mdot=mdot, dp=dp,
                dT_est=lambda Q: Q / max(mdot * p["cp"], 1e-9))

# ------------------------------------------------------------------ #
#  Architecture comparator                                            #
# ------------------------------------------------------------------ #
def compare_architectures(d, g, fl, masses, T_amb, C_duty=None) -> pd.DataFrame:
    d = dict(d, C1=(C_duty if C_duty else d["C1"]))
    loop = WATER_LOOP[d["loop_fluid"]]
    Pw = water_pump_power(d, g, loop)["P"]
    imm_mass = masses["m_oil"] + masses["m_tubes"] + masses["m_fins"]
    rows = []

    def imm_case(label, u, extraP, mass, note):
        dd = dict(d); dd["u_oil"] = u
        r = solve_steady(dd, g, fl, 1.0, T_amb, C_rate=d["C1"])
        rows.append(dict(Architecture=label, T_cell=r["T_b"], T_core=r["T_core"],
                         Parasitic_W=Pw + extraP, Thermal_mass_kg=mass, Notes=note))

    imm_case("Static immersion + internal HX", 0.0, 0.0, imm_mass,
             "this design; thermosiphon only, no moving parts in oil")
    imm_case("Stirred immersion + internal HX (5 cm/s)", 0.05,
             stirrer_power(d, g, fl, 0.05), imm_mass + 0.5,
             "sealed magnetically-coupled circulator")

    # bottom cold plate (dry pack): axial cell path + TIM + channel film
    Q_cell = q_gen_per_cell(d, d["C1"], 35.0)
    Q_tot = Q_cell * g["N"]
    R_ax = (d["h_cell"] / 2) / (28.0 * math.pi * d["d_cell"] ** 2 / 4)
    R_cp = R_ax + 0.8 + 1.0 / (3000.0 * d["pitch"] ** 2)
    dTw = Q_tot / max((d["flow_lpm"] / 60 * loop["rho"] / 1000) * loop["cp"], 1e-9)
    T_cp = d["T_water_in"] + 0.5 * dTw + Q_cell * R_cp
    plate_mass = g["Lx"] * g["Ly"] * 0.006 * 2700 + 2.0
    rows.append(dict(Architecture="Bottom cold plate (dry pack)",
                     T_cell=T_cp, T_core=T_cp + Q_cell * r_core(d),
                     Parasitic_W=Pw, Thermal_mass_kg=plate_mass,
                     Notes="axial path ~3.6 K/W + TIM 0.8 + channel film; "
                           "1-2 mm pitch possible (no gap rule)"))

    # pumped dielectric + external HX (AMG HPB80 style)
    p35 = film_props(fl, 35.0)
    u_p = 0.20
    hf = nu_crossflow_cb(u_p * d["d_cell"] / p35["nu"], p35["Pr"]) * p35["k"] \
         / d["d_cell"] * d.get("cal_h", 1.0)
    mdot_oil = Q_tot / (fl["cp"] * 5.0)            # sized for 5 K oil rise
    T_oil_mean = d["T_water_in"] + 3.0 + 2.5       # HX approach + half rise
    T_pp = T_oil_mean + Q_cell / (hf * math.pi * d["d_cell"] * d["h_cell"])
    P_oil = (mdot_oil / fl["rho"]) * 30000.0 / 0.40
    pumped_mass = 0.025 * g["N"] * fl["rho"] / 1000 + 4.5   # ~25 mL/cell (HPB80 ratio) + HX/pump
    rows.append(dict(Architecture="Pumped dielectric + external HX (AMG-style)",
                     T_cell=T_pp, T_core=T_pp + Q_cell * r_core(d),
                     Parasitic_W=Pw + P_oil, Thermal_mass_kg=pumped_mass,
                     Notes=f"oil at {u_p*100:.0f} cm/s past cells, h ~ {hf:.0f}; "
                           "pump, plumbing, filter, de-aeration"))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ #
#  Thermal-runaway screening (order of magnitude)                     #
# ------------------------------------------------------------------ #
def runaway_screen(d, g, fl, masses) -> dict:
    p_, rp = d["pitch"], g["row_pitch"]
    r_zone = d["zone_pitches"] * p_
    plan = math.pi * r_zone ** 2
    n_in = max(plan / (p_ * rp) - 1.0, 0.0)
    oil_frac = max(1.0 - (math.pi * d["d_cell"] ** 2 / 4) / (p_ * rp), 0.05)
    V_zone = plan * d["h_cell"] * oil_frac + plan * d["tube_zone"]
    C_zone = V_zone * fl["rho"] * fl["cp"] + n_in * d["m_cell"] * d["cp_cell"]
    dT_zone = d["frac_oil"] * d["E_tr"] * 1000.0 / max(C_zone, 1.0)
    dT_bulk = d["E_tr"] * 1000.0 / (masses["C_oil"] + masses["C_batt"])
    V_hs = max(g["Lx"] * g["Ly"] * d["gas_gap"], 1e-5)
    P_final = 101325.0 * (V_hs + d["vent_L"] / 1000.0) / V_hs * (380.0 / 298.0)
    return dict(n_in=n_in, V_zone_L=V_zone * 1000, C_zone=C_zone,
                dT_zone=dT_zone, dT_bulk=dT_bulk,
                P_bar_g=(P_final - 101325.0) / 1e5, V_hs_L=V_hs * 1000)

# ------------------------------------------------------------------ #
#  Plan-view layout figure                                            #
# ------------------------------------------------------------------ #
def layout_figure(d, g):
    xs, ys, cs = [], [], []
    cnt = 0
    for r in range(g["n_rows"]):
        for c in range(g["n_cols"]):
            if cnt >= g["N"]:
                break
            x = d["edge_margin"] + (c + 0.5) * d["pitch"] \
                + (d["pitch"] / 2 if (d["arrangement"] == "Hexagonal" and r % 2) else 0)
            y = d["edge_margin"] + d["pitch"] / 2 + r * g["row_pitch"]
            xs.append(x); ys.append(y)
            cx, cy = g["Lx"] / 2, g["Ly"] / 2
            cs.append(1.0 - math.hypot(x - cx, y - cy) / math.hypot(cx, cy))
            cnt += 1
    size = max(3.0, d["d_cell"] / max(g["Lx"], 1e-3) * 640)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=g["Lx"], y1=g["Ly"],
                  line=dict(color=INK, width=2), fillcolor="rgba(232,161,58,0.06)")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="cells",
                             marker=dict(size=size, color=cs, colorscale="RdYlBu_r",
                                         showscale=False, line=dict(width=0)),
                             hoverinfo="skip"))
    for j in range(d["n_tubes"]):
        yj = (j + 0.5) * g["Ly"] / d["n_tubes"]
        fig.add_trace(go.Scatter(x=[d["manifold_margin"], g["Lx"] - d["manifold_margin"]],
                                 y=[yj, yj], mode="lines", showlegend=False,
                                 line=dict(color="#0EA5E9", width=3), hoverinfo="skip"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"Plan view: {g['n_cols']} x {g['n_rows']} grid "
                            f"({d['arrangement'].lower()}), {d['n_tubes']} tube runs (blue) "
                            "in the zone above - cell colour hints centre-vs-edge tendency",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ------------------------------------------------------------------ #
#  Calibration against measured data                                  #
# ------------------------------------------------------------------ #
def fit_calibration(d, g, fl, masses, T_amb, t_arr, C_arr, t_m, T_m):
    """Single multiplier on both oil-film h values, fitted to a measured
    cell-temperature trace - same pattern as the spray app's SS-1.0 factor."""
    best = (1.0, 1e9)
    for c in np.linspace(0.4, 2.2, 19):
        dd = dict(d); dd["cal_h"] = float(c)
        tr = solve_transient(dd, g, fl, masses, T_amb, t_arr, C_arr)
        pred = np.interp(t_m, tr["t"], tr["T_b"])
        rmse = float(np.sqrt(np.mean((pred - np.asarray(T_m)) ** 2)))
        if rmse < best[1]:
            best = (float(c), rmse)
    return best

# ------------------------------------------------------------------ #
#  Keyed widgets (enables save/load of whole designs)                 #
# ------------------------------------------------------------------ #
def _w(fn, label, key, default, **kw):
    k = f"w_{key}"
    if k not in st.session_state:
        st.session_state[k] = default
    return fn(label, key=k, **kw)


def arch_tab(d, g, fl, masses, C_steady):
    if True:
        st.markdown(f"Same pack, same {C_steady:.2f}C-rms duty, same water loop - four ways "
                    "to build the thermal system. This is the 'is it worth doing' slide.")
        adf = compare_architectures(d, g, fl, masses, d["T_amb"], C_steady)
        st.dataframe(adf.round(1), hide_index=True, use_container_width=True)
        cA, cB = st.columns(2)
        with cA:
            figA = go.Figure(go.Bar(x=adf["Architecture"], y=adf["T_cell"],
                                    marker_color=[ACCENT, "#D97706", "#6366F1", "#10B981"],
                                    text=[f"{v:.1f}" for v in adf["T_cell"]],
                                    textposition="outside"))
            figA.add_hline(y=d["T_limit"], line_dash="dash", line_color="#B91C1C")
            figA.update_layout(height=380, title="Steady cell temperature [°C]",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_tickangle=-15)
            st.plotly_chart(figA, use_container_width=True)
        with cB:
            figP = go.Figure(go.Bar(x=adf["Architecture"], y=adf["Parasitic_W"],
                                    marker_color="#64748B",
                                    text=[f"{v:.0f} W" for v in adf["Parasitic_W"]],
                                    textposition="outside"))
            figP.update_layout(height=380, title="Parasitic power [W] (chiller excluded)",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_tickangle=-15)
            st.plotly_chart(figP, use_container_width=True)
        st.caption("Cold-plate constants: axial cell path ~3.6 K/W, TIM 0.8 K/W, channel film "
                   "h = 3000 W/m²·K - edit in code if you have better numbers. Pumped case: "
                   "20 cm/s past cells, oil sized for 5 K rise, 30 kPa loop at 40% pump "
                   "efficiency, ~25 mL of fluid per cell (the HPB80 ratio). Kelvin per watt: "
                   "work the bottleneck, then buy the cheapest watts.")


def coolant_tab(d, g, cool_df):
    if True:
        st.markdown("Every fluid in the reviewed table, run through the **same pack at the "
                    f"same duty ({d['C1']:.1f}C continuous)**, thermosiphon and DCIR(T) "
                    "included. Non-dielectric fluids are thermal references only.")
        rows = []
        for _, r in cool_df.iterrows():
            f2 = fluid_dict(r)
            try:
                r2 = solve_steady(d, g, f2, 1.0, d["T_amb"], C_rate=d["C1"])
                m2 = build_masses(d, g, f2, r2["fin"])
                flags = []
                if not f2["dielectric"]:
                    flags.append("NOT dielectric")
                if not math.isnan(f2["bp"]) and r2["T_b"] + 10 > f2["bp"]:
                    flags.append("near boiling")
                if not math.isnan(f2["flash"]) and f2["flash"] < 120:
                    flags.append("low flash")
                if any(s in str(r["family"]).lower() for s in ("fluor", "hfo", "hydrofluoro")):
                    flags.append("PFAS")
                rows.append(dict(Fluid=f2["name"], Family=r["family"], nu_cSt=r["nu_cSt"],
                                 k=r["k"], u_ts_mms=r2["u_ts"] * 1000, h_cell=r2["h_cell"],
                                 T_cell=r2["T_b"], oil_kg=m2["m_oil"], Whkg=m2["whkg_pack"],
                                 Flags=", ".join(flags)))
            except Exception:
                pass
        sdf = pd.DataFrame(rows).sort_values("T_cell")
        st.dataframe(sdf.round(1), hide_index=True, use_container_width=True, height=420)
        figS = go.Figure()
        lab_all = thin_labels(sdf["oil_kg"], -sdf["T_cell"], list(sdf["Fluid"]),
                              min_dx=0.06, min_dy=0.10)   # favour the coolest
        lab_map = dict(zip(sdf["Fluid"], lab_all))
        for fam, grp in sdf.groupby("Family"):
            figS.add_trace(go.Scatter(x=grp["oil_kg"], y=grp["T_cell"], mode="markers+text",
                                      text=[lab_map[f] for f in grp["Fluid"]],
                                      hovertext=grp["Fluid"], hoverinfo="text+x+y",
                                      textposition="top center", name=fam,
                                      textfont=dict(size=11),
                                      marker=dict(size=9 + 40 * grp["k"] / sdf["k"].max())))
        figS.add_hline(y=d["T_limit"], line_dash="dash", line_color="#B91C1C")
        figS.update_layout(height=460, title="Cooler is down, lighter is left "
                           "(size ~ conductivity; hover for every fluid)",
                           xaxis_title="Coolant mass on board [kg]",
                           yaxis_title="Steady cell temperature [°C]",
                           plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figS, use_container_width=True)
        st.caption("The fluorinated fluids win on h despite 5x lower conductivity (Ra ~ 1/nu, "
                   "and note their stronger thermosiphon in the u_ts column) but lose on "
                   "density, boiling point, cost and PFAS status (3M exited PFAS manufacture "
                   "end-2025). The esters are the pragmatic middle. Review notes from the "
                   "source spreadsheet apply.")


def learn_tab(d, g, fl, res, masses, cool_df, loop):
    if True:
        st.markdown("Work top to bottom: each panel is one physical idea, using **your live "
                    "design** for the numbers.")
        with st.expander("1. The whole story: two thin films own the problem", expanded=True):
            st.markdown(
                "Once heat is in the moving oil, buoyancy mixes it well; the bulk is nearly "
                "isothermal. All the temperature drop concentrates in two near-stagnant "
                "**boundary layers**: one on the cell wall, one on the tube wall. Across a "
                "film the only transport is conduction through oil, so")
            st.latex(r"h \approx \frac{k_{oil}}{\delta_{film}} \quad\Rightarrow\quad "
                     r"h \sim \frac{0.13\ \mathrm{W/mK}}{1\text{-}2\ \mathrm{mm}} "
                     r"\approx 60\text{-}130\ \mathrm{W/m^2K}")
            st.markdown(
                f"Right now your films give **h_cell = {res['h_cell']:.0f}** and "
                f"**h_tube = {res['h_tube']:.0f} W/m²·K**. Compare: 1.5 mm of stagnant oil has "
                "about **4000x** the resistance of 1 mm of copper wall. Both R = 1/(hA), so "
                "the fixes are exactly two: raise h (thin the film: stir, lower viscosity) or "
                "raise A (fins, more tubes).")
        with st.expander("2. Natural convection playground: Ra -> Nu -> h"):
            st.latex(r"Ra_L=\frac{g\,\beta\,\Delta T\,L^3}{\nu\,\alpha},\qquad "
                     r"Nu=\Big(0.825+\frac{0.387\,Ra^{1/6}}{[1+(0.492/Pr)^{9/16}]^{8/27}}\Big)^2,"
                     r"\qquad h=\frac{Nu\,k}{L}")
            cA2, cB2 = st.columns(2)
            dT_p = cA2.slider("Surface-to-bulk dT [K]", 1.0, 30.0, 8.0, 0.5)
            L_p = cB2.slider("Characteristic length [mm]", 10.0, 200.0, d["h_cell"] * 1000, 5.0) / 1000
            pf = film_props(fl, 35.0)
            Ra_p = rayleigh(pf, dT_p, L_p)
            h_p = nu_vertical_cc(Ra_p, pf["Pr"]) * pf["k"] / L_p
            st.markdown(f"**{fl['name']}**: Ra = {Ra_p:.2e} (laminar below ~1e9), Pr = "
                        f"{pf['Pr']:.0f}, **h = {h_p:.0f} W/m²·K**, and one 21700 sheds "
                        f"**{h_p * math.pi * 0.021 * 0.07 * dT_p:.1f} W** at that dT. Note "
                        "h ~ dT^(1/4): you cannot rescue a hot pack by letting it run hotter, "
                        "and h ~ (1/nu)^(1/4) via Ra, which is panel 3.")
        with st.expander("3. Why low viscosity beats high conductivity"):
            xs, ys, names, ks = [], [], [], []
            for _, r in cool_df.iterrows():
                f3 = fluid_dict(r)
                p3 = film_props(f3, 35.0)
                Ra3 = rayleigh(p3, 8.0, d["h_cell"])
                ys.append(nu_vertical_cc(Ra3, p3["Pr"]) * p3["k"] / d["h_cell"])
                xs.append(r["nu_cSt"]); names.append(r["name"]); ks.append(r["k"])
            figV = go.Figure(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                text=thin_labels(xs, ys, names, logx=True),
                hovertext=names, hoverinfo="text+x+y",
                textposition="top center", textfont=dict(size=11),
                marker=dict(size=8 + 60 * np.array(ks) / max(ks),
                            color=ks, colorscale="YlOrBr",
                            colorbar=dict(title="k [W/mK]"))))
            figV.update_layout(height=430, xaxis_type="log",
                               xaxis_title="Kinematic viscosity at 25 °C [cSt] (log)",
                               yaxis_title="Natural-convection h on a 21700 [W/m²·K]",
                               title="Ra ~ 1/nu, h ~ Ra^(1/4-1/6): viscosity is the strong "
                                     "axis (hover for every fluid)",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figV, use_container_width=True)
            st.caption("This reproduces Wang et al.'s HFE7100 result: 1% of the viscosity "
                       "beat 5x lower conductivity. But the whole y-axis spans barely a "
                       "factor of 3: fluid choice cannot buy fast charge. Geometry and "
                       "stirring can.")
        with st.expander("4. The cell-gap cliff (Wang Fig. 9)"):
            gaps = np.linspace(0.5, 10, 60)
            figG = go.Figure(go.Scatter(x=gaps, y=[gap_factor(x) for x in gaps],
                                        line=dict(color=ACCENT, width=3)))
            figG.add_vline(x=g["gap_mm"], line_dash="dash",
                           annotation_text=f"your gap {g['gap_mm']:.1f} mm")
            figG.update_layout(height=300, xaxis_title="Cell-to-cell gap [mm]",
                               yaxis_title="h penalty factor",
                               title="Below ~6 mm the buoyant flow in the gaps is throttled",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figG, use_container_width=True)
            st.caption("Wang et al. measured gap velocity falling 1.8 -> 0.5 mm/s as spacing "
                       "shrank 8 -> 2 mm; temperature climbed steeply below 6 mm. This is the "
                       "energy-density tax of static immersion: cold-plate packs run 1-2 mm "
                       "pitch. The penalty applies to the buoyant component only, so stirring "
                       "largely removes the cliff.")
        with st.expander("5. Fins: buying area where h is worst"):
            hs = np.linspace(0.002, 0.02, 40)
            eff, gain = [], []
            for hh in hs:
                fp = fin_pack(d["tube_od"], hh, d["fin_t"] if d["fins_on"] else 0.0005,
                              d["fin_p"] if d["fins_on"] else 0.004,
                              205.0, max(res["h_tube"], 30))
                eff.append(fp["eta"]); gain.append(fp["area_gain"])
            figF = go.Figure()
            figF.add_trace(go.Scatter(x=hs * 1000, y=gain, name="Area gain x",
                                      line=dict(color=INK, width=3)))
            figF.add_trace(go.Scatter(x=hs * 1000, y=eff, name="Fin efficiency",
                                      yaxis="y2", line=dict(color=ACCENT, width=3)))
            figF.update_layout(height=320, xaxis_title="Fin height [mm]",
                               yaxis_title="Effective area multiplier",
                               yaxis2=dict(title="Schmidt efficiency", overlaying="y",
                                           side="right", range=[0, 1.05]),
                               title="Oil's low h keeps even long thin fins ~90% efficient: "
                                     "fin hard",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(figF, use_container_width=True)
        with st.expander("6. Stirring and the thermosiphon floor"):
            us = np.linspace(0, 0.2, 50)
            hcs, hts = [], []
            for u in us:
                ue = max(u, res["u_ts"])
                hcs.append(h_cell_side(fl, res["T_b"], res["T_il"], d["h_cell"], d["d_cell"],
                                       g["gap_mm"], ue)["h"] * d["cal_h"])
                hts.append(h_tube_side(fl, res["T_il"], res["T_wall"], d["tube_od"], ue)["h"]
                           * d["cal_h"])
            figU = go.Figure()
            figU.add_trace(go.Scatter(x=us * 100, y=hcs, name="Cell film",
                                      line=dict(color="#EF4444", width=3)))
            figU.add_trace(go.Scatter(x=us * 100, y=hts, name="Tube film",
                                      line=dict(color="#6366F1", width=3)))
            figU.add_vline(x=d["u_oil"] * 100, line_dash="dash", annotation_text="your stirring")
            figU.add_vline(x=res["u_ts"] * 100, line_dash="dot", line_color="#10B981",
                           annotation_text=f"thermosiphon {res['u_ts']*1000:.1f} mm/s")
            figU.update_layout(height=320, xaxis_title="Oil velocity [cm/s]",
                               yaxis_title="h [W/m²·K]",
                               title="The pack stirs itself a little; a circulator does it "
                                     "properly",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(figU, use_container_width=True)
            st.caption(f"Predicted self-circulation: buoyant head rho*beta*g*H*dT against "
                       f"laminar loop friction; here {res['u_ts']*1000:.1f} mm/s and "
                       f"{res['dT_loop']:.1f} K top-to-bottom (Wang measured 0.5-1.8 mm/s). "
                       "Tube placement sets the head: put the cold plane high. A few cm/s of "
                       f"forced stirring ({stirrer_power(d, g, fl, 0.05):.1f} W at 5 cm/s) "
                       "dwarfs it.")
        with st.expander("7. Inside the tubes: the laminar plateau"):
            fls = np.linspace(0.5, 60, 80)
            hws, res_w = [], []
            for q in fls:
                md = q / 60 * loop["rho"] / 1000 / max(d["n_tubes"], 1)
                w = h_water_inside(loop, md, g["d_i"], g["L_tube"])
                hws.append(w["h"]); res_w.append(w["Re"])
            figW = go.Figure(go.Scatter(x=fls, y=hws, line=dict(color="#0EA5E9", width=3)))
            figW.add_vline(x=d["flow_lpm"], line_dash="dash", annotation_text="your flow")
            i2300 = int(np.argmin(np.abs(np.array(res_w) - 2300)))
            figW.add_vline(x=fls[i2300], line_dash="dot", line_color="#B91C1C",
                           annotation_text="Re 2300")
            figW.update_layout(height=320, xaxis_title="Total water flow [L/min]",
                               yaxis_title="h inside tube [W/m²·K]",
                               title="In laminar flow Nu is ~constant: pumping harder does "
                                     "nothing until you trip turbulence",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figW, use_container_width=True)
        with st.expander("8. Buffering: the oil is a thermal flywheel"):
            cQ, cT = st.columns(2)
            Q_ex = cQ.slider("Excess heat beyond removal [kW]", 0.1, 15.0, 3.0, 0.1)
            dT_h = cT.slider("Allowed temperature drift [K]", 2.0, 25.0, 10.0, 1.0)
            Ctot = masses["C_oil"] + masses["C_batt"]
            st.markdown(f"Thermal mass = oil {masses['C_oil']/1000:.0f} kJ/K + cells "
                        f"{masses['C_batt']/1000:.0f} kJ/K = **{Ctot/1000:.0f} kJ/K**. "
                        f"It absorbs {Q_ex:.1f} kW of excess for "
                        f"**{Ctot*dT_h/(Q_ex*1000)/60:.1f} minutes** per {dT_h:.0f} K of "
                        "drift. Size the steady HX for continuous duty and let the flywheel "
                        "eat the peaks.")
        with st.expander("9. Heat that fights back: DCIR(T) and the core"):
            Ts = np.linspace(0, 60, 61)
            figD = go.Figure(go.Scatter(x=Ts, y=[r_of_T(d, t) for t in Ts],
                                        line=dict(color=INK, width=3)))
            figD.add_vline(x=res["T_b"], line_dash="dash", annotation_text="your cell")
            figD.update_layout(height=300, xaxis_title="Cell temperature [°C]",
                               yaxis_title="DCIR [mΩ]",
                               title=f"R(T) = R25 exp(-{d['k_dcir']*100:.1f}%/K x (T-25))",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figD, use_container_width=True)
            st.markdown(f"Running at {res['T_b']:.0f} °C instead of 25 cuts heat generation "
                        f"by **{100*(1-r_of_T(d, res['T_b'])/d['r_dc']):.0f}%** at the same "
                        "current - the AMG 45 °C set-point logic, and why the transient "
                        "self-stabilises near the limit. Separately, the core runs "
                        f"**{res['dT_core']:.1f} K** above the can here "
                        f"(R_core = 1/(4 pi k_r H) = {r_core(d):.2f} K/W): no coolant choice "
                        "touches that term.")
        with st.expander("10. Safety, practicalities, and the production reference"):
            st.markdown(f"""
* **Water-in-oil leak** is the single-point failure: hold **oil pressure above water
  pressure** so leaks go oil-to-water, or double-walled tubes with leak detection
  (transformer practice).
* **Expansion**: beta = {fl['beta']:.1e} /K on {masses['V_oil_L']:.0f} L means
  ~{fl['beta']*masses['V_oil_L']*70:.1f} L over a -10 to 60 °C band. Bellows or bladder,
  not free air.
* **Materials**: seal/insulation compatibility, ester moisture uptake, copper oxidation
  catalysis (use inhibited fluids or plated tubes).
* **Venting**: a cell venting into a sealed flooded box is a pressure spike - see the
  runaway screening in Decide. The flip side: oxygen exclusion and the oil's heat
  absorption suppress propagation.
* **Fluid supply**: 3M exited PFAS manufacture end-2025; anchor the programme on esters.
* **Mercedes AMG HPB80** (batterydesign.net): 560 x 21700, 112S5P, 6.1 kWh, 89 kg,
  68.5 Wh/kg, 150 kW peak / 70 kW continuous, pumped dielectric (14 L) through an external
  dielectric-to-water HX, 10 kW cooling, 45 °C set point.
* **Sources**: Wang et al. 2023 (J. Energy Storage 62, 106821); Zou et al. 2024 (J. Energy
  Storage 83, 110634); Roe et al. 2022 (J. Power Sources 525, 231094); batterydesign.net;
  coolant_comparison_reviewed.xlsx.
""")


def improve_core(d, g, fl, masses, res, Q_duty, C_steady, cool_df):
    if True:
        if True:
            st.markdown("Auto-diagnosis of **this** design at **this** duty, then the "
                        "cheapest fixes ranked by what they actually buy.")
            for level, msg in diagnose(d, g, fl, masses, res, d["T_limit"], Q_duty):
                {"info": st.info, "do": st.success, "warn": st.warning,
                 "bad": st.error}[level](msg)
            st.markdown("---")
            st.markdown("**Sensitivity: change one thing, what happens to steady cell "
                        "temperature?**")
            with st.spinner("Re-solving perturbed designs..."):
                sens, base_T = sensitivity(d, g, fl, cool_df, d["T_amb"], C_steady, d["T_limit"])
            figSe = go.Figure(go.Bar(
                y=sens["change"], x=sens["dT"], orientation="h",
                marker_color=np.where(sens["dT"] < 0, "#10B981", "#EF4444"),
                text=[f"{v:+.1f} K" for v in sens["dT"]], textposition="outside"))
            figSe.add_vline(x=0, line_color=INK)
            figSe.update_layout(height=380, title=f"Change in steady cell T from "
                                f"{base_T:.1f} °C at {C_steady:.2f}C rms (left = cooler)",
                                xaxis_title="ΔT_cell [K]", plot_bgcolor="rgba(255,255,255,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=10, r=60, t=40, b=10))
            st.plotly_chart(figSe, use_container_width=True)

            with st.expander("Goal-seek: cheapest single lever to hit the limit"):
                gl = st.selectbox("Lever", list(GOAL_LEVERS))
                if st.button("Solve for the limit"):
                    key, lo_, hi_, isint = GOAL_LEVERS[gl]
                    inv = key == "T_water_in"
                    with st.spinner("Bisecting..."):
                        xstar, Tst = goal_seek(d, fl, key, lo_, hi_, isint,
                                               d["T_amb"], C_steady, d["T_limit"], inv)
                    if xstar is None:
                        st.error(f"This lever alone cannot reach {d['T_limit']:.0f} °C "
                                 f"(best achievable ~{Tst:.1f} °C). Combine levers.")
                    else:
                        shown = xstar * 1000 if key == "pitch" else xstar
                        st.success(f"{gl} = **{shown:.2f}** gives {Tst:.1f} °C at "
                                   f"{C_steady:.2f}C rms. Set it in the sidebar to adopt.")

            with st.expander("Two-lever sweep (heatmap)"):
                AXES = {"Water flow [L/min]": ("flow_lpm", 2.0, 40.0),
                        "Number of tubes": ("n_tubes", 4, 40),
                        "Stirring [m/s]": ("u_oil", 0.0, 0.12),
                        "Water inlet [°C]": ("T_water_in", 5.0, 35.0),
                        "Cell pitch [mm]": ("pitch", 0.023, 0.033)}
                cx, cy = st.columns(2)
                ax_x = cx.selectbox("X axis", list(AXES), 0)
                ax_y = cy.selectbox("Y axis", list(AXES), 1)
                if ax_x != ax_y and st.button("Run 7 x 7 sweep"):
                    kx, x0, x1 = AXES[ax_x]; ky, y0, y1 = AXES[ax_y]
                    xs = np.linspace(x0, x1, 7); ys = np.linspace(y0, y1, 7)
                    Z = np.zeros((7, 7))
                    prog = st.progress(0.0)
                    for i, yv in enumerate(ys):
                        for j, xv in enumerate(xs):
                            dd = dict(d)
                            dd[kx] = int(round(xv)) if kx == "n_tubes" else float(xv)
                            dd[ky] = int(round(yv)) if ky == "n_tubes" else float(yv)
                            gg = build_geometry(dd)
                            Z[i, j] = solve_steady(dd, gg, fl, 1.0, d["T_amb"],
                                                   C_rate=C_steady)["T_b"]
                        prog.progress((i + 1) / 7)
                    xs_d = xs * 1000 if kx == "pitch" else xs
                    ys_d = ys * 1000 if ky == "pitch" else ys
                    figH = go.Figure(go.Heatmap(x=xs_d, y=ys_d, z=Z, colorscale="RdYlBu_r",
                                                colorbar=dict(title="T_cell [°C]")))
                    figH.add_contour(x=xs_d, y=ys_d, z=Z, showscale=False,
                                     contours=dict(start=d["T_limit"], end=d["T_limit"],
                                                   coloring="lines"),
                                     line=dict(color="black", width=3))
                    figH.update_layout(height=460, xaxis_title=ax_x, yaxis_title=ax_y,
                                       title=f"Steady cell T at {C_steady:.2f}C rms - black "
                                             f"contour = {d['T_limit']:.0f} °C limit")
                    st.plotly_chart(figH, use_container_width=True)


def runaway_ui(d, g, fl, masses, res):
    if True:
        if True:
            with st.expander("Thermal-runaway screening (order of magnitude)"):
                cr1, cr2 = st.columns(2)
                with cr1:
                    d["E_tr"] = _w(st.slider, "Heat released per cell [kJ]", "etr", 55.0,
                                   min_value=20.0, max_value=120.0, step=5.0,
                                   help="21700 NMC total ~30-80 kJ depending on SoC")
                    d["frac_oil"] = _w(st.slider, "Fraction into local oil zone", "ftr", 0.6,
                                       min_value=0.2, max_value=1.0, step=0.05)
                with cr2:
                    d["zone_pitches"] = _w(st.slider, "Local zone radius [pitches]", "ztr", 1.5,
                                           min_value=1.0, max_value=3.0, step=0.25)
                    d["vent_L"] = _w(st.slider, "Vent gas at STP [L]", "vtr", 5.0,
                                     min_value=1.0, max_value=15.0, step=0.5)
                rw = runaway_screen(d, g, fl, masses)
                margin = 170.0 - (res["T_b"] + rw["dT_zone"])
                verdict = ("looks containable" if margin > 30
                           else "MARGINAL - add spacing, oil, or interstitial barriers")
                st.markdown(f"""
One cell lets go at {res['T_b']:.0f} °C operating temperature:

* Local zone ({d['zone_pitches']:.1f} pitches): **{rw['V_zone_L']:.1f} L of oil +
  {rw['n_in']:.0f} neighbour cells** -> zone rise **{rw['dT_zone']:.0f} K**, i.e.
  neighbours reach ~**{res['T_b']+rw['dT_zone']:.0f} °C** vs a ~170-200 °C trigger
  (margin {margin:+.0f} K, before venting jets - {verdict}).
* Spread over the whole pack it is only **{rw['dT_bulk']:.1f} K** - the flooded pack's
  big argument.
* {d['vent_L']:.0f} L of vent gas into the {rw['V_hs_L']:.0f} L headspace at ~380 K:
  **~{rw['P_bar_g']:.1f} bar gauge** - size the burst disc well below the lid's rating and
  expect oil ejection through it.

Screening numbers only: vent jets, ejecta and local boiling are not modelled.""")
    else:
        st.sidebar.info("Student version: Decide tab hidden.")

def bench_wang_tab():
    if True:
        st.markdown("The same correlations and two-node network, applied to the **exact rig "
                    "of Wang et al. (2023)**: six prismatic dummy cells (148 x 97 x 27 mm) in "
                    "transformer oil, four 6 mm copper tubes at the top, 5 °C water at "
                    "17.1 mL/s, 2C for 1800 s, 25 °C ambient. Fixed heater power, no "
                    "DCIR(T), calibration factor 1.0.")
        bm = benchmark_wang()
        st.dataframe(pd.DataFrame(bm["rows"], columns=["Quantity", "This app", "Paper"]),
                     hide_index=True, use_container_width=True)
        figB = go.Figure()
        figB.add_trace(go.Scatter(x=bm["t"] / 60, y=bm["T_b"], name="Cell (this app)",
                                  line=dict(color="#EF4444", width=3)))
        figB.add_trace(go.Scatter(x=bm["t"] / 60, y=bm["T_il"], name="Oil (this app)",
                                  line=dict(color=ACCENT, width=3)))
        figB.add_trace(go.Scatter(x=[30], y=[32.3], mode="markers",
                                  name="Paper: cell at 1800 s",
                                  marker=dict(color="#B91C1C", size=12, symbol="x")))
        figB.add_trace(go.Scatter(x=[30], y=[28.5], mode="markers",
                                  name="Paper: oil at 1800 s",
                                  marker=dict(color="#D97706", size=12, symbol="x")))
        figB.update_layout(height=380, xaxis_title="Time [min]",
                           yaxis_title="Temperature [°C]",
                           title="Transient rebuild of the paper's 2C experiment",
                           plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(figB, use_container_width=True)
        st.markdown("""
**How to read the agreement.** Resistances land within ~30-50% and end temperatures within
~1-2 K, with zero tuning. Known residuals: the paper's oil conductivity is internally
inconsistent (Table 3 in kelvin gives 0.30 W/mK, matching their Table 7, vs ~0.13 for real
oil); their h values are back-calculated through a lumped model; and the vertical-plate
correlation is slightly conservative. Treat this app's oil-side h as honest to ~+/-30% -
or pin it with the calibration fit in Results once your own rig data exists.""")

def bench_prod_tab(masses, Cmax):
    if True:

        st.markdown("---")
        st.markdown("**Production packs: where this design sits.** Teardown and "
                    "certification data (OEMs do not publish pack masses); peak power is "
                    "the vehicle rating, a proxy for the battery-side limit.")
        prows = []
        for p_ in PRODUCTION_PACKS:
            r_ = dict(p_)
            if r_["Pack"] == "This design":
                r_.update(kWh=round(masses["E_kwh"], 1), kg=round(masses["m_pack"]),
                          Whkg=round(masses["whkg_pack"]), WhL=round(masses["whl_pack"]),
                          kWpk=round(Cmax * masses["E_kwh"], 1))
                r_["Cooling"] += f" (continuous {Cmax:.1f}C; peak not rated)"
            prows.append(r_)
        pdf_ = pd.DataFrame(prows)
        pdf_["kW/kg"] = (pdf_["kWpk"] / pdf_["kg"]).round(2)
        st.dataframe(pdf_[["Pack", "kWh", "kg", "Whkg", "kW/kg", "Cooling"]],
                     hide_index=True, use_container_width=True)
        figPk = go.Figure()
        for _, r_ in pdf_.iterrows():
            figPk.add_trace(go.Scatter(
                x=[r_["Whkg"]], y=[r_["kW/kg"]], mode="markers+text", text=[r_["Pack"]],
                textposition="top center", showlegend=False,
                marker=dict(size=16 if r_["Pack"] == "This design" else 11,
                            color=ACCENT if r_["Pack"] == "This design" else "#64748B")))
        figPk.update_layout(height=420, xaxis_title="Pack energy density [Wh/kg]",
                            yaxis_title="Peak power density [kW/kg] (vehicle rating)",
                            title="Energy vs power density - immersion trades energy density "
                                  "for simplicity and abuse tolerance",
                            plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figPk, use_container_width=True)
        st.caption("Sources, retrieved July 2026: Model 3 2170 = Rickard teardown "
                   "(Teslarati/EVANNEX, 478 kg, 4416 cells, ~75 kWh usable, glycol ribbon "
                   "side cooling). Model Y 4680 = EU certification 447 kg at ~79 kWh gross "
                   "(Electrek, May 2026); Munro teardown confirms side ribbons retained and "
                   "cells structurally bonded (batterydesign.net). Plaid = Munro 181.5 Wh/kg "
                   "at ~99 kWh (mass derived, ~545 kg), 760 kW vehicle peak (EVKX), "
                   "7920 x 18650 in 110S72P (Ingineerix). Model 3 LFP = batterydesign.net "
                   "(438 kg, 55 kWh, bottom cold plate). AMG HPB80 = batterydesign.net. "
                   "This design's peak column shows thermal continuous rating, which "
                   "understates a 30 s peak.")


# ------------------------------------------------------------------ #
#  Smoke test                                                         #
# ------------------------------------------------------------------ #
def smoke():
    cool_df = _read_coolants()
    d = dict(DEFAULTS)
    fl = fluid_dict(cool_df[cool_df["name"] == d["coolant"]].iloc[0])
    g = build_geometry(d)
    res = solve_steady(d, g, fl, 1.0, d["T_amb"], C_rate=d["C1"])
    Q = res["Q_eff"]
    masses = build_masses(d, g, fl, res["fin"])
    Cmax = max_continuous_C(d, g, fl, d["T_amb"], d["T_limit"])
    t, C = duty_profile(d["duty"], d["duration"], d["C1"], d["t1"], d["C2"], d["t2"])
    tr = solve_transient(d, g, fl, masses, d["T_amb"], t, C)
    loop = WATER_LOOP[d["loop_fluid"]]
    print(f"Q@2C(DCIR-T)={Q:.0f} W  T_b={res['T_b']:.1f}  T_core={res['T_core']:.1f}  "
          f"u_ts={res['u_ts']*1000:.2f} mm/s  dT_loop={res['dT_loop']:.1f} K  "
          f"spread={res['spread']:.1f} K  Cmax={Cmax:.2f}C")
    print(f"parasitics: pump={water_pump_power(d, g, loop)['P']:.2f} W  "
          f"stir@5cm/s={stirrer_power(d, g, fl, 0.05):.2f} W")
    adf = compare_architectures(d, g, fl, masses, d["T_amb"])
    print(adf[["Architecture", "T_cell", "Parasitic_W", "Thermal_mass_kg"]].round(1)
          .to_string(index=False))
    rw = runaway_screen(d, g, fl, masses)
    print(f"runaway: zone dT={rw['dT_zone']:.0f} K  bulk dT={rw['dT_bulk']:.1f} K  "
          f"headspace={rw['P_bar_g']:.1f} bar g")
    layout_figure(d, g)
    # calibration self-test: synthesise 'measured' data at cal=1.35, refit
    dd = dict(d); dd["cal_h"] = 1.35
    t_s = np.arange(0, 1201.0, 2.0); C_s = np.full_like(t_s, d["C1"])
    tr_s = solve_transient(dd, g, fl, masses, d["T_amb"], t_s, C_s)
    tm = t_s[::30]; Tm = np.interp(tm, tr_s["t"], tr_s["T_b"]) + np.random.default_rng(1).normal(0, 0.1, len(tm))
    c_fit, rmse = fit_calibration(d, g, fl, masses, d["T_amb"], t_s, C_s, tm, Tm)
    print(f"calibration self-test: true 1.35 -> fitted {c_fit:.2f} (RMSE {rmse:.2f} K)")
    bm = benchmark_wang()
    print(f"benchmark: T_b(1800s)={bm['T_b'][-1]:.1f} (paper 32.3)")
    tp = dict(d); tp["tube_plane"] = "Below the cells"
    r_low = solve_steady(tp, g, fl, 1.0, d["T_amb"], C_rate=d["C1"])
    print(f"tubes-below check: u_ts={r_low['u_ts']*1000:.2f} mm/s  T_b={r_low['T_b']:.1f}")
    # ---- v3 checks ----
    print(f"weight: pack={masses['m_pack']:.0f} kg ({masses['whkg_pack']:.0f} Wh/kg, "
          f"{masses['whl_pack']:.0f} Wh/L)  enclosure={masses['m_struct']:.1f} kg "
          f"@{masses['enc']['t_mm']:.1f} mm  busbar={masses['m_bus']:.1f} kg  "
          f"holders={masses['m_holders']:.1f} kg")
    for nm in DRIVE_CYCLES:
        tc, vc, D = cycle_speed(nm)
        dist = np.trapezoid(vc, tc) / 1000
        err = 100 * (dist / D - 1)
        print(f"cycle {nm:<24s} {tc[-1]:5.0f} s  {dist:6.2f} km (official {D}, {err:+.2f}%)")
        assert abs(err) < 0.5, nm
    tc, vc, D = cycle_speed("WLTP Class 3b")
    P_b = vehicle_battery_power(tc, vc, d)
    tr_dc = simulate_pack(d, g, fl, masses, d["T_amb"], dict(kind="P", t=tc, P=P_b))
    print(f"WLTP: peak {tr_dc['C'].max():.2f}C / regen {-tr_dc['C'].min():.2f}C  "
          f"C_rms={tr_dc['C_rms']:.3f}  SoC {tr_dc['soc'][0]:.2f}->{tr_dc['soc'][-1]:.3f}  "
          f"T_b end {tr_dc['T_b'][-1]:.1f}")
    dch = dict(d); dch["soc0"] = 0.20; dch["T_start"] = 25.0
    tr_ch = simulate_pack(dch, g, fl, masses, d["T_amb"],
                          dict(kind="chg", t=np.arange(0.0, 7200.0, 2.0)))
    print(f"CC-CV 25C: SoC 0.20->{tr_ch['soc'][-1]:.3f}  peak chg {-tr_ch['C'].min():.2f}C  "
          f"end chg {-tr_ch['C'][-1]:.3f}C")
    assert tr_ch["soc"][-1] > 0.98 and -tr_ch["C"].min() <= d["C_chg"] + 1e-6
    dcold = dict(dch); dcold["T_start"] = 5.0
    tr_cd = simulate_pack(dcold, g, fl, masses, 5.0,
                          dict(kind="chg", t=np.arange(0.0, 3600.0, 2.0)))
    pk_cold = -tr_cd["C"].min()
    pk_early = -tr_cd["C"][: 40].min()          # first 80 s while still cold
    i08 = int(np.argmax(-tr_cd["C"] > 0.8))
    print(f"CC-CV 5°C start: first-80s chg {pk_early:.2f}C (plating cap "
          f"{plating_frac(5.0)*d['C_chg']:.2f}C at 5 °C); self-heats to 0.8C "
          f"by t={tr_cd['t'][i08]/60:.0f} min; run peak {pk_cold:.2f}C")
    assert pk_early < 0.35 * d["C_chg"] and pk_cold > pk_early
    xs_, Ts_ = goal_seek(d, fl, "flow_lpm", 0.5, 60.0, False, d["T_amb"], 2.0,
                         d["T_limit"], False)
    print(f"goal-seek flow for {d['T_limit']:.0f} °C at 2C: {xs_ if xs_ is None else round(xs_,1)} L/min -> {Ts_:.1f}")
    html = report_html(d, g, fl, res, masses, Cmax, [])
    assert "<html" in html and len(html) > 1500
    adf2 = compare_architectures(d, g, fl, masses, d["T_amb"], 2.0)
    assert len(adf2) == 4
    d46 = dict(d); d46.update(FORMATS["4680"], fmt="4680", Ns=96, Np=2,
                              pitch=0.054, cap_Ah=FORMATS["4680"]["cap"],
                              r_dc=FORMATS["4680"]["rdc"], m_cell=FORMATS["4680"]["mcell"])
    g46 = build_geometry(d46)
    r46 = solve_steady(d46, g46, fl, 1.0, d["T_amb"], C_rate=2.0)
    print(f"4680 sanity: {g46['N']} cells  T_b={r46['T_b']:.1f}  dT_core={r46['dT_core']:.1f} K")
    assert 20 < res["T_b"] < 90 and 0.1 < Cmax < 12
    assert abs(bm["T_b"][-1] - 32.3) < 4.0, "benchmark drifted"
    assert abs(c_fit - 1.35) < 0.15, "calibration fit drifted"
    assert res["u_ts"] > 1e-4, "thermosiphon dead at defaults"
    # ---- v5 checks ----
    ch = chiller_model(res["Q_w"], d["T_water_in"], d["T_amb"])
    print(f"chiller: duty {res['Q_w']/1000:.2f} kW  COP {ch['COP']:.1f}  "
          f"P_el {ch['P_el']:.0f} W")
    assert 1.5 < ch["COP"] < 12
    Q_bus = (d['C1']*d['cap_Ah']*d['Np'])**2 * busbar_props(d, g)['R']
    fig_s = heat_sankey(res, Q, Q_bus, 0.1, 0.0, ch)
    assert len(fig_s.data) == 1
    sts, totdT = station_list(d, g, fl, res, masses, Q, Q_bus, 0.1, 0.0, ch)
    assert len(sts) == 7 and totdT > 5
    sc = scale_to_C(d, g, fl, masses, d["T_amb"], 4.0)
    print(f"scale-to-4C: T_b={sc['r0']['T_b']:.1f} °C  heat "
          f"{sc['r0']['Q_eff']/1000:.2f} kW  chiller {sc['chiller']['P_el']/1000:.2f} kW el")
    for nm, x, T in sc["fixes"]:
        print(f"   4C single lever {nm:<28s} "
              f"{'insufficient alone' if x is None else f'{x:.2f} -> {T:.1f} °C'}")
    c_sweep_fig(d, g, fl, d["T_amb"], 2.0); setpoint_trade(d, g, fl, 2.0, d["T_amb"])
    figs = dict(sankey=fig_s, ladder=waterfall_chart(res, Q, d),
                resist=resistance_chart(res)[0], transient=fig_s,
                csweep=fig_s, setpoint=fig_s, pack3d=fig_s)
    spec_c = dict(kind="C", t=t, C=C)
    tr_s = simulate_pack(d, g, fl, masses, d["T_amb"], spec_c)
    secs = report_sections(d, g, fl, res, masses, tr_s, spec_c, Cmax, d["C1"],
                           Q, Q_bus, 0.1, 0.0, ch, figs)
    html = export_report_html(secs, figs)
    assert len(secs) == 7 and "<html" in html and len(html) > 8000
    print(f"report: {len(secs)} sections, HTML {len(html)//1000} kB")
    # ---- v7 checks: serpentine plates + circuit ----
    ds = dict(d, plate_on=True, u_oil=0.05, plate_t=0.0015, plate_contact=0.8)
    rs = solve_steady(ds, g, fl, 1.0, d["T_amb"], C_rate=d["C1"])
    A_pl, eta_pl, m_pl = plate_fin_area(ds, g, rs["h_tube"])
    sp = serpentine_pump(ds, g, fl, 0.05)
    dTp = sp["dT_est"](rs["Q_eff"])
    print(f"serpentine: T_b {res['T_b']:.1f} -> {rs['T_b']:.1f} °C  "
          f"plates +{A_pl:.1f} m² (eta {eta_pl:.2f}, {m_pl:.1f} kg)  "
          f"pump {sp['P']:.2f} W  channel dT {dTp:.1f} K  "
          f"spread {res['spread']:.1f} -> {rs['spread']:.1f} K")
    assert rs["T_b"] < res["T_b"] - 2.5 and sp["P"] < 5.0 and A_pl > 0.7
    assert dTp < 6.0, "parallel channels should carry the heat"
    fc = thermal_circuit_fig(d, g, fl, res, res["Q_eff"])
    assert len(fc.layout.shapes) >= 6
    nu_T_fig(fl)
    bmdf = load_benchmark()
    assert bmdf is not None and len(bmdf) >= 50
    assert bmdf["whkg"].notna().sum() >= 50 and bmdf["whl"].notna().sum() >= 50
    p_kg = 100 * (bmdf["whkg"].dropna() < masses["whkg_pack"]).mean()
    print(f"benchmark db: {len(bmdf)} packs  whkg {bmdf['whkg'].min():.0f}-"
          f"{bmdf['whkg'].max():.0f}  this design at {p_kg:.0f}th pct")
    print("SMOKE OK")


# ------------------------------------------------------------------ #
#  v3: Electrical model (OCV, entropic, plating derate, formats)      #
# ------------------------------------------------------------------ #
SOC_GRID = np.array([0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, 1.0])
OCV_GRID = np.array([3.00, 3.30, 3.45, 3.55, 3.61, 3.65, 3.69, 3.75, 3.83, 3.93,
                     4.03, 4.09, 4.20])          # generic graphite-NMC
DUDT_GRID = np.array([0.10, 0.05, 0.00, -0.05, -0.09, -0.11, -0.11, -0.10, -0.09,
                      -0.06, -0.03, -0.02, 0.00]) * 1e-3   # V/K, generic NMC shape
PLATE_T = np.array([-10, 0, 5, 10, 15, 20, 25, 45, 60])
PLATE_F = np.array([0.02, 0.08, 0.20, 0.45, 0.70, 0.90, 1.00, 1.00, 0.50])

def ocv(soc):  return float(np.interp(soc, SOC_GRID, OCV_GRID))
def dudt(soc): return float(np.interp(soc, SOC_GRID, DUDT_GRID))
def plating_frac(T_C): return float(np.interp(T_C, PLATE_T, PLATE_F))

# format presets: capacity/DCIR/mass from teardown literature
# (4680: About:Energy teardown 86.7 Wh -> ~24 Ah, 244 Wh/kg -> ~355 g;
#  DCIR ~5-6 mΩ mid-SoC is an estimate from published teardowns)
FORMATS = {
    "18650": dict(d_cell=0.018, h_cell=0.065, cap=3.4, rdc=30.0, mcell=0.048),
    "21700": dict(d_cell=0.021, h_cell=0.070, cap=5.0, rdc=25.0, mcell=0.070),
    "4680":  dict(d_cell=0.046, h_cell=0.080, cap=24.0, rdc=5.5, mcell=0.355),
}

# ------------------------------------------------------------------ #
#  v3: Drive-cycle library                                            #
#  Coarse (t [s], v [km/h]) breakpoints of the official profiles,     #
#  uniformly speed-scaled so the integrated distance matches the      #
#  official figure exactly. Adequate for thermal work (the pack       #
#  filters everything above ~0.01 Hz); swap in the official 1 Hz      #
#  trace via 'Custom CSV' for certification-grade inputs.             #
# ------------------------------------------------------------------ #
def _u(t0):   # one ECE-15 urban unit for NEDC
    return [(t0, 0), (t0+11, 0), (t0+15, 15), (t0+23, 15), (t0+28, 0), (t0+49, 0),
            (t0+54, 32), (t0+85, 32), (t0+96, 0), (t0+117, 0), (t0+122, 35),
            (t0+133, 50), (t0+155, 50), (t0+163, 35), (t0+176, 35), (t0+188, 0), (t0+195, 0)]

DRIVE_CYCLES = {
    "WLTP Class 3b": (23.27, [(0,0),(11,0),(26,25),(40,42),(48,48),(61,25),(96,0),(122,0),
        (135,30),(160,45),(175,56.5),(201,50),(240,25),(260,0),(300,0),(316,30),(345,49),
        (375,40),(390,20),(420,0),(465,0),(480,28),(511,46),(536,56.5),(556,45),(575,20),
        (589,0),(611,0),(640,40),(670,60),(700,76.6),(735,65),(770,50),(800,60),(840,70),
        (870,55),(900,40),(940,30),(980,45),(1005,25),(1022,0),(1060,0),(1090,40),(1130,65),
        (1180,85),(1230,97.4),(1290,90),(1350,80),(1400,60),(1440,30),(1477,0),(1500,20),
        (1530,60),(1570,90),(1620,110),(1660,125),(1700,131.3),(1740,120),(1765,90),
        (1785,40),(1800,0)]),
    "NEDC": (11.03, _u(0)+_u(195)+_u(390)+_u(585)+[(780,0),(790,35),(805,50),(830,70),
        (870,70),(880,50),(930,50),(940,70),(970,70),(985,100),(1035,100),(1050,120),
        (1090,120),(1120,80),(1160,30),(1180,0)]),
    "UDDS (FTP-75 city)": (12.07, [(0,0),(20,0),(35,40),(60,48),(90,25),(115,0),(125,0),
        (150,55),(185,75),(205,88),(230,91.2),(260,80),(300,60),(330,40),(345,0),(360,0),
        (380,45),(420,55),(450,40),(470,0),(500,30),(530,45),(560,30),(580,0),(610,40),
        (640,50),(670,35),(690,0),(720,40),(760,55),(800,45),(830,0),(860,35),(900,50),
        (930,30),(950,0),(980,40),(1020,55),(1060,45),(1090,25),(1110,0),(1140,35),
        (1180,48),(1220,40),(1260,55),(1300,45),(1340,20),(1369,0)]),
    "HWFET (highway)": (16.45, [(0,0),(30,40),(60,70),(100,80),(160,88),(220,78),(280,86),
        (340,92),(400,96.4),(460,88),(520,80),(580,86),(640,90),(700,75),(740,40),(765,0)]),
    "US06 (aggressive)": (12.89, [(0,0),(15,50),(30,90),(50,108),(70,112),(90,95),(110,60),
        (130,30),(145,0),(160,40),(180,80),(210,105),(250,120),(290,129.2),(330,125),
        (370,118),(410,125),(450,110),(480,80),(510,95),(540,60),(570,30),(600,0)]),
    "Artemis Motorway 130": (28.74, [(0,0),(30,60),(60,100),(100,118),(150,125),(200,131.8),
        (260,122),(320,128),(380,115),(430,125),(490,130),(550,118),(610,108),(660,90),
        (700,110),(760,125),(820,130),(880,120),(930,100),(980,70),(1030,30),(1068,0)]),
}

def cycle_speed(name):
    """1 s resampled speed [m/s], scaled to the official distance."""
    D_km, pts = DRIVE_CYCLES[name]
    tp = np.array([p[0] for p in pts], float)
    vp = np.array([p[1] for p in pts], float) / 3.6
    t = np.arange(0.0, tp[-1] + 1e-9, 1.0)
    v = np.interp(t, tp, vp)
    dist = np.trapezoid(v, t)
    v *= (D_km * 1000.0) / max(dist, 1.0)
    return t, v, D_km

def vehicle_battery_power(t, v, d):
    """Wheel power -> battery power [W] with drivetrain efficiency, capped
    regen, and constant accessory load. Positive = discharge."""
    a = np.gradient(v, t)
    P_wheel = d["veh_m"] * a * v + d["veh_m"] * G * d["Crr"] * v \
              + 0.5 * 1.20 * d["CdA"] * v ** 3
    P = np.where(P_wheel >= 0, P_wheel / max(d["eta_dt"], 0.05) + d["P_acc"],
                 P_wheel * d["eta_rg"] + d["P_acc"])
    return np.clip(P, -d["P_rg"] * 1000.0, None)

def busbar_props(d, g):
    """Total busbar resistance and mass. Auto: series run of length
    Ns x pitch at the design current density; override with R_bus > 0."""
    I_des = d["C1"] * d["cap_Ah"] * d["Np"]
    A_mm2 = max(I_des / d["bus_J"], 10.0)
    L = d["Ns"] * d["pitch"] * 1.15
    R_auto = 1.7e-8 * L / (A_mm2 * 1e-6)          # copper
    R = (d["R_bus"] * 1e-3) if d["R_bus"] > 0 else R_auto
    m = A_mm2 * 1e-6 * L * 8960.0 * 1.25          # + joints/terminals
    return dict(R=R, m=m, A_mm2=A_mm2)

def simulate_pack(d, g, fl, masses, T_amb, spec) -> dict:
    """v3 transient: electro-thermal. Tracks SoC, OCV, terminal voltage,
    CC-CV charge with plating-derated current, entropic heat, busbar heat.
    spec kinds: 'C' (array of C, thermal-only unless track_soc),
    'P' (battery power array, drive cycles), 'chg' (CC-CV), 'cyc' (cycling)."""
    loop = WATER_LOOP[d["loop_fluid"]]
    mdot = d["flow_lpm"] / 60.0 * loop["rho"] / 1000.0
    C_b, C_il = masses["C_batt"], masses["C_oil"]
    bus = busbar_props(d, g)
    t_arr = spec["t"]
    n = len(t_arr)
    T_b = np.full(n, d["T_start"]); T_il = np.full(n, d["T_start"])
    T_core = np.full(n, d["T_start"])
    soc = np.full(n, d["soc0"]); I_c = np.zeros(n); V_c = np.zeros(n)
    C_tr = np.zeros(n); Q_tr = np.zeros(n)
    rep = solve_steady(d, g, fl, 1.0, T_amb, C_rate=max(d["C1"], 0.5))
    R_wall, R_in, R_atm = rep["R_wall"], rep["R_in"], rep["R_atm"]
    ch = d.get("cal_h", 1.0)
    N, cap = g["N"], d["cap_Ah"]
    mode = "dis"; rest_t = 0.0; cyc_count = 0
    for i in range(1, n):
        dt = t_arr[i] - t_arr[i - 1]
        s = soc[i - 1]; Tb = T_b[i - 1]
        R_cell = r_of_T(d, Tb) * 1e-3
        U = ocv(s)
        # --- current demand per cell (discharge positive) ---
        if spec["kind"] == "C":
            I = spec["C"][i - 1] * cap * (1 if d["dirn"] == "Discharge" else -1)
        elif spec["kind"] == "P":
            P_cell = spec["P"][i - 1] / N
            Rq = d["chg_mult"] * R_cell if P_cell < 0 else R_cell
            disc = U * U - 4.0 * Rq * P_cell
            I = (U - math.sqrt(disc)) / (2.0 * Rq) if disc > 0 else U / (2.0 * Rq)
            if I < 0:   # regen: plating derate
                I = -min(-I, plating_frac(Tb) * d["C_chg"] * cap)
        elif spec["kind"] in ("chg", "cyc"):
            if spec["kind"] == "cyc":
                if mode == "dis" and s <= d["soc_min"]:
                    mode, rest_t = "rest1", 0.0
                elif mode == "rest1":
                    rest_t += dt
                    if rest_t >= d["cyc_rest"]: mode = "chg"
                elif mode == "chg" and s >= 0.999:
                    mode, rest_t, cyc_count = "rest2", 0.0, cyc_count + 1
                elif mode == "rest2":
                    rest_t += dt
                    if rest_t >= d["cyc_rest"]:
                        mode = "dis" if cyc_count < d["n_cyc"] else "done"
            else:
                mode = "chg" if s < 0.999 else "done"
            if mode == "dis":
                I = d["C1"] * cap
            elif mode == "chg":
                R_ch = d["chg_mult"] * R_cell
                I_cc = -plating_frac(Tb) * d["C_chg"] * cap
                V_at_cc = U - I_cc * R_ch
                if V_at_cc < d["v_max"]:
                    I = I_cc
                else:                       # CV phase
                    I = -(d["v_max"] - U) / R_ch
                    if -I < d["v_cut"] * cap:
                        I = 0.0
                        if spec["kind"] == "chg": mode = "done"
            else:
                I = 0.0
        # SoC bounds
        if spec["kind"] != "C" or d["track_soc"]:
            if (I > 0 and s <= 0.002) or (I < 0 and s >= 0.999 and spec["kind"] == "P"):
                I = 0.0
            soc[i] = min(max(s - I * dt / (3600.0 * cap), 0.0), 1.0)
        else:
            soc[i] = s
        Rq = d["chg_mult"] * R_cell if I < 0 else R_cell
        q_cell = I * I * Rq
        if d["entropic"]:
            q_cell -= I * (Tb + 273.15) * dudt(s)
        Q = N * q_cell + (I * d["Np"]) ** 2 * bus["R"]
        # --- thermal step (v2 core) ---
        u_ts, _ = thermosiphon_u(d, g, fl, max(Q, 1.0), 0.5 * (Tb + T_il[i-1]))
        u_eff = max(d["u_oil"], u_ts)
        cellf = h_cell_side(fl, Tb, T_il[i-1], d["h_cell"], d["d_cell"], g["gap_mm"], u_eff)
        T_wall_est = T_il[i-1] - 0.6 * (T_il[i-1] - d["T_water_in"])
        tubf = h_tube_side(fl, T_il[i-1], T_wall_est, d["tube_od"], u_eff)
        R_b = 1.0 / max(ch * cellf["h"] * g["A_cells"], 1e-9)
        R_ot = 1.0 / max(ch * tubf["h"] * rep["A_oilside"], 1e-9)
        Q_bi = (Tb - T_il[i-1]) / R_b
        Q_w = (T_il[i-1] - (d["T_water_in"] + 0.5 * max(Q_bi, 0) / max(mdot * loop["cp"], 1e-9))) \
              / (R_ot + R_wall + R_in)
        Q_a = (T_il[i-1] - T_amb) / R_atm
        T_b[i] = Tb + dt * (Q - Q_bi) / C_b
        T_il[i] = T_il[i-1] + dt * (Q_bi - Q_w - Q_a) / C_il
        T_core[i] = T_b[i] + (Q / N) * r_core(d)
        I_c[i] = I; V_c[i] = U - I * Rq; C_tr[i] = I / cap; Q_tr[i] = Q
    return dict(t=t_arr, T_b=T_b, T_il=T_il, T_core=T_core, soc=soc, I=I_c, V=V_c,
                C=C_tr, Q=Q_tr, C_rms=float(np.sqrt(np.mean(C_tr ** 2))),
                bus=bus, cycles_done=cyc_count)

# ------------------------------------------------------------------ #
#  v3: goal-seek, report export, production benchmarks                #
# ------------------------------------------------------------------ #
GOAL_LEVERS = {"Total water flow [L/min]": ("flow_lpm", 0.5, 60.0, False),
               "Number of tubes": ("n_tubes", 1, 60, True),
               "Stirring [m/s]": ("u_oil", 0.0, 0.20, False),
               "Water inlet [°C]": ("T_water_in", 0.0, 40.0, False),
               "Cell pitch [mm]": ("pitch", 0.022, 0.035, False)}

def goal_seek(d, fl, lever_key, lo, hi, is_int, T_amb, C_duty, T_target, invert):
    """Bisect one lever so steady cell T meets T_target. invert=True for
    levers where increasing the value makes the pack hotter (T_water_in)."""
    def T_at(x):
        dd = dict(d); dd[lever_key] = int(round(x)) if is_int else float(x)
        gg = build_geometry(dd)
        return solve_steady(dd, gg, fl, 1.0, T_amb, C_rate=C_duty)["T_b"]
    T_lo, T_hi = T_at(lo), T_at(hi)
    cool_end_is_hi = T_hi < T_lo
    if min(T_lo, T_hi) > T_target:
        return None, max(T_lo, T_hi) if invert else min(T_lo, T_hi)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if (T_at(mid) > T_target) == cool_end_is_hi:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    return (int(math.ceil(x)) if is_int else x), T_at(x)

# Production packs for the benchmark table. Sources (retrieved July 2026):
# teardown/certification data, not OEM datasheets - Tesla do not publish
# pack masses. M3 2170: Rickard teardown via Teslarati/EVANNEX (478 kg,
# 4416 cells, ~75 kWh usable). MY 4680: EU certification 447 kg / ~79 kWh
# gross (Electrek May 2026); Munro-derived 445 kg (batterydesign.net).
# Plaid: Munro 181.5 Wh/kg at ~99 kWh -> ~545 kg (derived); 760 kW vehicle
# peak (EVKX). M3 LFP: batterydesign.net 438 kg, 55 kWh, 125 Wh/kg.
# HPB80: batterydesign.net. Peak power = vehicle rating (battery-side
# peaks unpublished).
PRODUCTION_PACKS = [
    dict(Pack="This design", kWh=None, kg=None, Whkg=None, WhL=None, kWpk=None,
         Cooling="static immersion + internal water HX"),
    dict(Pack="Mercedes AMG HPB80", kWh=6.1, kg=89, Whkg=68.5, WhL=None, kWpk=150,
         Cooling="pumped dielectric immersion + external HX, 45 °C set point"),
    dict(Pack="Tesla Model 3 LR (2170)", kWh=75.0, kg=478, Whkg=157, WhL=None, kWpk=377,
         Cooling="glycol ribbon/serpentine side cooling, 4416 cells, 96S46P"),
    dict(Pack="Tesla Model Y (4680 structural)", kWh=79.0, kg=447, Whkg=177, WhL=None,
         kWpk=331, Cooling="glycol side ribbons retained; cells glued in steel tub"),
    dict(Pack="Tesla Model S Plaid (18650)", kWh=99.0, kg=545, Whkg=181.5, WhL=None,
         kWpk=760, Cooling="micro-channel glycol ribbons, 7920 cells, 110S72P"),
    dict(Pack="Tesla Model 3 LFP (CATL prismatic)", kWh=55.0, kg=438, Whkg=125, WhL=None,
         kWpk=239, Cooling="bottom cold plate under prismatic cells"),
]

def report_html(d, g, fl, res, masses, Cmax, figs) -> str:
    rows = [
        ("Pack", f"{masses['E_kwh']:.1f} kWh, {d['Ns']}S{d['Np']}P = {g['N']} x "
                 f"{d['fmt']}, {d['Ns']*d['v_nom']:.0f} V"),
        ("Coolant", f"{fl['name']}, {masses['V_oil_L']:.0f} L / {masses['m_oil']:.0f} kg, "
                    f"stirring {d['u_oil']*100:.1f} cm/s"),
        ("Internal HX", f"{d['n_tubes']} x {d['tube_od']*1000:.0f} mm {d['tube_mat']} tubes, "
                        f"{d['tube_plane']}, fins {'on' if d['fins_on'] else 'off'}"),
        ("Water loop", f"{d['flow_lpm']:.0f} L/min at {d['T_water_in']:.0f} °C"),
        ("Steady at duty", f"can {res['T_b']:.1f} / core {res['T_core']:.1f} °C at "
                           f"C_rms, spread {res['spread']:.1f} K, "
                           f"thermosiphon {res['u_ts']*1000:.1f} mm/s"),
        ("Capability", f"max continuous {Cmax:.2f}C to {d['T_limit']:.0f} °C "
                       f"({'core' if d['limit_core'] else 'can'})"),
        ("Mass", f"pack {masses['m_pack']:.0f} kg -> {masses['whkg_pack']:.0f} Wh/kg, "
                 f"{masses['whl_pack']:.0f} Wh/L (enclosure {masses['m_struct']:.0f} kg at "
                 f"{masses['enc']['t_mm']:.1f} mm eff.)"),
        ("Calibration", f"oil-film factor {d['cal_h']:.2f}"),
    ]
    tab = "".join(f"<tr><th style='text-align:left;padding:4px 12px 4px 0'>{k}</th>"
                  f"<td>{v}</td></tr>" for k, v in rows)
    parts = "".join(f.to_html(full_html=False, include_plotlyjs=False) for f in figs)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<title>Immersion Pack Lab report</title>
<style>body{{font-family:Georgia,serif;max-width:960px;margin:2em auto;color:#1F2933}}
h1{{border-bottom:3px solid #F59E0B}}table{{border-collapse:collapse;margin:1em 0}}</style>
</head><body><h1>Immersion Pack Lab - design report</h1>
<p>Generated by the app; model honesty: oil-side h +/-30% unless calibrated.
Wang et al. 2023 benchmark: 33.0 vs 32.3 °C measured.</p>
<table>{tab}</table>{parts}
<p style="font-size:0.85em;color:#666">Two-node network; Churchill-Chu / Churchill-Bernstein /
Hausen-Gnielinski correlations; DCIR(T), entropic heat, thermosiphon and busbar terms included.
Sources: Wang 2023 (J. Energy Storage 62, 106821); coolant_comparison_reviewed.xlsx.</p>
</body></html>"""


# ------------------------------------------------------------------ #
#  v4: 3D pack view                                                   #
# ------------------------------------------------------------------ #
def _add_cyl_z(V, F, I, xc, yc, z0, z1, r, val, n=10):
    """Append a vertical cylinder (side + top cap) to vertex/face lists."""
    b = len(V)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([xc + r * np.cos(th), yc + r * np.sin(th)])
    for x, y in ring: V.append((x, y, z0))
    for x, y in ring: V.append((x, y, z1))
    V.append((xc, yc, z1))
    I.extend([val] * (2 * n + 1))
    for k in range(n):
        k2 = (k + 1) % n
        F.append((b + k, b + k2, b + n + k))
        F.append((b + k2, b + n + k2, b + n + k))
        F.append((b + n + k, b + n + k2, b + 2 * n))

def _add_cyl_x(V, F, I, x0, x1, yc, zc, r, val, n=12):
    """Append a horizontal (along-x) open cylinder."""
    b = len(V)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([yc + r * np.cos(th), zc + r * np.sin(th)])
    for y, z in ring: V.append((x0, y, z))
    for y, z in ring: V.append((x1, y, z))
    I.extend([val] * (2 * n))
    for k in range(n):
        k2 = (k + 1) % n
        F.append((b + k, b + k2, b + n + k))
        F.append((b + k2, b + n + k2, b + n + k))

def _mesh(V, F, I, **kw):
    V = np.array(V); F = np.array(F)
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                     i=F[:, 0], j=F[:, 1], k=F[:, 2],
                     intensity=np.array(I), flatshading=True, **kw)

def pack_3d_figure(d, g, show_oil=True, show_tubes=True, show_box=True,
                   show_fins=True):
    fig = go.Figure()
    # --- cells, coloured by centre-vs-edge tendency ---
    V, F, I = [], [], []
    z0 = d["bottom_gap"]; z1 = z0 + d["h_cell"]
    cx, cy = g["Lx"] / 2, g["Ly"] / 2
    cnt = 0
    for r_ in range(g["n_rows"]):
        for c_ in range(g["n_cols"]):
            if cnt >= g["N"]: break
            x = d["edge_margin"] + (c_ + 0.5) * d["pitch"] \
                + (d["pitch"] / 2 if (d["arrangement"] == "Hexagonal" and r_ % 2) else 0)
            y = d["edge_margin"] + d["pitch"] / 2 + r_ * g["row_pitch"]
            val = 1.0 - math.hypot(x - cx, y - cy) / math.hypot(cx, cy)
            _add_cyl_z(V, F, I, x, y, z0, z1, d["d_cell"] / 2, val, n=10)
            cnt += 1
    fig.add_trace(_mesh(V, F, I, colorscale="RdYlBu_r", showscale=False,
                        name="cells", lighting=dict(ambient=0.55, diffuse=0.7)))
    # --- tubes (+ translucent fin envelope) ---
    if show_tubes:
        Vt, Ft, It = [], [], []
        Vf, Ff, If_ = [], [], []
        inter = d.get("tube_plane") == "Interstitial (between rows)"
        x0, x1 = d["manifold_margin"], g["Lx"] - d["manifold_margin"]
        for j_ in range(d["n_tubes"]):
            if inter:
                gaps = max(g["n_rows"] - 1, 1)
                yj = d["edge_margin"] + d["pitch"] / 2 \
                     + (j_ % gaps + 0.5) * g["row_pitch"] * (g["n_rows"] - 1) / gaps
                zj = z0 + d["h_cell"] * (0.3 + 0.4 * ((j_ // gaps) % 2))
            else:
                yj = (j_ + 0.5) * g["Ly"] / d["n_tubes"]
                zj = z1 + d["tube_zone"] / 2
            _add_cyl_x(Vt, Ft, It, x0, x1, yj, zj, d["tube_od"] / 2, 1.0)
            if d["fins_on"] and show_fins:
                _add_cyl_x(Vf, Ff, If_, x0, x1, yj, zj,
                           d["tube_od"] / 2 + d["fin_h"], 1.0, n=10)
        fig.add_trace(_mesh(Vt, Ft, It, colorscale=[[0, "#D97706"], [1, "#D97706"]],
                            showscale=False, name="tubes"))
        if Vf:
            fig.add_trace(_mesh(Vf, Ff, If_, colorscale=[[0, "#A5B4CC"], [1, "#A5B4CC"]],
                                showscale=False, opacity=0.28, name="fin envelope"))
    # --- oil fill ---
    if show_oil:
        fz = g["fill_h"]
        xs = [0, g["Lx"], g["Lx"], 0, 0, g["Lx"], g["Lx"], 0]
        ys = [0, 0, g["Ly"], g["Ly"], 0, 0, g["Ly"], g["Ly"]]
        zs = [0, 0, 0, 0, fz, fz, fz, fz]
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, alphahull=0, opacity=0.13,
                                color="#F59E0B", name="oil", hoverinfo="skip"))
    # --- enclosure wireframe ---
    if show_box:
        Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
        E = [((0,0,0),(Lx,0,0)),((Lx,0,0),(Lx,Ly,0)),((Lx,Ly,0),(0,Ly,0)),((0,Ly,0),(0,0,0)),
             ((0,0,Lz),(Lx,0,Lz)),((Lx,0,Lz),(Lx,Ly,Lz)),((Lx,Ly,Lz),(0,Ly,Lz)),((0,Ly,Lz),(0,0,Lz)),
             ((0,0,0),(0,0,Lz)),((Lx,0,0),(Lx,0,Lz)),((Lx,Ly,0),(Lx,Ly,Lz)),((0,Ly,0),(0,Ly,Lz))]
        ex, ey, ez = [], [], []
        for (a, b) in E:
            ex += [a[0], b[0], None]; ey += [a[1], b[1], None]; ez += [a[2], b[2], None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                   line=dict(color=INK, width=3),
                                   name="enclosure", hoverinfo="skip"))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.45, y=1.25, z=0.85)),
                   bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"{g['N']} x {d['fmt']} cells, {d['n_tubes']} tubes "
              f"({d['tube_plane'].lower()}), oil to {g['fill_h']*1000:.0f} mm")
    return fig


def thin_labels(xs, ys, names, logx=False, min_dx=0.10, min_dy=0.12):
    """Keep a label only if it does not crowd an already-kept one; everything
    remains identifiable on hover. Distances are fractions of axis span
    (log10 span when logx)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    xv = np.log10(np.maximum(xs, 1e-9)) if logx else xs
    xspan = max(xv.max() - xv.min(), 1e-9); yspan = max(ys.max() - ys.min(), 1e-9)
    order = np.argsort(-ys)                      # label the notable (high) first
    kept, out = [], [""] * len(xs)
    for idx in order:
        if all(abs(xv[idx] - xv[j]) / xspan > min_dx
               or abs(ys[idx] - ys[j]) / yspan > min_dy for j in kept):
            kept.append(idx); out[idx] = names[idx]
    return out

# ------------------------------------------------------------------ #
#  v4: input panels (workbench layout)                                #
# ------------------------------------------------------------------ #
def design_inputs(cool_df) -> dict:
    d = {}
    c1 = c2 = c3 = contextlib.nullcontext()
    with c1, st.container(border=True):
        st.markdown("##### Cells")
        d["fmt"] = _w(st.selectbox, "Format", "fmt", "21700",
                      options=list(FORMATS) + ["Custom"],
                      help="Presets from teardown data; 4680 DCIR is an estimate.")
        if d["fmt"] != "Custom" and st.button(f"Apply {d['fmt']} preset"):
            f_ = FORMATS[d["fmt"]]
            st.session_state["_pending"] = dict(
                w_dcell=f_["d_cell"]*1000, w_hcell=f_["h_cell"]*1000,
                w_cap=f_["cap"], w_rdc=f_["rdc"], w_mcell=f_["mcell"],
                w_pitch=max(st.session_state.get("w_pitch", 27.0),
                            f_["d_cell"]*1000 + 6.0))
            st.rerun()
        d["d_cell"] = _w(st.slider, "Diameter [mm]", "dcell", 21.0, min_value=10.0, max_value=60.0, step=0.5) / 1000
        d["h_cell"] = _w(st.slider, "Height [mm]", "hcell", 70.0, min_value=40.0, max_value=130.0, step=1.0) / 1000
        d["cap_Ah"] = _w(st.number_input, "Capacity [Ah]", "cap", 5.0, min_value=1.0, max_value=30.0, step=0.1)
        d["v_nom"] = _w(st.number_input, "Nominal V", "vnom", 3.7, min_value=3.0, max_value=4.0, step=0.05)
        d["r_dc"] = _w(st.number_input, "DCIR at 25 °C [mΩ]", "rdc", 25.0, min_value=2.0, max_value=80.0, step=0.5)
        d["k_dcir"] = _w(st.slider, "DCIR fall [%/K]", "kdcir", 1.2, min_value=0.0, max_value=3.0, step=0.1) / 100.0
        d["m_cell"] = _w(st.number_input, "Mass [kg]", "mcell", 0.070, min_value=0.03, max_value=0.6, step=0.001, format="%.3f")
        d["cp_cell"] = _w(st.number_input, "cp [J/kgK]", "cpcell", 950.0, min_value=700.0, max_value=1200.0, step=10.0)
        d["k_rad"] = _w(st.slider, "Radial k_r [W/mK]", "krad", 0.9, min_value=0.3, max_value=2.0, step=0.1)
        with st.expander("Electrical detail"):
            d["v_max"] = _w(st.slider, "Charge V limit", "vmax", 4.2, min_value=4.0, max_value=4.4, step=0.05)
            d["chg_mult"] = _w(st.slider, "Charge DCIR x", "chgm", 1.10, min_value=1.0, max_value=1.5, step=0.05)
            d["v_cut"] = _w(st.slider, "CV cut-off [C]", "vcut", 0.05, min_value=0.02, max_value=0.2, step=0.01)
            d["entropic"] = _w(st.checkbox, "Entropic heat", "entro", True)
        d["Ns"] = _w(st.number_input, "Series (S)", "Ns", 108, min_value=1, max_value=300)
        d["Np"] = _w(st.number_input, "Parallel (P)", "Np", 10, min_value=1, max_value=60)
        with st.popover("Suggest S x P"):
            tE = st.number_input("Target kWh", 1.0, 200.0, 20.0, 0.5)
            tV = st.number_input("Target V", 48.0, 900.0, 400.0, 10.0)
            if st.button("Apply suggestion"):
                vn = st.session_state.get("w_vnom", 3.7); cp_ = st.session_state.get("w_cap", 5.0)
                ns = max(int(round(tV / vn)), 1)
                st.session_state["_pending"] = dict(
                    w_Ns=ns, w_Np=max(int(round(tE * 1000 / (ns * vn * cp_))), 1))
                st.rerun()
    with c2, st.container(border=True):
        st.markdown("##### Layout and coolant")
        d["arrangement"] = _w(st.radio, "Arrangement", "arr", "Square",
                              options=["Square", "Hexagonal"], horizontal=True)
        d["pitch"] = _w(st.slider, "Cell pitch [mm]", "pitch", 27.0, min_value=22.0, max_value=60.0, step=0.5) / 1000
        d["edge_margin"] = _w(st.slider, "Edge margin [mm]", "edge", 10.0, min_value=5.0, max_value=40.0, step=1.0) / 1000
        d["tube_zone"] = _w(st.slider, "Tube zone height [mm]", "tz", 35.0, min_value=15.0, max_value=80.0, step=1.0) / 1000
        d["gas_gap"] = _w(st.slider, "Headspace [mm]", "gas", 10.0, min_value=0.0, max_value=40.0, step=1.0) / 1000
        d["end_fraction"] = _w(st.slider, "End-caps wetted", "ends", 0.0, min_value=0.0, max_value=1.0, step=0.1)
        names = list(cool_df["name"])
        d["coolant"] = _w(st.selectbox, "Fluid", "fluid",
                          "MIVOLT DF7" if "MIVOLT DF7" in names else names[0], options=names)
        with st.expander("ν(T) - viscosity falls with temperature (in the solver)"):
            st.plotly_chart(nu_T_fig(fluid_dict(
                cool_df[cool_df["name"] == d["coolant"]].iloc[0])),
                use_container_width=True, key="nuT")
        d["circ"] = _w(st.selectbox, "Circulation method", "circ",
                       "Thermosiphon only",
                       options=["Thermosiphon only", "Open stirring",
                                "Serpentine plates (guided)"],
                       help="Realistic options are compared in Learn and Ideas.")
        d["u_oil"] = 0.0
        d["plate_on"] = False
        if d["circ"] == "Open stirring":
            d["u_oil"] = _w(st.slider, "Stirring velocity [m/s]", "uoil", 0.0,
                            min_value=0.0, max_value=0.20, step=0.005)
        if d["circ"] == "Serpentine plates (guided)":
            d["plate_on"] = True
            d["u_guided"] = _w(st.slider, "Guided channel velocity [m/s]", "ugd",
                               0.05, min_value=0.0, max_value=0.15, step=0.005)
            d["u_oil"] = d["u_guided"]
            d["plate_t"] = _w(st.select_slider, "Plate thickness [mm]", "plt", 1.5,
                              options=[1.0, 1.5, 2.0]) / 1000
            d["plate_mat"] = _w(st.selectbox, "Plate material", "plm", "Aluminium",
                                options=["Aluminium", "Copper"])
            d["plate_contact"] = _w(st.slider, "Plate-to-tube contact factor",
                                    "plc", 0.8, min_value=0.4, max_value=1.0,
                                    step=0.05,
                                    help="Brazed ~0.9, clamped ~0.6")
        with st.expander("Holders"):
            d["m_holder_g"] = _w(st.slider, "Holder mass [g/cell]", "mhold", 8.0, min_value=0.0, max_value=25.0, step=1.0)
            d["holder_block"] = _w(st.slider, "Gap-flow blockage", "hblk", 0.20, min_value=0.0, max_value=0.6, step=0.05)
    with c3, st.container(border=True):
        st.markdown("##### Heat exchanger, water and structure")
        d["n_tubes"] = _w(st.slider, "Tubes", "ntub", 16, min_value=1, max_value=60)
        d["tube_od"] = _w(st.slider, "Tube OD [mm]", "tod", 10.0, min_value=4.0, max_value=25.0, step=0.5) / 1000
        d["tube_wall"] = _w(st.slider, "Wall [mm]", "twall", 1.0, min_value=0.5, max_value=3.0, step=0.25) / 1000
        d["tube_mat"] = _w(st.selectbox, "Tube material", "tmat", "Copper", options=list(K_TUBE))
        d["passes"] = _w(st.slider, "Passes", "pass", 1, min_value=1, max_value=4)
        d["tube_plane"] = _w(st.selectbox, "Tube plane", "tplane", "Top of pack",
                             options=["Top of pack", "Interstitial (between rows)",
                                      "Mid-height", "Below the cells"])
        d["fins_on"] = _w(st.checkbox, "Annular fins", "fins", True)
        if d["fins_on"]:
            d["fin_h"] = _w(st.slider, "Fin height [mm]", "finh", 8.0, min_value=2.0, max_value=20.0, step=0.5) / 1000
            d["fin_t"] = _w(st.slider, "Fin thickness [mm]", "fint", 0.5, min_value=0.2, max_value=1.5, step=0.1) / 1000
            d["fin_p"] = _w(st.slider, "Fin pitch [mm]", "finp", 4.0, min_value=2.0, max_value=12.0, step=0.5) / 1000
            d["fin_mat"] = _w(st.selectbox, "Fin material", "finm", "Aluminium", options=["Aluminium", "Copper"])
        d["loop_fluid"] = _w(st.selectbox, "Water loop fluid", "loopf", "Water", options=list(WATER_LOOP))
        d["flow_lpm"] = _w(st.slider, "Flow [L/min]", "flow", 10.0, min_value=0.5, max_value=60.0, step=0.5)
        d["T_water_in"] = _w(st.slider, "Inlet T [°C]", "twin", 20.0, min_value=0.0, max_value=40.0, step=1.0)
        with st.expander("Structure and busbars"):
            d["R_bus"] = _w(st.number_input, "Busbar R [mΩ] (0=auto)", "rbus", 0.0, min_value=0.0, max_value=20.0, step=0.1)
            d["bus_J"] = _w(st.slider, "Busbar J [A/mm²]", "busj", 5.0, min_value=2.0, max_value=10.0, step=0.5)
            d["sigma_MPa"] = _w(st.slider, "Allowable stress [MPa]", "sigma", 80.0, min_value=30.0, max_value=200.0, step=5.0)
            d["stiff"] = _w(st.slider, "Stiffening knock-down", "stiff", 0.45, min_value=0.2, max_value=1.0, step=0.05)
            d["p_des_bar"] = _w(st.slider, "Design pressure [bar g]", "pdes", 0.5, min_value=0.1, max_value=2.0, step=0.1)
    return d

def duty_inputs() -> dict:
    d = {}
    d["duty"] = _w(st.selectbox, "Duty profile", "duty", "Constant C",
                   options=["Constant C", "Drive cycle", "Charge (CC-CV)",
                            "Cycling (dis/chg x N)", "Fast charge then rest",
                            "Pulse train", "CSV upload"])
    d["C1"] = _w(st.slider, "C-rate (primary / discharge)", "c1", 2.0, min_value=0.2, max_value=8.0, step=0.1)
    if d["duty"] == "Constant C":
        d["dirn"] = _w(st.radio, "Direction", "dirn", "Discharge",
                       options=["Discharge", "Charge"], horizontal=True)
        d["track_soc"] = _w(st.checkbox, "Track SoC (off = thermal-only)", "tsoc", False)
    if d["duty"] == "Drive cycle":
        d["cycle"] = _w(st.selectbox, "Cycle", "cycle", "WLTP Class 3b", options=list(DRIVE_CYCLES),
                        help="Breakpoint profiles scaled to official distance; CSV upload for 1 Hz traces.")
        d["repeat_cyc"] = _w(st.checkbox, "Repeat to fill simulation", "repcyc", True)
        with st.expander("Vehicle"):
            d["veh_m"] = _w(st.number_input, "Mass [kg]", "vehm", 1900.0, min_value=600.0, max_value=4000.0, step=50.0)
            d["CdA"] = _w(st.number_input, "Cd x A [m2]", "cda", 0.62, min_value=0.3, max_value=1.5, step=0.01)
            d["Crr"] = _w(st.number_input, "Crr", "crr", 0.009, min_value=0.005, max_value=0.02, step=0.001, format="%.3f")
            d["eta_dt"] = _w(st.slider, "Drivetrain eff.", "etad", 0.92, min_value=0.7, max_value=0.98, step=0.01)
            d["eta_rg"] = _w(st.slider, "Regen recovery", "etar", 0.65, min_value=0.0, max_value=0.95, step=0.05)
            d["P_rg"] = _w(st.slider, "Regen cap [kW]", "prg", 60.0, min_value=0.0, max_value=300.0, step=5.0)
            d["P_acc"] = _w(st.number_input, "Accessories [W]", "pacc", 500.0, min_value=0.0, max_value=5000.0, step=50.0)
    if d["duty"] == "Cycling (dis/chg x N)":
        d["n_cyc"] = _w(st.slider, "Cycles", "ncyc", 3, min_value=1, max_value=10)
        d["cyc_rest"] = _w(st.slider, "Rest [s]", "crest", 600.0, min_value=0.0, max_value=3600.0, step=60.0)
        d["soc_min"] = _w(st.slider, "Discharge to SoC", "socmin", 0.10, min_value=0.0, max_value=0.5, step=0.05)
    if d["duty"] in ("Fast charge then rest", "Pulse train"):
        d["t1"] = _w(st.slider, "Primary phase [s]", "t1", 900.0, min_value=60.0, max_value=3600.0, step=30.0)
    if d["duty"] == "Pulse train":
        d["C2"] = _w(st.slider, "C-rate (secondary)", "c2", 0.5, min_value=0.0, max_value=4.0, step=0.1)
        d["t2"] = _w(st.slider, "Secondary phase [s]", "t2", 600.0, min_value=30.0, max_value=3600.0, step=30.0)
    if d["duty"] == "CSV upload":
        dup = st.file_uploader("Duty CSV: t_s and C (or P_kW)", type="csv", key="duty_up")
        if dup is not None:
            try:
                probe = dict(DEFAULTS)
                probe.update({k: st.session_state.get(f"w_{k2}", probe[k]) for k, k2 in
                              [("Ns", "Ns"), ("Np", "Np"), ("v_nom", "vnom"), ("cap_Ah", "cap")]})
                st.session_state["duty_csv"] = duty_from_csv(dup, probe)
                tt, cc = st.session_state["duty_csv"]
                st.caption(f"{len(tt)} points, {tt[-1]:.0f} s, peak {cc.max():.1f}C.")
            except Exception as e:
                st.error(f"Could not parse: {e}")
    if d["duty"] in ("Charge (CC-CV)", "Cycling (dis/chg x N)", "Drive cycle"):
        d["C_chg"] = _w(st.slider, "Charge rating (CC) [C]", "cchg", 1.0, min_value=0.2, max_value=4.0, step=0.1,
                        help="Plating map derates below 25 °C; also caps regen.")
        d["soc0"] = _w(st.slider, "Start SoC", "soc0", 0.90 if d["duty"] != "Charge (CC-CV)" else 0.20,
                       min_value=0.0, max_value=1.0, step=0.05)
    st.markdown("---")
    d["duration"] = _w(st.slider, "Simulation length [s]", "dur", 3600.0, min_value=300.0, max_value=14400.0, step=300.0)
    d["T_start"] = _w(st.slider, "Start temperature [°C]", "tstart", 25.0, min_value=-10.0, max_value=45.0, step=1.0)
    d["T_limit"] = _w(st.slider, "Cell limit [°C]", "tlim", 45.0, min_value=35.0, max_value=60.0, step=1.0)
    d["limit_core"] = _w(st.checkbox, "Apply limit to core", "limcore", False)
    d["T_amb"] = _w(st.slider, "Ambient [°C]", "tamb", 25.0, min_value=-10.0, max_value=45.0, step=1.0)
    return d

# ------------------------------------------------------------------ #
#  v4 main: workbench                                                 #
# ------------------------------------------------------------------ #
def main():
    st.set_page_config(page_title="Immersion Pack Lab", layout="wide", page_icon=None)
    st.markdown(CSS, unsafe_allow_html=True)
    for _k, _v in st.session_state.pop("_pending", {}).items():
        st.session_state[_k] = _v
    hero_box = st.container()
    kpi_box = st.container()

    if "cool_df" not in st.session_state:
        st.session_state.cool_df = _read_coolants()
    cool_df = st.session_state.cool_df

    tabs = st.tabs(["Design", "Duty", "Results", "Improve", "Ideas", "Safety",
                    "Compare", "Learn", "Validate", "Report"])

    with tabs[0]:
        colL, colR = st.columns([1.15, 1], gap="large")
        with colL:
            st.caption("Define the pack. The live view on the right follows "
                       "every change and stays with you as you scroll.")
            d = dict(DEFAULTS)
            d.update(design_inputs(cool_df))
    with tabs[1]:
        cduty, cplot = st.columns([1, 1.4], gap="large")
        with cduty:
            st.caption("What you ask of the pack. Steady state uses the duty's RMS C-rate.")
            d.update(duty_inputs())
    # model-tuning keys (widgets live in tab 8, read here pre-compute)
    for key, wkey in [("cal_h", "cal_h"), ("K_loop", "kloop"), ("h_ext", "hext"),
                      ("bottom_gap", "bgap"), ("manifold_margin", "mman"),
                      ("T_service_max", "tserv")]:
        default = DEFAULTS[key]
        v = st.session_state.get(f"w_{wkey}", default * (1000 if key in
                                 ("bottom_gap", "manifold_margin") else 1))
        d[key] = v / 1000 if key in ("bottom_gap", "manifold_margin") else v

    # ---------------- compute ---------------- #
    fl = fluid_dict(cool_df[cool_df["name"] == d["coolant"]].iloc[0])
    g = build_geometry(d)
    res0 = solve_steady(d, g, fl, 1.0, d["T_amb"], C_rate=d["C1"])
    masses = build_masses(d, g, fl, res0["fin"])
    Cmax = max_continuous_C(d, g, fl, d["T_amb"], d["T_limit"])
    if d["duty"] == "Drive cycle":
        tc, vc, D_km = cycle_speed(d["cycle"])
        if d["repeat_cyc"] and tc[-1] < d["duration"]:
            reps = int(math.ceil(d["duration"] / tc[-1]))
            vc = np.tile(vc, reps)[: int(d["duration"]) + 1]
            tc = np.arange(0.0, len(vc), 1.0)
        spec = dict(kind="P", t=tc, P=vehicle_battery_power(tc, vc, d), v=vc, D_km=D_km)
    elif d["duty"] == "Charge (CC-CV)":
        spec = dict(kind="chg", t=np.arange(0.0, d["duration"] + 1e-9, 2.0))
    elif d["duty"] == "Cycling (dis/chg x N)":
        spec = dict(kind="cyc", t=np.arange(0.0, d["duration"] + 1e-9, 2.0))
    else:
        csv_tc = st.session_state.get("duty_csv") if d["duty"] == "CSV upload" else None
        t0, C0 = duty_profile(d["duty"], d["duration"], d["C1"], d["t1"], d["C2"],
                              d["t2"], csv_tc)
        spec = dict(kind="C", t=t0, C=C0)
    tr = simulate_pack(d, g, fl, masses, d["T_amb"], spec)
    t_arr, C_arr = tr["t"], tr["C"]
    C_steady = max(tr["C_rms"], 0.05)
    res = solve_steady(d, g, fl, 1.0, d["T_amb"], C_rate=C_steady)
    Q_duty = res["Q_eff"]
    loop = WATER_LOOP[d["loop_fluid"]]
    P_pump = water_pump_power(d, g, loop)["P"]
    if d.get("plate_on"):
        _sp = serpentine_pump(d, g, fl, d["u_oil"])
        P_stir = _sp["P"]
    else:
        P_stir = stirrer_power(d, g, fl, d["u_oil"])
    Q_bus = (C_steady * d["cap_Ah"] * d["Np"]) ** 2 * tr["bus"]["R"]
    chil = chiller_model(res["Q_w"] + P_pump, d["T_water_in"], d["T_amb"])

    # ---------------- hero + KPI cards ---------------- #
    T_gov = res["T_core"] if d["limit_core"] else res["T_b"]
    ok = T_gov <= d["T_limit"]
    with hero_box:
        st.markdown(
            f"<div class='hero'><h1>Immersion Pack Lab</h1>"
            f"<p>{masses['E_kwh']:.1f} kWh / {d['Ns']*d['v_nom']:.0f} V - "
            f"{g['N']} x {d['fmt']} in {fl['name']} - {d['duty']}"
            f"{(' / ' + d['cycle']) if d['duty'] == 'Drive cycle' else ''}</p>"
            f"<span class='chip {'ok' if ok else 'bad'}'>"
            f"{'WITHIN LIMIT' if ok else 'OVER LIMIT'} - "
            f"{T_gov:.1f} / {d['T_limit']:.0f} °C at {C_steady:.2f}C rms"
            f"</span></div>", unsafe_allow_html=True)
    with kpi_box:
        kpi_cards([
            ("Cell can / core", f"{res['T_b']:.1f} / {res['T_core']:.1f}",
             f"{T_gov-d['T_limit']:+.1f} K vs limit", "ok" if ok else "bad"),
            ("Max continuous", f"{Cmax:.2f} C", f"to {d['T_limit']:.0f} °C",
             "brand"),
            ("Heat at duty", f"{Q_duty/1000:.2f} kW",
             f"chiller ~{chil['P_el']/1000:.2f} kW el", ""),
            ("Parasitics", f"{P_pump+P_stir:.0f} W",
             f"pump {P_pump:.1f}" + (f" + stir {P_stir:.1f}" if P_stir > 0 else ""),
             ""),
            ("Coolant", f"{masses['V_oil_L']:.0f} L",
             f"{masses['m_oil']:.0f} kg {fl['name'].split('(')[0].strip()}", ""),
            ("Pack mass", f"{masses['m_pack']:.0f} kg",
             f"{masses['whkg_pack']:.0f} Wh/kg | {masses['whl_pack']:.0f} Wh/L",
             ""),
        ])

    # ---------------- sidebar: status, save/load, report ---------------- #
    sb = st.sidebar
    sb.markdown(
        f"<div class='kpi'><div class='l'>Design status</div>"
        f"<div class='v {'ok' if ok else 'bad'}'>"
        f"{'Within limit' if ok else 'Over limit'}</div>"
        f"<div class='s'>{res['T_b']:.1f} °C at {C_steady:.2f}C rms - "
        f"thermosiphon {res['u_ts']*1000:.1f} mm/s - spread "
        f"{res['spread']:.1f} K</div></div>", unsafe_allow_html=True)
    if not ok:
        sb.error("Over limit - see Improve tab.")
    sb.markdown("---")
    with sb.expander("Save / load design"):
        state = {k: v for k, v in st.session_state.items()
                 if k.startswith("w_") and isinstance(v, (int, float, str, bool))}
        st.download_button("Download design (.json)", data=pd.Series(state).to_json(),
                           file_name="pack_design.json", use_container_width=True)
        up = st.file_uploader("Load design", type="json", key="design_up")
        if up is not None:
            sig = up.name + str(up.size)
            if st.session_state.get("applied_design") != sig:
                for k, v in pd.read_json(up, typ="series").to_dict().items():
                    if k.startswith("w_"):
                        st.session_state[k] = v
                st.session_state["applied_design"] = sig
                st.rerun()
    # (full narrative report lives in tab 9; sidebar keeps a shortcut)
    sb.caption("Full narrative report: Report tab.")
    sb.caption("Wang et al. 2023 benchmark: 33.0 vs 32.3 °C. Oil-side h honest to "
               "+/-30% until calibrated (tab 8).")

    # ---------------- Design tab: sticky live view ---------------- #
    with colR:
        with st.container(key="liveview"):
            with st.container(border=True):
                v1, v2, v3 = st.columns([1.4, 1, 1])
                view = v1.radio("View", ["3D", "Plan"], horizontal=True,
                                label_visibility="collapsed")
                s_oil = v2.toggle("Oil", True)
                s_box = v3.toggle("Box", True)
                s_tub, s_fin = True, True
                if view == "3D":
                    st.plotly_chart(pack_3d_figure(d, g, s_oil, s_tub, s_box,
                                                   s_fin),
                                    use_container_width=True, key="live3d",
                                    config=PLOTCFG)
                else:
                    st.plotly_chart(layout_figure(d, g),
                                    use_container_width=True, key="liveplan")
                T_gov0 = res["T_core"] if d["limit_core"] else res["T_b"]
                ok0 = T_gov0 <= d["T_limit"]
                pin = st.session_state.get("pinned")
                def _dlt(cur, key, fmt="{:+.1f}"):
                    if not pin or key not in pin: return ""
                    return f" ({fmt.format(cur - pin[key])} vs A)"
                kpi_cards([
                    ("Can / core", f"{res['T_b']:.1f} / {res['T_core']:.1f}°C",
                     f"{T_gov0-d['T_limit']:+.1f} K margin"
                     + _dlt(res['T_b'], 'T_can'), "ok" if ok0 else "bad"),
                    ("Spread", f"{res['spread']:.1f} K",
                     "criterion 5 K" + _dlt(res['spread'], 'Spread_K'), ""),
                    ("Mass", f"{masses['m_pack']:.0f} kg",
                     f"{masses['whkg_pack']:.0f} Wh/kg"
                     + _dlt(masses['m_pack'], 'Pack_kg', '{:+.0f} kg'), ""),
                ])
                st.caption(f"Box {g['Lx']*1000:.0f}×{g['Ly']*1000:.0f}×"
                           f"{g['Lz']*1000:.0f} mm - gap {g['gap_mm']:.1f} mm - "
                           f"circulation: {d['circ'].lower()}"
                           + (f" - plates +{res.get('A_plate',0):.1f} m² eff."
                              if d.get('plate_on') else ""))
        with st.expander("Mass and packaging audit"):
            mdf2 = pd.DataFrame([
                ["Cells", masses["m_cells"]], ["Coolant", masses["m_oil"]],
                ["Tubes", masses["m_tubes"]], ["Fins", masses["m_fins"]],
                ["Busbars", masses["m_bus"]], ["Cell holders", masses["m_holders"]],
                [f"Enclosure ({masses['enc']['t_mm']:.1f} mm eff.)",
                 masses["m_struct"]],
            ], columns=["Item", "kg"])
            mdf2["% of pack"] = 100 * mdf2["kg"] / masses["m_pack"]
            st.dataframe(mdf2.round(1), hide_index=True, use_container_width=True)
            st.caption(f"Worst cell ~{res['T_worst']:.1f} °C, best "
                       f"~{res['T_best']:.1f} (spread {res['spread']:.1f} K vs "
                       f"5 K). Buffer "
                       f"{(masses['C_oil']+masses['C_batt'])/1e3:.0f} kJ/K.")

    # ---------------- Duty tab: response ---------------- #
    with tabs[1]:
        with cplot:
            if spec["kind"] == "P":
                figV = go.Figure(go.Scatter(x=spec["t"]/60, y=spec["v"]*3.6,
                                            line=dict(color="#6366F1", width=2)))
                figV.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10),
                                   title=f"{d['cycle']} speed trace",
                                   xaxis_title="min", yaxis_title="km/h",
                                   plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figV, use_container_width=True)
            band = st.checkbox("Show +/-30% oil-film band", False)
            figT = go.Figure()
            if band:
                trs = []
                for cf in (0.7, 1.3):
                    dd = dict(d); dd["cal_h"] = d["cal_h"] * cf
                    trs.append(simulate_pack(dd, g, fl, masses, d["T_amb"], spec))
                figT.add_trace(go.Scatter(
                    x=np.concatenate([tr["t"], tr["t"][::-1]])/60,
                    y=np.concatenate([trs[0]["T_b"], trs[1]["T_b"][::-1]]),
                    fill="toself", fillcolor="rgba(192,57,43,0.12)",
                    line=dict(width=0), name="h +/-30%", hoverinfo="skip"))
            figT.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_core"], name="Core",
                                      line=dict(color="#B91C1C", width=2, dash="dot")))
            figT.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_b"], name="Cell surface",
                                      line=dict(color="#EF4444", width=3)))
            figT.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_il"], name="Bulk oil",
                                      line=dict(color=ACCENT, width=3)))
            figT.add_hline(y=d["T_limit"], line_dash="dash", line_color="#B91C1C")
            figT.add_trace(go.Scatter(x=t_arr/60, y=C_arr, name="C-rate", yaxis="y2",
                                      line=dict(color="#94A3B8", dash="dot")))
            if d["duty"] != "Constant C" or d.get("track_soc"):
                figT.add_trace(go.Scatter(x=tr["t"]/60, y=tr["soc"]*4, yaxis="y2",
                                          name="SoC (x4)", line=dict(color="#10B981", width=2)))
            figT.update_layout(height=380, xaxis_title="Time [min]",
                               yaxis_title="Temperature [°C]",
                               yaxis2=dict(title="C / SoCx4", overlaying="y", side="right",
                                           showgrid=False,
                                           range=[min(0, float(C_arr.min())*1.2),
                                                  max(float(np.abs(C_arr).max())*1.6, 4.2)]),
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend=dict(orientation="h", y=1.14),
                               hovermode="x unified",
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figT, use_container_width=True, config=PLOTCFG)
            if spec["kind"] == "P":
                st.caption(f"Official distance {spec['D_km']:.2f} km/cycle; peak "
                           f"{tr['C'].max():.2f}C / regen {-tr['C'].min():.2f}C; "
                           f"C_rms {tr['C_rms']:.2f}; SoC {tr['soc'][0]*100:.0f} -> "
                           f"{tr['soc'][-1]*100:.0f}%.")
            elif spec["kind"] in ("chg", "cyc"):
                st.caption(f"SoC {tr['soc'][0]*100:.0f} -> {tr['soc'][-1]*100:.0f}%; peak "
                           f"charge {-tr['C'].min():.2f}C (plating-derated); cycles "
                           f"{tr['cycles_done']}.")

    # ---------------- Results ---------------- #
    with tabs[2]:
        st.plotly_chart(heat_sankey(res, Q_duty, Q_bus, P_pump, P_stir, chil),
                        use_container_width=True, key="res_sankey", config=PLOTCFG)
        st.plotly_chart(thermal_circuit_fig(d, g, fl, res, Q_duty),
                        use_container_width=True, key="res_circuit",
                        config=PLOTCFG)
        st.plotly_chart(waterfall_chart(res, Q_duty, d),
                        use_container_width=True, key="res_ladder")
        st.markdown("#### The heat journey, station by station")
        st.caption("Follow the watts from inside the cell to the chiller. "
                   "Each card: what happens here, the numbers, and how to "
                   "improve this stage.")
        stations, totdT = station_list(d, g, fl, res, masses, Q_duty, Q_bus,
                                       P_pump, P_stir, chil)
        for name, dT, nums, imp in stations:
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                h1.markdown(f"**{name}**")
                if dT > 0:
                    h2.progress(min(dT / max(totdT, 1e-9), 1.0),
                                text=f"{dT:.1f} K ({100*dT/max(totdT,1e-9):.0f}%)")
                st.markdown(nums)
                with st.expander("How to improve this stage"):
                    st.markdown(imp)
        st.plotly_chart(setpoint_trade(d, g, fl, C_steady, d["T_amb"]),
                        use_container_width=True, key="res_setpoint")

    # ---------------- Improve ---------------- #
    with tabs[3]:
        with st.container(border=True):
            st.markdown("#### Predictor - what happens if...")
            st.caption("Play with the levers; nothing is saved to the design. "
                       "Every slider move re-solves instantly.")
            p1, p2, p3, p4, p5 = st.columns(5)
            pf_flow = p1.slider("Flow [L/min]", 0.5, 60.0, float(d["flow_lpm"]), 0.5)
            pf_tubes = p2.slider("Tubes", 1, 60, int(d["n_tubes"]))
            pf_u = p3.slider("Oil velocity [m/s]", 0.0, 0.15, float(d["u_oil"]), 0.005)
            pf_tin = p4.slider("Inlet [°C]", 0.0, 40.0, float(d["T_water_in"]), 1.0)
            pf_C = p5.slider("C-rate", 0.2, 6.0, float(round(C_steady, 1)), 0.1)
            dp_ = dict(d, flow_lpm=pf_flow, n_tubes=int(pf_tubes), u_oil=pf_u,
                       T_water_in=pf_tin)
            gp_ = build_geometry(dp_)
            rp_ = solve_steady(dp_, gp_, fl, 1.0, d["T_amb"], C_rate=pf_C)
            okp = rp_["T_b"] <= d["T_limit"]
            kpi_cards([
                ("Predicted can / core",
                 f"{rp_['T_b']:.1f} / {rp_['T_core']:.1f}°C",
                 f"{rp_['T_b']-res['T_b']:+.1f} K vs current",
                 "ok" if okp else "bad"),
                ("Margin to limit", f"{d['T_limit']-rp_['T_b']:+.1f} K",
                 f"at {pf_C:.1f}C continuous", "ok" if okp else "bad"),
                ("Spread", f"{rp_['spread']:.1f} K",
                 f"{rp_['spread']-res['spread']:+.1f} vs current", ""),
                ("Heat", f"{rp_['Q_eff']/1000:.2f} kW",
                 f"chiller ~{chiller_model(rp_['Q_w'], pf_tin, d['T_amb'])['P_el']:.0f} W el",
                 ""),
            ])
        cpin1, cpin2, _sp = st.columns([1, 1, 4])
        summary = dict(Fluid=fl["name"], T_can=round(res["T_b"], 1),
                       T_core=round(res["T_core"], 1), Cmax=round(Cmax, 2),
                       Spread_K=round(res["spread"], 1), Pack_kg=round(masses["m_pack"]),
                       Whkg=round(masses["whkg_pack"]), WhL=round(masses["whl_pack"]),
                       Parasitic_W=round(P_pump + P_stir, 1), Tubes=d["n_tubes"],
                       Flow_lpm=d["flow_lpm"], Stir=d["u_oil"])
        if cpin1.button("Pin design as A"):
            st.session_state["pinned"] = summary
        if "pinned" in st.session_state and cpin2.button("Clear pin"):
            del st.session_state["pinned"]
        if "pinned" in st.session_state:
            st.dataframe(pd.DataFrame([st.session_state["pinned"], summary],
                                      index=["A (pinned)", "B (current)"]),
                         use_container_width=True)
        with st.container(border=True):
            st.markdown("#### Scale the system to a target C-rate")
            st.caption("Pick a continuous C and see what each subsystem "
                       "needs - chiller, flow, tubes, stirring, busbars, "
                       "core, charging.")
            C_t = st.slider("Target continuous C", 0.5, 6.0,
                            float(min(max(d["C1"], 0.5), 6.0)), 0.5)
            if st.button("Analyse target C"):
                with st.spinner("Solving current design and single-lever "
                                "fixes at target C..."):
                    sc = scale_to_C(d, g, fl, masses, d["T_amb"], C_t)
                st.dataframe(pd.DataFrame(sc["rows"],
                             columns=["Subsystem", "At this design", "Change needed / note"]),
                             hide_index=True, use_container_width=True)
            st.plotly_chart(c_sweep_fig(d, g, fl, d["T_amb"], C_steady),
                            use_container_width=True, key="imp_csweep")
        improve_core(d, g, fl, masses, res, Q_duty, C_steady, cool_df)

    # ---------------- Ideas ---------------- #
    with tabs[4]:
        st.caption("Concepts tried against the live design. Baseline = this "
                   "design with no forced circulation. Adopt a winner via "
                   "Design - Circulation method.")
        with st.container(border=True):
            st.markdown("#### Idea A - serpentine plate circulation (yours)")
            st.markdown(
                "Thin plates hang from the tubes between cell rows: they act "
                "as **conduction fins** (both faces wetted, no water inside) "
                "and guide a **small pump** to push oil along every row "
                "channel - forced flow exactly where the gaps are tightest, "
                "immune to the sub-6 mm gap penalty. Channels are **parallel "
                "and manifolded**; a single long serpentine would heat the "
                "oil tens of kelvin end to end.")
            i1, i2, i3 = st.columns(3)
            iu = i1.slider("Channel velocity [m/s]", 0.01, 0.15, 0.05, 0.005,
                           key="id_u")
            it = i2.select_slider("Plate thickness [mm]", [1.0, 1.5, 2.0], 1.5,
                                  key="id_t")
            ic = i3.slider("Plate-tube contact", 0.4, 1.0, 0.8, 0.05, key="id_c")
            base = dict(d, plate_on=False, u_oil=0.0)
            stir = dict(d, plate_on=False, u_oil=iu)
            serp = dict(d, plate_on=True, u_oil=iu, plate_t=it / 1000,
                        plate_contact=ic)
            rows_i, bars = [], []
            for nm, dd_ in [("Baseline (thermosiphon)", base),
                            ("Open stirring", stir),
                            ("Serpentine plates", serp)]:
                r_ = solve_steady(dd_, g, fl, 1.0, d["T_amb"], C_rate=C_steady)
                if dd_.get("plate_on"):
                    P_ = serpentine_pump(dd_, g, fl, iu)["P"]
                    dm_ = plate_fin_area(dd_, g, r_["h_tube"])[2]
                else:
                    P_ = stirrer_power(dd_, g, fl, dd_["u_oil"])
                    dm_ = 0.0
                rows_i.append(dict(Concept=nm, T_can=round(r_["T_b"], 1),
                                   T_core=round(r_["T_core"], 1),
                                   Spread_K=round(r_["spread"], 1),
                                   Pump_W=round(P_, 1),
                                   Added_kg=round(dm_, 1),
                                   A_sink_m2=round(r_["A_oilside"], 1)))
                bars.append((nm, r_["T_b"], P_, dm_))
            idf = pd.DataFrame(rows_i)
            st.dataframe(idf, hide_index=True, use_container_width=True)
            figI = go.Figure()
            names_i = [b[0] for b in bars]
            figI.add_trace(go.Bar(name="Cell T [°C]", x=names_i,
                                  y=[b[1] for b in bars],
                                  marker_color="#EF4444",
                                  text=[f"{b[1]:.1f}" for b in bars],
                                  textposition="outside"))
            figI.add_trace(go.Bar(name="Pump [W]", x=names_i,
                                  y=[b[2] for b in bars],
                                  marker_color="#6366F1",
                                  text=[f"{b[2]:.1f}" for b in bars],
                                  textposition="outside"))
            figI.update_layout(barmode="group", height=320,
                               title=f"Concepts at {C_steady:.2f}C rms "
                                     f"(limit {d['T_limit']:.0f} °C)",
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(figI, use_container_width=True, key="ideas_bar")
            dT_serp = rows_i[0]["T_can"] - rows_i[2]["T_can"]
            dT_stir = rows_i[0]["T_can"] - rows_i[1]["T_can"]
            st.markdown(
                f"**Verdict at this operating point:** serpentine buys "
                f"**{dT_serp:.1f} K** over baseline for "
                f"{rows_i[2]['Pump_W']:.1f} W and {rows_i[2]['Added_kg']:.0f} kg "
                f"of plates (open stirring buys {dT_stir:.1f} K for "
                f"{rows_i[1]['Pump_W']:.1f} W). The plates also cut the spread "
                f"({rows_i[2]['Spread_K']:.1f} vs {rows_i[0]['Spread_K']:.1f} K) "
                "because the sink is distributed. Risks: plate-to-tube joint "
                "quality (the contact slider), channel blockage, manifold "
                "design, and serviceability.")

    # ---------------- Safety ---------------- #
    with tabs[5]:
        st.markdown("Order-of-magnitude screening plus the engineering checklist. "
                    "Nothing here replaces abuse testing.")
        runaway_ui(d, g, fl, masses, res)
        exp_L = fl["beta"] * masses["V_oil_L"] * (d["T_service_max"] - 0.0)
        st.markdown(f"""
**Expansion and leaks.**
* Oil expansion over a 0 to {d['T_service_max']:.0f} °C service band:
  **~{exp_L:.1f} L** on {masses['V_oil_L']:.0f} L - bellows or bladder, never free air.
* Hold **oil pressure above water pressure** so any tube leak goes oil-to-water
  (transformer practice); or double-walled tubes with interstitial leak detection.
* Burst disc set at {d['p_des_bar']:.1f} bar g (the enclosure is sized for this in
  Design); route the vent away from occupants and expect oil ejection.
* Ester fluids: monitor moisture; copper: use inhibited oil or plated tubes.""")

    # ---------------- Compare ---------------- #
    with tabs[6]:
        st.subheader("Architectures")
        arch_tab(d, g, fl, masses, C_steady)
        st.markdown("---")
        st.subheader("Coolant shoot-out")
        coolant_tab(d, g, cool_df)
        st.markdown("---")
        st.subheader("BEV road-car pack database")
        fig_bm = benchmark_db_tab(d, g, masses, Cmax, res)
        st.markdown("---")
        st.subheader("Teardown cooling references")
        bench_prod_tab(masses, Cmax)

    # ---------------- Learn ---------------- #
    with tabs[7]:
        learn_tab(d, g, fl, res, masses, cool_df, loop)
        with st.expander("11. Moving the oil: the realistic options"):
            st.markdown(f"""
| Option | How | Power | Reaches tight gaps? | Notes |
|---|---|---|---|---|
| Thermosiphon only | buoyancy loop | 0 W | weakly ({res['u_ts']*1000:.1f} mm/s here) | free; dies if the cold plane is low |
| Magnetically coupled stirrer | sealed impeller in the bulk | ~1-3 W | no - flow bypasses the gaps | cheapest forced option; what "Open stirring" models |
| Pump + jet manifold | small pump, nozzles along a wall | ~2-5 W | partially | directional; nozzle fouling |
| **Serpentine plates (guided)** | plates between rows form parallel channels; small pump | **~0.5 W at 5 cm/s** | **yes - every gap sees forced flow** | plates double as conduction fins (+~2 m² eff.); needs manifold + brazed joints |
| Oscillating plate | reciprocating drive | ~2-10 W | partially | seals and fatigue |
| Full pumped immersion | external HX loop | 20+ W | yes | the AMG HPB80 architecture |

The guided-serpentine option is your Idea A: compare it live in the **Ideas**
tab and adopt it via **Design - Circulation method**. Viscosity falls with
temperature (Andrade law) and the solver already evaluates every film at its
local film temperature - see the ν(T) curve in Design.""")

    # ---------------- Report ---------------- #
    figs_r = dict(
        sankey=heat_sankey(res, Q_duty, Q_bus, P_pump, P_stir, chil),
        ladder=waterfall_chart(res, Q_duty, d),
        resist=resistance_chart(res)[0],
        transient=None,   # filled below from the Duty chart data
        csweep=c_sweep_fig(d, g, fl, d["T_amb"], C_steady),
        setpoint=setpoint_trade(d, g, fl, C_steady, d["T_amb"]),
        pack3d=pack_3d_figure(d, g),
    )
    figT_r = go.Figure()
    figT_r.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_core"], name="Core",
                                line=dict(color="#B91C1C", width=2, dash="dot")))
    figT_r.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_b"], name="Cell",
                                line=dict(color="#EF4444", width=3)))
    figT_r.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_il"], name="Oil",
                                line=dict(color=ACCENT, width=3)))
    figT_r.add_hline(y=d["T_limit"], line_dash="dash", line_color="#B91C1C")
    figT_r.update_layout(height=320, xaxis_title="min",
                         yaxis_title="°C", plot_bgcolor="rgba(255,255,255,0)",
                         paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.15))
    figs_r["transient"] = figT_r
    secs = report_sections(d, g, fl, res, masses, tr, spec, Cmax, C_steady,
                           Q_duty, Q_bus, P_pump, P_stir, chil, figs_r)
    with tabs[9]:
        cbt, _ = st.columns([1, 3])
        cbt.download_button("Download this report (.html)",
                            data=export_report_html(secs, figs_r),
                            file_name="pack_design_report.html",
                            mime="text/html", use_container_width=True)
        render_report_tab(secs, figs_r)

    # ---------------- Validate and tune ---------------- #
    with tabs[8]:
        st.subheader("Benchmark: Wang et al. 2023")
        bench_wang_tab()
        st.markdown("---")
        st.subheader("Calibrate to your rig")
        mup = st.file_uploader("Measured CSV: columns t_s, T_cell", type="csv", key="meas_up")
        if mup is not None:
            try:
                mdf = pd.read_csv(mup)
                cols = {c.lower().strip(): c for c in mdf.columns}
                st.session_state["meas_data"] = (
                    mdf[cols["t_s"]].to_numpy(float), mdf[cols["t_cell"]].to_numpy(float))
                st.success(f"Loaded {len(mdf)} points.")
            except Exception as e:
                st.error(f"Could not parse: {e}")
        meas = st.session_state.get("meas_data")
        if meas is not None:
            figM = go.Figure()
            figM.add_trace(go.Scatter(x=tr["t"]/60, y=tr["T_b"], name="Model",
                                      line=dict(color="#EF4444", width=3)))
            figM.add_trace(go.Scatter(x=meas[0]/60, y=meas[1], mode="markers",
                                      name="Measured", marker=dict(color=INK, size=5)))
            figM.update_layout(height=300, xaxis_title="min", yaxis_title="°C",
                               plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figM, use_container_width=True)
            if st.button("Fit oil-film calibration factor"):
                with st.spinner("Sweeping 0.4-2.2..."):
                    cb, rm = fit_calibration(d, g, fl, masses, d["T_amb"],
                                             tr["t"], np.abs(tr["C"]), meas[0], meas[1])
                st.session_state["_pending"] = dict(w_cal_h=cb)
                st.success(f"cal_h = {cb:.2f} (RMSE {rm:.2f} K) - applied below.")
                st.rerun()
        st.markdown("---")
        st.subheader("FEA studies (pack_fea_v1)")
        f1, f2, f3 = st.columns(3)
        with f1:
            figF1 = go.Figure(go.Bar(
                x=["Flat lid (FE)", "Best grillage (FE)", "App assumption"],
                y=[25.9, 14.6, 11.7], marker_color=["#94A3B8", "#6366F1", "#F59E0B"],
                text=["25.9", "14.6", "11.7"], textposition="outside"))
            figF1.update_layout(height=260, title="Lid mass [kg] at 0.5 bar g",
                                margin=dict(l=10, r=10, t=44, b=10))
            st.plotly_chart(figF1, use_container_width=True, key="fea1")
            st.caption("Roark validated to +0.0%. Honest knock-down 0.56 vs "
                       "the app's 0.45 (composite rib action would close the "
                       "gap) - consider 0.50.")
        with f2:
            figF2 = go.Figure()
            figF2.add_trace(go.Bar(name="App formula", x=["21700 2C", "4680 2C"],
                                   y=[2.85, 14.0], marker_color="#F59E0B"))
            figF2.add_trace(go.Bar(name="FE (mandrel + wetted ends)",
                                   x=["21700 2C", "4680 2C"], y=[2.28, 9.30],
                                   marker_color="#6366F1"))
            figF2.update_layout(barmode="group", height=260,
                                title="Core-to-can ΔT [K]",
                                legend=dict(orientation="h", y=1.2),
                                margin=dict(l=10, r=10, t=44, b=10))
            st.plotly_chart(figF2, use_container_width=True, key="fea2")
            st.caption("Exact-solution validation -0.1%. The app is 20-34% "
                       "conservative on real cells.")
        with f3:
            figF3 = go.Figure(go.Bar(
                x=["App estimate", "Section model, co-flow", "Counterflow"],
                y=[3.25, 1.08, 0.06],
                marker_color=["#F59E0B", "#6366F1", "#10B981"],
                text=["3.25", "1.08", "0.06"], textposition="outside"))
            figF3.update_layout(height=260,
                                title="Water-rise cell spread [K]",
                                margin=dict(l=10, r=10, t=44, b=10))
            st.plotly_chart(figF3, use_container_width=True, key="fea3")
            st.caption("Energy closure 0.00%. Counterflow plumbing of "
                       "alternate tubes removes the term for free.")
        st.caption("Full self-validating scripts and figures: pack_fea_v1.zip "
                   "(fea1_lid.py, fea2_cell.py, fea3_pack.py, REPORT.md).")
        st.markdown("---")
        st.subheader("Model tuning")
        t1_, t2_, t3_ = st.columns(3)
        with t1_:
            _w(st.slider, "Oil-film calibration factor", "cal_h", 1.0, min_value=0.4, max_value=2.2, step=0.01)
            _w(st.slider, "Thermosiphon minor-loss K", "kloop", 5.0, min_value=0.0, max_value=20.0, step=0.5)
        with t2_:
            _w(st.slider, "Casing-to-ambient h [W/m²·K]", "hext", 5.0, min_value=0.0, max_value=25.0, step=0.5)
            _w(st.slider, "Gap under cells [mm]", "bgap", 5.0, min_value=0.0, max_value=20.0, step=1.0)
        with t3_:
            _w(st.slider, "Manifold margin [mm]", "mman", 20.0, min_value=0.0, max_value=60.0, step=5.0)
            _w(st.slider, "Max service T (expansion) [°C]", "tserv", 60.0, min_value=40.0, max_value=90.0, step=5.0)
        if st.checkbox("Edit fluid property table", False):
            st.session_state.cool_df = st.data_editor(cool_df, num_rows="dynamic", height=260)
        st.caption("These knobs change the physics for every tab. cal_h multiplies both "
                   "oil films; fit it above once rig data exists (spray-app pattern).")

# ------------------------------------------------------------------ #
#  v5: chiller model, heat Sankey, station cards, scale-to-C          #
# ------------------------------------------------------------------ #
def chiller_model(Q_w, T_w_in, T_amb, eta=0.45, approach=5.0, cond_dT=10.0):
    """Vapour-compression estimate: COP = eta_carnot x T_evap / lift.
    Evaporator = inlet - approach; condenser = ambient + cond_dT."""
    T_c = T_w_in - approach + 273.15
    T_h = T_amb + cond_dT + 273.15
    lift = max(T_h - T_c, 3.0)
    COP = max(eta * T_c / lift, 0.4)
    return dict(COP=COP, P_el=Q_w / COP, lift=lift)

def heat_sankey(res, Q_duty, Q_bus, P_pump, P_stir, chil):
    """Left = sources, right = sinks. Link width = watts; colours follow the
    physical stream (red heat, amber oil, blue water, grey rejection)."""
    kw = lambda v: f"{v/1000:.2f} kW" if v >= 100 else f"{v:.0f} W"
    Qc = max(Q_duty - Q_bus, 1e-3)
    lab = [f"Cells<br>{kw(Qc)}", f"Busbars<br>{kw(Q_bus)}",
           f"Bulk oil<br>{kw(Q_duty)}", f"Water loop<br>{kw(res['Q_w'])}",
           f"Casing loss<br>{kw(res['Q_atm'])}",
           f"Chiller<br>{kw(res['Q_w']+P_pump)} duty",
           f"To ambient<br>{kw(res['Q_w']+P_pump+chil['P_el'])}",
           f"Chiller electricity<br>{kw(chil['P_el'])}"]
    node_c = ["#EF4444", "#D97706", "#F59E0B", "#0EA5E9", "#94A3B8",
              "#6366F1", "#64748B", "#10B981"]
    x = [0.01, 0.01, 0.28, 0.55, 0.99, 0.78, 0.99, 0.55]
    y = [0.30, 0.75, 0.42, 0.35, 0.88, 0.42, 0.32, 0.80]
    srcs = [0, 1, 2, 2, 3, 7, 5]
    dsts = [2, 2, 3, 4, 5, 5, 6]
    vals = [Qc, max(Q_bus, 1e-3), max(res["Q_w"], 1e-3),
            max(res["Q_atm"], 1e-3), max(res["Q_w"] + P_pump, 1e-3),
            max(chil["P_el"], 1e-3),
            max(res["Q_w"] + P_pump + chil["P_el"], 1e-3)]
    def rgba(hx, a=0.35):
        return f"rgba({int(hx[1:3],16)},{int(hx[3:5],16)},{int(hx[5:7],16)},{a})"
    link_c = [rgba(node_c[s]) for s in srcs]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=lab, x=x, y=y, pad=22, thickness=20, color=node_c,
                  line=dict(width=0),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=srcs, target=dsts, value=vals, color=link_c,
                  hovertemplate="%{source.label} to %{target.label}: "
                                "%{value:.0f} W<extra></extra>")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=56, b=10),
                      font=dict(size=12),
                      title="Heat flow map: every watt from the cells to the "
                            "air outside (hover any ribbon)")
    return fig


def thermal_circuit_fig(d, g, fl, res, Q):
    """The cell-to-water chain as an interactive circuit. Box height = share
    of total resistance; colour = transport mode; hover = the governing law
    with live numbers; the weakest link is flagged."""
    el = [
        ("Cell interior", "conduction", res["dT_core"] / max(Q, 1e-9),
         f"Radial conduction through the jellyroll<br>"
         f"R = 1/(4*pi*k_r*H) = {r_core(d)*1000:.1f} mK/W per cell<br>"
         f"k_r = {d['k_rad']:.1f} W/m*K -> DT = {res['dT_core']:.1f} K"),
        ("Can to oil film", "convection", res["R_b"],
         f"Natural/forced convection film<br>h = {res['h_cell']:.0f} W/m²·K "
         f"over A = {g['A_cells']:.1f} m²<br>R = 1/(hA) = "
         f"{res['R_b']*1000:.2f} mK/W -> DT = {Q*res['R_b']:.1f} K"),
        ("Oil to tube+fins" + ("+plates" if d.get("plate_on") else ""),
         "convection", res["R_ot"],
         f"Convection onto the finned sink<br>h = {res['h_tube']:.0f} W/m²·K, "
         f"A_eff = {res['A_oilside']:.1f} m²"
         + (f" (plates add {res.get('A_plate',0):.1f} m2, eta "
            f"{res.get('eta_plate',0):.2f})" if d.get("plate_on") else "")
         + f"<br>R = {res['R_ot']*1000:.2f} mK/W -> DT = {Q*res['R_ot']:.1f} K"),
        ("Tube wall", "conduction", res["R_wall"],
         f"Conduction through {d['tube_mat'].lower()}<br>"
         f"R = ln(r_o/r_i)/(2*pi*k*L*n) = {res['R_wall']*1000:.3f} mK/W "
         f"-> DT = {Q*res['R_wall']*1000:.0f} mK"),
        ("Water film", "convection", res["R_in"],
         f"{res['water_regime']} internal flow, Re = {res['Re_water']:.0f}<br>"
         f"h = {res['h_water']:.0f} W/m²·K -> R = {res['R_in']*1000:.2f} mK/W "
         f"-> DT = {Q*res['R_in']:.1f} K"),
        ("Water stream", "advection", 0.5 * res["dT_water"] / max(Q, 1e-9),
         f"Heat carried away: DT = Q/(m_dot*c_p)<br>rise "
         f"{res['dT_water']:.1f} K inlet to outlet (+{0.5*res['dT_water']:.1f} "
         "K at the mean cell)"),
    ]
    Rtot = sum(e[2] for e in el)
    films = [e for e in el if e[1] == "convection"]
    weakest = max(films, key=lambda e: e[2])[0]
    mode_c = dict(conduction="#94A3B8", convection="#6366F1", advection="#0EA5E9")
    fig = go.Figure()
    x = 0.0
    xw = 1.0 / len(el)
    hover_x, hover_y, hover_t = [], [], []
    for name, mode, R, tip in el:
        share = R / max(Rtot, 1e-12)
        h = 0.16 + 0.74 * share
        is_w = name == weakest
        fig.add_shape(type="rect", x0=x + 0.06 * xw, x1=x + 0.94 * xw,
                      y0=0.5 - h / 2, y1=0.5 + h / 2,
                      line=dict(color="#EF4444" if is_w else "#E2E8F0",
                                width=3 if is_w else 1),
                      fillcolor=mode_c[mode], opacity=0.9 if is_w else 0.75)
        fig.add_annotation(x=x + 0.5 * xw, y=0.5 + h / 2 + 0.07,
                           text=f"<b>{Q*R:.1f} K</b>" if Q * R >= 0.05
                                else f"{Q*R*1000:.0f} mK",
                           showarrow=False, font=dict(size=12, color="#0F172A"))
        fig.add_annotation(x=x + 0.5 * xw, y=0.5 - h / 2 - 0.08,
                           text=name.replace(" ", "<br>", 1), showarrow=False,
                           font=dict(size=10.5, color="#475569"))
        if is_w:
            fig.add_annotation(x=x + 0.5 * xw, y=0.97, text="WEAKEST LINK",
                               showarrow=False,
                               font=dict(size=10, color="#B91C1C"))
        hover_x.append(x + 0.5 * xw); hover_y.append(0.5)
        hover_t.append(f"<b>{name}</b> - {mode}<br>{tip}<br>"
                       f"share of chain: {100*share:.0f}%")
        if x > 0:
            pass
        x += xw
    for i in range(len(el) - 1):
        fig.add_annotation(x=(i + 1) * xw, y=0.5, text="", showarrow=True,
                           ax=(i + 0.94) * xw, ay=0.5, axref="x", ayref="y",
                           arrowhead=2, arrowwidth=2, arrowcolor="#CBD5E1")
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers",
                             marker=dict(size=42, opacity=0.0),
                             hovertext=hover_t, hoverinfo="text",
                             showlegend=False))
    for mode, c in mode_c.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=c, symbol="square"),
                                 name=mode))
    fig.update_layout(height=330, xaxis=dict(visible=False, range=[0, 1]),
                      yaxis=dict(visible=False, range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.05),
                      margin=dict(l=10, r=10, t=54, b=10),
                      title=f"The thermal circuit at {Q/1000:.2f} kW: box "
                            "height = share of resistance, hover for the law")
    return fig

def nu_T_fig(fl):
    Ts = np.linspace(0, 80, 60)
    nus = [film_props(fl, t)["nu"] * 1e6 for t in Ts]
    fig = go.Figure(go.Scatter(x=Ts, y=nus, line=dict(color="#F59E0B", width=3)))
    fig.update_layout(height=210, xaxis_title="Oil temperature [°C]",
                      yaxis_title="ν [cSt]",
                      title=f"{fl['name']}: viscosity vs temperature "
                            "(Andrade law - the solver uses this at film T)",
                      margin=dict(l=10, r=10, t=44, b=10))
    return fig

def station_list(d, g, fl, res, masses, Q_duty, Q_bus, P_pump, P_stir, chil,
                 cf_tip=True):
    """The heat journey as (name, dT_or_metric, numbers_md, improve_md)."""
    Q = Q_duty
    dTs = dict(core=res["dT_core"], f1=Q * res["R_b"], f2=Q * res["R_ot"],
               wall=Q * res["R_wall"], f3=Q * res["R_in"],
               rise=0.5 * res["dT_water"])
    tot = sum(dTs.values())
    S = []
    S.append(("1. Heat source - inside the cell",
        dTs["core"],
        f"**{Q/1000:.2f} kW** at {res['T_b']:.1f} °C "
        f"(DCIR {r_of_T(d, res['T_b']):.1f} mΩ vs {d['r_dc']:.0f} at 25; the "
        f"hot pack makes {100*(1-r_of_T(d,res['T_b'])/d['r_dc']):.0f}% less heat). "
        f"Busbars add {Q_bus:.0f} W. Core runs **{res['dT_core']:.1f} K** above "
        f"the can (R = 1/(4 pi k_r H)).",
        f"Lower-DCIR cells cut heat at the source; a warmer set point does too "
        f"(the AMG 45 °C logic). Busbars: thicker section (now sized at "
        f"{d['bus_J']:.0f} A/mm²). The core term ignores the mandrel and wetted "
        f"ends - FEA shows the real figure is ~20% lower (21700) to ~34% (4680); "
        f"only lower current or axial extraction move it further."))
    S.append(("2. Can to oil - the first film",
        dTs["f1"],
        f"h_cell = **{res['h_cell']:.0f} W/m²·K** over {g['A_cells']:.1f} m² -> "
        f"**{dTs['f1']:.1f} K**. Gap factor {res['gapf']:.2f} at "
        f"{g['gap_mm']:.1f} mm; flow component from "
        f"{'stirring' if d['u_oil']>res['u_ts'] else 'thermosiphon'} "
        f"({max(d['u_oil'],res['u_ts'])*1000:.1f} mm/s).",
        "Levers in order: keep the gap at or above 6 mm; stir a few cm/s "
        f"({stirrer_power(d,g,fl,0.05):.1f} W buys the biggest single film gain); "
        "a lower-viscosity fluid adds ~x1.3 at most (Ra ~ 1/nu, h ~ Ra^0.25)."))
    S.append(("3. Bulk oil - the mixer and flywheel",
        0.0,
        f"Self-circulation **{res['u_ts']*1000:.1f} mm/s**, stratification "
        f"~{res['dT_loop']:.1f} K, best-to-worst cell spread "
        f"**{res['spread']:.1f} K** (criterion 5 K). Thermal buffer "
        f"{(masses['C_oil']+masses['C_batt'])/1e3:.0f} kJ/K absorbs peaks.",
        "Keep the cold plane high (tube placement drives the loop head). "
        + ("**Plumb alternate tubes in opposite directions**: the section model "
           "shows counterflow collapses the water-rise spread from ~1.1 K to "
           "~0.06 K, free. " if cf_tip else "")
        + "Holders currently block "
        f"{d['holder_block']*100:.0f}% of the riser - open them up."))
    S.append(("4. Oil to tube - the finned second film",
        dTs["f2"],
        f"h_tube = **{res['h_tube']:.0f} W/m²·K**; fins eta "
        f"{res['fin']['eta']:.2f}, area x{res['fin']['area_gain']:.1f} -> "
        f"A_eff {res['A_oilside']:.1f} m² -> **{dTs['f2']:.1f} K**.",
        "Fin harder (oil's low h keeps long fins ~90% efficient), add tubes, "
        "or go interstitial to distribute the sink. This film shares the "
        "same stirring lever as station 2."))
    S.append(("5. Tube wall",
        dTs["wall"],
        f"{d['tube_mat']}, {d['tube_wall']*1000:.1f} mm -> "
        f"**{dTs['wall']*1000:.0f} mK**. Negligible.",
        f"Material is free here: aluminium instead of copper adds only "
        f"~{dTs['wall']*1000*(K_TUBE['Copper']/K_TUBE['Aluminium']-1):.0f} mK. "
        "Choose tubes for corrosion and joining, not conductivity."))
    S.append(("6. Water film and flow",
        dTs["f3"] + dTs["rise"],
        f"**{res['water_regime']}**, Re = {res['Re_water']:.0f}, h = "
        f"{res['h_water']:.0f} -> film **{dTs['f3']:.1f} K**; inlet-to-outlet "
        f"rise {res['dT_water']:.1f} K (mean +{dTs['rise']:.1f} K). Pump "
        f"{P_pump:.1f} W.",
        "In laminar flow more velocity does nothing - trip turbulence "
        "(Re > 3000) with more flow or fewer/narrower parallel paths, or "
        "split the water into counterflowing halves to also fix uniformity."))
    S.append(("7. Chiller - closing the loop",
        0.0,
        f"Duty **{(res['Q_w']+P_pump)/1000:.2f} kW** at {d['T_water_in']:.0f} °C "
        f"inlet; est. COP **{chil['COP']:.1f}** (lift {chil['lift']:.0f} K) -> "
        f"electrical **{chil['P_el']:.0f} W**. Casing sheds {res['Q_atm']:.0f} W "
        "for free.",
        "Warmer inlet is a double win up to the cell limit: COP rises and "
        "DCIR(T) cuts the heat itself - see the set-point trade curve below. "
        "Size the chiller for continuous duty and let the oil flywheel eat "
        "the peaks."))
    return S, tot

def setpoint_trade(d, g, fl, C_duty, T_amb):
    rows = []
    for Tin in (5, 10, 15, 20, 25, 30):
        dd = dict(d); dd["T_water_in"] = float(Tin)
        r = solve_steady(dd, g, fl, 1.0, T_amb, C_rate=C_duty)
        ch = chiller_model(r["Q_w"], Tin, T_amb)
        rows.append((Tin, r["T_b"], r["Q_eff"], ch["P_el"], ch["COP"]))
    a = np.array(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a[:, 0], y=a[:, 3], name="Chiller electrical [W]",
                             line=dict(color="#6366F1", width=3)))
    fig.add_trace(go.Scatter(x=a[:, 0], y=a[:, 1], name="Cell T [°C]",
                             yaxis="y2", line=dict(color="#EF4444", width=3)))
    fig.add_trace(go.Scatter(x=[a[0, 0], a[-1, 0]], y=[d["T_limit"]] * 2,
                             yaxis="y2", mode="lines", name="limit",
                             line=dict(color="#B91C1C", dash="dash")))
    fig.update_layout(height=300, xaxis_title="Water inlet set point [°C]",
                      yaxis_title="Chiller electrical [W]",
                      yaxis2=dict(title="Cell T [°C]", overlaying="y",
                                  side="right", showgrid=False),
                      title=f"Set-point trade at {C_duty:.2f}C: COP vs DCIR",
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.2),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

def c_sweep_fig(d, g, fl, T_amb, C_now):
    Cs = np.linspace(0.5, 6.0, 12)
    Tb, Qs, Pel = [], [], []
    for C in Cs:
        r = solve_steady(d, g, fl, 1.0, T_amb, C_rate=float(C))
        Tb.append(r["T_b"]); Qs.append(r["Q_eff"] / 1000)
        Pel.append(chiller_model(r["Q_w"], d["T_water_in"], T_amb)["P_el"] / 1000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Cs, y=Qs, name="Heat [kW]",
                             line=dict(color="#EF4444", width=3)))
    fig.add_trace(go.Scatter(x=Cs, y=Pel, name="Chiller electrical [kW]",
                             line=dict(color="#6366F1", width=3)))
    fig.add_trace(go.Scatter(x=Cs, y=Tb, name="Cell T [°C]", yaxis="y2",
                             line=dict(color="#F59E0B", width=3)))
    fig.add_trace(go.Scatter(x=[Cs[0], Cs[-1]], y=[d["T_limit"]] * 2,
                             yaxis="y2", mode="lines", name="limit",
                             line=dict(color="#B91C1C", dash="dash")))
    fig.add_vline(x=C_now, line_dash="dot", annotation_text=f"duty {C_now:.2f}C")
    fig.update_layout(height=340, xaxis_title="Continuous C-rate",
                      yaxis_title="kW",
                      yaxis2=dict(title="Cell T [°C]", overlaying="y",
                                  side="right", showgrid=False),
                      title="Scaling the same hardware with C "
                            "(heat ~ C^2, softened by DCIR(T))",
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.15),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig

def scale_to_C(d, g, fl, masses, T_amb, C_t):
    """What must change, subsystem by subsystem, to run C_t continuously."""
    r0 = solve_steady(d, g, fl, 1.0, T_amb, C_rate=C_t)
    ok0 = r0["T_b"] <= d["T_limit"]
    ch = chiller_model(r0["Q_w"], d["T_water_in"], T_amb)
    I_pack = C_t * d["cap_Ah"] * d["Np"]
    bus = busbar_props(dict(d, C1=C_t), g)
    rows = [("Heat to remove", f"{r0['Q_eff']/1000:.2f} kW at {r0['T_b']:.1f} °C "
             f"({'OK' if ok0 else 'OVER LIMIT'})",
             "size everything below for this, continuously")]
    rows.append(("Chiller", f"duty {(r0['Q_w'])/1000:.2f} kW, COP {ch['COP']:.1f} "
                 f"-> {ch['P_el']/1000:.2f} kW electrical",
                 "or raise the set point and re-check (COP + DCIR both improve)"))
    fixes = []
    for name, (key, lo, hi, isint) in GOAL_LEVERS.items():
        inv = key == "T_water_in"
        x, T = goal_seek(d, fl, key, lo, hi, isint, T_amb, C_t, d["T_limit"], inv)
        fixes.append((name, x, T))
        cur = d[key] * (1000 if key == "pitch" else 1)
        shown = "insufficient alone" if x is None else \
                f"{x*1000:.1f}" if key == "pitch" else f"{x:.1f}"
        rows.append((f"Single lever: {name}",
                     f"now {cur:.1f}", f"needs {shown}"
                     + ("" if x is None else f" -> {T:.1f} °C")))
    # stirred combination if static fails everywhere useful
    dd = dict(d, u_oil=max(d["u_oil"], 0.05))
    r_st = solve_steady(dd, g, fl, 1.0, T_amb, C_rate=C_t)
    rows.append(("Combination: stir 5 cm/s",
                 f"{r_st['T_b']:.1f} °C ({stirrer_power(d,g,fl,0.05):.1f} W)",
                 "then re-run the single levers from there"))
    rows.append(("Busbars", f"{I_pack:.0f} A pack current",
                 f"section {bus['A_mm2']:.0f} mm² at {d['bus_J']:.0f} A/mm2, "
                 f"{bus['m']:.1f} kg, adds {(I_pack**2*bus['R']):.0f} W"))
    rows.append(("Water side", f"Re = {r0['Re_water']:.0f} ({r0['water_regime']})",
                 "target Re > 3000; laminar extra flow is wasted"))
    rows.append(("Cell core", f"+{r0['dT_core']:.1f} K core-to-can at {C_t:.1f}C "
                 "(formula; FEA says ~20% less for a real 21700)",
                 "no coolant touches this - format and current only"))
    chg_ok = plating_frac(25.0) * d["C_chg"]
    rows.append(("Charging at this C", f"cell rating {d['C_chg']:.1f}C CC; at "
                 f"{C_t:.1f}C the cell must be rated for it and warm "
                 f"(map allows {plating_frac(15)*C_t:.1f}C at 15 °C)",
                 "preheat to 25 °C+ or accept CC derating; CV tail unchanged"))
    return dict(rows=rows, r0=r0, chiller=ch, fixes=fixes)

# ------------------------------------------------------------------ #
#  v5: full narrative report (in-app tab + HTML export)               #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  v7.1: BEV road-car pack benchmark database (user-supplied)         #
# ------------------------------------------------------------------ #
_BM_COLS = {
 "Model/Pack": "model", "Energy": "E_use", "Unnamed: 2": "E_tot",
 "Nominal Capacity": "cap_Ah", "Power": "P10s_kW", "Pack Mass": "m_pack",
 "Cell Mass": "m_cells", "Pack-Cells": "m_noncell",
 "Cell to Pack Ratio": "ctp_m", "Pack Dimensions": "dx", "Unnamed: 10": "dy",
 "Unnamed: 11": "dz", "Pack Volume": "V_pack", "Cell Total Volume": "V_cells",
 "Cell to Pack Ratio.1": "ctp_v", "Pack-Cells.1": "V_noncell",
 "Gravimetric Energy Density": "whkg", "Volumetric Energy Density": "whl",
 "Gravimetric Power Density": "w10s_kg", "Unnamed: 19": "wcont_kg_dis",
 "Unnamed: 20": "wcont_kg_chg", "Volumetric Power Density": "w10s_l",
 "Unnamed: 22": "wcont_l"}

def load_benchmark(path="pack_benchmark.xlsx"):
    try:
        df = pd.read_excel(path, sheet_name="Sheet1").rename(columns=_BM_COLS)
    except Exception:
        return None
    df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:
        if c != "model":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["whkg"] = df["whkg"].fillna(df["E_tot"] * 1000 / df["m_pack"])
    df["whl"] = df["whl"].fillna(df["E_tot"] * 1000 / df["V_pack"])
    df["kW_kg_10s"] = df["P10s_kW"] / df["m_pack"] * 1000
    df["C10s"] = df["P10s_kW"] / df["E_tot"]
    return df

def _pct(series, val):
    s = series.dropna()
    return 100.0 * (s < val).mean() if len(s) else float("nan")

def benchmark_db_tab(d, g, masses, Cmax, res):
    bm = load_benchmark()
    if bm is None:
        st.warning("pack_benchmark.xlsx not found next to app.py.")
        return None
    st.caption(f"{len(bm)} BEV road-car packs from your file "
               "(batterydesign.net database, filtered). Star = this design; "
               "hollow star = this design with the FEA-honest ribbed "
               "enclosure (knock-down 0.56 instead of 0.45).")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    e_rng = f1.slider("Pack energy filter [kWh]", 0.0,
                      float(np.ceil(bm["E_tot"].max() / 10) * 10),
                      (0.0, float(np.ceil(bm["E_tot"].max() / 10) * 10)),
                      key="bm_e")
    q = f2.text_input("Search model", "", key="bm_q")
    only_p = f3.toggle("Only packs with 10 s power", False, key="bm_p")
    v = bm[(bm["E_tot"] >= e_rng[0]) & (bm["E_tot"] <= e_rng[1])]
    if q:
        v = v[v["model"].str.contains(q, case=False, na=False)]
    if only_p:
        v = v[v["P10s_kW"].notna()]
    my_whkg, my_whl = masses["whkg_pack"], masses["whl_pack"]
    m_rib = masses["m_pack"] - masses["m_struct"] + masses["m_struct"] * 0.56 / 0.45
    whkg_rib = masses["E_kwh"] * 1000 / m_rib
    kpi_cards([
        ("Packs shown", f"{len(v)}", f"of {len(bm)} in the file", ""),
        ("Wh/kg percentile", f"{_pct(bm['whkg'], my_whkg):.0f}%",
         f"this design: {my_whkg:.0f} Wh/kg", "brand"),
        ("Wh/L percentile", f"{_pct(bm['whl'], my_whl):.0f}%",
         f"this design: {my_whl:.0f} Wh/L", "brand"),
        ("Cell-to-pack mass", f"{100*masses['m_cells']/masses['m_pack']:.0f}%",
         f"database median {100*bm['ctp_m'].median():.0f}%", ""),
    ])
    # ---- A: energy density map ----
    figA = go.Figure()
    figA.add_trace(go.Scatter(
        x=v["whkg"], y=v["whl"], mode="markers", name="BEV packs",
        marker=dict(size=6 + 22 * v["E_tot"] / bm["E_tot"].max(),
                    color=v["C10s"], colorscale="Viridis",
                    colorbar=dict(title="10 s C-rate"), opacity=0.85,
                    line=dict(color="white", width=1)),
        hovertext=[f"<b>{m}</b><br>{e:.0f} kWh, {mm:.0f} kg<br>"
                   f"{wk:.0f} Wh/kg, {wl:.0f} Wh/L"
                   + (f"<br>{p:.0f} kW 10 s ({c:.1f}C)" if p == p else "")
                   for m, e, mm, wk, wl, p, c in zip(
                       v["model"], v["E_tot"], v["m_pack"], v["whkg"],
                       v["whl"], v["P10s_kW"], v["C10s"])],
        hoverinfo="text"))
    figA.add_trace(go.Scatter(x=[my_whkg], y=[my_whl], mode="markers+text",
        text=["This design"], textposition="top center",
        marker=dict(symbol="star", size=22, color="#EF4444",
                    line=dict(color="white", width=2)), name="This design"))
    figA.add_trace(go.Scatter(x=[whkg_rib], y=[my_whl * masses["m_pack"] / m_rib * 0 + my_whl],
        mode="markers", name="FEA-honest enclosure",
        marker=dict(symbol="star-open", size=18, color="#EF4444",
                    line=dict(width=2))))
    figA.update_layout(height=460, xaxis_title="Gravimetric [Wh/kg]",
                       yaxis_title="Volumetric [Wh/L]",
                       title="Energy-density map (bubble size = pack kWh, "
                             "colour = 10 s C-rate)",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figA, use_container_width=True, key="bm_A", config=PLOTCFG)
    cB, cC = st.columns(2)
    with cB:
        vp = v[v["kW_kg_10s"].notna()]
        figB = go.Figure()
        figB.add_trace(go.Scatter(x=vp["whkg"], y=vp["kW_kg_10s"], mode="markers",
            name="10 s (database)", marker=dict(size=9, color="#6366F1",
            opacity=0.8), hovertext=vp["model"], hoverinfo="text+x+y"))
        vc = v[v["wcont_kg_dis"].notna()]
        figB.add_trace(go.Scatter(x=vc["whkg"], y=vc["wcont_kg_dis"],
            mode="markers", name="continuous (database)",
            marker=dict(size=10, color="#0EA5E9", symbol="diamond"),
            hovertext=vc["model"], hoverinfo="text+x+y"))
        my_cont = Cmax * masses["E_kwh"] * 1000 / masses["m_pack"]
        figB.add_trace(go.Scatter(x=[my_whkg], y=[my_cont], mode="markers+text",
            text=["this (cont.)"], textposition="top center",
            marker=dict(symbol="star", size=18, color="#EF4444"),
            name="This design, continuous"))
        figB.add_trace(go.Scatter(x=[68.5], y=[150000 / 89], mode="markers+text",
            text=["AMG HPB80 (peak)"], textposition="bottom center",
            marker=dict(symbol="x", size=12, color="#D97706"), name="HPB80"))
        figB.update_layout(height=380, xaxis_title="Wh/kg",
                           yaxis_title="W/kg",
                           title="Power vs energy density - 10 s ratings vs "
                                 "this design's thermal continuous",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(figB, use_container_width=True, key="bm_B",
                        config=PLOTCFG)
        st.caption("Honest comparison note: database power is a 10 s rating; "
                   "the red star is this design's thermally continuous "
                   "capability, which is the harder number.")
    with cC:
        vr = v.dropna(subset=["whkg"]).sort_values("whkg", ascending=True)
        vr = pd.concat([vr.tail(18)])
        figC = go.Figure(go.Bar(y=vr["model"], x=vr["whkg"], orientation="h",
                                marker_color="#94A3B8"))
        figC.add_vline(x=my_whkg, line_color="#EF4444", line_width=3,
                       annotation_text=f"this design {my_whkg:.0f}")
        figC.add_vline(x=68.5, line_color="#D97706", line_dash="dot",
                       annotation_text="HPB80")
        figC.update_layout(height=380, xaxis_title="Wh/kg",
                           title="Top of the field vs this design")
        st.plotly_chart(figC, use_container_width=True, key="bm_C")
    with st.expander("Cell-to-pack ratios and the full table"):
        vm = bm.dropna(subset=["ctp_m"]).sort_values("ctp_m")
        figD = go.Figure(go.Bar(x=vm["model"], y=100 * vm["ctp_m"],
                                marker_color="#0EA5E9"))
        figD.add_hline(y=100 * masses["m_cells"] / masses["m_pack"],
                       line_color="#EF4444", line_width=3,
                       annotation_text=f"this design "
                       f"{100*masses['m_cells']/masses['m_pack']:.0f}%")
        figD.update_layout(height=320, yaxis_title="cell mass / pack mass [%]",
                           title="Cell-to-pack mass ratio (higher = less "
                                 "overhead)", xaxis_tickangle=-40)
        st.plotly_chart(figD, use_container_width=True, key="bm_D")
        show = v[["model", "E_use", "E_tot", "m_pack", "whkg", "whl",
                  "P10s_kW", "C10s", "ctp_m", "V_pack"]].round(1)
        st.dataframe(show, hide_index=True, use_container_width=True,
                     height=340)
        st.download_button("Download filtered set (.csv)",
                           show.to_csv(index=False), "pack_benchmark_view.csv",
                           key="bm_dl")
    near = bm.dropna(subset=["whkg", "whl"]).copy()
    near["dist"] = np.hypot((near["whkg"] - my_whkg) / bm["whkg"].std(),
                            (near["whl"] - my_whl) / bm["whl"].std())
    nn = near.nsmallest(3, "dist")["model"].tolist()
    st.markdown(f"**Where this design sits:** {my_whkg:.0f} Wh/kg puts it at "
                f"the **{_pct(bm['whkg'], my_whkg):.0f}th percentile** of BEV "
                f"road-car packs and {my_whl:.0f} Wh/L at the "
                f"**{_pct(bm['whl'], my_whl):.0f}th**; its nearest neighbours "
                f"in the database are {', '.join(nn)}. The gap to the field "
                "is the immersion tax: oil "
                f"({masses['m_oil']:.0f} kg) and the pressure-rated enclosure "
                f"({masses['m_struct']:.0f} kg). The counter-argument this "
                "database cannot show is thermal: abuse tolerance, "
                "uniformity, and the flywheel.")
    return figA

def report_sections(d, g, fl, res, masses, tr, spec, Cmax, C_steady, Q_duty,
                    Q_bus, P_pump, P_stir, chil, figs):
    ok = (res["T_core"] if d["limit_core"] else res["T_b"]) <= d["T_limit"]
    stations, totdT = station_list(d, g, fl, res, masses, Q_duty, Q_bus,
                                   P_pump, P_stir, chil)
    st_md = ""
    for name, dT, nums, imp in stations:
        share = f" - **{dT:.1f} K** ({100*dT/max(totdT,1e-9):.0f}% of the ladder)" \
                if dT > 0 else ""
        st_md += f"\n**{name}**{share}\n\n{nums}\n\n*Improve:* {imp}\n"
    S = []
    S.append(("Executive summary", f"""
{masses['E_kwh']:.1f} kWh / {d['Ns']*d['v_nom']:.0f} V pack of {g['N']} x
{d['fmt']} cells in static {fl['name']}, cooled by {d['n_tubes']} internal
water tubes ({d['tube_plane'].lower()}). At the duty's {C_steady:.2f}C RMS the
cells sit at **{res['T_b']:.1f} °C can / {res['T_core']:.1f} core** against a
{d['T_limit']:.0f} °C limit: **{'WITHIN LIMIT' if ok else 'OVER LIMIT'}**.
Max continuous **{Cmax:.2f}C**. Pack **{masses['m_pack']:.0f} kg**
({masses['whkg_pack']:.0f} Wh/kg, {masses['whl_pack']:.0f} Wh/L). Chiller duty
{res['Q_w']/1000:.2f} kW -> ~{chil['P_el']/1000:.2f} kW electrical at COP
{chil['COP']:.1f}. Parasitics {P_pump+P_stir:.0f} W.""", ["sankey"]))
    S.append(("The design", f"""
Box {g['Lx']*1000:.0f} x {g['Ly']*1000:.0f} x {g['Lz']*1000:.0f} mm,
{g['n_cols']} x {g['n_rows']} {d['arrangement'].lower()} grid at
{d['pitch']*1000:.1f} mm pitch (gap {g['gap_mm']:.1f} mm), oil to
{g['fill_h']*1000:.0f} mm with {d['gas_gap']*1000:.0f} mm headspace.
Enclosure sized for {d['p_des_bar']:.1f} bar g: {masses['enc']['t_mm']:.1f} mm
effective wall, {masses['m_struct']:.0f} kg. Mass ledger: cells
{masses['m_cells']:.0f}, oil {masses['m_oil']:.0f}, enclosure
{masses['m_struct']:.0f}, holders {masses['m_holders']:.1f}, tubes+fins
{masses['m_tubes']+masses['m_fins']:.1f}, busbars {masses['m_bus']:.1f} kg.
*Improve:* the enclosure is the second-heaviest non-cell item; FEA supports a
stiffened knock-down of ~0.50-0.56 (ribbed), or drop the burst set pressure
and re-check runaway venting.""", ["pack3d"]))
    S.append(("The heat journey, station by station", st_md, ["ladder", "resist"]))
    S.append(("Duty and transient response", f"""
Duty: **{d['duty']}**{(' - ' + d['cycle']) if d['duty']=='Drive cycle' else ''},
{spec['t'][-1]:.0f} s simulated, RMS {C_steady:.2f}C, peak
{np.abs(tr['C']).max():.2f}C. SoC {tr['soc'][0]*100:.0f} ->
{tr['soc'][-1]*100:.0f}%. The oil flywheel
({(masses['C_oil']+masses['C_batt'])/1e3:.0f} kJ/K) filters peaks; size steady
hardware for the RMS, not the spike.
*Improve:* if transients ever graze the limit, buffer harder (more oil) before
buying chiller.""", ["transient"]))
    S.append(("Scaling with C-rate", f"""
Heat scales ~C^2, softened by DCIR(T). This hardware holds
{d['T_limit']:.0f} °C up to **{Cmax:.2f}C** continuous. The chart shows
heat, chiller electricity and cell temperature as C rises; the Improve tab's
scale-to-C tool lists the cheapest change per subsystem for any target.
*Improve at 4C-class duties:* stirring becomes mandatory (static is a 1-2C
architecture), water must be turbulent, busbars resize with I, and the core
term grows to ~{4*4/(C_steady**2+1e-9)*res['dT_core']:.0f} K-class - check the
core limit, and note charging additionally needs the plating map satisfied
(warm cells).""", ["csweep", "setpoint"]))
    S.append(("Uniformity and safety", f"""
Best-to-worst spread estimate **{res['spread']:.1f} K** (criterion 5 K); the
section model shows the water-rise part is ~3x conservative and that
**counterflow plumbing removes it almost entirely**. Runaway screening: one
cell's {d['E_tr']:.0f} kJ heats its local zone modestly and the whole pack by
under a kelvin, but venting pressurises the headspace - burst disc at
{d['p_des_bar']:.1f} bar g, oil-above-water pressure rule, expansion bellows
sized for ~{fl['beta']*masses['V_oil_L']*(d['T_service_max']):.1f} L.""", []))
    S.append(("Validation and honesty", f"""
Wang et al. 2023 rebuild: 33.0 vs 32.3 °C measured, resistances within
30-50%, zero tuning. FD/FE studies: lid coefficients match Roark (+0.0%),
cell core matches the exact solution (-0.1%) and shows the app conservative
by 20-34% on real geometry; the pack-section model closes energy to 0.00%.
Oil-side h honest to +/-30% until the calibration factor
(currently {d['cal_h']:.2f}) is fitted to rig data. Constants open to
challenge are listed in the README.""", []))
    return S

def render_report_tab(secs, figs):
    st.caption("The same report exports as a standalone HTML from the sidebar "
               "or the button below - figures stay interactive.")
    for title, md, fkeys in secs:
        with st.container(border=True):
            st.subheader(title)
            st.markdown(md)
            for k in fkeys:
                if k in figs:
                    st.plotly_chart(figs[k], use_container_width=True,
                                    key=f"rep_{title[:12]}_{k}")

def export_report_html(secs, figs) -> str:
    body = ""
    for title, md, fkeys in secs:
        html_md = md.replace("**", "").replace("*Improve:*", "<b>Improve:</b>")
        body += f"<h2>{title}</h2><p>{html_md}</p>"
        for k in fkeys:
            if k in figs:
                body += figs[k].to_html(full_html=False, include_plotlyjs=False)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<title>Immersion Pack Lab - design report</title>
<style>body{{font-family:Georgia,serif;max-width:1000px;margin:2em auto;
color:#1F2933;line-height:1.5}}h1{{border-bottom:3px solid #F59E0B}}
h2{{color:#4F46E5;margin-top:1.6em}}</style></head><body>
<h1>Immersion Pack Lab - design report</h1>{body}
<p style="font-size:.85em;color:#666">Generated by Immersion Pack Lab v5.
Sources: Wang 2023 (J. Energy Storage 62, 106821); teardown data per the
Compare tab; pack_fea_v1 studies.</p></body></html>"""

if __name__ == "__main__":
    if os.environ.get("SMOKE"):
        smoke()
    else:
        main()
