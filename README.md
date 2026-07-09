# Immersion Pack Lab v4

Parametric design and teaching app for a **static or stirred immersion-cooled
21700 battery pack** with an **internal water-cooled tube heat exchanger**
(the ICDC architecture of Wang et al. 2023).

## Run
    pip install -r requirements.txt
    streamlit run app.py

Keep `coolants.csv` next to `app.py`. `SMOKE=1 python app.py` runs the physics
self-test (Wang benchmark, thermosiphon sanity, calibration round-trip).

## Model
Two-node thermal network (cells, bulk oil) with water and ambient boundaries;
film coefficients recomputed each step from local dT.

* Cell side: Churchill-Chu vertical plate x confinement penalty below 6 mm gap
  (calibrated to Wang Fig. 9). Tube side: Churchill-Chu horizontal cylinder +
  Schmidt annular fins. Forced/mixed flow: Churchill-Bernstein, Nu^3 blend.
* Water side: Hausen / Gnielinski with laminar-turbulent bridge and axial rise.
* **Thermosiphon (v2)**: buoyant head rho*beta*g*H_loop*dT against laminar bank
  friction + minor losses (K_loop, default 5), solved by bisection; H_loop set
  by the tube-plane position. Feeds the films as the floor on oil velocity.
  Order-of-magnitude validation: Wang measured 0.5-1.8 mm/s; app predicts
  ~4 mm/s for the default 20 kWh pack (bigger head, bigger Q).
* **DCIR(T) (v2)**: R = R25 exp(-k(T-25)), default 1.2 %/K, coupled into heat
  generation each iteration/step. Set k = 0 to disable (recovers v1 numbers).
* **Core temperature (v2)**: quasi-steady radial, R_core = 1/(4 pi k_r H),
  k_r default 0.9 W/mK; peak (centreline) value, mean is half.
* **Worst cell (v2)**: +/- (water rise + stratification)/2 about the mean,
  checked against the 5 K uniformity criterion. An estimate, not a CFD result.

## v2 features
Predicted thermosiphon and tube-plane placement; DCIR(T); core temperature
and core-limit option; worst-cell spread; plan-view Layout tab; Architecture
comparator (static / stirred / bottom cold plate / pumped external HX, with
parasitic power and thermal-system mass); water-pump and stirrer power;
thermal-runaway screening (zone heating, bulk rise, headspace pressure);
design save/load as JSON; A/B pin-and-compare; duty-cycle CSV import (t_s +
C or P_kW); measured-data overlay with one-click calibration-factor fit
(multiplies both oil films, the spray-app pattern); 7x7 two-lever sweep
heatmap with limit contour; student mode (hides Decide).

## Fixed engineering constants (edit in code if you have better numbers)
Cold plate: axial cell path k_ax = 28 W/mK, TIM 0.8 K/W, channel film
3000 W/m2K. Pumped case: 20 cm/s past cells, oil sized for 5 K rise, 30 kPa
at 40% pump efficiency, 25 mL fluid/cell (HPB80 ratio). Runaway: vent gas
heated to 380 K, default 55 kJ/cell with 60% into the local zone. Pump/stirrer
efficiencies 35%/30% (fluid power only; controller overheads excluded).

## Validation
Benchmark tab rebuilds Wang's rig with fixed heater power, cal = 1, no DCIR:
cell 33.0 degC at 1800 s vs 32.3 measured, oil 28.8 vs ~28.5, resistances
within ~30-50% of their Table 6. Treat oil-side h as +/-30% until calibrated.

## Sources
Wang et al. 2023, J. Energy Storage 62, 106821. Zou et al. 2024, J. Energy
Storage 83, 110634. Roe et al. 2022, J. Power Sources 525, 231094.
batterydesign.net (AMG HPB80; Dielectric Immersion Cooling).
coolant_comparison_reviewed.xlsx (reviewed fluid table).

## v3 additions
Electrical: SoC tracking, generic NMC OCV(SoC) and entropic heat, CC-CV
charge with plating-derated current (generic derate map, editable arrays at
the top of app.py), charge/discharge DCIR split, repeated cycling duty.
Drive cycles: WLTP 3b, NEDC, UDDS, HWFET, US06, Artemis Motorway 130 as
coarse breakpoint profiles uniformly speed-scaled to the official distance
(exact by construction; thermally adequate since the pack filters >~0.01 Hz;
use CSV upload for official 1 Hz traces). Vehicle model: mass, CdA, Crr,
drivetrain efficiency, capped regen, accessories. Steady state uses the
duty's RMS C-rate. Structural: cell format presets (18650/21700/4680 from
teardown data; 4680 DCIR is an estimate), busbar heat and mass (auto-sized
at a set current density), cell holders (mass + thermosiphon blockage),
calculated enclosure (stiffened-plate sizing at the burst-disc pressure)
giving Wh/L and an honest total weight, interstitial tube routing option.
Workflow: S x P suggester, one-click HTML design report, +/-30% uncertainty
band, goal-seek on a single lever, production-pack benchmark table
(Tesla 2170 / 4680 / Plaid / LFP, AMG HPB80) with teardown sources.

## v3 constants to challenge
Enclosure: sigma_allow 80 MPa, stiffening knock-down 0.45, design pressure =
burst-disc 0.5 bar g. Busbars: copper at 5 A/mm2, length 1.15 x Ns x pitch.
Holders: 8 g/cell, 20% flow blockage. Interstitial mode: thermosiphon head
8 mm, stratification x0.35 (distributed sinks) - an engineering judgement,
not CFD. OCV/entropic/plating arrays are generic NMC shapes. Vehicle
defaults: 1900 kg, CdA 0.62, Crr 0.009, eta 0.92, regen 0.65 capped 60 kW.

## v4: workbench restructure and 3D view
Layout reorganised around the workflow: 1 Design (all pack/cooling/structure
inputs in three domain columns, with a live 3D or plan view), 2 Duty (load
definition beside the transient response), 3 Results, 4 Improve (diagnosis,
sensitivity, goal-seek, sweep, A/B pin), 5 Safety (runaway screening,
expansion, leak checklist), 6 Compare (architectures, coolant shoot-out,
production packs), 7 Learn, 8 Validate and tune (Wang benchmark, rig
calibration, model-tuning knobs, fluid table). The sidebar is now a status
card plus save/load and report export only. Student version removed.
The 3D view renders every cell (coloured by centre-vs-edge tendency), tube
runs with translucent fin envelopes, the oil fill level, and the enclosure;
it follows format, pitch, tube plane (including interstitial) live.
Old saved designs load unchanged (same widget keys).

## v4.1 fixes
Apply-suggestion / apply-preset / calibration-fit buttons now write widget
values through a pending queue drained at the top of the script, removing the
StreamlitAPIException (state written after widget instantiation). Fluid
scatter charts (Learn panel 3, Coolant shoot-out) auto-thin their labels to
avoid pile-ups; every point keeps its name on hover.
