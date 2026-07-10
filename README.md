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

## v5: heat-journey Results, scale-to-C, full report tab
Results is now the heat journey: a Sankey of where the watts go
(cells/busbars -> oil -> water/casing -> chiller -> ambient), the
temperature ladder and resistance ranking, then seven station cards
(source, first film, bulk oil, finned film, wall, water, chiller) each with
its numbers, its share of the ladder, and a quantified "how to improve this
stage" note, closing with the water set-point trade curve (COP vs DCIR).
A chiller model (COP = 0.45 x Carnot on the actual lift) sizes electrical
power. Improve gains "Scale to a target C": solves the design at the target,
tables every subsystem's required change (chiller, single-lever fixes via
goal-seek, stirred combination, busbar resize, water regime, core, plating/
charging), plus a heat/chiller/temperature vs C sweep chart. Tab 9 renders a
full narrative design report (executive summary, design, heat journey, duty,
scaling, uniformity and safety, validation - each with improvement actions
and live figures) and exports it as standalone interactive HTML.
Mass audit moved to Design. Sidebar simplified.

## v6: visual redesign
One design system, applied everywhere: gradient hero header with a live
status chip, KPI cards with large numerals (replacing default metrics),
pill-style tab navigation, soft-shadow white cards for every section,
Inter typography, and a single global plotly template (clean light grid,
modern colorway) so all charts share one look. Chart semantics kept but
refreshed: cell red #EF4444, oil amber #F59E0B, water sky #0EA5E9, brand
indigo/cyan for structure. Sidebar status is a card; station progress bars
use the brand gradient; the HTML report export matches. No physics or
widget-key changes: saved designs load unchanged. All styling lives in the
single CSS block and the "packlab" plotly template at the top of app.py,
so re-theming (including dark mode) is a one-block edit.

## v7: clarity, interactivity, and the serpentine-plate idea
Clearer physics graphics: the heat-flow Sankey is rebuilt with fixed columns,
kW-labelled nodes and stream-coloured ribbons; a new interactive thermal
circuit shows every interface in the cell-to-water chain sized by its share
of resistance, coloured by transport mode (conduction / convection /
advection), with the governing law and live numbers on hover and the weakest
link flagged. Units are typeset properly throughout (m², W/m²·K, °C, mΩ, ν).
Design now has a sticky live panel: the 3D/plan view plus a change summary
(temperatures, margin, spread, mass, deltas vs the pinned design) stays on
screen while you edit; circulation is a first-class design choice
(thermosiphon / open stirring / guided serpentine plates) with a ν(T) curve
for the chosen fluid. Improve opens with a fully interactive Predictor
(what-if sliders, instant re-solve, delta cards). A new Ideas tab compares
concepts against the live baseline, seeded with the serpentine-plate idea:
plates between rows act as conduction fins (fin-efficiency model, contact
factor) and guide parallel manifolded channels driven by a ~0.5 W pump
(slot-flow pressure drop); forced flow reaches every tight gap and the
distributed sink cuts stratification. FEA findings now render as three
charts in Validate. Chart toolbars gain drawing tools; time charts use
unified hover. Learn gains a circulation-options panel.

## v7.1: BEV pack benchmark database
Compare now loads pack_benchmark.xlsx (user-supplied batterydesign.net
export, 58 BEV road-car packs): filterable energy-density map (Wh/kg vs
Wh/L, bubble = kWh, colour = 10 s C-rate), power-vs-energy chart with the
honest 10 s vs continuous caveat, ranked field vs this design, cell-to-pack
ratio comparison, full table with CSV export, percentile cards and a
nearest-neighbour summary. The FEA-honest enclosure variant plots as a
second marker. Keep pack_benchmark.xlsx next to app.py.

## v7.2: clearer heat-flow map and engineering triptych
Sankey rebuilt: single-line kW-in-label nodes with clean typography, short
casing branch (no cross-canvas ribbon), watt-labelled hovers. The live
design panel defaults to a three-view engineering drawing - plan plus two
true-scale sections - showing tubes as circles with their fin annuli,
serpentine plates, oil level and headspace, bottom gap, enclosure wall
thickness and outer dimensions; interstitial routing draws correctly. 3D
stays as an option, resized to fit the panel with a livelier camera and
lighting; plan view height also fits. All three views follow every design
change live.

## v8: the Live Pack and direct-manipulation UX
Results now opens on the Live Pack: a client-side animated cross-section
(pure canvas, no reruns) where oil particles circulate at the solved
velocity - thermosiphon loop, stirred, or guided serpentine lanes with
plates drawn - water beads travel and warm along the tubes, cells and cores
are coloured by their real solved temperatures with heat glow scaled by Q,
and the weakest link is tagged on the picture. Play/speed/layer/exaggerate
controls run in the browser instantly. The station wall is now a stepper:
click a box in the thermal circuit (the chart is selection-enabled) or pick
a chip, and that one station opens with its numbers and improvement moves;
the Sankey, ladder and set-point trade tuck into an expander. The transient
gains a time scrubber with a live readout card (can/core, C, SoC, oil,
heat at the picked instant). Design's three input groups are collapsed
expanders whose titles show the live key values. No API or internet is
required for any of this; livepack.py ships alongside app.py.

## v8.1: units in °C, richer Compare, plain-words Learn
Temperature differences now read in °C everywhere (spread, margins,
core-to-can, water rise, station tolls); compound units like W/m²·K and
kJ/K stay, and the Learn narrative explains that a 1 K difference equals a
1 °C difference. Compare's density map is grouped by energy class (City /
Mid / Large / Flagship) with legend on/off and double-click isolation, plus
toggleable median-crosshair and interquartile zones; a new explorer plots
any two database quantities with per-class colours, optional log axis and a
trend line, with this design overlaid where comparable; a head-to-head
radar duels up to four chosen packs against this design across five
normalised metrics (with the continuous-vs-10 s power caveat stated). Learn
opens with "Where the heat goes - the whole story in plain words": a
seven-step narrative using the live numbers, ending with what matters,
what doesn't, and the current biggest toll.

## v8.2: Live Pack hardened; native theme
The Live Pack component is rewritten to be self-diagnosing: any browser
error paints its own message onto the canvas and status line instead of a
blank panel, and the corner badges are pre-filled server-side so the solved
numbers show even if a browser blocks the script. The canvas is retina-safe
(CSS height + devicePixelRatio buffer scaling, ResizeObserver refit) and
the script passes an automated Node syntax + stub-DOM runtime harness in
the test suite. The toolbar is a floating translucent glass bar on a dark
canvas. A .streamlit/config.toml theme now ships (keep the folder next to
app.py) so native widgets adopt the brand palette on any Streamlit build,
and the tab-pill CSS carries selectors for multiple Streamlit versions.

## v8.3: benchmark comparison always visible; version badge
The header now shows the app version so it is always clear which bundle is
running. The BEV benchmark section no longer hides anything in collapsed
expanders: a sub-navigation (Density map / Power / Rankings / Explorer /
Head-to-head / Full table) shows every comparison full-size, with a green
DATA LOADED chip confirming the source (58 packs) or, if the file is
missing, a clear error plus an in-app upload fallback that works for the
session. Power and Explorer views are grouped by energy class like the map.

## v8.5: readable charts, compact header, gentle white theme
Chart titles and legends no longer collide: the global template puts every
title top-left and every horizontal legend top-right on its own band with
extra headroom, applied to all charts at once. The Ideas comparison puts
pump power on its own right-hand axis so half-watt bars are actually
visible. The oversized banner and the KPI card row are merged into one
compact header band (title, version, status chip, and seven small stats in
two lines). Background is now pure white with a gentler palette: softer
indigo/sky brand, lighter borders and shadows, calmer hero gradient, and a
matching .streamlit theme. The sticky live design panel gains stronger CSS
(column-level :has() selectors, overflow fixes, self-scrolling) so it stays
on screen while scrolling across more Streamlit builds.
