# Pack FEA studies - findings and implications for Immersion Pack Lab

Three numerical models testing the app's closed-form constants. Each script
is self-validating (run `python3 fea*.py`); all figures regenerate.

## 1. Enclosure lid (fea1_lid.py, lid_results.png)

Kirchhoff plate as an FD biharmonic (Kronecker assembly), ribs as line
bending stiffness. Verification: deflection and stress coefficients vs
Roark, simply supported alpha +0.1% / beta +2.3%, clamped alpha +0.5% /
beta +0.0%.

Default lid (916 x 889 mm, 0.5 bar g, sigma_allow 80 MPa, aluminium):

| ribs/dir | skin t | rib h | lid mass | t_equiv |
|---|---|---|---|---|
| 0 (flat) | 11.8 mm | - | 26.0 kg | 11.8 mm |
| 2 | 4.0 mm | 100 mm | 14.8 kg | 6.7 mm |
| 4 | 2.4 mm | 80 mm | 14.6 kg | 6.7 mm |

Honest mass knock-down vs flat = **0.56**, against the app's **0.45**.
Caveat in the app's favour: ribs were modelled about their own neutral
axis; composite T-beam action with the skin raises rib stiffness 2-3x, so
the true optimum lies between 0.45 and 0.56.

**Recommendation:** set the stiffening knock-down default to **0.50** and
keep the slider; the app's flat-plate coefficient 0.31 is confirmed
(clamped square beta = 0.308).

## 2. Cell core temperature (fea2_cell.py, cell_results.png)

Axisymmetric anisotropic conduction (k_r = 0.9, k_z = 30 W/mK), mandrel
hole, convective side and optional ends. Verification vs exact solutions:
solid -0.08%, hollow -0.02%, film rise exact.

| case | FD dT core | app 1/(4 pi k_r H) |
|---|---|---|
| 21700 2C solid, adiabatic ends | 2.85 K | 2.85 K (0%) |
| + 2 mm mandrel | 2.50 K | -12% |
| + mandrel + wetted ends (h=80) | 2.28 K | **-20%** |
| 21700 4C, mandrel + ends | 9.19 K | -20% |
| 4680 2C solid, adiabatic | 13.98 K | 14.00 K (0%) |
| 4680 2C, 4 mm mandrel + wetted ends | 9.30 K | **-34%** |

**Recommendation:** the formula is exact for its own assumptions and
conservative for real cells. Either keep it (safe-side) with a note, or add
a geometry factor ~0.80 (21700) / ~0.66 (4680, ends wetted). The 4680 core
problem is real but 9 K, not 14 K, when the ends are in the oil.

## 3. Worst-cell spread (fea3_pack.py, pack_spread.png)

Cell-resolved plan network: 48 oil bins along the tubes, per-tube water
marching through the app's chain resistance, thermosiphon dispersion
D_eff = u_ts x H_loop (swept x0.2/x1/x5), casing end losses. Pure
conduction FE would be dishonest here (the oil moves); the dispersion term
carries the mixing. Verification: mixing limit recovers uniform oil
(0.004 K) and energy closes to 0.00%.

At the default 2C case (app's water-rise contribution to spread = 3.25 K):

| flow plumbing | D_eff x0.2 | x1 | x5 |
|---|---|---|---|
| co-flow (all tubes same direction) | 2.11 K | **1.08 K** | 0.32 K |
| counterflow (alternate directions) | 0.20 K | **0.06 K** | 0.01 K |

**Findings:** (a) the app's +/- dT_water/2 water-rise term is ~3x
conservative at nominal mixing, because the thermosiphon smears the axial
gradient; a realistic co-flow total spread is ~1.1 + 0.9 (stratification)
= ~2.0 K, not 4.2 K. (b) **Counterflow plumbing of alternate tubes kills
the water-rise term almost entirely, for free.**

**Recommendation:** keep the app's estimate as the conservative bound but
add a "counterflow tube plumbing" toggle (halves? no - removes the water
term) and a diagnose() tip recommending it whenever spread > 3 K.

## Limits of these models

No CFD: natural convection enters only through the app's film correlations
and the dispersion coefficient. Rib composite action neglected
(conservative). Cell k_r/k_z are literature values, not measured. None of
this replaces the mini-rig; all of it sharpens what the rig should measure
(lid strain at the panel centre, a core thermocouple down the mandrel, and
inlet-end vs outlet-end cell temperatures under co- vs counterflow).
