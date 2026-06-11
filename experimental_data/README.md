# Experimental Data

This directory contains the experimental data recorded on the Navion and OctoMag eMNS platforms and used to produce the figures in the accompanying paper, together with the plotting scripts that generated them. The plotting scripts also serve as postprocessing examples.

## Directory Structure

```
experimental_data/
│
├── calibration/
│   ├── NavionDenseQuadropole.yaml          MPEM calibration used during experiments
│   └── Navion_1_2_Calibration_old.yaml     Earlier calibration (used for torque discrepancy analysis)
│
├── data/
│   ├── 1/ … 6/                             Per-trial logged data (six ILC iterations)
│   │   ├── actual_currents.csv             Coil currents commanded at each timestep
│   │   ├── iteration_data.csv              State, reference, feedforward and feedback signals
│   │   └── target_currents.csv             Desired currents from the ILC feedforward
│   ├── repeatability/
│   │   ├── fb/                             Closed-loop (LQR trajectory feedback, Navion)
│   │   │   ├── merged_data.csv             Merged repeatability trials
│   │   │   └── optimal.npz                 Reference trajectory used in these trials
│   │   └── no_fb/                          Open-loop (no feedback, OctoMag)
│   │       ├── merged_data.csv
│   │       └── optimal.npz
│   └── torque_discrepancy_analysis.csv     Predicted vs. learned torque correction per timestep
│
├── plots/                                  Output figures (EPS/PDF) as included in the paper
│   ├── fig4_reference_trajectory.eps
│   ├── fig5_fb_comparison.pdf
│   ├── fig6_iterations.eps
│   ├── fig7_stabilization.pdf
│   └── fig8_torque_discrepancy.eps
│
├── parameters/                             Physical parameters (copy from src/) used by the scripts
│   ├── general.py
│   └── pendulum.py
│
├── geometry_jit.py                         Geometry utilities (copy from src/) used by the scripts
├── setup_plt.py                            Matplotlib style settings shared across all scripts
│
├── plot_optimal.py                         Fig. 4 — optimal reference trajectory
├── plot_fb.py                              Fig. 5 — open-loop vs. closed-loop repeatability
├── plot_iterations.py                      Fig. 6 — ILC learning progression over six trials
├── plot_stabilization.py                   Fig. 7 — final trial: swing-up + balancing
├── plot_torque_discrepancy.py              Fig. 8 — learned correction vs. MPEM torque prediction
```

## Running the Scripts

From inside this directory:

```bash
python plot_optimal.py
python plot_iterations.py
python plot_fb.py
python plot_stabilization.py
python plot_torque_discrepancy.py
```

Each script reads from `data/` and writes the corresponding figure to `plots/`. Dependencies: `numpy matplotlib scipy pandas`.
