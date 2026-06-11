# -*- coding: utf-8 -*-
"""
Fig. 8 — Torque discrepancy: the learned ILC correction signal closely matches
         the torque difference predicted by the high-fidelity MPEM calibration
         relative to the initial calibration, supporting the interpretation that
         ILC learned to compensate for field-model mismatch.

Computes the discrepancy from raw sensor/current data, then produces the figure.
Also saves an intermediate CSV: data/torque_discrepancy_analysis.csv.
"""
import numpy as np
import pandas as pd
import numba
from pathlib import Path
from scipy import stats
from setup_plt import plt

from geometry_jit import (
    get_normal_vector_from_quaternion,
    magnetic_interaction_from_dipole_moment,
    jacobian_torqueforce_to_torque,
    get_normal_angles_from_normal_vector,
)
import parameters.general as params
from mag_manip import mag_manip

# --- Configuration -------------------------------------------------------

DATA_TRIAL    = "data/6/iteration_data_uncut.csv"
CURRENTS_FILE = "data/6/target_currents.csv"
CAL_DIR       = "calibration"
CAL_OLD       = "Navion_1_2_Calibration_old.yaml"
CAL_NEW       = "NavionDenseQuadropole.yaml"

# ILC active only up to the equilibrium controller switch
EQ_SWITCH_TIME = 0.585   # s

# =========================================================================
# 1. Load and merge sensor / current logs
# =========================================================================

sensors_df  = pd.read_csv(DATA_TRIAL)
currents_df = pd.read_csv(CURRENTS_FILE)
sensors_df.columns  = sensors_df.columns.str.strip()
currents_df.columns = currents_df.columns.str.strip()

merged_df = pd.merge_asof(
    sensors_df.sort_values("timestamp (ns)"),
    currents_df.sort_values("timestamp (ns)"),
    on="timestamp (ns)",
    direction="forward",
    suffixes=("_sensor", "_current"),
)

# =========================================================================
# 2. Pre-compute geometry (Numba JIT)
# =========================================================================

qx = merged_df["qx_act"].to_numpy(dtype=np.float64)
qy = merged_df["qy_act"].to_numpy(dtype=np.float64)
qz = merged_df["qz_act"].to_numpy(dtype=np.float64)
qw = merged_df["qw_act"].to_numpy(dtype=np.float64)
tx = merged_df["tx_act"].to_numpy(dtype=np.float64)
ty = merged_df["ty_act"].to_numpy(dtype=np.float64)
tz = merged_df["tz_act"].to_numpy(dtype=np.float64)
I0 = merged_df["I0"].to_numpy(dtype=np.float64)
I1 = merged_df["I1"].to_numpy(dtype=np.float64)
I2 = merged_df["I2"].to_numpy(dtype=np.float64)


@numba.njit
def compute_magnet_positions(qx, qy, qz, qw, tx, ty, tz, length):
    n   = len(qx)
    pos = np.empty((n, 3))
    for i in range(n):
        q  = np.array([qx[i], qy[i], qz[i], qw[i]], dtype=np.float64)
        nv = -get_normal_vector_from_quaternion(q)
        pos[i, 0] = tx[i] - length * nv[0]
        pos[i, 1] = ty[i] - length * nv[1]
        pos[i, 2] = tz[i] - length * nv[2]
    return pos


@numba.njit
def precompute_geometry(qx, qy, qz, qw, m_tilde, l_m):
    n       = len(qx)
    J_stack = np.zeros((n, 2, 6))
    M_stack = np.zeros((n, 6, 8))
    for i in range(n):
        q       = np.array([qx[i], qy[i], qz[i], qw[i]], dtype=np.float64)
        nv      = -get_normal_vector_from_quaternion(q)
        M_stack[i] = magnetic_interaction_from_dipole_moment(m_tilde * nv)
        beta, alpha = get_normal_angles_from_normal_vector(nv)
        J_stack[i]  = jacobian_torqueforce_to_torque(beta=beta, alpha=alpha, l_mag=l_m)[:2, :]
    return J_stack, M_stack


print("Pre-computing geometry...")
mag_pos  = compute_magnet_positions(qx, qy, qz, qw, tx, ty, tz, params.l_act_marker2magnet)
J_series, M_series = precompute_geometry(qx, qy, qz, qw, params.m_tilde, params.l_m)

# =========================================================================
# 3. Compute torques for both calibrations
# =========================================================================

def calculate_torques(cal_file, I0, I1, I2):
    cal_path = Path(CAL_DIR) / cal_file
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")
    model = mag_manip.ForwardModelMPEM()
    model.setCalibrationFile(str(cal_path))
    n       = len(I0)
    torques = np.zeros((n, 2))
    for i in range(n):
        A          = model.getActuationMatrix(np.ascontiguousarray(mag_pos[i]))
        torques[i] = J_series[i] @ M_series[i] @ A @ np.array([I0[i], I1[i], I2[i]])
    return torques


print("Computing torques — initial calibration...")
torques_old = calculate_torques(CAL_OLD, I0, -I1, I2)   # I1 sign flip for old cal

print("Computing torques — high-fidelity calibration...")
torques_new = calculate_torques(CAL_NEW, I0, I1, I2)

# =========================================================================
# 4. Align ILC signal with torque discrepancy
# =========================================================================

t_ns  = merged_df["timestamp (ns)"].to_numpy()
t_rel = (t_ns - t_ns[0]) / 1e9

u_ilc_all = merged_df["u_ilc"].to_numpy()
t_start   = t_rel[u_ilc_all != 0].min()
t_end     = t_rel[u_ilc_all != 0].max()

active = (t_rel >= t_start) & (t_rel <= t_end)
# Skip first 4 steps: u_ilc[0] is pre-applied before the trajectory starts
t_ilc    = t_rel[active][4:]
u_ilc    = u_ilc_all[active][4:]
u_ilc_fb = (merged_df["u_ilc"] + merged_df["u_tracking_fb"]).to_numpy()[active][4:]

torque_diff  = torques_old[:, 1] - torques_new[:, 1]
diff_aligned = np.interp(t_ilc, t_rel, torque_diff)

t_plot       = t_ilc - t_ilc[0]
keep         = t_plot <= EQ_SWITCH_TIME
t_plot       = t_plot[keep]
diff_aligned = diff_aligned[keep]
u_ilc        = u_ilc[keep]
u_ilc_fb     = u_ilc_fb[keep]

# =========================================================================
# 5. Statistics and intermediate CSV
# =========================================================================

r_full, p_full = stats.pearsonr(diff_aligned, u_ilc)
print(f"\nPearson r (full):     r={r_full:.3f}  p={p_full:.2e}")

mask_015 = t_plot >= 0.15
r_015, p_015 = stats.pearsonr(diff_aligned[mask_015], u_ilc[mask_015])
print(f"Pearson r (t≥0.15s): r={r_015:.3f}  p={p_015:.2e}")

pd.DataFrame({
    "time_s":         t_plot,
    "torque_diff_nm": diff_aligned,
    "u_ilc":          u_ilc,
    "u_ilc_plus_fb":  u_ilc_fb,
}).to_csv("data/torque_discrepancy_analysis.csv", index=False)

# =========================================================================
# 6. Paper figure
# =========================================================================

color_1 = "#5e85ff"
color_2 = "#ff4b6f"

fig, ax = plt.subplots(figsize=(3.5, 3.5 * 0.4))
ax.plot(t_plot, u_ilc,        color=color_2, label=r"$\mathrm{u}_\text{ILC}$")
ax.plot(t_plot, diff_aligned, color=color_1, label=r"$\Delta\tau_\text{c}$")

ax.set_xlabel(r"\text{time [s]}")
ax.set_ylabel(r"\text{torque [Nm]}")
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim(-0.05, 0.65)
ax.grid(True)
ax.legend()

save_path = Path("plots/fig8_torque_discrepancy.eps")
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, format="eps", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Figure saved to {save_path}")
