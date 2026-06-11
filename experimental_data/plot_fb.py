# -*- coding: utf-8 -*-
"""
Fig. 5 — Repeatability comparison: open-loop (OctoMag, no feedback)
         vs. closed-loop (Navion, with LQR trajectory feedback).

Also prints final-state dispersion statistics used in the paper text.

Requires: prep_repeatability_data.py must be run first to generate
          data/repeatability/no_fb/merged_data.csv.
"""
from setup_plt import plt
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration -------------------------------------------------------

# OctoMag open-loop: all 17 recorded trials
NO_FB_TRIALS = range(1, 18)
# Navion closed-loop: 11 repeatability trials (attempt_ix 1–11 in merged_data.csv)
FB_ATTEMPTS  = range(1, 12)

# Trajectory window in no_fb time coordinates (k=0..59 → 0.04..0.63 s)
T_START, T_END = 0.04, 0.64
t_common = np.linspace(0, 0.6, 300)   # common interpolation grid
dt = 0.01

color_main  = '#5e85ff'
color_ref   = '#ff4b6f'
SHADE_ALPHA = 0.25

# --- Data loading --------------------------------------------------------

def load_no_fb(trials):
    df = pd.read_csv("data/repeatability/no_fb/merged_data.csv")
    alphas, phis = [], []
    for i in trials:
        trial = df[(df["attempt_ix"] == i) &
                   (df["time"] >= T_START) & (df["time"] <= T_END)]
        if trial.empty:
            continue
        t = trial["time"].values - T_START
        alphas.append(np.interp(t_common, t, np.rad2deg(trial["alpha"].values)))
        phis.append(  np.interp(t_common, t, np.rad2deg(trial["phi"].values)))
    return np.array(alphas), np.array(phis)


def load_fb(attempts):
    df = pd.read_csv("data/repeatability/fb/merged_data.csv")
    alphas, phis = [], []
    for i in attempts:
        trial = df[(df["attempt_ix"] == i) & (df["current_k"] >= 0)]
        if trial.empty:
            continue
        t = trial["current_k"].values * dt
        alphas.append(np.interp(t_common, t, np.rad2deg(trial["alpha"].values)))
        phis.append(  np.interp(t_common, t, np.rad2deg(trial["phi"].values)))
    return np.array(alphas), np.array(phis)


# --- Dispersion statistics -----------------------------------------------

def print_dispersion(label, alphas_arr, phis_arr):
    print(f"\n--- {label} (n={len(alphas_arr)}) ---")
    for name, vals in [("alpha", alphas_arr[:, -1]), ("phi", phis_arr[:, -1])]:
        print(f"  {name}: mean={np.mean(vals):.2f}°  "
              f"std=±{np.std(vals):.2f}°  "
              f"half-range=±{(np.max(vals)-np.min(vals))/2:.2f}°")


# --- Plotting helper -----------------------------------------------------

def plot_envelope(ax, data_arr, color):
    lo, hi = np.min(data_arr, axis=0), np.max(data_arr, axis=0)
    ax.fill_between(t_common, lo, hi, color=color, alpha=SHADE_ALPHA, lw=0)
    ax.plot(t_common, hi, color=color, lw=1.0)
    ax.plot(t_common, lo, color=color, lw=1.0)


# --- Main ----------------------------------------------------------------

alphas_no, phis_no = load_no_fb(NO_FB_TRIALS)
alphas_fb, phis_fb = load_fb(FB_ATTEMPTS)

print_dispersion("(a) OctoMag — no feedback", alphas_no, phis_no)
print_dispersion("(b) Navion  — with feedback", alphas_fb, phis_fb)

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5 * 0.8), sharex=True, sharey="row")
fig.subplots_adjust(hspace=0.05, wspace=0.1)

plot_envelope(axes[0, 0], alphas_no, color_main)
plot_envelope(axes[1, 0], phis_no,   color_main)
plot_envelope(axes[0, 1], alphas_fb, color_main)
plot_envelope(axes[1, 1], phis_fb,   color_main)

# Reference trajectories
ref_no = np.load("data/repeatability/no_fb/optimal.npz")
ref_t_no = np.arange(ref_no["X"].shape[1]) * (dt / 2)   # trajectory planned at half dt
axes[0, 0].plot(ref_t_no, np.rad2deg(ref_no["X"][0]), color=color_ref, lw=1, label="reference")
axes[1, 0].plot(ref_t_no, np.rad2deg(ref_no["X"][1]), color=color_ref, lw=1)

ref_fb = np.load("data/repeatability/fb/optimal.npz")
ref_t_fb = np.arange(ref_fb["X"].shape[1]) * dt
axes[0, 1].plot(ref_t_fb, np.rad2deg(ref_fb["X"][0]), color=color_ref, lw=1, label="reference")
axes[1, 1].plot(ref_t_fb, np.rad2deg(ref_fb["X"][1]), color=color_ref, lw=1)

# Formatting
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        ax.grid(True)
        ax.set_xlim(-0.02, 0.62)
        ax.set_xticks([0, 0.3, 0.6])
        if i == 0:
            label = r"(a) without feedback" if j == 0 else r"(b) with feedback"
            ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel(r"time [s]")
        if j == 1:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        if j == 0:
            axes[0, j].set_ylabel(r"$\alpha$ [deg]")
            axes[1, j].set_ylabel(r"$\varphi$ [deg]")

handle_ref   = axes[0, 1].get_lines()[-1]
handle_shade = axes[0, 1].collections[0]
fig.legend(
    handles=[handle_ref, handle_shade],
    labels=["optimal", "experimental range"],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False,
    fontsize=9,
)
fig.subplots_adjust(bottom=0.15)

save_path = Path("plots/fig5_fb_comparison.pdf")
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
print(f"\nFigure saved to {save_path}")
