# -*- coding: utf-8 -*-
"""
Fig. 7 — Final iteration (trial 6): swing-up tracking and subsequent
         balancing, showing the two-phase control architecture.
         Top/Middle: alpha and phi vs. reference. Bottom: coil currents.
"""
from setup_plt import plt
import numpy as np
import pandas as pd
from pathlib import Path

fig, axes = plt.subplots(3, 1, figsize=(7, 7 * 0.5), sharex=True)
fig.subplots_adjust(hspace=0.02)

dt         = 1 / 100
final_time = 2.5
t_switch   = 0.58   # s — empirical region-of-attraction boundary for balancing handoff

color_1 = '#5e85ff'
color_2 = '#ff4b6f'
color_3 = '#00c92b'

# --- Angle trajectories --------------------------------------------------

df = pd.read_csv("data/6/iteration_data.csv")
df = df[(df['k'] >= 0) & (df['k'] <= final_time / dt)]

axes[0].plot(df["k"] * dt, np.rad2deg(df['alpha_optimal']),
             color=color_2, label=r'$\alpha^*$', linewidth=2)
axes[1].plot(df["k"] * dt, np.rad2deg(df['phi_optimal']),
             color=color_2, label=r'$\varphi^*$', linewidth=2)
axes[0].plot(df["k"] * dt, np.rad2deg(df['alpha']),
             color='black', label=r'$\alpha$', linewidth=1)
axes[1].plot(df["k"] * dt, np.rad2deg(df['phi']),
             color='black', label=r'$\varphi$', linewidth=1)

# --- Coil currents -------------------------------------------------------

df_curr = pd.read_csv("data/6/actual_currents.csv")
df_curr = df_curr[(df_curr['relative_time'] >= 0.04) &
                  (df_curr['relative_time'] <= final_time + 0.04)]
df_curr['relative_time'] -= 0.04   # align to trajectory start

for i, color in enumerate([color_1, color_2, color_3]):
    axes[2].plot(df_curr["relative_time"], df_curr[f'I{i}'],
                 label=fr'$i_{{{i}}}$', color=color, linewidth=1)

# --- Phase annotations ---------------------------------------------------

y_text = 1.1
y_line = 1.08
mid_tracking = t_switch / 2
mid_stab     = t_switch + (final_time - t_switch) / 2

axes[0].text(mid_tracking, y_text, r'swing-up phase',
             transform=axes[0].get_xaxis_transform(),
             ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0].text(mid_stab, y_text, r'balancing phase',
             transform=axes[0].get_xaxis_transform(),
             ha='center', va='bottom', fontweight='bold', fontsize=9)

axes[0].annotate('', xy=(-0.01, y_line), xytext=(t_switch + 0.01, y_line),
                 xycoords=axes[0].get_xaxis_transform(),
                 arrowprops=dict(arrowstyle='<|-|>', color='black', lw=0.5))
axes[0].annotate('', xy=(t_switch - 0.01, y_line), xytext=(final_time + 0.01, y_line),
                 xycoords=axes[0].get_xaxis_transform(),
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=0.5))

for x in [0, t_switch]:
    axes[0].plot([x, x], [y_line - 0.08, y_line + 0.05],
                 transform=axes[0].get_xaxis_transform(),
                 color='#b0b0b0', lw=0.8, clip_on=False)

# --- Formatting ----------------------------------------------------------

for ax in axes:
    ax.axvline(x=t_switch, color='#b0b0b0', linestyle='-', linewidth=0.8)
    ax.set_xlim(-0.1, final_time)
    ax.grid(True)
    ax.set_xticks([0, 1, 2])
    ax.set_xticks([t_switch], minor=True)
    ax.set_xticklabels(["0.58"], minor=True)

axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axes[0].set_ylabel(r"$\alpha$ [deg]")
axes[1].set_ylabel(r"$\varphi$ [deg]")
axes[2].set_ylabel(r"$i$ [A]")
axes[1].set_xlabel(r"time [s]")

axes[0].legend()
axes[1].legend()
axes[2].legend(ncols=3)

# --- Save ----------------------------------------------------------------

save_path = Path("plots/fig7_stabilization.pdf")
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
print(f"Figure saved to {save_path}")
