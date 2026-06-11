# -*- coding: utf-8 -*-
"""
Fig. 6 — ILC learning progression over six iterations on the Navion eMNS.
         Actuator angle alpha (top), pendulum angle phi (middle), and
         feedforward correction u_ILC (bottom) relative to the reference.
"""
from setup_plt import plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplot2tikz import save

fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.5 * 1.5), sharex=True)
fig.subplots_adjust(hspace=0.02)

iterations = range(1, 7)
dt         = 1 / 100
reference_color = '#ff4b6f'

n_iters = len(iterations)
colors  = [(1 - (i + 1) / (n_iters + 1),) * 3 for i in range(n_iters)]

# --- Data loading & plotting ---------------------------------------------

try:
    df_ref = pd.read_csv("data/1/iteration_data.csv")
    df_ref = df_ref[(df_ref['k'] >= 0) & (df_ref['k'] <= 60)]

    axes[0].plot(df_ref["k"] * dt, np.rad2deg(df_ref['alpha_optimal']),
                 color=reference_color, label="optimal", linewidth=2)
    axes[1].plot(df_ref["k"] * dt, np.rad2deg(df_ref['phi_optimal']),
                 color=reference_color, label="optimal", linewidth=2)
    axes[2].plot(df_ref["k"] * dt, np.zeros(len(df_ref)),
                 color=reference_color, label="optimal", linewidth=2)

    for i, it in enumerate(iterations):
        df = pd.read_csv(f"data/{it}/iteration_data.csv")
        n_j = int(df['N_j'].iloc[0]) if 'N_j' in df.columns else 60
        df  = df[(df['k'] >= 0) & (df['k'] <= n_j)]

        is_first = (i == 0)
        is_last  = (i == n_iters - 1)
        color     = '#5e85ff' if is_first else colors[i]
        linestyle = '-' if (is_first or is_last) else ':'
        label     = f"{i + 1 if not is_first else '1 (no learning)'}"

        axes[0].plot(df["k"] * dt, np.rad2deg(df['alpha']),
                     color=color, label=label, linestyle=linestyle, linewidth=1)
        axes[1].plot(df["k"] * dt, np.rad2deg(df['phi']),
                     color=color, label=label, linestyle=linestyle, linewidth=1)
        axes[2].plot(df["k"] * dt, df['u_ilc'],
                     color=color, label=label, linestyle=linestyle, linewidth=1)

except Exception as e:
    print(f"Data not found, plotting empty axes: {e}")

# --- Formatting ----------------------------------------------------------

for i, ax in enumerate(axes):
    ax.grid(True)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xlim(-0.05, 0.65)
    if i < 2:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

axes[2].ticklabel_format(useOffset=False, style='plain', axis='both')
axes[0].set_ylabel(r"$\alpha$ [deg]")
axes[1].set_ylabel(r"$\varphi$ [deg]")
axes[2].set_ylabel(r"$\mathrm{u}_\text{ILC}$ [Nm]")
axes[2].set_xlabel(r"time [s]")

axes[2].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.35),
    ncol=4,
    fontsize=8,
    frameon=False,
)

# --- Save ----------------------------------------------------------------

save_path = Path("plots/fig6_iterations.eps")
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f"Figure saved to {save_path}")
