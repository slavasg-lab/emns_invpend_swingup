# -*- coding: utf-8 -*-
"""
Fig. 4 — Optimal swing-up trajectory: reference angles alpha* and phi* (top),
         angular rates (middle), and commanded input u* with simulated
         actual torque tau_c* reconstructed via RK4 integration (bottom).
"""
from setup_plt import plt
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_tau_trajectory(u_sequence, tau_0, dt, omega_tau, n_substeps=5):
    """
    Integrates tau_dot = omega_tau * (u - tau) using RK4.
    """
    tau = np.zeros(len(u_sequence) + 1)
    tau[0] = tau_0
    inner_dt = dt / n_substeps
    
    for k in range(len(u_sequence)):
        tau_k = tau[k]
        u_k = u_sequence[k]
        
        # Sub-stepping to match the optimizer's integration density
        for _ in range(n_substeps):
            def f(t_val): return omega_tau * (u_k - t_val)
            
            k1 = f(tau_k)
            k2 = f(tau_k + (inner_dt / 2) * k1)
            k3 = f(tau_k + (inner_dt / 2) * k2)
            k4 = f(tau_k + inner_dt * k3)
            
            tau_k = tau_k + (inner_dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
        tau[k+1] = tau_k
        
    return tau

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.5 * 1), sharex=True)

# Remove the gap between plots
fig.subplots_adjust(hspace=0.02)

iterations = range(1, 7)
color_delta = 0
dt = 1 / 100

# --- Data Loading & Plotting ---
# Note: Ensure data/1/iteration_data.csv exists in your local directory
df_optimal = pd.read_csv("data/1/iteration_data.csv")
df_optimal = df_optimal[(df_optimal['k'] <= 60) & (df_optimal['k'] >= 0)]

from parameters.pendulum import *
from parameters.general import g

alpha_0 = df_optimal['alpha_optimal'].iloc[0]
phi_0 = df_optimal['phi_optimal'].iloc[0]
tau_0 = - eta_alpha * np.sin(alpha_0) - eta_phi  * np.sin(phi_0)

tau_traj = calculate_tau_trajectory(df_optimal['u_optimal'].values, tau_0, dt=0.01, omega_tau=omega_tau)

color_1 = '#5e85ff'
color_2 = '#ff4b6f'

# Plotting on Axis 0
axes[0].plot(df_optimal["k"] * dt, np.rad2deg(df_optimal['alpha_optimal']), 
                label=r'$\alpha^*$', linewidth=1, color=color_1)
axes[0].plot(df_optimal["k"] * dt, np.rad2deg(df_optimal['phi_optimal']), 
                label=r'$\varphi^*$', linewidth=1, color=color_2)
axes[0].legend(loc='upper right', frameon=True)
axes[0].legend(loc='best', frameon=True)

# Plotting on Axis 1
axes[1].plot(df_optimal["k"] * dt, np.rad2deg(df_optimal['alphaD_optimal']), 
                label=r'$\dot{\alpha}^*$', linewidth=1, color=color_1)
axes[1].plot(df_optimal["k"] * dt, np.rad2deg(df_optimal['phiD_optimal']), 
                label=r'$\dot{\varphi}^*$', linewidth=1, color=color_2)
axes[1].legend(loc='best', frameon=True)

# Plotting on Axis 2
u_values = df_optimal['u_optimal'].values[:-1]
tau_traj = calculate_tau_trajectory(u_values, tau_0, dt=dt, omega_tau=omega_tau)

axes[2].plot(df_optimal["k"] * dt, df_optimal['u_optimal'], 
            label=r'$\mathrm{u}^*$', linewidth=1, color=color_1)
axes[2].plot(df_optimal["k"] * dt, tau_traj, 
            label=r'$\tau_\text{c}^*$', linewidth=1, color=color_2)
axes[2].legend(loc='best', frameon=True)

for i, ax in enumerate(axes):
    ax.grid(True)
    ax.set_xlim(-0.05, 0.65)
    
    # Custom tick locations
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # Hide x-ticks labels for top plots, but keep the grid
    if i < 2:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Axis Labels
axes[0].set_ylabel(r"angle [deg]")
axes[1].set_ylabel(r"rate [deg s$^{-1}$]")
axes[2].set_ylabel(r"torque [Nm]")
axes[2].set_xlabel(r"time [s]")

# Force plain numbering on the bottom plot
axes[2].ticklabel_format(useOffset=False, style='plain', axis='both')

# Adjust layout
# plt.subplots_adjust(bottom=0.15, hspace=0.05)

# --- Save and Export ---
save_path = Path("plots/fig4_reference_trajectory.eps")
save_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    save_path,
    format='eps',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.05
)

print(f"Figure saved to {save_path}")
plt.show()