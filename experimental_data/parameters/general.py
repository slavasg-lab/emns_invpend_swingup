import numpy as np

# --- Fundamental consts --- #

g = 9.81 # [m/s^2]

# Permeability constant
mu_0 = 4*np.pi * 1e-7 # [H/m]

# --- Mechanical parameters --- #

rho_carbon = 0.01095 #   [kg/m]

## magnets and pressfit

d_a_magnet = 20e-3 #   [m]
d_i_magnet = 4.2e-3 #   [m]
h_magnet = 5e-3 #     [m]
m_magnet = 11.4e-3 #     [kg/-]

N_magnets = 3 # [-]

m_m = N_magnets * m_magnet #    [kg]

# l_m = 38e-3 # [m]
l_m = 13e-3 + N_magnets / 2 * h_magnet #   [m]

## markers
m_4_markers = 1.1e-3 # [kg]
m_3_markers = 1.2e-3 # [kg]

## actuator


l_1 = 222e-3 #  [m]
m_1 = l_1 * rho_carbon #        [kg]

l_act_marker = 106e-3 # [m]
m_act_marker = 0.0e-3 # [kg]

weight_attached = True
l_act_weight = 197.5e-3 # [m]
m_act_weight = 4.435e-3 if weight_attached else 0.0 # [kg]

# l_weight_tape = 175e-3 # [m]
# m_weight_tape = 0.15e-3 if weight_attached else 0.0 # [kg]

# l_act_marker2magnet = 116e-3 # [m]
l_act_marker2magnet = l_act_marker - l_m # [m]
## pendulum

l_2 = 405e-3 #  [m]
m_2 = l_2 * rho_carbon #        [kg]

# l_pend_marker = 302.5e-3
l_pend_marker = 300e-3 # [m]
m_pend_marker = 0.0e-3 # [kg]

## joint

m_j = 2e-3 #                    [kg]

## other
joint_switch_arg_scaling = 1e3
joint_damping_effect = 0.05




# --- Mechanical parameters aggregation --- #





# --- Magnetic parameters --- #

B_r = 1.345 # [T]

V_magnet = np.pi * h_magnet * (d_a_magnet**2 - d_i_magnet**2) / 4 # [m^3]
m_tilde = (B_r * V_magnet / mu_0) * N_magnets # [A*m^2]

# --- System parameters --- #
frequency = 100 # [Hz]
print(f"VICON FREQUENCY SET TO: {frequency} Hz.")

Ts = 1 / frequency
omega_tau = 2 * np.pi * 20

# --- Trajectory current calculation --- #
pivot_point = np.array([0.0, 10e-2, 0.0]) # [m]