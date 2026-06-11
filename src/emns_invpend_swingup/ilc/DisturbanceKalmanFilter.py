"""
Iteration-domain Kalman filter for disturbance estimation.

Implements Eqs. (22)-(23) from:
    Schoellig, A.P., Mueller, F.L., D'Andrea, R. (2012).
    Optimization-based iterative learning for precise quadrocopter trajectory tracking.
    Auton Robot 33:103-127. DOI 10.1007/s10514-012-9283-2
"""
import numpy as np

# TODO: decrease epsilon with iterations to prevent overfitting to noise;
# Schoellig et al. 2012, Sect. 2.2, Eq. (24): Ω_j = ε_j I with ε_j decreasing over j
class DisturbanceKalmanFilter:
    def __init__(
            self,
            n_x,
            n_y,
            P_0,
            OMEGAs,
            Ms,
            G
        ):
        self.G = G
        self.n_x = n_x
        self.n_y = n_y

        self.P_j_minus_1 = P_0
        self.N_j_minus_1 = 0

        self.P_0 = P_0
        
        self.OMEGA_history = OMEGAs
        self.M_history = Ms

        self.j = 0




    def do_step(self, N_j):
        # Called before each ILC to update the gain
        # 1. Step on j

        self.j += 1

        # 2. Get previous P variables and extend/cut them

        P_j_minus_1 = self.P_j_minus_1
        N_j_minus_1 = self.N_j_minus_1

        # Align P from the previous trial to the current trial length N_j.
        # Steps already estimated carry their covariance; steps beyond N_j_minus_1
        # (horizon extended or first trial) are initialized with P_0 — maximum uncertainty.
        P_j_minus_1_extended = np.zeros((N_j * self.n_x, N_j * self.n_x))
        P_0 = self.P_0
        P_j_minus_1_extended[self.n_x * min(N_j_minus_1, N_j):, self.n_x * min(N_j_minus_1, N_j):] = P_0[self.n_x * min(N_j_minus_1, N_j):self.n_x * N_j, self.n_x * min(N_j_minus_1, N_j):self.n_x * N_j]
        P_j_minus_1_extended[:min(N_j_minus_1, N_j) * self.n_x, :min(N_j_minus_1, N_j) * self.n_x] = P_j_minus_1[:min(N_j_minus_1, N_j) * self.n_x, :min(N_j_minus_1, N_j) * self.n_x]

        # Ω uses index j-1: disturbance evolves as d_j = d_{j-1} + ω_{j-1}, Eq. (16)
        OMEGA_j_minus_1_cut = self.OMEGA_history[self.j - 1][:N_j * self.n_x, :N_j * self.n_x]

        G_cut = self.G[:N_j*self.n_y, :N_j*self.n_x]

        # M uses index j: measurement noise covariance for the current trial, Eq. (22)
        M_j_cut = self.M_history[self.j][:N_j*self.n_y, :N_j*self.n_y]

        # 3. Kalman filter update, Eq. (22): S_j = P_{j-1} + Ω_{j-1},
        #    K_j = S_j G^T (G S_j G^T + M_j)^{-1},  P_j = (I - K_j G) S_j
        S_j = P_j_minus_1_extended + OMEGA_j_minus_1_cut
        K_j = S_j @ G_cut.T @ np.linalg.inv( G_cut @ S_j @ G_cut.T + M_j_cut)
        I = np.eye(N_j * self.n_x)
        P_j = (I - K_j @ G_cut) @ S_j

        # 4. Store updated variables

        self.P_j_minus_1 = P_j[:, :]
        self.N_j_minus_1 = N_j


        return K_j