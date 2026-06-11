"""
Iterative Learning Controller based on the algorithm of Schoellig et al. 2012.

Reference:
    Schoellig, A.P., Mueller, F.L., D'Andrea, R. (2012).
    Optimization-based iterative learning for precise quadrocopter trajectory tracking.
    Auton Robot 33:103-127. DOI 10.1007/s10514-012-9283-2
"""
import numpy as np

import matplotlib.pyplot as plt

from emns_invpend_swingup.helpers import get_lifted_representation
from emns_invpend_swingup.ilc.DisturbanceKalmanFilter import DisturbanceKalmanFilter

import cvxpy as cp


def get_fd_matrix(n: int) -> np.ndarray:
    D = np.eye(n)
    sub_diagonal_matrix = np.eye(n, k=-1) * -1
    D = D + sub_diagonal_matrix
    
    return D


def ilc_minimize(F_cut, d_est_cut, u_tilde_min_cut, u_tilde_max_cut, S=None, alpha=None):
    """Solve the ILC input update, Eq. (28) from Schoellig et al. 2012.

    min_{u_{j+1}}  ||S (F u_{j+1} + d̂_j)||_2  +  α ||D u_{j+1}||_2

    Args:
        F_cut (np.ndarray): Lifted input-to-state matrix F, shape (N_j*n_x, N_j*n_u), Eq. (14).
        d_est_cut (np.ndarray): Disturbance estimate d̂_j, shape (N_j*n_x, 1), Eq. (23).
        u_tilde_min_cut (np.ndarray): Lower bound on ũ_{j+1}.
        u_tilde_max_cut (np.ndarray): Upper bound on ũ_{j+1}.
        S (np.ndarray): State error scaling matrix S = T_W S_W S_x, Eq. (29).
        alpha (float): Regularization weight α on the input derivative norm, Eq. (28).

    Returns:
        np.ndarray: Optimized input correction ũ_{j+1}.
    """

    if S is None:
        S = np.eye(F_cut.shape[0])
    
    if alpha is None:
        alpha = 0.0

    # Define the decision variable
    u_j_plus_1 = cp.Variable((F_cut.shape[1], 1))
    
    D = get_fd_matrix(F_cut.shape[1])
    # Define the objective function
    state_error = F_cut @ u_j_plus_1 + d_est_cut
    cost = cp.norm(S @ state_error, 2) + alpha * cp.norm(D @ u_j_plus_1, 2)
    
    # Define the constraints
    constraints = [
        u_j_plus_1 >= u_tilde_min_cut,
        u_j_plus_1 <= u_tilde_max_cut
    ]
    
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"ILC optimization failed with status: {problem.status}")

    return u_j_plus_1.value


class IterativeLearningController:

    def __init__(
            self,
            x_optimal,
            y_optimal,
            u_optimal,
            P_0,
            OMEGAs,
            Ms,
            d_0,
            u_min,
            u_max,
            s_x,
            s_w,
            T_w,
            Ks,
            get_discrete_state_space,
            alpha = None,
            get_is_terminated=None
        ):
        # Step 1: Compute lifted representation
        n_x = x_optimal.shape[0]
        n_y = y_optimal.shape[0]
        n_u = u_optimal.shape[0]
        N = u_optimal.shape[1]

        self.N = N
        self.n_x = n_x
        self.n_y = n_y
        self.n_u = n_u
        self.x_optimal = x_optimal


        F, G = get_lifted_representation(
            x_trajectory=x_optimal,
            u_trajectory=u_optimal,
            get_discrete_state_space=get_discrete_state_space,
            Ks = Ks
        )

        self.F = F
        self.G = G

        # Fortran (column-major) order stacks inputs time-step-by-time-step into
        # a single column vector, matching the lifted-domain convention.
        self.u_tilde = np.zeros_like(u_optimal).reshape(-1, 1, order='F')

        self.disturbance_kalman_filter = DisturbanceKalmanFilter(
            n_x = n_x,
            n_y = n_y,
            P_0 = P_0,
            OMEGAs = OMEGAs,
            Ms = Ms,
            G = G
        )

        self.u_optimal = u_optimal
        self.y_optimal = y_optimal 

        self.d_0 = d_0
        self.d_j_minus_1 = d_0
        self.N_j_minus_1 = 0

        self.y_history = []
        self.u_ilc_history = []
        self.u_applied_history = []

        self.u_tilde_min = (u_min - u_optimal).reshape(-1, 1, order='F')
        self.u_tilde_max = (u_max - u_optimal).reshape(-1, 1, order='F')

        
        # Build S = T_W S_W S_x (Eq. 29): s_x scales each state to the same magnitude,
        # s_w weights which deviations matter more, T_w shapes emphasis along the trajectory.
        S_x = np.diag(s_x * N)
        S_w = np.diag(s_w * N)

        self.S = T_w @ S_w @ S_x

        self.get_is_terminated = get_is_terminated
        self.alpha = alpha

        self.j = 1

        self.u_ilc_history.append(self.u_tilde)

    def update_input(self, y_hat, N_j):
        """Run one ILC iteration: update disturbance estimate and compute new input.

        Implements the two-step update from Schoellig et al. 2012, Sects. 2.2-2.3:
          1. Disturbance estimation via Kalman filter (Eqs. 22-23):
               d̂_j = d̂_{j-1} + K_j (ỹ_j - G d̂_{j-1} - G F ũ_j)
          2. Input update via constrained optimization (Eq. 28):
               min ||S (F ũ_{j+1} + d̂_j)||_2 + α ||D ũ_{j+1}||_2

        Args:
            y_hat (np.ndarray): Measured output trajectory y_j, shape (n_y, N+1).
            N_j (int): Effective trial length (may be shorter than N if terminated early).

        Returns:
            np.ndarray: Updated ILC input correction ũ_{j+1}, shape (1, N*n_u).
        """




        
        # N_j = self.get_N_j(y_unfiltered)
        y_deviation = y_hat - self.y_optimal
        
        # Saving data from previous trial

        d_j_minus_1 = self.d_j_minus_1
        N_j_minus_1 = self.N_j_minus_1
        
        d_j_minus_1_extended = np.zeros((self.n_x * N_j, 1))
        
        d_j_minus_1_extended[min(N_j_minus_1, N_j) * self.n_x:, :] = self.d_0[min(N_j_minus_1, N_j) * self.n_x:N_j * self.n_x, :]
        d_j_minus_1_extended[:min(N_j_minus_1, N_j) * self.n_x, :] = d_j_minus_1[:min(N_j_minus_1, N_j)  * self.n_x, :]

        # Output deviation ỹ_j: skip k=0 because the initial state is fixed (not a free output),
        # then flatten in Fortran order to match the lifted-domain column vector convention.
        y_tilde_j = y_deviation[:, 1:(N_j + 1)].reshape(-1, 1, order='F')

        K = self.disturbance_kalman_filter.do_step(N_j)


        # Calculating for the next one

        G_cut = self.G[:N_j*self.n_y, :N_j*self.n_x]
        F_cut = self.F[:N_j*self.n_x, :N_j*self.n_u]
        


        d_hat = d_j_minus_1_extended + K @ (y_tilde_j - G_cut @ d_j_minus_1_extended - G_cut @ F_cut @ self.u_ilc_history[-1].reshape(-1, 1, order='F')[:N_j*self.n_u, :])


        u_tilde_sol_cut = ilc_minimize(
            F_cut=F_cut,
            d_est_cut=d_hat,
            u_tilde_min_cut=self.u_tilde_min[:self.n_u * N_j, :],
            u_tilde_max_cut=self.u_tilde_max[:self.n_u * N_j, :],
            S = self.S[:self.n_x * N_j, :self.n_x * N_j],
            alpha=self.alpha
        ).reshape((1, self.n_u * N_j))

        u_tilde_full = np.zeros( (1, self.n_u * self.N ))

        u_tilde_full[:, :N_j] = u_tilde_sol_cut

        self.d_j_minus_1 = d_hat
        self.N_j_minus_1 = N_j
        
        self.j += 1
        self.u_ilc_history.append(u_tilde_full)

        return u_tilde_full
    
    def get_N_j(self, y_hat):
        y_deviation = y_hat - self.y_optimal
        if self.get_is_terminated is not None:
            for k in range(y_deviation.shape[1]):
                if self.get_is_terminated( y_deviation[:, k] ):
                    return k
                
        return y_deviation.shape[1] - 1
