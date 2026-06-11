import numpy as np
import emns_invpend_swingup.parameters.general as parameters 
from control_utils.general.torque_utils import matrix_gradfield_to_torqueforce
from control_utils.general.utilities import init_MPEM_model, get_actuation_matrix_from_model
import cvxpy as cp
from emns_invpend_swingup.geometry_jit import jacobian_torqueforce_to_torque



def calc_trajectory_feedback_lqr_gains(x_trajectory, u_trajectory, get_discrete_state_space_matrices, Q, R, Q_f):
    N = u_trajectory.shape[1]

    As = []
    Bs = []
    for k in range(N):
        x_k = x_trajectory[:, k]
        u_k = u_trajectory[:, k]
        A_k, B_k, _, __ = get_discrete_state_space_matrices(x_k, u_k)
        As.append(A_k)
        Bs.append(B_k)
         
    Ps = [None] * (N + 1)
    Ps[N] = Q_f

    # Backward-pass Riccati recursion: gains at step k depend on the cost-to-go
    # from k+1 onward, so we must sweep from the terminal condition back to k=0.
    for k in range(N, 0, -1):
        P_k = Ps[k]
        A_k_minus_1 = As[k - 1]
        B_k_minus_1 = Bs[k - 1]

        P_k_minus_1 = Q + A_k_minus_1.T @ P_k @ A_k_minus_1 - A_k_minus_1.T @ P_k @ B_k_minus_1 @ np.linalg.inv(R + B_k_minus_1.T @ P_k @ B_k_minus_1) @ B_k_minus_1.T @ P_k @ A_k_minus_1

        Ps[k - 1] = P_k_minus_1

    Ks = [None] * N
    for k in range(N):
        A_k = As[k]
        B_k = Bs[k]
        P_k_plus_1 = Ps[k + 1]

        K_k = -np.linalg.inv(R + B_k.T @ P_k_plus_1 @ B_k) @ B_k.T @ P_k_plus_1 @ A_k

        Ks[k] = K_k

    return Ks


def get_lifted_representation(x_trajectory, u_trajectory, get_discrete_state_space, Ks, x0_tilde=None):
    """ Computes the lifted representation (F, G, H, d0) of the system along the given trajectory.
    
    Args:   
        x_trajectory (np.ndarray): State trajectory of shape (n_x, N+1).
        u_trajectory (np.ndarray): Input trajectory of shape (n_u, N).
        get_discrete_state_space (function): Function that returns the discrete state-space matrices 
                                             A_d, B_d, C_d, D_d for given state and input.

    Returns:
        F (np.ndarray): Lifted input-to-state matrix of shape (N*n_x, N*n_u).
        G (np.ndarray): Lifted state-to-output matrix of shape (N*n_y, N*n_x).
    """
    if x0_tilde is None:
        x0_tilde = np.zeros((x_trajectory.shape[0], 1))
    else:
        raise NotImplementedError("x0_tilde other than zero is not implemented yet.")


    A_d_trajectory = []
    B_d_trajectory = []
    C_d_trajectory = []

    N = u_trajectory.shape[1]



    for k in range(N):
        x_k = x_trajectory[:, k]
        u_k = u_trajectory[:, k]
        K_k = Ks[k]

        A_d_k, B_d_k, C_d_k, _ = get_discrete_state_space(x_k, u_k)
        # Closed-loop A: feedback gain K shifts the poles of the open-loop system
        A_d_trajectory.append(A_d_k + B_d_k @ K_k)
        B_d_trajectory.append(B_d_k)
        C_d_trajectory.append(C_d_k)
    

    n_x = A_d_k.shape[0]
    n_u = B_d_k.shape[1]
    n_y = C_d_k.shape[0]

    # COMPUTING F — lower-triangular block matrix (Eq. 14, Schoellig 2012).
    # F[l,m] = A(l-1)…A(m) B(m-1) if m < l  (input m propagates to step l)
    #        = B(m-1)              if m == l  (direct feedthrough)
    #        = 0                   if m > l  (future inputs cannot affect past states)
    F = [[None for _ in range(N)] for _ in range(N)]

    for l in range(1, N + 1):
        for m in range(1, N + 1):
            if m < l:
                # Start with identity matrix of the same shape as A_D[0]
                prod = np.eye(A_d_trajectory[0].shape[0])
                # Multiply A_D(l-1) * ... * A_D(m)
                for k in range(l - 1, m - 1, -1):  # descending from l-1 to m
                    prod = prod @ A_d_trajectory[k]
                F[l - 1][m - 1] = prod @ B_d_trajectory[m - 1]
            elif m == l:
                F[l - 1][m - 1] = B_d_trajectory[m - 1]
            else:  # m > l
                F[l - 1][m - 1] = np.zeros_like(B_d_trajectory[0])

    F = np.block(F)


    # COMPUTING G
    G = np.zeros(( N * n_y, N *n_x ))
    for l in range(1, N + 1):
        G[(l - 1) * n_y : l * n_y, (l - 1) * n_x : l * n_x] = C_d_trajectory[l-1]

    return F, G

def calculate_trajectory_currents(alpha_trajectory: np.ndarray, torque_trajectory: np.ndarray, system_type, calibration_file_name):
    currents = []

    model = init_MPEM_model(calibration_file_name) 
    
    for k, (alpha_k, torque_k) in enumerate(zip(alpha_trajectory, torque_trajectory[0, :])):
        
        torque_vec = np.array([0, float(torque_k), 0])

        normal_vector_actuator = np.array([np.sin(alpha_k), 0, np.cos(alpha_k)])
        magnet_position = parameters.pivot_point + normal_vector_actuator * parameters.l_m

        J = jacobian_torqueforce_to_torque(beta = 0.0, alpha = alpha_k, l_mag = parameters.l_m)
        J = J[:2, :]

        magnetic_dipole_moment = parameters.m_tilde * normal_vector_actuator
        M = matrix_gradfield_to_torqueforce(magnetic_dipole_moment)

        A = get_actuation_matrix_from_model(model, magnet_position) # init actuation matrix
        allocation_matrix = J @ M @ A

        currents_k = np.linalg.pinv(allocation_matrix) @ torque_vec[:2]

        currents.append(currents_k)
    

    return np.vstack(currents).T

def calculate_max_trajectory_current(alpha_trajectory: np.ndarray, torque_trajectory: np.ndarray, system_type, calibration_file_name):
    return np.max(np.abs(calculate_trajectory_currents(alpha_trajectory=alpha_trajectory, torque_trajectory=torque_trajectory, system_type=system_type, calibration_file_name=calibration_file_name)))