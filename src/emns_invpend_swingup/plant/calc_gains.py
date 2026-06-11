import numpy as np
import control

from emns_invpend_swingup.plant.InvertedPendulum import InvertedPendulum
import emns_invpend_swingup.parameters.general as gen




def calc_inf_lqr_gains_pendulum(Q, R):
    inverted_pendulum = InvertedPendulum()
    operating_point = np.array([0.0, 0.0, 0.0, 0.0])
    torque = 0.0 # Around eq. point torque doesn't play any role
    A_d, B_d, _ , __ = inverted_pendulum.get_discrete_state_space_matrices(operating_point, [torque])

    K, _, _ = control.dlqr(
        A_d,
        B_d,
        Q,
        R,
    )

    return K

def calc_inf_lqr_gains_actuator(Q, R):
    """Compute infinite-horizon discrete LQR gains for the actuator (lower arm).

    The linearized 2-state model is evaluated at alpha=0 (upright equilibrium).
    States: [alpha, alphaD]. Input: torque.
    """
    # Moment of inertia about pivot: rod (1/3 m l²) + point masses (magnet, marker, weight)
    J_actuator = (
        1/3 * gen.m_1 * gen.l_1**2
        + gen.m_m * gen.l_m**2
        + gen.m_act_marker * gen.l_act_marker**2
        + gen.m_act_weight * gen.l_act_weight**2
    )
    # Gravitational restoring torque coefficient: sum of (mass × distance-to-CoM)
    eta_actuator = gen.g * (
        gen.m_m * gen.l_m
        + 1/2 * gen.m_1 * gen.l_1
        + gen.m_act_marker * gen.l_act_marker
        + gen.m_act_weight * gen.l_act_weight
    )

    A = np.array([[0, 1],
                  [eta_actuator / J_actuator, 0]])
    B = np.array([[0],
                  [1 / J_actuator]])

    sys_c = control.ss(A, B, np.eye(2), np.zeros((2, 1)))
    sys_d = control.c2d(sys_c, gen.Ts, 'zoh')
    A_d = np.array(sys_d.A)
    B_d = np.array(sys_d.B)

    K, _, _ = control.dlqr(A_d, B_d, Q, R)
    return K
