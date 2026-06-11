import parameters.general as parameters

J_alpha = (
    parameters.m_m * parameters.l_m**2 + 
    1/4 * parameters.m_1 * parameters.l_1**2 + 
    parameters.m_2 * parameters.l_1**2 + 
    parameters.m_j * parameters.l_1**2 + 
    parameters.m_act_marker * parameters.l_act_marker**2 + 
    parameters.m_pend_marker * parameters.l_1**2
    )
J_phi = (
    1/3 * parameters.m_2 * parameters.l_2**2 + 
    parameters.m_pend_marker * parameters.l_pend_marker**2
    )
J_alpha_phi = (
    parameters.m_2 * parameters.l_1 * parameters.l_2 + 
    2 * parameters.l_1 * parameters.l_pend_marker*parameters.m_pend_marker
    )

eta_alpha = parameters.g * (
    parameters.m_2 * parameters.l_1 + 
    parameters.m_j * parameters.l_1 + 
    parameters.m_m * parameters.l_m + 
    1/2 * parameters.m_1 * parameters.l_1 + 
    parameters.m_act_marker * parameters.l_act_marker + 
    parameters.m_pend_marker * parameters.l_1
    )
eta_phi = parameters.g * (
    1/2 * parameters.m_2 * parameters.l_2 + 
    parameters.m_pend_marker * parameters.l_pend_marker
    )

Ts = parameters.Ts
omega_tau = parameters.omega_tau

joint_damping_effect = parameters.joint_damping_effect
joint_switch_arg_scaling = parameters.joint_switch_arg_scaling

l_1 = parameters.l_1
l_2 = parameters.l_2