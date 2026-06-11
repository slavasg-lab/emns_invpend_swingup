import numpy.typing as np_t
import numpy as np
import control
from sympy import Matrix, symbols, cos, sin, simplify, diff, lambdify

import emns_invpend_swingup.parameters.pendulum as parameters




class InvertedPendulum:
    def __init__(self):
        a, p = symbols('a p')
        a_dot, p_dot= symbols('aD pD')

        J_a, J_p, J_ap = symbols('J_a J_p J_ap')
        eta_a, eta_p = symbols('eta_a eta_p')
        t_a = symbols('t_a')
        M11 = J_a
        M12 = .5 * J_ap * cos(a - p)
        M21 = .5 * J_ap * cos(a - p)
        M22 = J_p
        M = Matrix([[M11, M12], [M21, M22]])

        F1 = (
            - .5 * p_dot**2 * sin(a - p) * J_ap
            + sin(a) * eta_a
            + t_a
        )
        F2 = (
            + .5 * a_dot**2 * sin(a - p) * J_ap
            + sin(p) * eta_p
        )
        F = Matrix([F1, F2])

        eom = simplify( M.inv() * F )

        A11 = 0
        A12 = 0
        A13 = 1
        A14 = 0

        A21 = 0
        A22 = 0
        A23 = 0
        A24 = 1

        a_ddot = eom[0]
        A31 = diff(a_ddot, a)
        A32 = diff(a_ddot, p)
        A33 = diff(a_ddot, a_dot)
        A34 = diff(a_ddot, p_dot)

        p_ddot = eom[1]
        A41 = diff(p_ddot, a)
        A42 = diff(p_ddot, p)
        A43 = diff(p_ddot, a_dot)
        A44 = diff(p_ddot, p_dot)

        A = Matrix([
            [A11, A12, A13, A14], 
            [A21, A22, A23, A24], 
            [A31, A32, A33, A34], 
            [A41, A42, A43, A44], 
        ])

        B11 = 0
        B21 = 0
        B31 = diff(a_ddot, t_a)
        B41 = diff(p_ddot, t_a)

        B = Matrix([
            [B11], 
            [B21], 
            [B31], 
            [B41], 
        ])

        params = {
            J_a: parameters.J_alpha,
            J_p: parameters.J_phi,
            J_ap: parameters.J_alpha_phi,
            eta_a: parameters.eta_alpha,
            eta_p: parameters.eta_phi
        }

        self.calc_A = lambdify([a, p, a_dot, p_dot, t_a], A.subs(params), 'numpy')
        self.calc_B = lambdify([a, p, a_dot, p_dot, t_a], B.subs(params), 'numpy')


    def get_continuous_state_space_matrices(self, operating_point: np_t.NDArray, inputs: np_t.NDArray):
        C = np.eye(4)
        D = np.zeros((4, 1))
        return self.calc_A(*operating_point, *inputs), self.calc_B(*operating_point, *inputs), C, D
    


    
    def get_discrete_state_space_matrices(self, operating_point: np_t.NDArray, inputs: np_t.NDArray):
        A, B, C, D = self.get_continuous_state_space_matrices(operating_point, inputs)

        system_conti = control.ss(A, B, C, D)

        system_discrete = control.c2d(system_conti, parameters.Ts, 'zoh')

        A_d, B_d, C_d, D_d = control.ssdata(system_discrete)

        return A_d, B_d, C_d, D_d
