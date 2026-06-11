import casadi as ca
import numpy as np
import rospkg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import emns_invpend_swingup.parameters.pendulum as pend_parameters
import emns_invpend_swingup.parameters.trajectory as traj_parameters

from emns_invpend_swingup.helpers import calculate_trajectory_currents


def get_casadi_eom(CONSTRAINT):

    def eom(x, u):
        # Correct way to access elements of a CasADi matrix
        alpha = x[0]
        phi = x[1]
        alpha_dot = x[2]
        phi_dot = x[3]
        torque = x[4]
        
        torque_sp = u[0] # Assuming u is also a CasADi MX with one element

        # Use CasADi's symbolic functions for trigonometric and matrix operations
        M11 = pend_parameters.J_alpha        
        M12 = 0.5 * pend_parameters.J_alpha_phi * ca.cos(alpha - phi) # Use ca.cos
        M21 = 0.5 * pend_parameters.J_alpha_phi * ca.cos(alpha - phi) # Use ca.cos
        M22 = pend_parameters.J_phi
        
        # Use ca.vertcat and ca.horzcat for constructing symbolic matrices
        M = ca.vertcat(
            ca.horzcat(M11, M12),
            ca.horzcat(M21, M22)
        )

        # Angle between the two arms: 180° when fully extended (straight line),
        # decreasing as the joint bends. CONSTRAINT is the physical hard stop.
        joint_angle = np.deg2rad(180) - phi + alpha
        angle_violation = CONSTRAINT - joint_angle

        # Smooth contact model using arctan: switches from 0 to 1 as the joint
        # approaches/exceeds the constraint, avoiding the discontinuity of a hard clamp.
        switch_term_overall = (np.arctan(angle_violation * pend_parameters.joint_switch_arg_scaling) + np.pi/2) / np.pi

        # Damping only acts when the joint is closing (positive velocity violation).
        velocity_violation = phi_dot - alpha_dot
        switch_term_velocity = (np.arctan(velocity_violation * pend_parameters.joint_switch_arg_scaling) + np.pi/2) / np.pi

        joint_damping_force = pend_parameters.joint_damping_effect * velocity_violation

        # Net joint torque is zero unless both the angle AND velocity push into the constraint.
        joint_torque = switch_term_overall * switch_term_velocity * joint_damping_force

        F1 = (
            - 0.5 * phi_dot**2 * ca.sin(alpha - phi) * pend_parameters.J_alpha_phi
            + ca.sin(alpha) * pend_parameters.eta_alpha
            # - 2 * parameters.d * alpha_dot
            # + parameters.d * phi_dot
            + torque
            + joint_torque
        )
        F2 = (
            + 0.5 * alpha_dot**2 * ca.sin(alpha - phi) * pend_parameters.J_alpha_phi
            + ca.sin(phi) * pend_parameters.eta_phi
            # - parameters.d * phi_dot
            # + parameters.d * alpha_dot
            - joint_torque
        )
        F = ca.vertcat(F1, F2) # Use ca.vertcat

        # Use CasADi for matrix inversion and multiplication
        accelerations = ca.mtimes(ca.inv(M), F)

        alpha_ddot = accelerations[0]
        phi_ddot = accelerations[1]

        tau_dot = pend_parameters.omega_tau * (torque_sp - torque)

        return ca.vertcat(alpha_dot, phi_dot, alpha_ddot, phi_ddot, tau_dot)

    return eom


class TrajectoryPlannerPendulum:
    def __init__(self, start_alpha, start_phi, package_name, relative_path_to_trajectory,
                 system_type="Navion", calibration_file_name="Navion_1_2_Calibration_old.yaml"):

        CONSTRAINT = np.deg2rad(180) - start_phi + start_alpha

        print(f"JOINT CONSTRAINT: {np.rad2deg(CONSTRAINT)} deg")

        self.system_type = system_type
        self.calibration_file_name = calibration_file_name

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path(package_name)
        self.traj_filename = self.pkg_path + relative_path_to_trajectory

        # Initial guess (most probably adequate)
        alpha_initial = np.linspace(start_alpha, 0, traj_parameters.N + 1)
        phi_initial = np.linspace(start_phi, 0, traj_parameters.N + 1)
        alphaD_initial = np.zeros(traj_parameters.N + 1)
        phiD_initial = np.zeros(traj_parameters.N + 1)
        torques_SP_initial = np.ones(traj_parameters.N) * -traj_parameters.max_torque * np.sign(start_alpha)
        torques_initial = np.ones(traj_parameters.N + 1) * -traj_parameters.max_torque * np.sign(start_alpha)
        
        initial_X = np.array([alpha_initial, phi_initial, alphaD_initial, phiD_initial, torques_initial])
        initial_U = torques_SP_initial
            
        
        x = ca.MX.sym('x', 5)
        u = ca.MX.sym('u', 1)
        
        casadi_eom = get_casadi_eom(CONSTRAINT)
        f = ca.Function('f', [x, u], [casadi_eom(x, u)])

        self.N = traj_parameters.N

        opti = ca.Opti()

        X = opti.variable(5, traj_parameters.N + 1)
        U = opti.variable(1, traj_parameters.N)

        T = traj_parameters.T
        
        alpha, phi, alpha_dot, phi_dot = X[0,:], X[1,:], X[2,:], X[3,:]
        torque = X[4, :]
        torque_sp = U[0,:]

        opti.subject_to(alpha[0] == start_alpha)
        opti.subject_to(phi[0] == start_phi)
        opti.subject_to(alpha_dot[0] == 0)
        opti.subject_to(phi_dot[0] == 0)
        opti.subject_to(torque[0] == -np.sin(start_alpha) * pend_parameters.eta_alpha - np.sin(start_phi) * pend_parameters.eta_phi)


        if start_alpha > 0:
            opti.subject_to(alpha <= start_alpha + np.deg2rad(.1))
        else:
            opti.subject_to(alpha >= start_alpha + np.deg2rad(-.1))


        opti.subject_to(alpha[-1] == 0)
        opti.subject_to(phi[-1] == 0)

        opti.subject_to(alpha_dot[-1] == 0)
        opti.subject_to(alpha_dot[-1] - alpha_dot[-2] == 0)
        opti.subject_to(phi_dot[-1] == 0)
        opti.subject_to(phi_dot[-1] - phi_dot[-2] == 0)
        opti.subject_to(torque_sp[-1] == 0)

        opti.subject_to(opti.bounded(-traj_parameters.max_torque, torque_sp, traj_parameters.max_torque))

        # RK4 with sub-steps: the trajectory horizon is coarse (large dt), so
        # sub-integrating improves accuracy without adding extra decision variables.
        N_subintegrals = 5
        dt = T / traj_parameters.N / N_subintegrals

        for k in range(traj_parameters.N):
            x_k = X[:, k]
            u_k = U[:, k]
            for _ in range(N_subintegrals):
                k1 = f(x_k, u_k)
                k2 = f(x_k + dt/2*k1, u_k)
                k3 = f(x_k + dt/2*k2, u_k)
                k4 = f(x_k + dt*k3, u_k)
                x_k = x_k + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k+1] == x_k)

        opti.minimize(ca.sumsqr(U))

        opti.solver('ipopt')

        opti.set_initial(X, initial_X)
        opti.set_initial(U, initial_U)

        self.opti = opti
        self.X = X
        self.U = U
        self.T = T
        
    def solve(self) -> bool:
        try:
            self.solution = self.opti.solve()
            self.__save_results()
            return True
            
        except RuntimeError as e:
            return False
        
    def __save_results(self):
        self.X = np.array(self.solution.value(self.X))
        self.U = np.array(self.solution.value(self.U))
        self.U = self.U.reshape((1, self.N))

        self.dt = self.T / traj_parameters.N

        np.savez(
            self.traj_filename,
            T = self.T,
            X = self.X[:4, :],
            U = self.U
        )

        self.__visualize()

    def __visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        max_length = pend_parameters.l_1 + pend_parameters.l_2 + 100e-3

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-max_length, max_length])
        ax.set_ylim([-max_length, max_length])
        ax.grid(True)
        ax.set_title("Inverted Pendulum Trajectory")
        ax.set_xlabel("X   [m]")
        ax.set_ylabel("Z   [m]")

        line1, = ax.plot([], [], 'o-', lw=2, color='black', label='Limb 1')
        line2, = ax.plot([], [], 'o-', lw=2, color='black', label='Limb 2')
        # self.ax.legend()

        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

        def update(frame):
            alpha = self.X[0, frame]
            phi = self.X[1, frame]
            current_time = frame * self.dt

            x0 = 0
            y0 = 0

            x1 = pend_parameters.l_1 * np.sin(alpha)
            y1 = pend_parameters.l_1 * np.cos(alpha)

            x2 = x1 + pend_parameters.l_2 * np.sin(phi)
            y2 = y1 + pend_parameters.l_2 * np.cos(phi)


            line1.set_data([x0, x1], [y0, y1])
            line2.set_data([x1, x2], [y1, y2])
            time_text.set_text(f'Time: {current_time:.2f} s')

            return line1, line2, time_text
        
        animation = FuncAnimation(
            fig,
            update,
            frames=np.linspace(0, self.X.shape[1] - 1, 50, dtype=int),
            blit=True,
            interval=self.dt * 1000,  # Interval in milliseconds
            repeat=False
        )

        animation.save(filename=self.pkg_path + "/analysis/trajectory/animation.gif")
        plt.close()

        _, axs = plt.subplots(4, 1, sharex=True, figsize=(24, 16))
        time_points_states = np.arange(self.X.shape[1]) * pend_parameters.Ts
    

        plt.grid(True)

        axs[0].plot(time_points_states, np.rad2deg(self.X[0, :]), label='alpha')
        axs[0].plot(time_points_states, np.rad2deg(self.X[1, :]), label='phi')

        axs[0].set_ylabel('Angles   [deg]')
        axs[0].legend() 

        axs[1].plot(time_points_states, np.rad2deg(self.X[2, :]), label='alphaD')
        axs[1].plot(time_points_states, np.rad2deg(self.X[3, :]), label='phiD')

        axs[1].set_ylabel('Angular velocities   [deg/s]')
        axs[1].legend() 

        axs[2].plot(time_points_states, self.X[4, :], label='tau')
        axs[2].set_ylabel('Torque   [Nm]')
        axs[2].legend()

        time_points_states = np.arange(self.U.shape[1]) * pend_parameters.Ts
        axs[2].plot(time_points_states, self.U[0, :], label='tau_sp')
        axs[2].set_ylabel('Torque setpoint   [Nm]')

        currents = calculate_trajectory_currents(self.X[0, :-1], self.U, self.system_type, self.calibration_file_name)
        

        time_points_states = np.arange(currents.shape[1]) * self.T / self.N

        for i in range(currents.shape[0]):
            axs[3].plot(time_points_states, currents[i, :], label=f"i_{i}")
        axs[3].set_ylabel('Current   [A]')
        axs[3].legend()

        plt.grid(True)
        plt.savefig(self.pkg_path + "/analysis/trajectory/plots.png")
        plt.close()