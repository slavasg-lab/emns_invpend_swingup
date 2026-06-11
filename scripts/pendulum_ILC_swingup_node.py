#!/usr/bin/env python3
import rospy
import numpy as np
import time as timing
import rospkg

from geometry_msgs.msg import TransformStamped
from control_utils.msg import ScalarStamped
from std_srvs.srv import Empty, EmptyResponse

from control_utils.general.utilities import init_MPEM_model, get_actuation_matrix_from_model
from control_utils.general.utilities import init_system
from control_utils.feedback_ctrl.state_feedback import StateFeedbackController


from emns_invpend_swingup.utils import gains2csv, csv2nparr
from emns_invpend_swingup.plant.calc_gains import calc_inf_lqr_gains_pendulum
from emns_invpend_swingup.plant.calc_gains import calc_inf_lqr_gains_actuator
from emns_invpend_swingup.plant.InvertedPendulum import InvertedPendulum
from emns_invpend_swingup.plant.TrajectoryPlanner import TrajectoryPlannerPendulum
from emns_invpend_swingup.ilc.IterativeLearningController import IterativeLearningController
import emns_invpend_swingup.parameters.trajectory as trajectory_params
import emns_invpend_swingup.parameters.general as params
from emns_invpend_swingup.ilc.DataLogger import DataLogger
from emns_invpend_swingup.helpers import calculate_max_trajectory_current, calc_trajectory_feedback_lqr_gains

from emns_invpend_swingup.geometry_jit import get_normal_vector_from_quaternion, get_normal_angles_from_quaternion, get_normal_angles_from_normal_vector, magnetic_interaction_from_dipole_moment, jacobian_torqueforce_to_torque
from emns_invpend_swingup.numerical import numba_pinv, numba_clip

from emns_invpend_swingup.srv import SetBoolean, SetBooleanResponse



class InvertedPendulum3DControlNode:
    def __init__(self):
        rospy.init_node('oct_3d_pend_node', anonymous=True)

        rospy.loginfo("Initializing the control node...")


        # Initializing the trajectory and ILC
        rospy.Service('~ilc_init', Empty, self.init_ilc_callback)
        rospy.Service('~start_trial', Empty, self.start_trial_callback)
        rospy.Service('~ilc_step', Empty, self.ilc_step_callback)
        rospy.Service('~switch_eq_feedback', SetBoolean, self.set_eq_fb)
        rospy.Service('~process_current_trial', Empty, self.process_current_trial_callback)

        # Pre-trajectory warmup: apply the initial input for this many steps before
        # k=0 so the magnet field is already ramped up when recording starts.
        self.initial_k = -4

        self.read_ROS_parameters()
        self.init_payload_variables()
        self.init_publishers()

        if self.system_type == "JECB":
            self.active_drivers = [0, 1, 2,  3, 4, 5,  7, 8]
            self.model = init_MPEM_model(self.calibration_file_name)
            self.actuation_matrix = get_actuation_matrix_from_model(self.model, position = np.array([0.0, 0.0, 0.0]))  
            self.N_coils = 9
            self.CURRENT_MAX = 4.0 # [A]
            self.CURRENT_MIN = -self.CURRENT_MAX 
        elif self.system_type == "Navion":
            self.active_drivers = [0, 1, 2]
            self.model = init_MPEM_model(self.calibration_file_name)
            self.actuation_matrix = get_actuation_matrix_from_model(self.model, np.zeros(3)) 
            self.N_coils = 3
            self.CURRENT_MAX = 13.0 # [A]
            self.CURRENT_MIN = -self.CURRENT_MAX 
      
        self.current_SP = np.zeros(self.N_coils)


        if self.b_use_prerecoded_correction_angles:
            self.phi_ss, self.theta_ss = csv2nparr(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/correction_angles.csv")
        else:
            self.phi_ss, self.theta_ss = 0.0, 0.0

        
        if self.b_use_prerecoded_integrator_torques:
            self.torque_int_beta, self.torque_int_alpha = csv2nparr(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/int_torques.csv")
        else:
            self.torque_int_beta, self.torque_int_alpha = 0.0, 0.0
        
        self.eq_switched_on = False
        self.current_msg, self.current_SP_pub, self.publish_currents = init_system(self.system_type, self.b_hardware_connected, coil_nrs = self.active_drivers)
        
        rospy.loginfo("Control node initialized!")


    def set_eq_fb(self, req):
        self.equilibrium_feedback_enabled = req.value
        return SetBooleanResponse()
    

    def process_current_trial_callback(self, req):
        
        self.N_j = self.ilc.get_N_j(self.state_history)

        self.data_logger.log_current_trial(
            x_real = self.state_history,
            x_fitted = self.state_history,
            u = self.input_history,
            u_ilc = self.u_ilc,
            N_j = self.N_j
        )
        rospy.loginfo("Trial postprocessing done.")
        return EmptyResponse()
    
    def init_publishers(self):
        self.pub_alpha  = rospy.Publisher("/alpha", ScalarStamped, queue_size=10)
        self.pub_alpha_traj  = rospy.Publisher("/alpha_traj", ScalarStamped, queue_size=10)
        self.pub_alphaD = rospy.Publisher("/alphaD", ScalarStamped, queue_size=10)
        self.pub_alphaD_traj = rospy.Publisher("/alphaD_traj", ScalarStamped, queue_size=10)
        self.msg_alpha  = ScalarStamped()
        self.msg_alpha_traj  = ScalarStamped()
        self.msg_alphaD = ScalarStamped()
        self.msg_alphaD_traj = ScalarStamped()

        self.pub_phi  = rospy.Publisher("/phi", ScalarStamped, queue_size=10)
        self.pub_phi_traj  = rospy.Publisher("/phi_traj", ScalarStamped, queue_size=10)
        self.pub_phiD = rospy.Publisher("/phiD", ScalarStamped, queue_size=10)
        self.pub_phiD_traj = rospy.Publisher("/phiD_traj", ScalarStamped, queue_size=10)
        self.msg_phi  = ScalarStamped()
        self.msg_phi_traj  = ScalarStamped()
        self.msg_phiD = ScalarStamped()
        self.msg_phiD_traj = ScalarStamped()

        self.pub_phase = rospy.Publisher("/phase", ScalarStamped, queue_size=10) # publishing which phase the system is in (0 - not running, 1 - trajectory tracking, 2 - equilibrium stabilization)
        self.msg_phase = ScalarStamped()

        self.pub_j = rospy.Publisher("/current_iteration", ScalarStamped, queue_size=10) # publishing current iteration
        self.msg_j = ScalarStamped()
    
        self.pub_tau_eq = rospy.Publisher("/tau_eq", ScalarStamped, queue_size=10) # publishing equilibrium torque
        self.pub_tau_fb = rospy.Publisher("/tau_fb", ScalarStamped, queue_size=10) # publishing feedback torque
        self.pub_tau_traj = rospy.Publisher("/tau_traj", ScalarStamped, queue_size=10) # publishing trajectory torque
        self.pub_tau_ilc = rospy.Publisher("/tau_ilc", ScalarStamped, queue_size=10) # publishing ILC torque
        self.pub_tau_int = rospy.Publisher("/tau_int", ScalarStamped, queue_size=10) # publishing integrator torque

        self.msg_tau_eq = ScalarStamped()
        self.msg_tau_fb = ScalarStamped()
        self.msg_tau_traj = ScalarStamped()
        self.msg_tau_ilc = ScalarStamped()
        self.msg_tau_int = ScalarStamped()

        self.pub_process_time = rospy.Publisher("/process_time", ScalarStamped, queue_size=10)
        self.msg_process_time = ScalarStamped()

    def publish_vars(self, tau_eq, tau_fb, tau_int):
        if self.verbose:
            self.msg_alpha.header.stamp = rospy.Time.now()
            self.msg_alpha.scalar = self.alpha
            self.pub_alpha.publish(self.msg_alpha)

            self.msg_alphaD.header.stamp = rospy.Time.now()
            self.msg_alphaD.scalar = self.alphaD
            self.pub_alphaD.publish(self.msg_alphaD)

            self.msg_phi.header.stamp = rospy.Time.now()
            self.msg_phi.scalar = self.phi
            self.pub_phi.publish(self.msg_phi)

            self.msg_phiD.header.stamp = rospy.Time.now()
            self.msg_phiD.scalar = self.phiD
            self.pub_phiD.publish(self.msg_phiD)

            self.msg_phase.header.stamp = rospy.Time.now()
            self.msg_phase.scalar = self.phase
            self.pub_phase.publish(self.msg_phase)

            self.msg_j.header.stamp = rospy.Time.now()
            self.msg_j.scalar = 0 if self.ilc is None else self.ilc.j
            self.pub_j.publish(self.msg_j)

            self.msg_alpha_traj.header.stamp = rospy.Time.now()
            self.msg_phi_traj.header.stamp = rospy.Time.now()
            self.msg_alphaD_traj.header.stamp = rospy.Time.now()
            self.msg_phiD_traj.header.stamp = rospy.Time.now()
            self.msg_tau_traj.header.stamp = rospy.Time.now()
            self.msg_tau_ilc.header.stamp = rospy.Time.now()

            # k can be negative during pre-warmup; clamp to 0 for safe trajectory indexing
            if self.k is not None and self.x_optimal is not None:
                self.msg_alpha_traj.scalar = self.x_optimal[0, max(self.k, 0)]
                self.msg_phi_traj.scalar = self.x_optimal[1, max(self.k, 0)]
                self.msg_alphaD_traj.scalar = self.x_optimal[2, max(self.k, 0)]
                self.msg_phiD_traj.scalar = self.x_optimal[3, max(self.k, 0)]
                self.msg_tau_traj.scalar = self.u_optimal[0, max(self.k, 0)]
                self.msg_tau_ilc.scalar = self.u_ilc[0, max(self.k, 0)]
            else:
                self.msg_alpha_traj.scalar = 0.0
                self.msg_phi_traj.scalar = 0.0
                self.msg_alphaD_traj.scalar = 0.0
                self.msg_phiD_traj.scalar = 0.0
                self.msg_tau_traj.scalar = 0.0
                self.msg_tau_ilc.scalar = 0.0

            self.pub_alpha_traj.publish(self.msg_alpha_traj)
            self.pub_phi_traj.publish(self.msg_phi_traj)
            self.pub_alphaD_traj.publish(self.msg_alphaD_traj)
            self.pub_phiD_traj.publish(self.msg_phiD_traj)
            self.pub_tau_traj.publish(self.msg_tau_traj)
            self.pub_tau_ilc.publish(self.msg_tau_ilc)

            self.msg_tau_eq.header.stamp = rospy.Time.now()
            self.msg_tau_eq.scalar = tau_eq
            self.pub_tau_eq.publish(self.msg_tau_eq)

            self.msg_tau_fb.header.stamp = rospy.Time.now()
            self.msg_tau_fb.scalar = tau_fb
            self.pub_tau_fb.publish(self.msg_tau_fb)

            self.msg_tau_int.header.stamp = rospy.Time.now()
            self.msg_tau_int.scalar = tau_int
            self.pub_tau_int.publish(self.msg_tau_int)

        self.msg_process_time.header.stamp = rospy.Time.now()
        self.msg_process_time.scalar = timing.time() - self.current_vicon_timestamp
        self.pub_process_time.publish(self.msg_process_time)

    def init_ilc_callback(self, req):
        package_name = "emns_invpend_swingup"
        package_path = rospkg.RosPack().get_path(package_name)

        # 1. Initialize the trajectory
        if self.trajectory_path is None:
            self.trajectory_path = "/data/pendulum/swingup/trajectory.npz"
            trajectory_planner = TrajectoryPlannerPendulum(
                start_alpha=self.alpha,
                start_phi=self.phi,
                package_name=package_name,
                relative_path_to_trajectory=self.trajectory_path,
                system_type=self.system_type,
                calibration_file_name=self.calibration_file_name
            )

            if not trajectory_planner.solve():
                rospy.logerr("Failed to plan the trajectory.")
                return EmptyResponse()
        
        # 2. Load the trajectory
        trajectory_zip = np.load(package_path + self.trajectory_path)
        x_optimal, u_optimal, T_optimal = trajectory_zip["X"], trajectory_zip["U"], trajectory_zip["T"]
        y_optimal = x_optimal.copy()

        n_x = x_optimal.shape[0]
        n_y = x_optimal.shape[0]
        n_u = u_optimal.shape[0]
        N = u_optimal.shape[1]

        self.x_optimal = x_optimal
        self.u_optimal = u_optimal

        # 2.5 Calculate max current
        
        max_current = calculate_max_trajectory_current(alpha_trajectory=x_optimal[0, :-1],
                                                       torque_trajectory=u_optimal,
                                                       system_type=self.system_type,
                                                       calibration_file_name=self.calibration_file_name)
        rospy.loginfo(f"MAX TRAJECTORY CURRENT: {max_current} A.")
        # 3. Initialize ILC parameters
        
        # Initial error covariance P_0 = E[(d_0-d̂_0)(d_0-d̂_0)^T]; larger -> faster
        # initial adaptation (Schoellig 2012, Eq. 22)
        P_0 = np.eye( n_x * N ) * 1

        # Process noise covariance Ω_j = ε_j I (Schoellig 2012, Eq. 24).
        # ε controls how much the disturbance estimate can change between iterations;
        # larger ε -> faster adaptation, smaller ε -> smoother learning.
        epsilon = 2e-2
        OMEGA = np.eye( n_x * N ) * epsilon
        OMEGAs = [OMEGA for _ in range(30)]

        # Measurement noise covariance M_j = η I (Schoellig 2012, Eq. 25).
        # η captures sensor noise and unmodeled dynamics.
        eta = 1e-3
        M = np.eye( n_y * N ) * eta
        Ms = [M for _ in range(30)]

        # Initial disturbance estimate d̂_0
        d_0 = np.zeros(( N * n_x, 1 ))

        # Torque bounds
        u_min = trajectory_params.min_torque * np.ones((1, N))
        u_max = trajectory_params.max_torque * np.ones((1, N))

        # Regularization weight α on the input derivative norm (Schoellig 2012, Eq. 28).
        # Penalizes rapid changes in the ILC input; higher α -> smoother but slower learning.
        alpha = 0.05 / params.Ts

        # Scaling matrices forming S = T_W S_W S_x (Schoellig 2012, Eq. 29).
        # s_x normalizes each state dimension; s_w weights state deviations;
        # T_w weights trajectory segments (uniform here).
        s_x = [1 / np.max(np.abs(x_optimal[i,:])) for i in range(n_x)]
        s_w = [1.0, 1.0, 1.0, 1.0]
        T_w = np.eye( n_x * N )

        # Trial termination criteria
        def get_is_terminated(state_deviation):
            return np.abs(state_deviation[0]) > np.deg2rad(20) or np.abs(state_deviation[1]) > np.deg2rad(20)
        

        # Calculating trajectory tracking feedback gains
        Q_traj = np.diag([1, 1, 0.01, 0.01])
        R_traj = np.diag([1e4])
        Q_f_traj = 1 * Q_traj

        get_discrete_state_space = InvertedPendulum().get_discrete_state_space_matrices

        if self.trajectory_feedback_enabled:
            self.Ks = calc_trajectory_feedback_lqr_gains(
                x_trajectory=x_optimal,
                u_trajectory=u_optimal,
                get_discrete_state_space_matrices=get_discrete_state_space,
                Q = Q_traj,
                R = R_traj,
                Q_f = Q_f_traj
            )
        else:
            self.Ks = [np.zeros((1,4)) for _ in range(N)]

        # 4. Initialize the ILC controller
        self.ilc = IterativeLearningController(
            x_optimal = x_optimal,
            y_optimal = y_optimal,
            u_optimal = u_optimal,
            P_0 = P_0,
            OMEGAs = OMEGAs,
            Ms = Ms,
            d_0 = d_0,
            u_min = u_min, # Just a scalar for now
            u_max = u_max,
            s_x = s_x,
            s_w = s_w,
            T_w = T_w,
            Ks = self.Ks,
            alpha = alpha,
            get_discrete_state_space = get_discrete_state_space,
            get_is_terminated = get_is_terminated
        )

        # 5. Initialize trajectory tracking feedback controller for lateral deviations
        Q_beta = np.diag([1, 0.1])
        R_beta = np.diag([1e4])
        if self.trajectory_beta_enabled:
            K = calc_inf_lqr_gains_actuator(Q = Q_beta, R = R_beta)
        else:
            K = np.zeros((1,2))
        
        gains2csv(K, package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/actuator/balancing/inf_lqr_gains.csv")

        self.ctrl_b_trajectory = StateFeedbackController(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/actuator/balancing/inf_lqr_gains.csv") # _b = beta theta

        

        # 6. Initialize equilibrium balancing feedback controller
        Q_eq = np.diag([1, 1, 0.1, 0.1])
        R_eq = np.diag([1e4])

        K = calc_inf_lqr_gains_pendulum(Q = Q_eq, R = R_eq)
        gains2csv(K, package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/balancing/inf_lqr_gains.csv")

        self.inf_lqr = StateFeedbackController(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/balancing/inf_lqr_gains.csv") # _b = beta theta

        # 7. Initialize switching condition: switch to equilibrium LQR once near-upright
        # with low velocity. Velocity check prevents switching during a fast pass-through.
        self.switching_condition = lambda : np.abs(self.phi) < np.deg2rad(10) and np.abs(self.phiD) < np.deg2rad(100) and np.abs(self.alpha) < np.deg2rad(30)


        # 8. Save the data

        hyperparams = {
            # --- ILC-related ---
            "P_0": P_0,
            "OMEGAs": np.array(OMEGAs),
            "Ms": np.array(Ms),
            "d_0": d_0,
            "u_min": u_min,
            "u_max": u_max,
            "alpha": alpha,
            "s_x": np.array(s_x),
            "s_w": np.array(s_w),
            "T_w": T_w,

            # --- Trajectory feedback settings ---
            "trajectory_feedback_enabled": self.trajectory_feedback_enabled,
            "trajectory_beta_enabled": self.trajectory_beta_enabled,
            "is_3D": self.is_3D,
            "b_use_prerecoded_integrator_torques": self.b_use_prerecoded_integrator_torques,
            "b_use_prerecoded_correction_angles": self.b_use_prerecoded_correction_angles,
            "calibration_file_name": self.calibration_file_name,


            # --- Separate Q, R, Q_f matrices ---
            # For trajectory beta controller
            "Q_beta": Q_beta,
            "R_beta": R_beta,

            # For trajectory tracking controller
            "Q_tracking": Q_traj,
            "R_tracking": R_traj,
            "Qf_tracking": Q_f_traj,

            # For equilibrium stabilization controller
            "Q_eq": Q_eq,
            "R_eq": R_eq
        }

        self.data_logger = DataLogger(
            x_optimal=x_optimal,
            u_optimal=u_optimal,
            T_optimal=T_optimal,
            package_name="emns_invpend_swingup",
            relative_path="analysis/pendulum_ilc",
            hyperparameters=hyperparams
        )

        rospy.loginfo("All controllers initialized successfully.")

        return EmptyResponse()
    
    def start_trial_callback(self, req):
        self.trial_running = True
        self.prev_vicon_timestamp = None
        
        self.state_history = np.zeros((4, trajectory_params.N + 1))
        self.input_history = np.zeros((1, trajectory_params.N))


        return EmptyResponse()
    
    def ilc_step_callback(self, req):
        u_ilc_old = self.u_ilc
        self.u_ilc = self.ilc.update_input(self.state_history, self.N_j)
        self.data_logger.log_trial(
            x_real = self.state_history,
            x_fitted = self.state_history,
            u = self.input_history,
            u_ilc = u_ilc_old,
            N_j = self.N_j,
            j = self.ilc.j - 1
        )
        return EmptyResponse()
    
    def read_ROS_parameters(self):
        self.vicon_callback_topic_actuator      = rospy.get_param('~vicon_callback_topic_actuator', "/vicon/pendulum_marker/Root")
        self.vicon_callback_topic_pendulum      = rospy.get_param('~vicon_callback_topic_pendulum', "/vicon/arm_marker/Root")
        
        self.trajectory_beta_enabled        = rospy.get_param('~trajectory_beta_enabled', False)
        self.trajectory_feedback_enabled    = rospy.get_param('~trajectory_feedback_enabled', False)
        self.equilibrium_feedback_enabled   = rospy.get_param('~equilibrium_feedback_enabled', False)

        self.b_use_prerecoded_integrator_torques = rospy.get_param('~b_use_prerecoded_integrator_torques', False)
        self.b_use_prerecoded_correction_angles = rospy.get_param('~b_use_prerecoded_correction_angles', False)
        
        self.system_type                   = rospy.get_param('~system_type', "JECB")
        self.calibration_file_name         = rospy.get_param('~calibration_file_name', "octomag_5point.yaml")

        self.b_hardware_connected          = rospy.get_param('~b_hardware_connected', False)
        self.b_southpole_up                = rospy.get_param('~b_southpole_up', False)
        self.b_flip_actuation_matrix       = rospy.get_param('~b_flip_actuation_matrix', False)

        self.is_3D                          = rospy.get_param('~is_3D', True)

        self.trajectory_path                = rospy.get_param('~trajectory_path', None)
        self.verbose                        = rospy.get_param('~verbose', False)

        if self.system_type == "JECB" and self.calibration_file_name == "octomag_5point.yaml":
            self.b_flip_actuation_matrix = True
    
    def init_payload_variables(self):
        self.alpha          = np.deg2rad(60)
        self.alphaD         = 0.0
        self.alpha_hat      = 0.0
        self.prev_alpha_hat = 0.0

        self.phi            = self.alpha + np.deg2rad(90)
        self.phiD           = 0.0
        self.phi_hat        = 0.0
        self.prev_phi_hat   = 0.0

        self.beta           = 0.0
        self.betaD          = 0.0
        self.beta_hat       = 0.0
        self.prev_beta_hat  = 0.0

        self.theta            = 0.0
        self.thetaD           = 0.0
        self.theta_hat        = 0.0
        self.prev_theta_hat   = 0.0

        self.callback_time = None

        self.magnet_position = np.zeros(3)
        self.torque_vec      = np.zeros(3)

        self.ilc       = None
        self.ctrl_b_trajectory    = None
        self.inf_lqr = None
        self.u_optimal = None
        self.u_ilc = np.zeros((1, trajectory_params.N))
        self.k = None

        self.current_vicon_timestamp    = None
        self.prev_vicon_timestamp       = None

        self.trial_running = False
        self.recorded_k = []
        self.state_history = np.zeros((4, trajectory_params.N + 1))
        self.input_history = np.zeros((1, trajectory_params.N))
        self.N_j = trajectory_params.N

        self.normal_vector_actuator = np.array([0.0, 0.0, 1.0])

        self.phase = 0 # 0 - not running, 1 - trajectory tracking, 2 - equilibrium stabilization

        self.switching_condition = lambda : False

    def get_current_k(self):

        # Determine k:
        if not self.trial_running: # If it's not running, there is no k.
            return None

        if self.prev_vicon_timestamp is None: # If it's the first call after starting the trial and previous timestamp is not set -> it's the initial k. Reset recorded k list.
            self.recorded_k = []
            k = self.initial_k

        else: # Calculate how many time steps have passed since the last callback (in case of dropped frames due to occlusions)
            dk = np.round( (self.current_vicon_timestamp - self.prev_vicon_timestamp) / params.Ts ).astype(int)
            k = self.k + dk

        if k == trajectory_params.N: # Record last frame state and stop the trial
            x_a, x_b = self.get_state()
            self.state_history[:, self.k:self.k+1] = x_a
            self.recorded_k.append(k)
            

        if k >= trajectory_params.N: # If k exceeds the trajectory length -> stop the trial
            self.prev_vicon_timestamp = None
            self.trial_running = False

            rospy.loginfo(f"Trial finished. Missing frames: {list( set(range(trajectory_params.N + 1)) - set(self.recorded_k) )}")
            self.k = None

            return None
        
        self.recorded_k.append(k)
        self.prev_vicon_timestamp = self.current_vicon_timestamp
        self.k = k

        return k
    

    def callback_alpha_beta(self, msg):
        quat_rot =  msg.transform.rotation
        quat_rot = np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])
        self.normal_vector_actuator = -get_normal_vector_from_quaternion(quat_rot)

        self.beta, self.alpha = get_normal_angles_from_normal_vector(self.normal_vector_actuator)

        vicon_position = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        self.magnet_position = vicon_position - params.l_act_marker2magnet * self.normal_vector_actuator

        self.prev_alpha_hat = self.alpha_hat
        self.alpha_hat = self.alpha
        self.alphaD = (self.alpha_hat - self.prev_alpha_hat) / params.Ts

        self.prev_beta_hat = self.beta_hat
        self.beta_hat = self.beta
        self.betaD = (self.beta_hat - self.prev_beta_hat) / params.Ts



    def callback_phi_theta(self, msg):
        self.current_vicon_timestamp = timing.time()

        quat_rot =  msg.transform.rotation
        quat_rot = np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])

        self.theta, self.phi = get_normal_angles_from_quaternion(quat_rot, flip=True)

        self.prev_phi_hat = self.phi_hat
        self.phi_hat = self.phi
        self.phiD= (self.phi_hat - self.prev_phi_hat) / params.Ts

        self.prev_theta_hat = self.theta_hat
        self.theta_hat = self.theta
        self.thetaD = (self.theta_hat - self.prev_theta_hat) / params.Ts

        self.k = self.get_current_k()
        self.state_feedback_step()
        
        # --- not publishing ---
        if self.trial_running and self.k >= 0:
            x_a, x_b = self.get_state()
            self.state_history[:, self.k:self.k+1] = x_a
            if self.k < trajectory_params.N:
                # South-pole-up mounting flips the magnet polarity, so torque sign is reversed
                self.input_history[0, self.k] = self.torque_vec[1] if not self.b_southpole_up else -self.torque_vec[1]


    def state_feedback_step(self):
        x_a, x_b = self.get_state()

        to_switch = self.switching_condition()

        # If it's within radius of catching, equlibrium feedback is enabled and is initialized -> switch on
        if to_switch and self.equilibrium_feedback_enabled and self.inf_lqr is not None: 
            self.eq_switched_on = True

        # If equilibrium feedback is on, but it's too far away -> switch off the eq to not toss the pendulum around
        left_balancing_region = np.abs(self.phi) >= np.deg2rad(60)
        if self.eq_switched_on and left_balancing_region:
            self.eq_switched_on = False

        # Calculating torques

        tau_eq = 0.0
        tau_fb = 0.0
        tau_int = 0.0
        
        if self.eq_switched_on: # If eq feedback is on -> use it
            self.phase = 2
            torque_alpha = self.inf_lqr.run(x_SP = np.zeros_like(x_a), x = x_a) + self.torque_int_alpha

            tau_eq = torque_alpha
            tau_int = self.torque_int_alpha

            if self.is_3D:
                torque_beta = self.inf_lqr.run(x_SP = np.zeros_like(x_b), x = x_b) + self.torque_int_beta
            else:
                torque_beta = 0.0

        elif not self.trial_running: # If eq feedback is off and trial is not running -> chill
            self.phase = 0
            torque_alpha = 0.0 
            torque_beta  = 0.0

        elif self.k >= 0: # If trial is running -> use trajectory tracking + ILC
            self.phase = 1
            # Note: sign follows the derivation of the time-varying LQR (no minus here).
            # If trajectory feedback is disabled, Ks are zero matrices.
            fb_torque = np.matmul( self.Ks[self.k], (x_a - self.x_optimal[:, self.k:self.k+1]) )
            fb_torque = float(fb_torque)

            tau_fb = fb_torque

            torque_alpha = self.u_optimal[0, self.k] + self.u_ilc[0, self.k] + fb_torque

            x_b_actuator = np.array([[self.beta], [self.betaD]])
            torque_beta = self.ctrl_b_trajectory.run(x_SP = np.zeros_like(x_b_actuator), x = x_b_actuator)

        else: # If trial is running, but it's k < 0 -> preapplying initial torque
            self.phase = 1
            torque_alpha = self.u_optimal[0, 0] + self.u_ilc[0, 0]
            x_b_actuator = np.array([[self.beta], [self.betaD]])
            torque_beta = self.ctrl_b_trajectory.run(x_SP = np.zeros_like(x_b_actuator), x = x_b_actuator)


        self.torque_vec[0] = torque_beta  # rot x (lateral)
        self.torque_vec[1] = torque_alpha # rot y (swing)
        

        magnetic_dipole_moment = params.m_tilde * self.normal_vector_actuator

        M = magnetic_interaction_from_dipole_moment(magnetic_dipole_moment)

        J = jacobian_torqueforce_to_torque(beta = self.beta, alpha = self.alpha, l_mag=params.l_m)
        J = J[:2, :]

        if self.b_southpole_up:
            self.torque_vec = -1.0 * self.torque_vec

        actuation_matrix = get_actuation_matrix_from_model(self.model, self.magnet_position)

        allocation_matrix = J @ M @ actuation_matrix
        computed_current = numba_pinv(allocation_matrix) @ self.torque_vec[:2]
        self.current_SP[self.active_drivers] = computed_current



        for ii in range(self.N_coils):
            if self.current_SP[ii] < self.CURRENT_MIN or self.current_SP[ii] > self.CURRENT_MAX:
                rospy.logwarn(f"Current at index {ii} exceeds borders: {self.current_SP[ii]}")
            self.current_SP[ii] = numba_clip(self.current_SP[ii], self.CURRENT_MIN, self.CURRENT_MAX)

        # special case because wiring was changed in the navion: 
        if self.calibration_file_name == "Navion_1_2_Calibration_old.yaml":
            self.current_SP[1] = - self.current_SP[1]  # Flip coil 2 current 

        self.publish_currents(self.current_SP, self.current_msg, self.current_SP_pub)

        self.publish_vars(
            tau_eq = tau_eq,
            tau_fb = tau_fb,
            tau_int = tau_int
        )
 
    

    def get_state(self):
        x_a = np.array([[self.alpha - self.phi_ss], [self.phi - self.phi_ss], [self.alphaD], [self.phiD]])
        x_b = np.array([[self.beta - self.theta_ss], [self.theta - self.theta_ss], [self.betaD], [self.thetaD]])
        return x_a, x_b


    def run(self):
        rospy.Subscriber(self.vicon_callback_topic_actuator, TransformStamped, self.callback_alpha_beta, queue_size=1)
        rospy.Subscriber(self.vicon_callback_topic_pendulum, TransformStamped, self.callback_phi_theta, queue_size=1)
                    
        rospy.spin()


if __name__ == '__main__':

    node = InvertedPendulum3DControlNode()
    node.run()
