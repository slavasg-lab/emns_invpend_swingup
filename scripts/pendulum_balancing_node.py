#!/usr/bin/env python3
import rospy
import numpy as np
import time as timing

from geometry_msgs.msg import TransformStamped
from control_utils.msg import ScalarStamped, VectorStamped
from control_utils.feedback_ctrl.state_feedback import StateFeedbackController
from control_utils.feedback_ctrl.integrator import ScalarIntegralController

from control_utils.general.utilities import init_MPEM_model, get_actuation_matrix_from_model

from control_utils.general.utilities import init_system

from scipy.signal import iirfilter 
from control_utils.general.utilities import LiveLFilter

from std_srvs.srv import Empty, EmptyResponse

from emns_invpend_swingup.plant.calc_gains import calc_inf_lqr_gains_pendulum
import emns_invpend_swingup.parameters.general as parameters
from emns_invpend_swingup.utils import gains2csv, nparr2csv, csv2nparr
from emns_invpend_swingup.geometry_jit import get_normal_vector_from_quaternion, get_normal_angles_from_quaternion, get_normal_angles_from_normal_vector, magnetic_interaction_from_dipole_moment, jacobian_torqueforce_to_torque
from emns_invpend_swingup.numerical import numba_pinv, numba_clip




class InvertedPendulum3DControlNode:
    def __init__(self):
        rospy.init_node('oct_3d_pend_node', anonymous=True)

        # Compute LQR gains and save to disk so controllers can reload them
        K = calc_inf_lqr_gains_pendulum(Q = np.diag([1, 1, 0.1, 0.1]), R = np.diag([1e4]))
        gains2csv(K, package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/balancing/inf_lqr_gains.csv")

        rospy.loginfo("Pendulum gains obtained. Initializing the control node...")

        # Initializing parameters 
        self.l_mag =parameters.l_m
        self.m_tilde = parameters.m_tilde
        self.distance_mag_vicon = parameters.l_act_marker2magnet

        # integrator can be triggered using (e.g. obj 1): 
        """rosservice call /pendulum_balancing_node/trigger_integrator"""
        # for two nodes, use: 
        """rosservice call /pendulum_balancing_node/trigger_integrator & rosservice call /pendulum_balancing_node/trigger_integrator"""
        self.integrator_duration = 60.0  # [s] Duration to keep integrator active
        self.integrator_start_time = None
        self.integrator_service = rospy.Service('~trigger_integrator', Empty, self.trigger_integrator_callback)
        self.b_integral_control_enabled = False


        # angle offset correction can be triggered using:
        """rosservice call /pendulum_balancing_node/trigger_angle_correction"""
        # for two nodes, use:
        """rosservice call /pendulum_balancing_node/trigger_angle_correction & rosservice call /pendulum_balancing_node/trigger_angle_correction"""
        self.angle_correction_duration = 60.0  # [s] Duration to keep integrator active
        self.angle_correction_start_time = None
        self.angle_correction_service = rospy.Service('~trigger_angle_correction', Empty, self.trigger_angle_correction_callback)
        self.b_angle_correction_control_enabled = False  


        # no matter how many nodes, just use:
        """rosservice call /pendulum_balancing_node/trigger_angle_correction"""
        self.disturbance_start_time = None
        self.disturbance_service = rospy.Service('~trigger_disturbance', Empty, self.trigger_disturbance_callback)
        self.disturbance_duration = 0.4 # [s]
        self.disturbance_pause = 10 # [s]
        self.disturbance_current = 0.4 # [A]
        self.disturbance_coil_ix = 1 # [-]
        self.disturbance_count = 10 # [-]
        self.disturbance_total_cycle_duration = self.disturbance_duration + self.disturbance_pause
        self.disturbance_total_sequence_duration = self.disturbance_total_cycle_duration * self.disturbance_count

        self.is_disturbing = False


        self.read_ROS_parameters()
        self.init_publishers()
        self.init_payload_variables()
        self.init_controllers()
        self.init_angular_velocity_filters()
        self.init_offset_calibration()

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
            self.CURRENT_MAX = 10.0 # [A]
            self.CURRENT_MIN = -self.CURRENT_MAX 
      
        self.current_SP = np.zeros(self.N_coils)


        self.current_msg, self.current_SP_pub, self.publish_currents = init_system(self.system_type, self.b_hardware_connected, coil_nrs = self.active_drivers)

        rospy.loginfo("Node initialized!")




    def read_ROS_parameters(self):
        self.vicon_callback_topic_actuator = rospy.get_param('~vicon_callback_topic_actuator', "/vicon/arm_marker/Root")
        self.vicon_callback_topic_pendulum = rospy.get_param('~vicon_callback_topic_pendulum', "/vicon/pendulum_marker/Root")
        
        self.system_type                   = rospy.get_param('~system_type', "JECB")

        self.angle_actuator_topic_rot_y    = rospy.get_param('~angle_actuator_topic_rot_y', "/alpha")
        self.angle_actuator_topic_rot_x    = rospy.get_param('~angle_actuator_topic_rot_x', "/beta")

        self.torque_topic_rot_y            = rospy.get_param('~torque_topic_rot_y', "/torque_alpha")
        self.torque_topic_rot_x            = rospy.get_param('~torque_topic_rot_x', "/torque_beta")

        self.calibration_file_name         = rospy.get_param('~calibration_file_name', "octomag_5point.yaml")

        self.b_use_prerecoded_integrator_torques = rospy.get_param('~b_use_prerecoded_integrator_torques', False)
        self.b_use_prerecoded_correction_angles = rospy.get_param('~b_use_prerecoded_correction_angles', False)

        self.b_hardware_connected          = rospy.get_param('~b_hardware_connected', False)
        self.b_southpole_up                = rospy.get_param('~b_southpole_up', False)
        self.b_update_actuation_matrix     = rospy.get_param('~b_update_actuation_matrix', False)
        self.b_flip_actuation_matrix       = rospy.get_param('~b_flip_actuation_matrix', False)

        self.process_time_actuator_topic = rospy.get_param('~process_time_actuator_topic', "/process_time_act")
        self.process_time_pendulum_topic = rospy.get_param('~process_time_pendulum_topic', "/process_time_pend")

        self.verbose = rospy.get_param('~verbose', False)

        self.Ts = parameters.Ts

        self.condition_number_topic = rospy.get_param('~condition_number_topic', "/condition_number")

        if self.system_type == "JECB" and self.calibration_file_name == "octomag_5point.yaml":
            self.b_flip_actuation_matrix = True 

            
    def init_controllers(self):
        self.ctrl_a = StateFeedbackController(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/balancing/inf_lqr_gains.csv") # _a = alpha phi
        self.ctrl_b = StateFeedbackController(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/balancing/inf_lqr_gains.csv") # _b = beta theta


        # Reading integrator torques
        if self.b_use_prerecoded_integrator_torques:
            self.torque_int_beta, self.torque_int_alpha = csv2nparr(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/int_torques.csv")
        else:
            self.torque_int_beta, self.torque_int_alpha = 0.0, 0.0

        self.ctrl_int_a = ScalarIntegralController(-0.005, self.Ts, initial_value = self.torque_int_alpha)
        self.ctrl_int_b = ScalarIntegralController(-0.005, self.Ts, initial_value = self.torque_int_beta)
        
    def init_offset_calibration(self):
        b, a = iirfilter(2, Wn=0.05, fs=1.0/self.Ts, btype="low", ftype="butter") 
        self.livefilt_phi_ss = LiveLFilter(b, a)
        self.livefilt_theta_ss = LiveLFilter(b, a)

        if self.b_use_prerecoded_correction_angles:
            self.phi_ss, self.theta_ss = csv2nparr(package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/correction_angles.csv")
        else:
            self.phi_ss, self.theta_ss = 0.0, 0.0



        
    def init_payload_variables(self):
        
        # state variables: 
        self.alpha      = 0.0
        self.phi        = 0.0
        self.alphaD     = 0.0
        self.phiD       = 0.0
        self.prev_alpha_hat = 0.0
        self.prev_phi_hat   = 0.0

        self.beta       = 0.0
        self.theta      = 0.0
        self.betaD      = 0.0
        self.thetaD     = 0.0
        self.prev_beta_hat  = 0.0
        self.prev_theta_hat = 0.0

        self.alpha_hat = 0.0
        self.phi_hat = 0.0

        self.beta_hat = 0.0
        self.theta_hat = 0.0

        # processing times: 
        self.callback_time_act = 0.0
        self.callback_time_pend = 0.0

        # torque vars: 
        self.normal_vector_actuator = np.array([0.0, 0.0, 1.0])
        self.torque_vec = np.zeros(3)
        self.vicon_position = np.zeros(3)
        self.magnet_position = np.zeros(3)

        self.prev_callback_time_pend = None
        self.prev_callback_time_act = None
    
    def trigger_disturbance_callback(self, req):
        self.disturbance_start_time = timing.time()
        return EmptyResponse()

    def update_disturbance_enabled(self):
        if self.disturbance_start_time is not None:
            elapsed_time = timing.time() - self.disturbance_start_time

            if elapsed_time >= self.disturbance_total_sequence_duration:
                self.is_disturbing = False
                self.disturbance_start_time = None
                return
            
            time_in_current_cycle = elapsed_time % self.disturbance_total_cycle_duration

            if time_in_current_cycle < self.disturbance_duration:
                self.is_disturbing = True
            else:
                self.is_disturbing = False
            

    

    def trigger_angle_correction_callback(self, req):
        self.angle_correction_start_time = timing.time()
        self.b_angle_correction_control_enabled = True
        return EmptyResponse()
    

    def update_angle_correction_enabled(self):
        if self.b_angle_correction_control_enabled and self.angle_correction_start_time is not None:
            elapsed_time = timing.time() - self.angle_correction_start_time
            if elapsed_time >= self.angle_correction_duration:
                
                self.b_angle_correction_control_enabled = False
                self.angle_correction_start_time = None

                correction_angles = np.array([self.phi_ss, self.theta_ss])
                nparr2csv(correction_angles, package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/correction_angles.csv")
                rospy.loginfo("SAVED CORRECTION ANGLES")


    
    def trigger_integrator_callback(self, req):
        self.integrator_start_time = timing.time()
        self.b_integral_control_enabled = True
        return EmptyResponse()

    def update_integrator_enabled(self):
        if self.b_integral_control_enabled and self.integrator_start_time is not None:
            elapsed_time = timing.time() - self.integrator_start_time
            if elapsed_time >= self.integrator_duration:
                self.b_integral_control_enabled = False
                self.integrator_start_time = None

                # Save integrator values
                integrator_torques = np.array([self.torque_int_beta, self.torque_int_alpha])
                nparr2csv(integrator_torques, package_name="emns_invpend_swingup", relative_path_to_ctrl_prm="/data/pendulum/int_torques.csv")
                rospy.loginfo("SAVED INTEGRATOR TORQUES")
                

    def init_publishers(self): 
        self.pub_a  = rospy.Publisher(self.angle_actuator_topic_rot_y, ScalarStamped, queue_size=10) 
        self.pub_aD = rospy.Publisher(self.angle_actuator_topic_rot_y + "D", ScalarStamped, queue_size=10)
        self.msg_alpha  = ScalarStamped()
        self.msg_alphaD = ScalarStamped()

        self.pub_phi  = rospy.Publisher("/phi", ScalarStamped, queue_size=10) 
        self.pub_phiD = rospy.Publisher("/phi" + "D", ScalarStamped, queue_size=10)
        self.msg_phi  = ScalarStamped()
        self.msg_phiD = ScalarStamped()

        self.pub_b  = rospy.Publisher(self.angle_actuator_topic_rot_x, ScalarStamped, queue_size=10)
        self.pub_bD = rospy.Publisher(self.angle_actuator_topic_rot_x + "D", ScalarStamped, queue_size=10)
        self.msg_beta  = ScalarStamped()
        self.msg_betaD = ScalarStamped()

        self.pub_theta  = rospy.Publisher("/theta", ScalarStamped, queue_size=10) 
        self.pub_thetaD = rospy.Publisher("/theta" + "D", ScalarStamped, queue_size=10)
        self.msg_theta  = ScalarStamped()
        self.msg_thetaD = ScalarStamped()

        self.pub_a_hat = rospy.Publisher(self.angle_actuator_topic_rot_y + '_hat', ScalarStamped, queue_size=10)
        self.pub_b_hat = rospy.Publisher(self.angle_actuator_topic_rot_x + '_hat', ScalarStamped, queue_size=10)
        self.msg_alpha_hat = ScalarStamped()
        self.msg_beta_hat  = ScalarStamped()

        self.pub_phi_hat = rospy.Publisher("/phi" + '_hat', ScalarStamped, queue_size=10)
        self.pub_theta_hat = rospy.Publisher("/theta" + '_hat', ScalarStamped, queue_size=10)
        self.msg_phi_hat = ScalarStamped()
        self.msg_theta_hat  = ScalarStamped()

        self.pub_phi_ss = rospy.Publisher("/phi" + '_ss', ScalarStamped, queue_size=10)
        self.pub_theta_ss = rospy.Publisher("/theta" + '_ss', ScalarStamped, queue_size=10)
        self.msg_phi_ss = ScalarStamped()
        self.msg_theta_ss  = ScalarStamped()
        
        self.pub_torque_a  = rospy.Publisher(self.torque_topic_rot_y, ScalarStamped, queue_size=10)
        self.pub_torque_b      = rospy.Publisher(self.torque_topic_rot_x, ScalarStamped, queue_size=10)
        self.msg_torque_a  = ScalarStamped()
        self.msg_torque_b   = ScalarStamped()

        self.pub_torque_integral_a = rospy.Publisher("/torque_integral_a", ScalarStamped, queue_size=10)
        self.pub_torque_integral_b = rospy.Publisher("/torque_integral_b", ScalarStamped, queue_size=10)
        self.msg_torque_integral_a = ScalarStamped()
        self.msg_torque_integral_b = ScalarStamped()

        self.pub_act_process_time  = rospy.Publisher("/act_process_time", ScalarStamped, queue_size=10) 
        self.pub_pend_process_time  = rospy.Publisher("/pend_process_time", ScalarStamped, queue_size=10) 
        self.msg_act_process_time  = ScalarStamped()
        self.msg_pend_process_time  = ScalarStamped()

        self.pub_act_betw_callback_time  = rospy.Publisher("/act_betw_callback_time", ScalarStamped, queue_size=10) 
        self.pub_pend_betw_callback_time  = rospy.Publisher("/pend_betw_callback_time", ScalarStamped, queue_size=10) 
        self.msg_act_betw_callback_time  = ScalarStamped()
        self.msg_pend_betw_callback_time  = ScalarStamped()

        self.condition_number_pub = rospy.Publisher(self.condition_number_topic, ScalarStamped, queue_size = 10)
        self.msg_condition_number = ScalarStamped()

        self.magnet_pos_pub = rospy.Publisher("/magnet_position",VectorStamped, queue_size = 10)
        self.msg_magnet_pos = VectorStamped()

            
    def init_angular_velocity_filters(self):
        # Identity filter (pass-through) — finite-difference velocity is already
        # smooth enough at 100 Hz. Swap for an IIR Butterworth if needed:
        #   b, a = iirfilter(2, Wn=30.0, fs=1.0/self.Ts, btype="low", ftype="butter")
        b = [1.0]
        a = [1.0]

        self.livefilt_alpha_hat = LiveLFilter(b, a)
        self.livefilt_phi_hat = LiveLFilter(b, a)

        self.livefilt_beta_hat = LiveLFilter(b, a)
        self.livefilt_theta_hat = LiveLFilter(b, a)
    


    
    
    def callback_alpha_beta(self, msg):
        self.callback_time_act = timing.time()

        # read: 
        quat_rot =  msg.transform.rotation
        self.normal_vector_actuator = - get_normal_vector_from_quaternion(np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w]))

        self.beta, self.alpha = get_normal_angles_from_normal_vector(self.normal_vector_actuator)


        self.vicon_position = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        self.magnet_position = self.vicon_position - self.distance_mag_vicon * self.normal_vector_actuator
        
        # use filtered angles to compute angular velocities:
        self.prev_alpha_hat = self.alpha_hat
        self.alpha_hat = self.livefilt_alpha_hat(self.alpha)
        self.alphaD = (self.alpha_hat - self.prev_alpha_hat) / self.Ts

        self.prev_beta_hat = self.beta_hat
        self.beta_hat = self.livefilt_beta_hat(self.beta)
        self.betaD = (self.beta_hat - self.prev_beta_hat) / self.Ts
        
        self.publish_actuator_states()


    def publish_actuator_states(self):
        if self.verbose:
            # publish alpha: 
            self.msg_alpha.header.stamp = rospy.Time.now()
            self.msg_alpha.scalar = self.alpha 
            self.pub_a.publish(self.msg_alpha)

            # publish alphaD:
            self.msg_alphaD.header.stamp = rospy.Time.now()
            self.msg_alphaD.scalar = self.alphaD
            self.pub_aD.publish(self.msg_alphaD)

            # publish beta:
            self.msg_beta.header.stamp = rospy.Time.now()
            self.msg_beta.scalar = self.beta
            self.pub_b.publish(self.msg_beta)

            # publish betaD:
            self.msg_betaD.header.stamp = rospy.Time.now()
            self.msg_betaD.scalar = self.betaD
            self.pub_bD.publish(self.msg_betaD)

            # publish filtered angles (used for velocity estimation):
            self.msg_alpha_hat.header.stamp = rospy.Time.now()
            self.msg_alpha_hat.scalar = self.alpha_hat
            self.pub_a_hat.publish(self.msg_alpha_hat)

            self.msg_beta_hat.header.stamp = rospy.Time.now()
            self.msg_beta_hat.scalar = self.beta_hat
            self.pub_b_hat.publish(self.msg_beta_hat)

            # publish magnet position: 
            self.msg_magnet_pos.header.stamp = rospy.Time.now()
            self.msg_magnet_pos.vector = self.magnet_position
            self.magnet_pos_pub.publish(self.msg_magnet_pos)

        # publish process times:
        self.msg_act_process_time.header.stamp = rospy.Time.now()
        self.msg_act_process_time.scalar = timing.time() - self.callback_time_act
        self.pub_act_process_time.publish(self.msg_act_process_time)

        if self.prev_callback_time_act is not None:
            self.msg_act_betw_callback_time.header.stamp = rospy.Time.now()
            self.msg_act_betw_callback_time.scalar = timing.time() - self.prev_callback_time_act
            self.pub_act_betw_callback_time.publish(self.msg_act_betw_callback_time)
        
        self.prev_callback_time_act = timing.time()
    

    def publish_pendulum_states(self):
        if self.verbose:
            
            # publish phi: 
            self.msg_phi.header.stamp = rospy.Time.now()
            self.msg_phi.scalar = self.phi 
            self.pub_phi.publish(self.msg_phi)

            # publish phiD:
            self.msg_phiD.header.stamp = rospy.Time.now()
            self.msg_phiD.scalar = self.phiD
            self.pub_phiD.publish(self.msg_phiD)

            # publish phi_ss:
            self.msg_phi_ss.header.stamp = rospy.Time.now()
            self.msg_phi_ss.scalar = self.phi_ss
            self.pub_phi_ss.publish(self.msg_phi_ss)

            # publish theta:
            self.msg_theta.header.stamp = rospy.Time.now()
            self.msg_theta.scalar = self.theta
            self.pub_theta.publish(self.msg_theta)

            # publish thetaD:
            self.msg_thetaD.header.stamp = rospy.Time.now()
            self.msg_thetaD.scalar = self.thetaD
            self.pub_thetaD.publish(self.msg_thetaD)

            # publish theta_ss:
            self.msg_theta_ss.header.stamp = rospy.Time.now()
            self.msg_theta_ss.scalar = self.theta_ss
            self.pub_theta_ss.publish(self.msg_theta_ss)

            # publish filtered angles (used for velocity estimation):
            self.msg_phi_hat.header.stamp = rospy.Time.now()
            self.msg_phi_hat.scalar = self.phi_hat
            self.pub_phi_hat.publish(self.msg_phi_hat)

            self.msg_theta_hat.header.stamp = rospy.Time.now()
            self.msg_theta_hat.scalar = self.theta_hat
            self.pub_theta_hat.publish(self.msg_theta_hat)

        # publish process times:
        self.msg_pend_process_time.header.stamp = rospy.Time.now()
        self.msg_pend_process_time.scalar = timing.time() - self.callback_time_pend
        self.pub_pend_process_time.publish(self.msg_pend_process_time)

        if self.prev_callback_time_pend is not None:
            self.msg_pend_betw_callback_time.header.stamp = rospy.Time.now()
            self.msg_pend_betw_callback_time.scalar = timing.time() - self.prev_callback_time_pend
            self.pub_pend_betw_callback_time.publish(self.msg_pend_betw_callback_time)
        
        self.prev_callback_time_pend = timing.time()

        
    def get_state(self):
        x_a = np.array([[self.alpha - self.phi_ss], [self.phi - self.phi_ss], [self.alphaD], [self.phiD]])
        x_b = np.array([[self.beta - self.theta_ss], [self.theta - self.theta_ss], [self.betaD], [self.thetaD]])
        return x_a, x_b
        

    def state_feedback_step(self):       
        self.update_integrator_enabled()
        self.update_angle_correction_enabled()
        self.update_disturbance_enabled()

        x_a, x_b = self.get_state()

        # state feedback control:
        torque_alpha = self.ctrl_a.run(x_SP = np.zeros_like(x_a), x = x_a) 
        torque_beta  = self.ctrl_b.run(x_SP = np.zeros_like(x_b), x = x_b) 

        if self.b_integral_control_enabled: 
            self.torque_int_alpha = self.ctrl_int_a.run(y_SP = 0, y = self.alpha - self.phi_ss)
            # self.torque_int_beta  = self.ctrl_int_b.run(y_SP = 0, y = self.beta - self.theta_ss)

        magnetic_dipole_moment = self.m_tilde * self.normal_vector_actuator
        M = magnetic_interaction_from_dipole_moment(magnetic_dipole_moment)

        J = jacobian_torqueforce_to_torque(beta = self.beta, alpha = self.alpha, l_mag=self.l_mag)
        J = J[:2, :]

        
        self.torque_vec[0] = 0.0  # lateral (beta) torque disabled — 2D balancing only
        self.torque_vec[1] = torque_alpha + self.torque_int_alpha  # rot y

        if self.b_southpole_up:
            self.torque_vec = -1.0 * self.torque_vec

        self.actuation_matrix = get_actuation_matrix_from_model(self.model, self.magnet_position) # init actuation matrix

        if self.b_flip_actuation_matrix:
            self.actuation_matrix = -self.actuation_matrix
        allocation_matrix = J @ M @ self.actuation_matrix
        computed_current = numba_pinv(allocation_matrix) @ self.torque_vec[:2]
        self.current_SP[self.active_drivers] = computed_current
        
        if self.is_disturbing: 
            self.current_SP[self.disturbance_coil_ix] = self.disturbance_current

        for ii in range(self.N_coils):
            self.current_SP[ii] = numba_clip(self.current_SP[ii], self.CURRENT_MIN, self.CURRENT_MAX) # limit currents   
    

        # compute condition number:
        if self.verbose:
            condition_number = np.linalg.cond(allocation_matrix)
        else:
            condition_number = 0.0

        # special case because wiring was changed in the navion: 
        if self.calibration_file_name == "Navion_1_2_Calibration_old.yaml":
            self.current_SP[1] = - self.current_SP[1]  # Flip coil 2 current 

        self.publish_currents(self.current_SP, self.current_msg, self.current_SP_pub)
 
        self.publish_control_variables(torque_alpha, torque_beta, condition_number)


    def callback_phi_theta(self, msg):
        self.callback_time_pend = timing.time()

        # read: ScalarStampedPend
        quat_rot =  msg.transform.rotation
        self.theta, self.phi = get_normal_angles_from_quaternion(np.array([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w]), flip=True)
        
        self.prev_phi_hat = self.phi_hat
        self.phi_hat = self.livefilt_phi_hat(self.phi)
        self.phiD= (self.phi_hat - self.prev_phi_hat) / self.Ts

        self.prev_theta_hat = self.theta_hat
        self.theta_hat = self.livefilt_theta_hat(self.theta) 
        self.thetaD = (self.theta_hat - self.prev_theta_hat) / self.Ts

        if self.b_angle_correction_control_enabled:
            self.phi_ss = self.livefilt_phi_ss(self.phi) 
            self.theta_ss = self.livefilt_theta_ss(self.theta)

        self.state_feedback_step()
        self.publish_pendulum_states()


    def publish_control_variables(self, torque_alpha, torque_beta, condition_number):
        # publish control inputs: 

        if self.verbose:
            self.msg_torque_a.header.stamp = rospy.Time.now()
            self.msg_torque_a.scalar = torque_alpha
            self.pub_torque_a.publish(self.msg_torque_a)

            self.msg_torque_b.header.stamp = rospy.Time.now()
            self.msg_torque_b.scalar = torque_beta
            self.pub_torque_b.publish(self.msg_torque_b)

            self.msg_torque_integral_a.header.stamp = rospy.Time.now()
            self.msg_torque_integral_a.scalar = self.torque_int_alpha
            self.pub_torque_integral_a.publish(self.msg_torque_integral_a)

            self.msg_torque_integral_b.header.stamp = rospy.Time.now()
            self.msg_torque_integral_b.scalar = self.torque_int_beta
            self.pub_torque_integral_b.publish(self.msg_torque_integral_b)

            self.msg_condition_number.header.stamp = rospy.Time.now()
            self.msg_condition_number.scalar = condition_number
            self.condition_number_pub.publish(self.msg_condition_number)

    

    def run(self):
        rospy.Subscriber(self.vicon_callback_topic_actuator, TransformStamped, self.callback_alpha_beta, queue_size=1)
        rospy.Subscriber(self.vicon_callback_topic_pendulum, TransformStamped, self.callback_phi_theta, queue_size=1)
                    
        rospy.spin()


if __name__ == '__main__':

    node = InvertedPendulum3DControlNode()
    node.run()
