import rospkg
import rospy
import numpy as np
import os

import matplotlib.pyplot as plt

class DataLogger:
    def __init__(self, x_optimal, u_optimal, T_optimal, package_name, relative_path, hyperparameters):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path(package_name)
        base_path = os.path.join(pkg_path, relative_path)
        os.makedirs(base_path, exist_ok=True)

        # 2. Create timestamped trial folder using ROS time
        now = rospy.Time.now()
        timestamp = f"{now.secs}_{now.nsecs}"
        folder_name = os.path.join(base_path, f"trial_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # 3. Save hyperparameters to npz
        np.savez(os.path.join(folder_name, "hyperparameters.npz"), **hyperparameters)

        # 4. Create _optimal subfolder and save trajectory
        optimal_path = os.path.join(folder_name, "_optimal")
        os.makedirs(optimal_path, exist_ok=True)
        np.savez(os.path.join(optimal_path, "trajectory.npz"),
                X=x_optimal,
                U=u_optimal,
                T=T_optimal)

        # 5. Create _current subfolder
        current_path = os.path.join(folder_name, "_current")
        os.makedirs(current_path, exist_ok=True)

        self.n_x = x_optimal.shape[0]
        self.n_u = u_optimal.shape[0]

        self.folder_name = folder_name
    
    def log_current_trial(self, x_real, x_fitted, u, u_ilc, N_j):
        current_path = os.path.join(self.folder_name, "_current")

        np.savez(os.path.join(current_path, "trajectory.npz"),
            X_real = x_real,
            X_fitted = x_fitted,
            U = u,
            U_ilc = u_ilc,
            N_j = N_j
        )

        self.__plot_trial(current_path)


    def log_trial(self, j, x_real, x_fitted, u, u_ilc, N_j):
        trial_path = os.path.join(self.folder_name, f"{j}")
        os.makedirs(trial_path, exist_ok=True)

        np.savez(os.path.join(trial_path, "trajectory.npz"),
            X_real = x_real,
            X_fitted = x_fitted,
            U = u,
            U_ilc = u_ilc,
            N_j = N_j
        )

        self.__plot_trial(trial_path)

    def __plot_trial(self, iteration_folder):
        optimal_trajectory = np.load(os.path.join(self.folder_name, "_optimal", "trajectory.npz"))
        current_trajectory = np.load(os.path.join(iteration_folder, "trajectory.npz"))
        # Plot states

        _, axs = plt.subplots(self.n_x, 1, sharex=True, figsize=(20, 10))

        full_timesteps = np.arange(optimal_trajectory['X'].shape[1])
        timesteps = np.arange(current_trajectory['N_j'])

        for i_x in range(self.n_x):
            axs[i_x].plot(full_timesteps, optimal_trajectory['X'][i_x, :], color="red", label="Optimal")
            axs[i_x].plot(timesteps, current_trajectory['X_real'][i_x, :current_trajectory['N_j']], color="black", label="Tracked")
            axs[i_x].plot(timesteps, current_trajectory['X_fitted'][i_x, :current_trajectory['N_j']], color="blue", linestyle='dotted', label="Polynomial")
            axs[i_x].grid(True)
        axs[self.n_x - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=5)

        plt.tight_layout()
        plt.savefig(os.path.join(iteration_folder, "x.jpg"), dpi=300)
        plt.close()

        # Plot inputs
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

        full_timesteps = np.arange(optimal_trajectory['U'].shape[1])

        ax1.plot(full_timesteps, optimal_trajectory['U'][0, :], color="red", label="Optimal")
        ax2.plot(full_timesteps, np.zeros_like(optimal_trajectory['U'][0, :]), color="red", label="Optimal")
        
        ax1.grid(True)
        ax2.grid(True)
        
        ax1.plot(full_timesteps, current_trajectory['U'][0, :], color="black", label="Tracked")
        ax2.plot(full_timesteps, current_trajectory['U_ilc'][0, :], color="black", label="Tracked")

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=5)

        plt.tight_layout()
        plt.savefig(os.path.join(iteration_folder, "u.jpg"), dpi=300)
        plt.close()

