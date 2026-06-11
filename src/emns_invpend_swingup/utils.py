import os
import numpy as np
import numpy.typing as np_t
import rospkg

def gains2csv(K: np_t.NDArray, package_name: str, relative_path_to_ctrl_prm: str) -> None:
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(package_name)
    filename = pkg_path + relative_path_to_ctrl_prm
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, K, delimiter=",", fmt="%.8f")

def nparr2csv(torques: np_t.NDArray, package_name: str, relative_path_to_ctrl_prm: str) -> None:
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(package_name)
    filename = pkg_path + relative_path_to_ctrl_prm
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, torques, delimiter=",", fmt="%.8f")


def csv2nparr(package_name: str, relative_path_to_ctrl_prm: str) -> np.ndarray:
    rospack = rospkg.RosPack()
    pkg_path=rospack.get_path(package_name)
    filename = pkg_path + relative_path_to_ctrl_prm
    return np.genfromtxt(filename, delimiter=',')