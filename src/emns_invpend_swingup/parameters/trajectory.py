import numpy as np
import emns_invpend_swingup.parameters.general as parameters


max_torque = 0.12 # [Nm]
min_torque = -max_torque
T = 0.6 # [s]

N = np.round(parameters.frequency * T).astype(int)