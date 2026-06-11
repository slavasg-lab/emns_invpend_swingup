import numpy as np
import numba
import numpy.typing as np_t


"""
Numba-accelerated numerical utilities.

Adapted from the oct_levitation package by Neelaksh Singh et al.:

    Singh, N., Zughaibi, J., von Arx, D., Nelson, B. J., & Muehlebach, M. (2025).
    Remote Magnetic Levitation Using Reduced Attitude Control and Parametric Field Models.
    arXiv:2512.15207. https://arxiv.org/abs/2512.15207
"""

@numba.njit(cache=True)
def numba_pinv(A: np_t.NDArray) -> np_t.NDArray:
    """
    This function is expected to provide a 2x speedup over the normal numpy version.
    """
    return np.linalg.pinv(A)

numba_pinv(np.eye(3)) # Force compilation on import.

@numba.njit(cache=True)
def numba_clip(a, a_min, a_max):
    a = np.maximum(a, a_min)
    a = np.minimum(a, a_max)
    return a

numba_clip(np.array([1, 2, 3]), 0, 2) # Force compilation on import.