from collections import namedtuple
from typing import Union, List
import numpy as np
from math import pi

EchoInfo = namedtuple('EchoInfo', ['fc', 'tau', 'phi','alpha', 'beta', 'r', 'tanh_m'])

echo_info_min = EchoInfo(2.3e6, 0, 0, 2e12, 0, -1, 16)
echo_info_max = EchoInfo(5e6, 30e-6, 2*pi, 2e12, 10e-6, 1, 16)
echo_info_default = EchoInfo(2.5e6, 0, 0, 2e12, 1, 0, 16)

def echo_function(
    t: Union[float, int, np.ndarray, List[float]],
    tau: float,
    beta: float,
    fc: float = echo_info_default.fc,
    phi: float = echo_info_default.phi,
    alpha: float = echo_info_default.alpha,
    r: float = echo_info_default.r,
    tanh_m: float = echo_info_default.tanh_m,
) -> Union[float, np.ndarray, List[float]]:
    """Compute echo signal for scalar, numpy array, or list inputs.

    Behavior by input type of t:
    - float/int: returns a Python float
    - numpy.ndarray (any shape): returns an ndarray with the same shape
    - list[float]: returns a list[float] of the same length

    The signal model is:
        s(t) = beta * exp(-alpha * (1 - r * tanh(m * (t - tau))) * (t - tau)^2)
               * cos(2*pi*fc*(t - tau) + phi)
    """

    # Track original type to return type-preserving output
    is_np_array = isinstance(t, np.ndarray)
    is_list = isinstance(t, list)

    # Vectorize computation using numpy for all input kinds
    t_arr = np.asarray(t)
    dt = t_arr - tau
    env = beta * np.exp(-alpha * (1 - r * np.tanh(tanh_m * dt)) * (dt ** 2))
    s = env * np.cos(2 * np.pi * fc * dt + phi)

    if is_np_array:
        # Preserve ndarray shape and type
        return s
    if is_list:
        # Return Python list with same length
        return s.tolist()

    # Scalar case: return a Python float
    return float(np.asarray(s).item())