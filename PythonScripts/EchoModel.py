from collections import namedtuple
from typing import Union, List
import numpy as np
from math import pi

EchoInfo = namedtuple('EchoInfo', ['fc', 'beta', 'alpha', 'r', 'tau', 'psi', 'phi'])

# Define min, max, and default parameter sets
# tau min is set to 0, max should be set according to waveform length in usage
echo_info_min = EchoInfo(4e6, 0, 5e12, -0.9, 0, -10e12, 0)
echo_info_max = EchoInfo(5e6, 0.025, 10e12, 0.9, 0, 10e12, 2*pi)
echo_info_default = EchoInfo(4.3e6, 0.025, 5e12, 0, 0, 0, 0)
units = {
    'fc': 'Hz',
    'beta': 'Volt',
    'alpha': '1/seconds^2',
    'tau': 'seconds',
    'r': 'unitless',
    'psi': 'radians/seconds^2',
    'phi': 'radians',
}

def echo_function(
    t: Union[float, int, np.ndarray, List[float]],
    fc: float,
    beta: float,
    alpha: float,
    r: float,
    tau: float,
    psi: float,
    phi: float,
) -> Union[float, np.ndarray, List[float]]:
    """Compute echo signal(s) for scalar, numpy array, or list inputs.

    Behavior by input type of t:
    - float/int: returns a Python float
    - numpy.ndarray (any shape): returns an ndarray with the same shape
    - list[float]: returns a list[float] of the same length

    The signal model is

        s(t) = beta * exp(-alpha * (1 - r * tanh(m * (t - tau))) * (t - tau)^2)
               * cos(2*pi*fc*(t - tau) + psi * (t - tau)^2 + phi)
    """
    TANH_M = 1e6  # Fixed parameter for tanh steepness
    # Track original type to return type-preserving output
    is_np_array = isinstance(t, np.ndarray)
    is_list = isinstance(t, list)

    t_arr = np.asarray(t)
    dt = t_arr - tau
    env = beta * np.exp(-alpha * (1 - r * np.tanh(TANH_M * dt)) * (dt ** 2))
    s = env * np.cos(2 * np.pi * fc * dt + psi * (dt ** 2) + phi)

    if is_np_array:
        return s
    if is_list:
        return s.tolist()

    # Scalar case: return a Python float
    return float(np.asarray(s).item())