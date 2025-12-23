from scipy.interpolate import CubicSpline
from typing import Tuple
import numpy as np

def create_interpolators(time_s: np.ndarray, positions: np.ndarray, velocities: np.ndarray) -> Tuple[CubicSpline, CubicSpline]:
    pos_interp = CubicSpline(time_s, positions, axis=0)
    vel_interp = CubicSpline(time_s, velocities, axis=0)
    return pos_interp, vel_interp