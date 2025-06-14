"""Shared utility code for smap-loss-functions"""

import numpy as np


def zero_precipitation(t):
    """Precipitation function that always returns 0"""
    return np.zeros_like(t, dtype='float64') if isinstance(t, np.ndarray) else 0.0
