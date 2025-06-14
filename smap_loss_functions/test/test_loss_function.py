"""Tests for loss functions"""

import numpy as np

import pytest

from smap_loss_functions.loss_function import LossFunction


def test_loss_function_init():
    """
    Test LossFunction initialization with valid parameters.
    """
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    lf = LossFunction(Wmax, Wmin, LA, LB, LC)

    assert np.isclose(lf.W[0], Wmin)
    assert np.isclose(lf.W[-1], Wmax)
    assert np.isclose(lf.L[0], 0.0)
    assert np.isclose(lf.L[-1], Wmax / 24.0)

    # Check intermediate W values
    expected_WA = Wmin + 0.25 * (Wmax - Wmin)
    expected_WB = Wmin + 0.5 * (Wmax - Wmin)
    expected_WC = Wmin + 0.75 * (Wmax - Wmin)
    assert np.isclose(lf.W[1], expected_WA)
    assert np.isclose(lf.W[2], expected_WB)
    assert np.isclose(lf.W[3], expected_WC)

    # Check intermediate L values
    assert np.isclose(lf.L[1], LA)
    assert np.isclose(lf.L[2], LB)
    assert np.isclose(lf.L[3], LC)


def test_loss_function_call():
    """
    Test LossFunction's __call__ method for interpolation and extrapolation.
    """
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    lf = LossFunction(Wmax, Wmin, LA, LB, LC)

    # Test at Wmin and Wmax
    assert np.isclose(lf(Wmin), 0.0)
    assert np.isclose(lf(Wmax), Wmax / 24.0)

    # Test extrapolation below Wmin
    assert np.isclose(lf(Wmin - 0.05), 0.0)

    # Test extrapolation above Wmax
    assert np.isclose(lf(Wmax + 0.05), Wmax / 24.0)

    # Test interpolation at an intermediate point (e.g., halfway between Wmin and WA)
    mid_point = (lf.W[0] + lf.W[1]) / 2
    expected_l = np.interp(
        mid_point, lf.W, lf.L
    )  # Use numpy's interp for expected value
    assert np.isclose(lf(mid_point), expected_l)


def test_simulate_soil_moisture():
    """Test simulate_soil_moisture for a basic scenario."""
    Winit = 0.25
    Wmax = 0.5
    # Simple LossFunction (LA, LB, LC chosen for a plausible test)
    loss_func = LossFunction(Wmax=Wmax, Wmin=0.1, LA=0.01, LB=0.015, LC=0.02)

    def P_of_t_mm_d(t):
        """Constant precipitation of 24 mm/day (1 mm/hr)"""
        return np.full_like(t, 24.0)

    # Simulate for 3 hours (total 4 hourly points: t0, t1, t2, t3)
    ts_start = 1672555200.0  # Jan 1 2023 12:00:00 (epoch seconds)
    ts_thru = ts_start + 3 * 3600  # 3 hours later

    times, W = loss_func.simulate_soil_moisture(
        Winit, Wmax, P_of_t_mm_d, ts_start, ts_thru
    )

    assert len(times) == 3
    assert np.isclose(times[0], ts_start)  # Should be exactly the start hour
    assert np.isclose(times[-1], ts_thru)  # Should be exactly the thru hour

    assert np.isclose(W[0], Winit)

    # Manual calculation for the first step:
    # W[i + 1] = W[i] - L(W[i]) * Δt + Wadd
    # Δt = 1 hour, D = 50 mm, nd = 1.0 (default max_infiltration_h)
    # P_hourly = P_of_t_mm_d / 24 = 24.0 / 24 = 1.0 mm/hr
    # I = min(P_hourly, (Wmax - W[i]) * D / nd)
    # Wadd = I * Δt / D

    # For W[0] = 0.25:
    L_W0 = loss_func(
        Winit
    )  # Loss function value at Winit (interpolation from init test)
    # For Winit=0.25, it's between WA=0.2 and WB=0.3
    # L_values are [0.0, 0.01, 0.015, 0.02, 0.5/24]
    # L(0.25) is halfway between L(0.2)=0.01 and L(0.3)=0.015, so 0.0125
    assert np.isclose(L_W0, 0.0125)

    I_0 = min(1.0, (Wmax - Winit) * 50 / 1.0)  # min(1.0, (0.5 - 0.25) * 50 / 1.0)
    # min(1.0, 0.25 * 50) = min(1.0, 12.5) = 1.0
    Wadd_0 = I_0 * 1 / 50  # 1.0 / 50 = 0.02

    expected_W1 = Winit - L_W0 * 1 + Wadd_0  # 0.25 - 0.0125 + 0.02 = 0.2575
    assert np.isclose(W[1], expected_W1)
    assert not np.isnan(W).any()

    # Test zero-length simulation
    with pytest.raises(ValueError, match='Zero-length simulation'):
        loss_func.simulate_soil_moisture(
            Winit, Wmax, P_of_t_mm_d, ts_start, ts_start + 100
        )  # Less than one full hour
