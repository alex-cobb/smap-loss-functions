"""Tests for fit_loss_functions"""

import numpy as np

import pytest

from smap_loss_functions.loss_function_db import set_up_loss_function_db
from smap_loss_functions import fit_loss_functions as flf


def test_optimize_loss_function_parameters():
    """Test optimize_loss_function_parameters with a simple objective."""
    Lmax = 1.0

    # A simple objective function where the true minimum satisfies constraints
    # Minimum is at LA=0.1, LB=0.2, LC=0.3
    def simple_objective(params):
        LA, LB, LC = params
        return (LA - 0.1) ** 2 + (LB - 0.2) ** 2 + (LC - 0.3) ** 2

    result = flf.optimize_loss_function_parameters(
        Lmax, simple_objective, n=10, iters=1
    )

    assert result.success is True
    # shgo might not find the *exact* minimum in limited iterations, but it should be
    # close
    assert np.allclose(result.x, [0.1, 0.2, 0.3], atol=1e-3)
    assert np.isclose(result.fun, 0.0, atol=1e-6)

    # Test with an objective that initially violates constraints but should still
    # converge
    # e.g., unconstrained min would be LA=0.3, LB=0.1, LC=0.2 (violates LA <= LB)
    # The optimizer should find the best constrained solution.
    def constrained_violating_objective(params):
        LA, LB, LC = params
        return (LA - 0.3) ** 2 + (LB - 0.1) ** 2 + (LC - 0.2) ** 2

    result_constrained = flf.optimize_loss_function_parameters(
        Lmax, constrained_violating_objective, n=10, iters=1
    )
    assert result_constrained.success is True
    # Verify that the constraints are approximately respected
    assert round(result_constrained.x[0], 12) <= round(result_constrained.x[1], 12)
    assert round(result_constrained.x[1], 12) <= round(result_constrained.x[2], 12)
    assert 0 <= result_constrained.x[0] <= Lmax
    assert 0 <= result_constrained.x[1] <= Lmax
    assert 0 <= result_constrained.x[2] <= Lmax


@pytest.mark.filterwarnings(
    'ignore:Values in x were outside bounds during a minimize step, '
    'clipping to bounds:RuntimeWarning'
)
def test_get_optimized_loss_function():
    """Test get_optimized_loss_function with simplified inputs."""
    # Test case 1: Constant soil moisture (should result in Wmin=Wmax, LA=LB=LC=0)
    smap_time_const = np.array([1, 2, 3]) * 86400.0  # Daily timestamps
    soil_moisture_const = np.full_like(smap_time_const, 0.3)

    def P_of_t_mm_d(t):
        """Constant precipitation of 10 mm/day"""
        return np.full_like(t, 10.0)

    L_const, rmse_const = flf.get_optimized_loss_function(
        smap_time_const, soil_moisture_const, P_of_t_mm_d
    )
    assert isinstance(L_const, flf.LossFunction)
    assert np.isclose(L_const.W[0], L_const.W[-1])  # Wmin should be close to Wmax
    assert (
        L_const.L[1] == 0 and L_const.L[2] == 0 and L_const.L[3] == 0
    )  # LA, LB, LC should be 0
    assert rmse_const is None  # For constant SMAP, RMSE is not computed meaningfully

    # Test case 2: Increasing soil moisture (requires actual optimization)
    smap_time_inc = np.array([1, 2, 3, 4]) * 86400.0
    soil_moisture_inc = np.array([0.2, 0.25, 0.3, 0.35])

    L_inc, rmse_inc = flf.get_optimized_loss_function(
        smap_time_inc, soil_moisture_inc, P_of_t_mm_d
    )

    assert isinstance(L_inc, flf.LossFunction)
    assert rmse_inc is not None
    assert rmse_inc >= 0  # RMSE should be non-negative

    # Check that the fitted parameters adhere to constraints and general sanity
    # Wmin = 0.2, Wmax = 0.35 + 0.1*(0.35-0.2) = 0.35 + 0.015 = 0.365
    # Lmax for this scenario = Wmax / 24 = 0.365 / 24
    expected_Lmax = (
        soil_moisture_inc.max()
        + 0.1 * (soil_moisture_inc.max() - soil_moisture_inc.min())
    ) / 24.0
    assert 0 <= L_inc.L[1] <= expected_Lmax  # LA
    assert 0 <= L_inc.L[2] <= expected_Lmax  # LB
    assert 0 <= L_inc.L[3] <= expected_Lmax  # LC
    # Test that inequality constraints are approximately satisfied
    assert round(L_inc.L[1], 12) <= round(L_inc.L[2], 12)  # LA <= LB
    assert round(L_inc.L[2], 12) <= round(L_inc.L[3], 12)  # LB <= LC


def test_fit_loss_functions(in_memory_smap_db, in_memory_output_db):
    """Test the main fit_loss_functions workflow."""
    set_up_loss_function_db(in_memory_output_db)  # Ensure output table exists

    # Run the main function
    result_code = flf.fit_loss_functions(in_memory_smap_db, in_memory_output_db)
    assert result_code == 0  # Should return 0 on success

    # Verify data in the output database
    out_cursor = in_memory_output_db.cursor()
    out_cursor.execute(
        'SELECT ease_col, ease_row, Wmin, Wmax, LA, LB, LC, rmse FROM loss_function'
    )
    fitted_data = out_cursor.fetchall()
    out_cursor.close()

    # From the fixture, we inserted two distinct cells: (10, 20) and (11, 21).
    # Cell (11, 21) had only 1 SMAP data point, so it should be skipped by the
    # 'len(soil_moisture) < 2' check.
    # Therefore, only cell (10, 20) should have a fitted entry.
    assert len(fitted_data) == 1
    col, row, Wmin, Wmax, LA, LB, LC, rmse = fitted_data[0]

    assert col == 10
    assert row == 20
    # Assert Wmin and Wmax are derived from the input SMAP data for cell (10, 20)
    assert np.isclose(Wmin, 0.25)  # Minimum soil_moisture for (10,20)
    assert np.isclose(
        Wmax, 0.30 + 0.1 * (0.30 - 0.25)
    )  # Wmax formula: max + 0.1 * (max - min)
    assert LA >= 0 and LB >= 0 and LC >= 0
    assert LA <= LB <= LC  # Verify optimization constraints are respected
    assert rmse >= 0  # RMSE should be non-negative and finite
