"""Fit SMAP loss functions sensu Koster et al (2017)

Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A data-driven approach for
  daily real-time estimates and forecasts of near-surface soil moisture. Journal of
  Hydrometeorology, 18(3), 837–843. https://doi.org/10.1175/jhm-d-16-0285.1

"""

import logging

import numpy as np

from scipy.optimize import shgo

from .loss_function import LossFunction
from .smap_db import get_soil_moisture, get_imerg_interpolant


# Show info messages from this module but not from Scipy, which uses the root logger
LOG = logging.getLogger('fit_loss_functions')
LOG.setLevel(logging.INFO)


def fit_loss_functions(in_connection, out_connection):
    """Fit SMAP loss functions"""
    in_cursor = in_connection.cursor()
    in_cursor.execute(
        'SELECT DISTINCT ease_col, ease_row FROM smap_data ORDER BY ease_col, ease_row'
    )
    cells = in_cursor.fetchall()
    out_cursor = out_connection.cursor()

    for col, row in cells:
        smap_time, soil_moisture = get_soil_moisture(in_cursor, col, row)
        if len(soil_moisture) < 2:
            LOG.info(
                '%s SMAP values in col %s, row %s: nothing to fit',
                len(soil_moisture),
                col,
                row,
            )
            continue
        interpolate_imerg = get_imerg_interpolant(
            in_cursor,
            col,
            row,
            start_time=smap_time.min() - 172800,
            thru_time=smap_time.max() + 172800,
        )

        LOG.info('Fitting col %s, row %s', col, row)
        L, rmse = get_optimized_loss_function(
            smap_time, soil_moisture, interpolate_imerg
        )
        out_cursor.execute(
            """
        INSERT INTO loss_function
          (ease_col, ease_row, Wmin, Wmax, LA, LB, LC, rmse)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?)""",
            (col, row) + (L.W[0], L.W[-1]) + tuple(L.L[1:4]) + (rmse,),
        )
        del col, row

    in_cursor.close()
    out_cursor.close()
    return 0


def get_optimized_loss_function(smap_time, soil_moisture, interpolate_imerg):
    """Optimize parameters for a loss function fitting a soil moisture time seres

    Arguments:
      smap_time: NumPy vector of SMAP soil moisture timestamps.
      soil_moisture: Numpy vector of SMAP soil moisture values.
      interpolate_imerg: A callable returning IMERG precipitation for a given
        timestamp.

    Returns:
      An optimized loss function and the final root-mean-square from the fit

    """
    Wmin = soil_moisture.min()
    if soil_moisture.max() == soil_moisture.min():
        LOG.warning('Constant soil moisture, zero loss function for W >= WA')
        return (LossFunction(Wmin, Wmin, 0, 0, 0), None)
    Wmax = soil_moisture.max() + 0.1 * (soil_moisture.max() - Wmin)
    assert soil_moisture.max() < Wmax, (soil_moisture.max(), Wmax)

    def smap_rms_error(L_values):
        """Compute the root-mean-square error for loss function parameters"""
        LA, LB, LC = L_values
        L = LossFunction(Wmax, Wmin, LA, LB, LC)
        tsim, Wsim = L.simulate_soil_moisture(
            Winit=soil_moisture[0],
            Wmax=Wmax,
            P_of_t_mm_d=interpolate_imerg,
            ts_start=smap_time[0],
            ts_thru=smap_time[-1],
        )
        residual = np.interp(smap_time, tsim, Wsim) - soil_moisture
        rmse = (residual**2).mean() ** 0.5
        return rmse

    Lmax = Wmax / 24.0
    result = optimize_loss_function_parameters(
        Lmax=Lmax, objective=smap_rms_error, n=100, iters=5
    )
    LA, LB, LC = result.x
    if not result.success:
        LOG.warning('Optimization failed to converge.')
    LOG.info(
        'RMSE: %s, %s',
        result.fun,
        result.message,
    )
    return (LossFunction(Wmax, Wmin, LA, LB, LC), result.fun)


def optimize_loss_function_parameters(Lmax, objective, n=100, iters=1):
    """Globally optimize loss function parameters

    Globally minimizes the objective function by adjusting a vector of parameters (LA,
    LB, LC).

    Koster et al (2017) required that L is non-decreasing ("... limiting the search
    space by assuming that L never decreases with increasing soil moisture").  This
    results in the constraints
      0 <= LA <= LB <= LC <= Lmax

    Uses scipy.optimize.shgo.

    Args:
        Lmax (float): The maximum allowed value for LC.
        objective (callable): A function that takes a vector of floats
                              (LA, LB, LC), and returns the scalar objective.
        n (int): Number of sampling points, passed to shgo.
        iters (int): Number of local search iterations, passed to shgo.

    Returns:
        OptimizeResult:  Optimization result with attributes:
                           - x: A NumPy array [LA, LB, LC] representing the
                                optimal parameters.
                           - fun (float): The minimum value of the scalar objective
                                          function.
                           - success (bool): Indicates whether the optimizer
                                             converged.
                           - message (str): Describes the optimization outcome.

    """
    # All variables must be non-negative and cannot exceed Lmax
    bounds = [
        (0, Lmax),  # Bounds for LA
        (0, Lmax),  # Bounds for LB
        (0, Lmax),  # Bounds for LC
    ]

    # Ordering constraints LA <= LB <= LC <= Lmax.
    # Constraints must be in the form g(x) >= 0.
    constraints = [
        # LB >= LA  =>  LB - LA >= 0
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
        # LC >= LB  =>  LC - LB >= 0
        {'type': 'ineq', 'fun': lambda x: x[2] - x[1]},
    ]
    return shgo(objective, bounds, constraints=constraints, n=n, iters=iters)
