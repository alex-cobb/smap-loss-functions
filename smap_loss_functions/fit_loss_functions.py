"""Fit SMAP loss functions sensu Koster et al (2017)

Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A data-driven approach for
  daily real-time estimates and forecasts of near-surface soil moisture. Journal of
  Hydrometeorology, 18(3), 837–843. https://doi.org/10.1175/jhm-d-16-0285.1

"""

import datetime
import logging

import numpy as np

from scipy.optimize import shgo


# Show info messages from this module but not from Scipy, which uses the root logger
LOG = logging.getLogger('fit_loss_functions')
LOG.setLevel(logging.INFO)


def set_up_loss_function_db(out_connection):
    """Set up database for loss functions

    Creates table to store loss functions.

    """
    cursor = out_connection.cursor()
    cursor.execute("""
    CREATE TABLE loss_function (
      ease_col integer NOT NULL,
      ease_row integer NOT NULL,
      Wmin real NOT NULL,
      Wmax real NOT NULL,
      LA real NOT NULL,
      LB real NOT NULL,
      LC real NOT NULL,
      rmse real NULL,
      PRIMARY KEY (ease_col, ease_row)
    )""")
    cursor.close()


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
        tsim, Wsim = simulate_soil_moisture(
            Winit=soil_moisture[0],
            Wmax=Wmax,
            loss_function_h=L,
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


def simulate_soil_moisture(
    Winit, Wmax, loss_function_h, P_of_t_mm_d, ts_start, ts_thru, max_infiltration_h=1.0
):
    """Simulate soil moisture given precipitation and a loss function

    Simulates soil moisture following Koster et al (2017) section 2,
    equations 1--3.
    W[i + 1] = W[i] - L(W[i]) Δt + Wadd (eqn 1)
    Wadd = I Δt / D (eqn 2)
    I = min(P, (Wmax - W[i]) D / nd) (eqn 3)

    Δt = 1 because calculations are done on an hourly grid in time units of hours.
    D = 50 mm is the effective thickness of SMAP soil moisture observations.
    nd is a time scale limiting the infiltration rate.

    In Koster et al (2017), nd is set to one day (24 h) such that the maximum
    infiltration rate is "the rate [that] if it were to be applied over a full
    day... would exceed the current soil water deficit".  Because the time step is 1 h,
    this implies that soil moisture cannot reach full saturation (Wmax).  In peatlands,
    this assumption seems unrealistic and so the default for this parameter is 1.0,
    which simply limits infiltration to prevent the soil moisture from exceeding Wmax.
    The original behavior from Koster et al (2017) can be recovered by setting
    max_infiltration_h to 24.

    Inputs:
      1. Winit: Initial soil moisture Winit(ts_start).
      2. Wmax: Maximum soil moisture in the pixel.
      3. P_of_t_mm_d: Function returning precipitation (mm / d)
                      at time t (seconds since the epoch).
      4. loss_function: Loss function L in volumetric units per hour.
      5. ts_start: Timestamp at which to start simulations, in seconds
         since the UNIX epoch (1970-01-01 00:00:00Z).
      6. ts_thru: Timestamp through which to simulate.
      7. max_infiltration_h: Time scale limiting the infiltration rate (nd in eqn 3).
    Returns:
      (time, W)
    where time is in seconds since the epoch on an hourly grid and W is the simulated
    soil moisture.
    time[0] is the smallest round hour >= ts_start, and time[-1] is the largest
    round hour <= ts_thru.

    """
    # Calculations are done on an hourly grid, then converted back to seconds since the
    # epoch at exit.
    delta_t = 1
    D = 50  # mm
    L = loss_function_h
    nd = max_infiltration_h

    start_hour = int(np.ceil(ts_start / 3600.0))
    thru_hour = int(np.floor(ts_thru / 3600.0))
    if thru_hour <= start_hour:
        raise ValueError(
            'Zero-length simulation '
            f'from {datetime.datetime.fromtimestamp(ts_start, datetime.UTC)} '
            f'to {datetime.datetime.fromtimestamp(ts_thru, datetime.UTC)}'
        )
    t = np.arange(start_hour, thru_hour + 1, dtype='int').astype('float64')
    P = P_of_t_mm_d(t * 3600) / 24
    assert P.shape == t.shape
    W = np.empty(shape=t.shape, dtype='float64')
    W[:] = float('NaN')
    W[0] = Winit
    for i in range(len(t) - 1):
        I = min((P[i], (Wmax - W[i]) * D / nd))  # noqa: E741
        Wadd = I * delta_t / D
        W[i + 1] = W[i] - L(W[i]) * delta_t + Wadd
    assert not np.isnan(W).any(), W
    return (t * 3600, W)


class LossFunction:
    """A loss function sensu Koster et al (2017)

    A function L from equation 1 in Koster et al (2017) that accepts a SMAP
    retrieval W and returns a rate of decrease in soil moisture by evaporation and
    drainage.  Units of the returned loss function are 1 / h.

    As described in section 2.c of the paper, the loss function is specified by:
    1. A maximum soil moisture, Wmax.  Wmax = Whigh + 0.1 (Whigh - Wlow) where
       Whigh and Wlow are the highest and lowest soil moistures retrieved in the
       grid.  L(Wmax) = Wmax / (24 h).
    2. A minimum soil moisture, Wmin = Wlow.  L(Wmin) = 0.
    3. Parameters LA, LB, LC specifying the loss rate at three intermediate
       soil moisture values WA, WB, WC that divide the interval [Wmin, Wmax] into
       four equal segments:
       WA, WB, WC = [Wmin + f x (Wmax - Wmin) for f in (0.25, 0.5, 0.75)]
    4. L at intermediate soil moistures is commputed by linear interpolation.

    The loss function extrapolates values L(Wmin) for W < Wmin and L(Wmax)
    for W > WMax.

    """

    def __init__(self, Wmax, Wmin, LA, LB, LC):
        WA, WB, WC = [Wmin + f * (Wmax - Wmin) for f in (0.25, 0.5, 0.75)]
        W_values = np.array([Wmin, WA, WB, WC, Wmax], dtype='float64')
        assert (np.diff(W_values) >= 0).all(), W_values
        Lmin = (
            0.0  # "We set the value of the loss function at the low end L(Wmin) to 0"
        )
        Lmax = Wmax / 24.0  # "L(Wmax) = Wmax volumetric units per day"
        self.W = W_values
        self.L = np.array([Lmin, LA, LB, LC, Lmax], dtype='float64')

    def __call__(self, soil_moisture):
        """Compute loss function L(w) after Koster et al (2017) equation 1

        Accepts a SMAP retrieval W and returns a rate of decrease in soil moisture by
        evaporation and drainage.

        """
        return np.interp(soil_moisture, self.W, self.L)


def get_soil_moisture(cursor, col, row):
    """Retrieve soil moisture data time series from EASE column and row

    Returns two NumPy 1D float64 arrays giving time in seconds since the epoch and SMAP
    soil moisture, respectively.

    """
    cursor.execute(
        """
    SELECT
      unixepoch(start_datetime, 'subsecond') AS start_datetime,
      unixepoch(thru_datetime, 'subsecond') AS thru_datetime,
      soil_moisture
    FROM smap_data
    WHERE ease_col = ?
    AND ease_row = ?
    ORDER BY start_datetime""",
        (col, row),
    )
    smap_start, smap_thru, soil_moisture = [
        np.array(v, dtype='float64') for v in zip(*cursor.fetchall())
    ]
    for v in (
        smap_start,
        smap_thru,
        soil_moisture,
    ):
        assert np.isfinite(v).all()
    smap_time = (smap_start + smap_thru) / 2
    del smap_start, smap_thru
    return (smap_time, soil_moisture)


def get_imerg_interpolant(cursor, col, row, start_time, thru_time):
    """Get interpolant of IMERG data for the given time range"""
    imerg_start, precipitation = get_precipitation(cursor, col, row)
    # Clip IMERG data to a day before and after SMAP datetime range
    imerg_start, precipitation = clip_to_timestamp_range(
        time=imerg_start,
        value=precipitation,
        start_time=start_time,
        thru_time=thru_time,
    )
    return create_piecewise_constant_interpolant(imerg_start, precipitation)


def get_precipitation(cursor, col, row):
    """Retrieve precipitation time series from EASE column and row

    Returns two NumPy 1D float64 arrays giving time in seconds since the epoch and IMERG
    precipitation, respectively.

    """
    cursor.execute(
        """
    SELECT
      unixepoch(start_datetime, 'subsecond') AS start_datetime,
      precipitation
    FROM imerg_data
    WHERE ease_col = ?
    AND ease_row = ?
    ORDER BY start_datetime""",
        (col, row),
    )
    imerg_start, precipitation = [
        np.array(v, dtype='float64') for v in zip(*cursor.fetchall())
    ]
    for v in (
        imerg_start,
        precipitation,
    ):
        assert np.isfinite(v).all()
    assert (np.diff(imerg_start) == 86400).all(), (
        'IMERG time intervals not all one day (86400 s): '
        f'{sorted(set(np.diff(imerg_start)))}'
    )
    return imerg_start, precipitation


def create_piecewise_constant_interpolant(x, y):
    """Create a piecewise-constant interpolant from arrays x and y

    Returns an interpolant f(p) such that:
    - if p is between x[i] and x[i+1], f(p) = y[i].
    - For p < x[0], f(p) = y[0].
    - For p >= x[-1], f(p) = y[-1].

    x must be strictly increasing (checked by an assertion).

    """
    assert len(x) == len(y)
    assert (np.diff(x) > 0).all(), f'{x} is not strictly increasing'

    def interpolate(p):
        """Interpolate value at p by piecewise-constant interpolation"""
        idx = np.searchsorted(x, p, side='right') - 1
        idx = np.clip(idx, 0, len(y) - 1)
        return y[idx]

    return interpolate


def clip_to_timestamp_range(time, value, start_time, thru_time):
    """Clip t, y values to time range

    Returns vectors t[i:j] and y[i:j] such that t[i] >= start_time and t[j] <=
    thru_time.

    """
    mask = (time >= start_time) & (time <= thru_time)
    return time[mask], value[mask]
