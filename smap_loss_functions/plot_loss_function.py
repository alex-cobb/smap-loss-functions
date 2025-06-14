"""Plot loss functions sensu Koster et al (2017)

Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A data-driven approach for
  daily real-time estimates and forecasts of near-surface soil moisture. Journal of
  Hydrometeorology, 18(3), 837–843. https://doi.org/10.1175/jhm-d-16-0285.1

"""

import datetime

import matplotlib.pyplot as plt

import numpy as np


def plot_loss_function(loss_function, ease_col, ease_row):
    """Plot a loss function"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    W_grid = np.linspace(loss_function.W[0], loss_function.W[-1], 100)
    ax.plot(
        W_grid,
        [loss_function(W) for W in W_grid],
        'b-',
    )
    ax.set_xlabel('SMAP soil moisture')
    ax.set_ylabel('Loss rate, 1 / h')
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_loss_function_simulation(loss_function, smap_connection, ease_col, ease_row):
    """Plot loss function simulations against SMAP data

    Given a connection to a database with SMAP and IMERG data, simulates SMAP dynamics
    given their initial value driven by the IMERG precipitation and plots it together
    with the SMAP soil moisture in the database.

    """
    cursor = smap_connection.cursor()
    smap_time, soil_moisture = get_soil_moisture(cursor, ease_col, ease_row)
    if not len(smap_time) >= 2:
        raise ValueError(
            f'Only {len(smap_time)} SMAP values in col {ease_col} row {ease_row}; nothing to do'
        )
    imerg_start, precipitation = get_precipitation(cursor, ease_col, ease_row)
    # Clip IMERG data to a day before and after SMAP datetime range
    imerg_start, precipitation = clip_to_timestamp_range(
        time=imerg_start,
        value=precipitation,
        start_time=smap_time.min() - 86400,
        thru_time=smap_time.max() + 86400,
    )
    interpolate_imerg = create_piecewise_constant_interpolant(
        imerg_start, precipitation
    )
    tsim, Wsim = simulate_soil_moisture(
        Winit=soil_moisture[0],
        Wmax=loss_function.W[-1],
        loss_function_h=loss_function,
        P_of_t_mm_d=interpolate_imerg,
        ts_start=smap_time[0],
        ts_thru=smap_time[-1],
    )

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(
        [datetime.datetime.fromtimestamp(ts) for ts in smap_time],
        soil_moisture,
        'bo-',
    )
    axs[0].plot([datetime.datetime.fromtimestamp(ts) for ts in tsim], Wsim, 'r.-')
    time_grid = np.linspace(imerg_start.min(), imerg_start.max(), len(imerg_start) * 3)
    imerg_on_grid = interpolate_imerg(time_grid)
    # Piecewise-constant (stair) plot, imerg
    axs[1].step(
        [datetime.datetime.fromtimestamp(ts) for ts in imerg_start],
        precipitation,
        '-',
        where='post',
    )
    # Points, imerg_on_grid
    axs[1].plot(
        [datetime.datetime.fromtimestamp(ts) for ts in time_grid],
        imerg_on_grid,
        'k.',
    )
    axs[0].set_ylabel('SMAP soil moisture')
    axs[1].set_ylabel('GPM IMERG precipitation')
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def get_loss_function_from_db(connection, ease_col, ease_row):
    """Instantiate loss function for an EASE column and row from database

    Returns loss function and Wmax, which is needed for simulations.

    The parameters are computed based on the procedure of Koster et al (2017) and are
    therefore ignored.

    """
    parameters = connection.execute(
        """
    SELECT Wmin, Wmax, LA, LB, LC
    FROM loss_function
    WHERE ease_col = ? AND ease_row = ?""",
        (ease_col, ease_row),
    ).fetchone()
    if parameters is None:
        min_col, max_col, min_row, max_row = connection.execute(
            'SELECT min(ease_col), max(ease_col), min(ease_row), max(ease_row) '
            'FROM loss_function'
        ).fetchone()
        raise ValueError(
            f'EASE col={ease_col} row={ease_row} not in ranges: '
            f'{min_col}--{max_col}, {min_row}--{max_row}'
        )
    Wmin, Wmax, LA, LB, LC = parameters
    return LossFunction(Wmax, Wmin, LA, LB, LC)


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
            f'from {datetime.datetime.fromtimestamp(ts_start)} '
            f'to {datetime.datetime.fromtimestamp(ts_thru)}'
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
