"""Forecast soil moisture dynamics using a loss function with zero precipitation

Using a loss function database (sensu Koster et al 2017) and a SMAP soil moisture
database, take the latest soil moisture reading from the SMAP database in each EASE 2.0
grid row and column as an initial condition and simulate soil moisture for 5 days
forward, assuming no precipitation.  Results are written to a new soil moisture database.

Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A data-driven approach for
  daily real-time estimates and forecasts of near-surface soil moisture. Journal of
  Hydrometeorology, 18(3), 837–843. https://doi.org/10.1175/jhm-d-16-0285.1


"""

import datetime
import logging
import sqlite3
import sys
import urllib.parse

import numpy as np


LOG = logging.getLogger('forecast_smap')
LOG.setLevel(logging.INFO)


def forecast_smap(loss_function_db_path, smap_db_path, forecast_db_path):
    """Forecast SMAP soil moisture over 5 days with zero precipitation"""
    loss_function_db_uri = ensure_read_only_uri(loss_function_db_path)
    del loss_function_db_path
    smap_db_uri = ensure_read_only_uri(smap_db_path)

    with sqlite3.connect(loss_function_db_uri, uri=True) as loss_function_connection:
        colrows = get_distinct_col_row_pairs(
            loss_function_connection.cursor(), smap_db_path
        )
        del loss_function_connection
    del smap_db_path
    if not colrows:
        print(
            'No matching loss function and SMAP grid cells, exiting.', file=sys.stderr
        )
        return 1

    def zero_precipitation(t):
        """Returns 0.0 mm/day precipitation"""
        return np.zeros(t.shape, dtype='float64') if isinstance(t, np.ndarray) else 0.0

    simulation_duration_s = 5 * 24 * 3600  # 5 days

    with (
        sqlite3.connect(loss_function_db_uri, uri=True) as loss_function_connection,
        sqlite3.connect(smap_db_uri, uri=True) as smap_connection,
        sqlite3.connect(forecast_db_path) as forecast_connection,
    ):
        smap_cursor = smap_connection.cursor()
        forecast_connection.execute('PRAGMA enforce_strict_check_constraints = ON;')
        forecast_cursor = forecast_connection.cursor()
        create_smap_table(forecast_cursor)
        forecast_connection.commit()
        for col, row in colrows:
            LOG.info('Forecasting SMAP for col %s, row %s', col, row)
            loss_function = get_loss_function_from_db(
                connection=loss_function_connection,
                ease_col=col,
                ease_row=row,
            )
            initial_soil_moisture, initial_timestamp = get_latest_soil_moisture(
                smap_cursor,
                ease_col=col,
                ease_row=row,
            )
            tsim, Wsim = simulate_soil_moisture(
                Winit=initial_soil_moisture,
                Wmax=loss_function.W[-1],  # Use Wmax from loss function
                loss_function_h=loss_function,
                P_of_t_mm_d=zero_precipitation,
                ts_start=initial_timestamp,
                ts_thru=initial_timestamp + simulation_duration_s,
            )
            insert_simulated_data(forecast_cursor, col, row, tsim, Wsim)
        forecast_connection.commit()
    LOG.info('Done.')
    return 0


def create_smap_table(cursor):
    """Create smap_data table"""
    # start_datetime and thru_datetime are set to the same hourly timestamp
    cursor.execute("""
    CREATE TABLE smap_data (
        start_datetime timestamp NOT NULL,
        thru_datetime timestamp NOT NULL
          CHECK (thru_datetime = start_datetime),
        ease_col integer NOT NULL,
        ease_row integer NOT NULL,
        soil_moisture real NOT NULL,
        PRIMARY KEY (start_datetime, ease_col, ease_row)
    )""")


def insert_simulated_data(cursor, ease_col, ease_row, tsim, Wsim):
    """Inserts simulated soil moisture data into the smap_data table

    Args:
        cursor: SQLite database cursor for the output database.
        ease_col: EASE grid column.
        ease_row: EASE grid row.
        tsim: NumPy array of timestamps (seconds since epoch) for simulated data.
        Wsim: NumPy array of simulated soil moisture values.
    """
    rows = []
    for t, W in zip(tsim, Wsim):
        start_dt = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')
        # start_datetime and thru_datetime are set to the same hourly timestamp
        rows.append((start_dt, start_dt, ease_col, ease_row, W))
    cursor.executemany(
        """
        INSERT OR REPLACE INTO smap_data
        (start_datetime, thru_datetime, ease_col, ease_row, soil_moisture)
        VALUES (?, ?, ?, ?, ?)
    """,
        rows,
    )


def get_distinct_col_row_pairs(loss_function_db_cursor, smap_db_path):
    """Retrieve distinct (col, row) pairs that have loss functions and SMAP data"""
    loss_function_db_cursor.execute(f"ATTACH DATABASE '{smap_db_path}' AS smap_alias")
    loss_function_db_cursor.execute("""
        SELECT DISTINCT ease_col, ease_row
        FROM loss_function
        JOIN smap_alias.smap_data USING (ease_col, ease_row)
        ORDER BY ease_col, ease_row""")
    return loss_function_db_cursor.fetchall()


def get_loss_function_from_db(connection, ease_col, ease_row):
    """Instantiate loss function for an EASE column and row from database.

    Returns a LossFunction object.
    """
    parameters = connection.execute(
        """
    SELECT Wmin, Wmax, LA, LB, LC
    FROM loss_function
    WHERE ease_col = ? AND ease_row = ?""",
        (ease_col, ease_row),
    ).fetchone()

    if parameters is None:
        # Provide range of available columns and rows for better error message
        min_col, max_col, min_row, max_row = connection.execute(
            'SELECT min(ease_col), max(ease_col), min(ease_row), max(ease_row) '
            'FROM loss_function'
        ).fetchone()
        raise ValueError(
            f'EASE col={ease_col} row={ease_row} not found in loss_function database. '
            f'Available ranges: Col {min_col}-{max_col}, Row {min_row}-{max_row}.'
        )
    Wmin, Wmax, LA, LB, LC = parameters
    return LossFunction(Wmax, Wmin, LA, LB, LC)


def get_latest_soil_moisture(cursor, ease_col, ease_row):
    """Retrieve the latest soil moisture data point for a given EASE column and row.

    Returns a tuple (latest_soil_moisture, latest_timestamp) where timestamp is
    in seconds since the UNIX epoch.
    """
    cursor.execute(
        """
    SELECT
        unixepoch(start_datetime, 'subsecond') AS start_datetime,
        soil_moisture
    FROM smap_data
    WHERE ease_col = ?
    AND ease_row = ?
    ORDER BY start_datetime DESC
    LIMIT 1""",
        (ease_col, ease_row),
    )
    result = cursor.fetchone()
    assert result is not None, f'SMAP data missing for col={ease_col}, row={ease_row}'

    latest_timestamp, latest_soil_moisture = result
    assert np.isfinite(latest_timestamp), (
        f'Non-finite timestamp at col={ease_col}, row={ease_row}'
    )
    assert np.isfinite(latest_soil_moisture), (
        f'Non-finite soil moisture at col={ease_col}, row={ease_row}'
    )
    return latest_soil_moisture, latest_timestamp


def simulate_soil_moisture(
    Winit, Wmax, loss_function_h, P_of_t_mm_d, ts_start, ts_thru, max_infiltration_h=1.0
):
    """Simulate soil moisture given precipitation and a loss function.

    Simulates soil moisture following Koster et al (2017) section 2, equations 1--3.
    W[i + 1] = W[i] - L(W[i]) Δt + Wadd (eqn 1)
    Wadd = I Δt / D (eqn 2)
    I = min(P, (Wmax - W[i]) D / nd) (eqn 3)

    Δt = 1 because calculations are done on an hourly grid in time units of hours.
    D = 50 mm is the effective thickness of SMAP soil moisture observations.
    nd is a time scale limiting the infiltration rate.

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
    delta_t = 1  # hourly time step
    D = 50  # mm (effective thickness of SMAP soil moisture observations)
    L = loss_function_h
    nd = max_infiltration_h  # Time scale limiting infiltration rate

    # Determine start and end hours for simulation
    start_hour = int(np.ceil(ts_start / 3600.0))
    thru_hour = int(np.floor(ts_thru / 3600.0))

    if thru_hour <= start_hour:
        # This can happen if the initial timestamp is very close to the end of the simulation period
        # or if the initial_timestamp is after thru_timestamp for some reason.
        raise ValueError(
            'Simulation period is less than an hour, '
            f'from {datetime.datetime.fromtimestamp(ts_start)} '
            f'to {datetime.datetime.fromtimestamp(ts_thru)}'
        )

    # Create time array for hourly steps
    t = np.arange(start_hour, thru_hour + 1, dtype='int').astype('float64')

    # Get precipitation for each hour (convert mm/d to mm/hr)
    # P_of_t_mm_d expects time in seconds, returns mm/day
    P = P_of_t_mm_d(t * 3600) / 24.0  # Convert daily precipitation to hourly

    if P.shape != t.shape:
        raise ValueError(
            f'Precipitation array shape {P.shape} does not match time array shape {t.shape}.'
        )

    # Initialize soil moisture array
    W = np.empty(shape=t.shape, dtype='float64')
    W[0] = Winit

    # Simulate soil moisture hour by hour
    for i in range(len(t) - 1):
        # Infiltration I is the minimum of actual precipitation and maximum possible infiltration
        # The equation for I is `min(P, (Wmax - W[i]) D / nd)`
        # P[i] is hourly precipitation (mm/hr)
        # (Wmax - W[i]) is soil moisture deficit (mm/mm)
        # D is effective thickness (mm)
        # nd is max_infiltration_h (hours)
        # So (Wmax - W[i]) * D / nd is (mm/mm * mm / hr) = mm/hr, which matches P[i] units
        I = min((P[i], (Wmax - W[i]) * D / nd))

        # Wadd is the added water from infiltration
        # Wadd = I * delta_t / D (mm/hr * hr / mm) = mm/mm
        Wadd = I * delta_t / D

        # Update soil moisture: W[i+1] = W[i] - L(W[i]) * delta_t + Wadd
        # L(W[i]) is loss rate in 1/hr, so L(W[i]) * delta_t is unitless
        # W[i] and Wadd are mm/mm, so units are consistent
        W[i + 1] = W[i] - L(W[i]) * delta_t + Wadd

        # Ensure soil moisture does not go below Wmin or above Wmax (clipping)
        # Although the model should inherently prevent W > Wmax, explicit clipping is safer
        W[i + 1] = np.clip(W[i + 1], loss_function_h.W[0], Wmax)

    if np.isnan(W).any():
        print(
            f'Warning: NaN values encountered in simulated soil moisture. W: {W}',
            file=sys.stderr,
        )
        raise ValueError('Simulation resulted in non-finite soil moisture values.')

    return (t * 3600, W)  # Convert time back to seconds for output


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
    4. L at intermediate soil moistures is computed by linear interpolation.

    The loss function extrapolates values L(Wmin) for W < Wmin and L(Wmax)
    for W > WMax.
    """

    def __init__(self, Wmax, Wmin, LA, LB, LC):
        WA, WB, WC = [Wmin + f * (Wmax - Wmin) for f in (0.25, 0.5, 0.75)]
        self.W = np.array([Wmin, WA, WB, WC, Wmax], dtype='float64')

        # Assert W_values are strictly increasing
        if not (np.diff(self.W) >= 0).all():
            raise ValueError(
                f'W values for loss function are not monotonically increasing: {self.W}'
            )

        Lmin = (
            0.0  # "We set the value of the loss function at the low end L(Wmin) to 0"
        )
        Lmax = (
            Wmax / 24.0
        )  # "L(Wmax) = Wmax volumetric units per day" converted to per hour

        self.L = np.array([Lmin, LA, LB, LC, Lmax], dtype='float64')

    def __call__(self, soil_moisture):
        """Compute loss function L(w) after Koster et al (2017) equation 1.

        Accepts a SMAP retrieval W and returns a rate of decrease in soil moisture by
        evaporation and drainage (1/h).
        """
        # np.interp performs linear interpolation. It also handles extrapolation
        # for values outside the W range by using the nearest end-point L value.
        return np.interp(soil_moisture, self.W, self.L)


def ensure_read_only_uri(db_path):
    """Ensures a database path is a file URI with 'mode=ro'.

    Handles both plain paths and existing URIs by correctly setting or overriding the
    'mode' parameter to 'ro'.

    """
    parsed_url = urllib.parse.urlparse(db_path)
    if not parsed_url.scheme:  # Plain path
        return f'file:{db_path}?mode=ro'
    query_params = urllib.parse.parse_qs(parsed_url.query) if parsed_url.query else {}
    query_params['mode'] = ['ro']
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    return urllib.parse.urlunparse(parsed_url._replace(query=new_query))
