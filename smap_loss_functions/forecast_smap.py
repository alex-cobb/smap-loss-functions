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

from .loss_function import LossFunction
from .utils import zero_precipitation


LOG = logging.getLogger('forecast_smap')
LOG.setLevel(logging.INFO)


def forecast_smap(
    loss_function_db_path,
    smap_db_path,
    forecast_db_path,
    # Default 5 day simulation
    simulation_duration_s=5 * 24 * 3600,
):
    """Forecast SMAP soil moisture over 5 days with zero precipitation"""
    with sqlite3.connect(
        ensure_read_only_uri(loss_function_db_path), uri=True
    ) as loss_function_connection:
        colrows = get_distinct_col_row_pairs(
            loss_function_connection.cursor(), smap_db_path
        )
        del loss_function_connection
    if not colrows:
        print(
            'No matching loss function and SMAP grid cells, exiting.', file=sys.stderr
        )
        return 1

    with (
        sqlite3.connect(
            ensure_read_only_uri(loss_function_db_path), uri=True
        ) as loss_function_connection,
        sqlite3.connect(
            ensure_read_only_uri(smap_db_path), uri=True
        ) as smap_connection,
        sqlite3.connect(forecast_db_path) as forecast_connection,
    ):
        forecast_connection.execute('PRAGMA enforce_strict_check_constraints = ON;')
        create_smap_table(forecast_connection.cursor())
        forecast_connection.commit()
        for col, row in colrows:
            LOG.info('Forecasting SMAP for col %s, row %s', col, row)
            loss_function = get_loss_function_from_db(
                connection=loss_function_connection,
                ease_col=col,
                ease_row=row,
            )
            initial_soil_moisture, initial_timestamp = get_latest_soil_moisture(
                smap_connection.cursor(),
                ease_col=col,
                ease_row=row,
            )
            tsim, Wsim = loss_function.simulate_soil_moisture(
                Winit=initial_soil_moisture,
                Wmax=loss_function.W[-1],  # Use Wmax from loss function
                P_of_t_mm_d=zero_precipitation,
                ts_start=initial_timestamp,
                ts_thru=initial_timestamp + simulation_duration_s,
            )
            insert_simulated_data(forecast_connection.cursor(), col, row, tsim, Wsim)
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
