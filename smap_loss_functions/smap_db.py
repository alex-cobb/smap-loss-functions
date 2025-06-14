"""SMAP and IMERG SQLite database code"""

import numpy as np


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
