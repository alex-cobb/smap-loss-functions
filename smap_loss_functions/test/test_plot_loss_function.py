"""Tests for plot_loss_function"""

import pytest
import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt

from smap_loss_functions.plot_loss_function import (
    plot_loss_function,
    plot_loss_function_simulation,
    get_loss_function_from_db,
    simulate_soil_moisture,
    LossFunction,
    get_soil_moisture,
    get_precipitation,
    create_piecewise_constant_interpolant,
    clip_to_timestamp_range,
)


@pytest.fixture
def in_memory_db():
    """
    Fixture to create an in-memory SQLite database with sample data.
    """
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create loss_function table
    cursor.execute("""
        CREATE TABLE loss_function (
            ease_col INTEGER,
            ease_row INTEGER,
            Wmin REAL,
            Wmax REAL,
            LA REAL,
            LB REAL,
            LC REAL,
            PRIMARY KEY (ease_col, ease_row)
        )
    """)
    cursor.execute(
        'INSERT INTO loss_function VALUES (?, ?, ?, ?, ?, ?, ?)',
        (1, 1, 0.1, 0.9, 0.01, 0.05, 0.1),
    )
    cursor.execute(
        'INSERT INTO loss_function VALUES (?, ?, ?, ?, ?, ?, ?)',
        (2, 2, 0.2, 0.8, 0.02, 0.06, 0.12),
    )
    conn.commit()

    # Create smap_data table
    cursor.execute("""
        CREATE TABLE smap_data (
            ease_col INTEGER,
            ease_row INTEGER,
            start_datetime TEXT,
            thru_datetime TEXT,
            soil_moisture REAL
        )
    """)
    # Sample SMAP data (timestamps in 'seconds since epoch' for unixepoch function)
    # Using real datetimes for sqlite3's unixepoch
    smap_data_entries = [
        (1, 1, '2023-01-01 00:00:00', '2023-01-01 01:00:00', 0.3),
        (1, 1, '2023-01-01 01:00:00', '2023-01-01 02:00:00', 0.35),
        (1, 1, '2023-01-01 02:00:00', '2023-01-01 03:00:00', 0.4),
        (
            1,
            1,
            '2023-01-02 00:00:00',
            '2023-01-02 01:00:00',
            0.45,
        ),  # for plot_loss_function_simulation
        (
            1,
            1,
            '2023-01-03 00:00:00',
            '2023-01-03 01:00:00',
            0.5,
        ),  # for plot_loss_function_simulation
    ]
    for entry in smap_data_entries:
        cursor.execute('INSERT INTO smap_data VALUES (?, ?, ?, ?, ?)', entry)
    conn.commit()

    # Create imerg_data table
    cursor.execute("""
        CREATE TABLE imerg_data (
            ease_col INTEGER,
            ease_row INTEGER,
            start_datetime TEXT,
            precipitation REAL
        )
    """)
    # Sample IMERG data (daily precipitation)
    imerg_data_entries = [
        (1, 1, '2022-12-31 00:00:00', 0.0),  # Day before SMAP start
        (1, 1, '2023-01-01 00:00:00', 5.0),
        (1, 1, '2023-01-02 00:00:00', 10.0),
        (1, 1, '2023-01-03 00:00:00', 2.0),
        (1, 1, '2023-01-04 00:00:00', 0.0),  # Day after SMAP end
    ]
    for entry in imerg_data_entries:
        cursor.execute('INSERT INTO imerg_data VALUES (?, ?, ?, ?)', entry)
    conn.commit()

    yield conn  # Provide the connection to the tests

    conn.close()  # Close connection after tests


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


def test_get_loss_function_from_db(in_memory_db):
    """
    Test get_loss_function_from_db with valid and invalid inputs.
    """
    conn = in_memory_db
    cursor = conn.cursor()

    # Test valid case
    lf = get_loss_function_from_db(conn, 1, 1)
    assert isinstance(lf, LossFunction)
    assert np.isclose(lf.W[0], 0.1)
    assert np.isclose(lf.W[-1], 0.9)

    # Test invalid case (non-existent ease_col, ease_row)
    with pytest.raises(ValueError, match='EASE col=99 row=99 not in ranges: '):
        get_loss_function_from_db(conn, 99, 99)


def test_get_soil_moisture(in_memory_db):
    """
    Test get_soil_moisture function.
    """
    conn = in_memory_db
    cursor = conn.cursor()

    time, sm = get_soil_moisture(cursor, 1, 1)
    assert isinstance(time, np.ndarray)
    assert isinstance(sm, np.ndarray)
    assert len(time) == 5
    assert len(sm) == 5
    assert np.isclose(sm[0], 0.3)
    assert np.isclose(sm[-1], 0.5)
    # Check that timestamps are increasing
    assert (np.diff(time) > 0).all()


def test_get_precipitation(in_memory_db):
    """
    Test get_precipitation function.
    """
    conn = in_memory_db
    cursor = conn.cursor()

    time, precip = get_precipitation(cursor, 1, 1)
    assert isinstance(time, np.ndarray)
    assert isinstance(precip, np.ndarray)
    assert len(time) == 5
    assert len(precip) == 5
    assert np.isclose(precip[0], 0.0)
    assert np.isclose(precip[1], 5.0)
    assert np.isclose(precip[-1], 0.0)
    # Check that timestamps are daily intervals
    assert (np.diff(time) == 86400).all()


def test_create_piecewise_constant_interpolant():
    """
    Test create_piecewise_constant_interpolant.
    """
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 20, 30, 40])
    interpolant = create_piecewise_constant_interpolant(x, y)

    # Test within intervals
    assert np.isclose(interpolant(1.5), 10)
    assert np.isclose(
        interpolant(2.0), 20
    )  # At the boundary, should take the value to the right
    assert np.isclose(interpolant(2.5), 20)
    assert np.isclose(interpolant(3.9), 30)

    # Test extrapolation below
    assert np.isclose(interpolant(0.5), 10)
    assert np.isclose(interpolant(-10), 10)

    # Test extrapolation above
    assert np.isclose(interpolant(4.0), 40)
    assert np.isclose(interpolant(5.0), 40)
    assert np.isclose(interpolant(100), 40)

    # Test assertion for non-strictly increasing x
    with pytest.raises(AssertionError):
        create_piecewise_constant_interpolant(
            np.array([1, 2, 2, 3]), np.array([1, 2, 3, 4])
        )
    with pytest.raises(AssertionError):
        create_piecewise_constant_interpolant(np.array([3, 2, 1]), np.array([1, 2, 3]))

    # Test with empty arrays
    interpolant_empty = create_piecewise_constant_interpolant(
        np.array([]), np.array([])
    )
    with pytest.raises(IndexError):  # np.clip will try to access empty array
        interpolant_empty(5)


def test_clip_to_timestamp_range():
    """
    Test clip_to_timestamp_range function.
    """
    time = np.array([10, 20, 30, 40, 50, 60, 70])
    value = np.array([1, 2, 3, 4, 5, 6, 7])

    # Clip from both ends
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 25, 55)
    assert np.array_equal(clipped_time, np.array([30, 40, 50]))
    assert np.array_equal(clipped_value, np.array([3, 4, 5]))

    # Clip from left only
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 45, 70)
    assert np.array_equal(clipped_time, np.array([50, 60, 70]))
    assert np.array_equal(clipped_value, np.array([5, 6, 7]))

    # Clip from right only
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 10, 35)
    assert np.array_equal(clipped_time, np.array([10, 20, 30]))
    assert np.array_equal(clipped_value, np.array([1, 2, 3]))

    # No clipping
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 10, 70)
    assert np.array_equal(clipped_time, time)
    assert np.array_equal(clipped_value, value)

    # Resulting in empty arrays
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 80, 90)
    assert len(clipped_time) == 0
    assert len(clipped_value) == 0

    # Range that includes no points (start > thru)
    clipped_time, clipped_value = clip_to_timestamp_range(time, value, 40, 30)
    assert len(clipped_time) == 0
    assert len(clipped_value) == 0


@pytest.mark.mpl_image_compare
def test_plot_loss_function_execution():
    """
    Test that plot_loss_function executes and returns a matplotlib Figure.
    """
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)
    ease_col, ease_row = 1, 1

    # plot_loss_function should now return the figure object
    fig = plot_loss_function(loss_func, ease_col, ease_row)
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_loss_function_simulation_execution(in_memory_db):
    """
    Test that plot_loss_function_simulation executes and returns a matplotlib Figure.
    """
    conn = in_memory_db
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)
    ease_col, ease_row = 1, 1

    # plot_loss_function_simulation should now return the figure object
    fig = plot_loss_function_simulation(loss_func, conn, ease_col, ease_row)
    assert isinstance(fig, plt.Figure)
    return fig


def test_plot_loss_function_simulation_insufficient_smap_data(in_memory_db):
    """
    Test plot_loss_function_simulation with insufficient SMAP data.
    """
    conn = in_memory_db
    cursor = conn.cursor()
    # Delete all but one SMAP data entry for col=1, row=1
    cursor.execute(
        "DELETE FROM smap_data WHERE ease_col = ? AND ease_row = ? AND start_datetime != '2023-01-01 00:00:00'",
        (1, 1),
    )
    conn.commit()

    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)
    ease_col, ease_row = 1, 1

    with pytest.raises(
        ValueError, match='Only 1 SMAP values in col 1 row 1; nothing to do'
    ):
        plot_loss_function_simulation(loss_func, conn, ease_col, ease_row)


def test_simulate_soil_moisture_basic_case():
    """
    Test simulate_soil_moisture with a simple scenario.
    """
    Winit = 0.5
    Wmax = 0.9
    # Create a dummy LossFunction
    loss_func = LossFunction(Wmax=0.9, Wmin=0.1, LA=0.01, LB=0.05, LC=0.1)

    # Simple P_of_t_mm_d: constant precipitation
    def constant_precipitation(t):
        return np.full_like(t, 24.0)  # 24 mm/day = 1 mm/hour

    ts_start = datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp()
    ts_thru = datetime.datetime(2023, 1, 1, 3, 0, 0).timestamp()  # 3-hour simulation

    time_sim, W_sim = simulate_soil_moisture(
        Winit=Winit,
        Wmax=Wmax,
        loss_function_h=loss_func,
        P_of_t_mm_d=constant_precipitation,
        ts_start=ts_start,
        ts_thru=ts_thru,
        max_infiltration_h=1.0,  # Default value
    )

    # Check time array
    expected_start_sec = int(np.ceil(ts_start / 3600.0)) * 3600
    expected_thru_sec = int(np.floor(ts_thru / 3600.0)) * 3600
    expected_time_sim = np.arange(
        expected_start_sec, expected_thru_sec + 3600, 3600, dtype='float64'
    )
    assert np.array_equal(time_sim, expected_time_sim)
    assert len(W_sim) == len(time_sim)

    # Check W values (rough check, specific values depend on loss_func and P)
    assert not np.isnan(W_sim).any()
    assert W_sim[0] == Winit
    # Expect W to change over time
    assert not np.allclose(W_sim[0], W_sim[1:])


def test_simulate_soil_moisture_zero_length_simulation():
    """
    Test simulate_soil_moisture raises ValueError for zero-length simulation.
    """
    Winit = 0.5
    Wmax = 0.9
    loss_func = LossFunction(Wmax=0.9, Wmin=0.1, LA=0.01, LB=0.05, LC=0.1)

    def dummy_precipitation(t):
        return np.array([0.0])

    ts_start = datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp()
    ts_thru = datetime.datetime(
        2023, 1, 1, 0, 30, 0
    ).timestamp()  # thru_hour <= start_hour

    with pytest.raises(ValueError, match='Zero-length simulation'):
        simulate_soil_moisture(
            Winit=Winit,
            Wmax=Wmax,
            loss_function_h=loss_func,
            P_of_t_mm_d=dummy_precipitation,
            ts_start=ts_start,
            ts_thru=ts_thru,
        )
