"""Tests for forecast_smap"""

import pytest
import sqlite3
import datetime
import numpy as np
import sys

from smap_loss_functions.forecast_smap import (
    forecast_smap,
    create_smap_table,
    insert_simulated_data,
    get_distinct_col_row_pairs,
    get_loss_function_from_db,
    get_latest_soil_moisture,
    simulate_soil_moisture,
    LossFunction,
    ensure_read_only_uri,
)


def create_mock_db(db_path, schema_sql, initial_data):
    """
    Helper to create a temporary SQLite database with given schema and data.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(schema_sql)
        for table, rows in initial_data.items():
            if rows:
                placeholders = ','.join(['?'] * len(rows[0]))
                cursor.executemany(f'INSERT INTO {table} VALUES ({placeholders})', rows)
        conn.commit()


def create_loss_function_db(db_path, data):
    """
    Creates a mock loss function database.
    Data format: [(Wmin, Wmax, LA, LB, LC, ease_col, ease_row), ...]
    """
    schema = """
    CREATE TABLE loss_function (
        Wmin REAL,
        Wmax REAL,
        LA REAL,
        LB REAL,
        LC REAL,
        ease_col INTEGER,
        ease_row INTEGER,
        PRIMARY KEY (ease_col, ease_row)
    );
    """
    initial_data = {'loss_function': data}
    create_mock_db(db_path, schema, initial_data)


def create_smap_db(db_path, data):
    """
    Creates a mock SMAP data database.
    Data format: [(start_datetime_str, thru_datetime_str, ease_col, ease_row, soil_moisture), ...]
    Timestamps should be strings in 'YYYY-MM-DD HH:MM:SS.ffffff' format.
    """
    schema = """
    CREATE TABLE smap_data (
        start_datetime TIMESTAMP NOT NULL,
        thru_datetime TIMESTAMP NOT NULL CHECK (thru_datetime = start_datetime),
        ease_col INTEGER NOT NULL,
        ease_row INTEGER NOT NULL,
        soil_moisture REAL NOT NULL,
        PRIMARY KEY (start_datetime, ease_col, ease_row)
    );
    """
    initial_data = {'smap_data': data}
    create_mock_db(db_path, schema, initial_data)


def zero_precipitation(t):
    """Precipitation function that always returns 0"""
    return np.zeros_like(t, dtype='float64') if isinstance(t, np.ndarray) else 0.0


@pytest.fixture
def loss_function_db_path(tmp_path):
    """Fixture for a loss function database."""
    db_file = tmp_path / 'loss_function.db'
    data = [
        (0.05, 0.45, 0.001, 0.005, 0.015, 10, 20),
        (0.03, 0.40, 0.002, 0.006, 0.018, 11, 21),
        (0.01, 0.30, 0.0005, 0.002, 0.008, 12, 22),
    ]
    create_loss_function_db(db_file, data)
    return str(db_file)


@pytest.fixture
def smap_db_path(tmp_path):
    """Fixture for a SMAP data database."""
    db_file = tmp_path / 'smap.db'
    now_ts = datetime.datetime.now().timestamp()
    data = [
        (
            datetime.datetime.fromtimestamp(now_ts - 3600 * 2).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            datetime.datetime.fromtimestamp(now_ts - 3600 * 2).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            10,
            20,
            0.25,
        ),
        (
            datetime.datetime.fromtimestamp(now_ts - 3600).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            datetime.datetime.fromtimestamp(now_ts - 3600).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            10,
            20,
            0.28,
        ),  # Latest for 10, 20
        (
            datetime.datetime.fromtimestamp(now_ts - 3600 * 3).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            datetime.datetime.fromtimestamp(now_ts - 3600 * 3).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            11,
            21,
            0.15,
        ),
        (
            datetime.datetime.fromtimestamp(now_ts - 3600 * 0.5).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            datetime.datetime.fromtimestamp(now_ts - 3600 * 0.5).strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            ),
            12,
            22,
            0.18,
        ),  # Latest for 12, 22 (non-matching in LF DB initially)
    ]
    create_smap_db(db_file, data)
    return str(db_file)


@pytest.fixture
def empty_smap_db_path(tmp_path):
    """Fixture for an empty SMAP data database."""
    db_file = tmp_path / 'empty_smap.db'
    create_smap_db(db_file, [])
    return str(db_file)


@pytest.fixture
def empty_loss_function_db_path(tmp_path):
    """Fixture for an empty loss function database."""
    db_file = tmp_path / 'empty_loss_function.db'
    create_loss_function_db(db_file, [])
    return str(db_file)


# Unit tests


def test_ensure_read_only_uri_plain_path():
    """Test converting a plain path to a read-only URI."""
    path = '/path/to/my/database.db'
    expected = 'file:/path/to/my/database.db?mode=ro'
    assert ensure_read_only_uri(path) == expected


def test_ensure_read_only_uri_existing_uri_no_mode():
    """Test converting an existing URI without mode to a read-only URI."""
    uri = 'file:///absolute/path/db.db?cache=shared'
    expected = 'file:///absolute/path/db.db?cache=shared&mode=ro'
    assert ensure_read_only_uri(uri) == expected


def test_ensure_read_only_uri_existing_uri_with_rw_mode():
    """Test converting an existing URI with read/write mode to a read-only URI."""
    uri = 'file:data.db?mode=rw'
    expected = 'file:data.db?mode=ro'  # mode=ro should override mode=rw
    assert ensure_read_only_uri(uri) == expected


def test_ensure_read_only_uri_existing_uri_with_ro_mode():
    """Test converting an existing URI already with read-only mode."""
    uri = 'file:another.db?mode=ro'
    expected = 'file:another.db?mode=ro'
    assert ensure_read_only_uri(uri) == expected


def test_create_smap_table(tmp_path):
    """Test if the smap_data table is created correctly."""
    db_file = tmp_path / 'test_create.db'
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        create_smap_table(cursor)
        conn.commit()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='smap_data';"
        )
        assert cursor.fetchone() is not None, 'Table does not exist'

        cursor.execute('PRAGMA table_info(smap_data);')
        columns = [row[1] for row in cursor.fetchall()]
        expected_columns = [
            'start_datetime',
            'thru_datetime',
            'ease_col',
            'ease_row',
            'soil_moisture',
        ]
        assert all(col in columns for col in expected_columns), 'Bad table schema'


def test_insert_simulated_data(tmp_path):
    """Test inserting simulated data and verifying its presence."""
    db_file = tmp_path / 'test_insert.db'
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        create_smap_table(cursor)
        conn.commit()

        col, row = 1, 1
        tsim = np.array(
            [
                datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp(),
                datetime.datetime(2023, 1, 1, 1, 0, 0).timestamp(),
            ]
        )
        Wsim = np.array([0.15, 0.16])

        insert_simulated_data(cursor, col, row, tsim, Wsim)
        conn.commit()

        cursor.execute(
            'SELECT start_datetime, ease_col, ease_row, soil_moisture '
            'FROM smap_data ORDER BY start_datetime;'
        )
        results = cursor.fetchall()

        assert len(results) == 2, 'Data insertion failure'
        # SQLite stores timestamps as strings by default when inserted this way.
        assert results[0] == ('2023-01-01 00:00:00.000000', 1, 1, 0.15)
        assert results[1] == ('2023-01-01 01:00:00.000000', 1, 1, 0.16)


def test_get_distinct_col_row_pairs_match(tmp_path):
    """Test retrieving distinct (col, row) pairs that exist in both DBs."""
    loss_db = tmp_path / 'lf_match.db'
    smap_db = tmp_path / 'smap_match.db'

    create_loss_function_db(
        loss_db,
        [
            (0.1, 0.5, 0.01, 0.02, 0.03, 10, 20),
            (0.05, 0.4, 0.005, 0.01, 0.015, 11, 21),
            (0.0, 0.3, 0.001, 0.002, 0.003, 12, 22),  # Only in LF
        ],
    )
    now_ts = datetime.datetime.now().timestamp()
    create_smap_db(
        smap_db,
        [
            (
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                10,
                20,
                0.2,
            ),
            (
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                11,
                21,
                0.3,
            ),
            (
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                13,
                23,
                0.4,
            ),  # Only in SMAP
        ],
    )

    with sqlite3.connect(loss_db) as conn:
        cursor = conn.cursor()
        result = get_distinct_col_row_pairs(cursor, str(smap_db))
        assert sorted(result) == sorted([(10, 20), (11, 21)])


def test_get_distinct_col_row_pairs_no_match(tmp_path):
    """Test when no common (col, row) pairs exist."""
    loss_db = tmp_path / 'lf_no_match.db'
    smap_db = tmp_path / 'smap_no_match.db'

    create_loss_function_db(
        loss_db,
        [
            (0.1, 0.5, 0.01, 0.02, 0.03, 10, 20),
        ],
    )
    now_ts = datetime.datetime.now().timestamp()
    create_smap_db(
        smap_db,
        [
            (
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                datetime.datetime.fromtimestamp(now_ts).strftime(
                    '%Y-%m-%d %H:%M:%S.%f'
                ),
                100,
                200,
                0.2,
            ),
        ],
    )

    with sqlite3.connect(loss_db) as conn:
        cursor = conn.cursor()
        result = get_distinct_col_row_pairs(cursor, str(smap_db))
        assert result == []


def test_get_distinct_col_row_pairs_empty_dbs(
    tmp_path, empty_loss_function_db_path, empty_smap_db_path
):
    """Test with empty loss function and SMAP databases."""
    with sqlite3.connect(empty_loss_function_db_path) as conn:
        cursor = conn.cursor()
        result = get_distinct_col_row_pairs(cursor, empty_smap_db_path)
        assert result == []


def test_get_loss_function_from_db_success(loss_function_db_path):
    """Test successful retrieval and instantiation of a LossFunction."""
    with sqlite3.connect(loss_function_db_path) as conn:
        lf = get_loss_function_from_db(conn, 10, 20)
        assert isinstance(lf, LossFunction)
        assert lf.W[-1] == pytest.approx(0.45)
        assert lf.W[0] == pytest.approx(0.05)
        assert lf.L[1] == pytest.approx(0.001)  # LA


def test_get_loss_function_from_db_not_found(loss_function_db_path):
    """Test retrieval for a non-existent (col, row) pair."""
    with sqlite3.connect(loss_function_db_path) as conn:
        with pytest.raises(ValueError, match='EASE col=99 row=99 not found'):
            get_loss_function_from_db(conn, 99, 99)


def test_get_latest_soil_moisture_success(smap_db_path):
    """Test successful retrieval of the latest soil moisture."""
    with sqlite3.connect(smap_db_path) as conn:
        cursor = conn.cursor()
        moisture, timestamp = get_latest_soil_moisture(cursor, 10, 20)
        assert moisture == pytest.approx(0.28)  # The latest value
        # Check if timestamp is finite (actual value depends on when fixture was created)
        assert np.isfinite(timestamp)


def test_get_latest_soil_moisture_missing_data(empty_smap_db_path):
    """Test retrieval when no SMAP data exists for the given cell."""
    with sqlite3.connect(empty_smap_db_path) as conn:
        cursor = conn.cursor()
        with pytest.raises(AssertionError, match='SMAP data missing for col=1, row=1'):
            get_latest_soil_moisture(cursor, 1, 1)


# Simplified LossFunction creation for testing
@pytest.fixture
def sample_loss_function():
    return LossFunction(Wmax=0.45, Wmin=0.05, LA=0.001, LB=0.005, LC=0.015)


def test_LossFunction_init_valid():
    """Test valid initialization of LossFunction."""
    lf = LossFunction(Wmax=0.45, Wmin=0.05, LA=0.001, LB=0.005, LC=0.015)
    assert np.allclose(lf.W, [0.05, 0.15, 0.25, 0.35, 0.45])
    assert np.allclose(lf.L, [0.0, 0.001, 0.005, 0.015, 0.45 / 24.0])


def test_LossFunction_init_invalid_W_order():
    """Test LossFunction initialization with decreasing W values"""
    with pytest.raises(
        ValueError, match='W values for loss function are not monotonically increasing'
    ):
        LossFunction(Wmax=0.05, Wmin=0.45, LA=0.001, LB=0.005, LC=0.015)  # Wmax < Wmin


def test_LossFunction_call_interpolation(sample_loss_function):
    """Test interpolation within the W range."""
    lf = sample_loss_function
    # Test values between Wmin and WA (0.05 and 0.15)
    # L(0.10) should be halfway between Lmin (0.0) and LA (0.001)
    assert lf(0.10) == pytest.approx(0.0005)
    # Test values between WA and WB (0.15 and 0.25)
    # L(0.20) should be halfway between LA (0.001) and LB (0.005)
    assert lf(0.20) == pytest.approx(0.003)


def test_LossFunction_call_extrapolation_below(sample_loss_function):
    """Test extrapolation for soil moisture below Wmin."""
    lf = sample_loss_function
    # L(Wmin) is 0.0, so anything below Wmin should also be 0.0
    assert lf(0.01) == pytest.approx(0.0)
    assert lf(-0.1) == pytest.approx(0.0)


def test_LossFunction_call_extrapolation_above(sample_loss_function):
    """Test extrapolation for soil moisture above Wmax."""
    lf = sample_loss_function
    # L(Wmax) = Wmax / 24.0 = 0.45 / 24 = 0.01875
    assert lf(0.50) == pytest.approx(0.45 / 24.0)
    assert lf(1.0) == pytest.approx(0.45 / 24.0)


def test_simulate_soil_moisture_basic_zero_precipitation(sample_loss_function):
    """
    Test a basic simulation with zero precipitation.
    Soil moisture should only decrease due to loss function.
    """
    Winit = 0.30
    Wmax = sample_loss_function.W[-1]
    loss_function_h = sample_loss_function
    P_of_t_mm_d = zero_precipitation
    ts_start = datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp()
    ts_thru = datetime.datetime(2023, 1, 1, 5, 0, 0).timestamp()  # Simulate for 5 hours

    tsim, Wsim = simulate_soil_moisture(
        Winit=Winit,
        Wmax=Wmax,
        loss_function_h=loss_function_h,
        P_of_t_mm_d=P_of_t_mm_d,
        ts_start=ts_start,
        ts_thru=ts_thru,
    )

    assert (
        len(tsim) == 6
    )  # 0, 1, 2, 3, 4, 5 hours (ts_start to ts_thru inclusive hours)
    assert len(Wsim) == 6
    assert Wsim[0] == pytest.approx(Winit)  # Initial condition

    # Expect soil moisture to decrease or stay the same (if at Wmin or L(W)=0)
    assert np.all(np.diff(Wsim) <= 0)  # Should be non-increasing for zero precipitation

    # Check a few values manually or against a simple model expectation
    # At Winit=0.30, L(0.30) is between L(WB=0.25)=0.005 and L(WC=0.35)=0.015.
    # L(0.30) = 0.005 + (0.30 - 0.25)/(0.35 - 0.25) * (0.015 - 0.005)
    # L(0.30) = 0.005 + 0.05/0.10 * 0.01 = 0.005 + 0.5 * 0.01 = 0.005 + 0.005 = 0.01
    # W[i+1] = W[i] - L(W[i]) * 1 (delta_t=1 hour)
    # W[1] = W[0] - L(W[0]) = 0.30 - 0.01 = 0.29
    assert Wsim[1] == pytest.approx(0.29)


def test_simulate_soil_moisture_error_zero_or_negative_duration():
    """Test `ValueError` for zero or negative simulation duration."""
    Winit = 0.20
    Wmax = 0.40
    lf = LossFunction(Wmax=Wmax, Wmin=0.10, LA=0.001, LB=0.002, LC=0.003)
    P_of_t_mm_d = zero_precipitation
    ts_start = datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp()
    ts_thru_equal = ts_start
    ts_thru_negative = ts_start - 3600

    with pytest.raises(ValueError, match='Simulation period is less than an hour'):
        simulate_soil_moisture(
            Winit=Winit,
            Wmax=Wmax,
            loss_function_h=lf,
            P_of_t_mm_d=P_of_t_mm_d,
            ts_start=ts_start,
            ts_thru=ts_thru_equal,
        )
    with pytest.raises(ValueError, match='Simulation period is less than an hour'):
        simulate_soil_moisture(
            Winit=Winit,
            Wmax=Wmax,
            loss_function_h=lf,
            P_of_t_mm_d=P_of_t_mm_d,
            ts_start=ts_start,
            ts_thru=ts_thru_negative,
        )


def test_simulate_soil_moisture_clipping_at_Wmin():
    """Test that soil moisture does not go below Wmin."""
    Winit = 0.06  # Slightly above Wmin
    Wmax = 0.40
    lf = LossFunction(Wmax=Wmax, Wmin=0.05, LA=0.001, LB=0.002, LC=0.003)  # Wmin = 0.05
    P_of_t_mm_d = zero_precipitation
    ts_start = datetime.datetime(2023, 1, 1, 0, 0, 0).timestamp()
    ts_thru = datetime.datetime(2023, 1, 1, 5, 0, 0).timestamp()

    tsim, Wsim = simulate_soil_moisture(
        Winit=Winit,
        Wmax=Wmax,
        loss_function_h=lf,
        P_of_t_mm_d=P_of_t_mm_d,
        ts_start=ts_start,
        ts_thru=ts_thru,
    )
    # After some time, Wsim should reach Wmin and stay there or very close
    assert np.all(Wsim >= lf.W[0] - 1e-9)  # Allow for floating point precision

    # Test directly setting Winit to Wmin, expect it to stay at Wmin (since L(Wmin)=0 and P=0)
    Winit_at_Wmin = lf.W[0]
    tsim_min, Wsim_min = simulate_soil_moisture(
        Winit=Winit_at_Wmin,
        Wmax=Wmax,
        loss_function_h=lf,
        P_of_t_mm_d=P_of_t_mm_d,
        ts_start=ts_start,
        ts_thru=ts_thru,
    )
    assert np.allclose(Wsim_min, lf.W[0])


# Test main function


def test_forecast_smap_success(loss_function_db_path, smap_db_path, tmp_path, capsys):
    """
    Test the main forecast_smap function for a successful run.
    Verifies that the forecast database is created and populated.
    """
    forecast_db_path = tmp_path / 'forecast.db'

    # Capture stdout/stderr for checking log messages
    original_stderr = sys.stderr
    sys.stderr = sys.stdout  # Redirect stderr to stdout for capsys

    try:
        # Run the forecast
        result = forecast_smap(
            loss_function_db_path=loss_function_db_path,
            smap_db_path=smap_db_path,
            forecast_db_path=str(forecast_db_path),
        )
    finally:
        sys.stderr = original_stderr  # Restore stderr

    assert result == 0  # Expect successful execution

    # Verify the forecast database exists
    assert forecast_db_path.exists()

    # Connect to the forecast database and check its content
    with sqlite3.connect(forecast_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT ease_col, ease_row, COUNT(*) FROM smap_data GROUP BY ease_col, ease_row;'
        )
        counts = cursor.fetchall()

        expected_points_per_cell = (
            5 * 24
        )  # 120 points for 5 full days (120 hours) simulation

        assert len(counts) == 3
        # Assuming (10, 20) and (11, 21) from fixtures are the ones matched
        for col, row, count in counts:
            if (col, row) in [(10, 20), (11, 21), (12, 22)]:
                assert count == expected_points_per_cell, (
                    f'Mismatch for ({col}, {row}): '
                    f'Expected {expected_points_per_cell}, got {count}'
                )
            else:
                pytest.fail(f'Unexpected cell ({col}, {row}) in forecast database.')

        cursor.execute(
            'SELECT soil_moisture FROM smap_data '
            'WHERE ease_col = 10 AND ease_row = 20 '
            'ORDER BY start_datetime LIMIT 5;'
        )
        some_data = cursor.fetchall()
        assert len(some_data) == 5
        assert some_data[0][0] == pytest.approx(
            0.28
        )  # Initial value from fixture for (10, 20)
        assert all(
            some_data[i][0] >= some_data[i + 1][0] for i in range(len(some_data) - 1)
        )  # Non-increasing with zero precipitation


def test_forecast_smap_no_matching_col_row(
    empty_loss_function_db_path, empty_smap_db_path, tmp_path, capsys
):
    """
    Test the main forecast_smap function when no matching (col, row) pairs are found.
    Should return 1 and print an error message.
    """
    forecast_db_path = tmp_path / 'forecast_empty_output.db'

    # Capture stdout/stderr to check printed messages
    original_stderr = sys.stderr
    sys.stderr = sys.stdout  # Redirect stderr to stdout for capsys

    try:
        result = forecast_smap(
            loss_function_db_path=empty_loss_function_db_path,
            smap_db_path=empty_smap_db_path,
            forecast_db_path=str(forecast_db_path),
        )
    finally:
        sys.stderr = original_stderr  # Restore stderr

    assert result == 1  # Expect a non-zero return for no matching data

    captured = capsys.readouterr()
    assert 'No matching loss function and SMAP grid cells, exiting.' in captured.out
    assert (
        not forecast_db_path.exists()
    )  # Forecast DB should not be created or should be empty if created
