"""Shared fixtures"""

import sqlite3

import pytest


@pytest.fixture
def in_memory_smap_db():
    """Fixture for an in-memory SQLite SMAP input database"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

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
    # Create imerg_data table
    cursor.execute("""
        CREATE TABLE imerg_data (
            ease_col INTEGER,
            ease_row INTEGER,
            start_datetime TEXT,
            precipitation REAL
        )
    """)

    # Populate with some dummy data for testing
    # SMAP data (daily, for simplicity, using 12:00:00 for start and thru)
    # These timestamps are for 2023-01-01 12:00:00 UTC onwards
    smap_data_entries = [
        (10, 20, '2023-01-01 12:00:00', '2023-01-01 12:00:00', 0.25),
        (10, 20, '2023-01-02 12:00:00', '2023-01-02 12:00:00', 0.28),
        (10, 20, '2023-01-03 12:00:00', '2023-01-03 12:00:00', 0.27),
        (10, 20, '2023-01-04 12:00:00', '2023-01-04 12:00:00', 0.30),
        # Another cell with less than 2 data points to test the 'skip' condition
        (11, 21, '2023-01-01 12:00:00', '2023-01-01 12:00:00', 0.15),
    ]
    cursor.executemany(
        'INSERT INTO smap_data VALUES (?, ?, ?, ?, ?)', smap_data_entries
    )

    # IMERG data (daily, precipitation in mm/d)
    # Timestamps are for 2022-12-31 00:00:00 UTC onwards (daily values)
    imerg_data_entries = [
        (10, 20, '2022-12-31 00:00:00', 5.0),  # Day before SMAP data starts
        (10, 20, '2023-01-01 00:00:00', 10.0),
        (10, 20, '2023-01-02 00:00:00', 1.0),
        (10, 20, '2023-01-03 00:00:00', 0.0),
        (10, 20, '2023-01-04 00:00:00', 20.0),
        (10, 20, '2023-01-05 00:00:00', 2.0),  # Day after SMAP data ends
        # For the other cell (11, 21)
        (11, 21, '2022-12-31 00:00:00', 3.0),
        (11, 21, '2023-01-01 00:00:00', 7.0),
        (11, 21, '2023-01-02 00:00:00', 0.5),
    ]
    cursor.executemany('INSERT INTO imerg_data VALUES (?, ?, ?, ?)', imerg_data_entries)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def in_memory_loss_function_db():
    """Fixture: in-memory SQLite loss function database with sample data"""
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
    yield conn
    conn.close()


@pytest.fixture
def in_memory_output_db():
    """Fixture :unpopulated in-memory SQLite output database"""
    conn = sqlite3.connect(':memory:')
    yield conn
    conn.close()
