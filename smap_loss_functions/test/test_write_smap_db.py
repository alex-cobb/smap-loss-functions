"""Tests for write-smap-db"""

import pytest
import sqlite3
import io

from smap_loss_functions.write_smap_db import write_smap_db


@pytest.fixture
def db_connection():
    """In-memory SQLite database connection for each test"""
    conn = sqlite3.connect(':memory:')
    yield conn
    conn.close()


@pytest.fixture
def smap_test_data():
    """Dummy SMAP data as an io.StringIO object, simulating a file

    Includes a header and multiple data rows.

    """
    data = (
        'start_datetime,thru_datetime,column,row,value\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,0.25\n'
        '2023-01-01 00:30:00,2023-01-01 01:00:00,11,21,0.30\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,21,0.26\n'
    )
    return io.StringIO(data)


@pytest.fixture
def imerg_test_data():
    """IMERG data as an io.StringIO object, simulating a file

    Includes a header and multiple data rows.

    """
    data = (
        'start_datetime,thru_datetime,column,row,value\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,1.5\n'
        '2023-01-01 00:30:00,2023-01-01 01:00:00,11,21,2.0\n'
    )
    return io.StringIO(data)


# --- Unit Tests ---


def test_write_smap_db_success(db_connection, smap_test_data, imerg_test_data):
    """Test that write_smap_db function creates SMAP and IMERG tables

    Then verifies the count and content of the inserted rows.

    """
    # Call the function under test
    result = write_smap_db(smap_test_data, imerg_test_data, db_connection)
    assert result == 0, 'Function should return 0 on success'

    cursor = db_connection.cursor()

    # Verify SMAP data insertion
    cursor.execute(
        'SELECT * FROM smap_data ORDER BY start_datetime, ease_col, ease_row'
    )
    smap_rows = cursor.fetchall()
    assert len(smap_rows) == 3, 'Expected 3 rows in smap_data table'
    # Check specific row content (ordered by primary key)
    assert smap_rows[0] == ('2023-01-01 00:00:00', '2023-01-01 00:30:00', 10, 20, 0.25)
    assert smap_rows[1] == ('2023-01-01 00:00:00', '2023-01-01 00:30:00', 10, 21, 0.26)
    assert smap_rows[2] == ('2023-01-01 00:30:00', '2023-01-01 01:00:00', 11, 21, 0.3)

    # Verify IMERG data insertion
    cursor.execute(
        'SELECT * FROM imerg_data ORDER BY start_datetime, ease_col, ease_row'
    )
    imerg_rows = cursor.fetchall()
    assert len(imerg_rows) == 2, 'Expected 2 rows in imerg_data table'
    # Check specific row content (ordered by primary key)
    assert imerg_rows[0] == ('2023-01-01 00:00:00', '2023-01-01 00:30:00', 10, 20, 1.5)
    assert imerg_rows[1] == ('2023-01-01 00:30:00', '2023-01-01 01:00:00', 11, 21, 2.0)


def test_write_smap_db_smap_header_failure(db_connection, imerg_test_data):
    """
    Tests that an AssertionError is raised if the SMAP input file's header is incorrect.
    """
    bad_smap_data = io.StringIO(
        'incorrect_header\n2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,0.25'
    )
    with pytest.raises(AssertionError) as excinfo:
        write_smap_db(bad_smap_data, imerg_test_data, db_connection)
    assert 'start_datetime,thru_datetime,column,row,value' in str(excinfo.value)


def test_write_smap_db_imerg_header_failure(db_connection, smap_test_data):
    """
    Tests that an AssertionError is raised if the IMERG input file's header is incorrect.
    This test ensures SMAP processing can complete before the IMERG header check.
    """
    bad_imerg_data = io.StringIO(
        'wrong_header_for_imerg\n2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,1.5'
    )
    with pytest.raises(AssertionError) as excinfo:
        write_smap_db(smap_test_data, bad_imerg_data, db_connection)
    assert 'start_datetime,thru_datetime,column,row,value' in str(excinfo.value)


def test_write_smap_db_empty_smap_data(db_connection, imerg_test_data):
    """
    Tests behavior when the SMAP input file contains only the header
    and no data rows. Verifies no SMAP data is inserted, but IMERG data is.
    """
    empty_smap = io.StringIO('start_datetime,thru_datetime,column,row,value\n')
    write_smap_db(empty_smap, imerg_test_data, db_connection)

    cursor = db_connection.cursor()
    cursor.execute('SELECT COUNT(*) FROM smap_data')
    assert cursor.fetchone()[0] == 0, (
        'No SMAP rows should be inserted for empty data file'
    )

    cursor.execute('SELECT COUNT(*) FROM imerg_data')
    assert cursor.fetchone()[0] == 2, 'IMERG data should still be written'


def test_write_smap_db_empty_imerg_data(db_connection, smap_test_data):
    """
    Tests behavior when the IMERG input file contains only the header
    and no data rows. Verifies no IMERG data is inserted, but SMAP data is.
    """
    empty_imerg = io.StringIO('start_datetime,thru_datetime,column,row,value\n')
    write_smap_db(smap_test_data, empty_imerg, db_connection)

    cursor = db_connection.cursor()
    cursor.execute('SELECT COUNT(*) FROM imerg_data')
    assert cursor.fetchone()[0] == 0, (
        'No IMERG rows should be inserted for empty data file'
    )

    cursor.execute('SELECT COUNT(*) FROM smap_data')
    assert cursor.fetchone()[0] == 3, 'SMAP data should still be written'


def test_write_smap_db_smap_pk_violation(db_connection, imerg_test_data):
    """
    Tests that an IntegrityError is raised when attempting to insert SMAP
    data with duplicate primary keys. This validates the database schema's
    primary key constraint.
    """
    duplicate_smap_data = (
        'start_datetime,thru_datetime,column,row,value\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,0.25\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,0.28\n'  # Duplicated PK
    )
    smap_file = io.StringIO(duplicate_smap_data)
    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        write_smap_db(smap_file, imerg_test_data, db_connection)
    assert 'UNIQUE constraint failed' in str(excinfo.value)


def test_write_smap_db_imerg_pk_violation(db_connection, smap_test_data):
    """
    Tests that an IntegrityError is raised when attempting to insert IMERG
    data with duplicate primary keys, validating the constraint.
    """
    duplicate_imerg_data = (
        'start_datetime,thru_datetime,column,row,value\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,1.5\n'
        '2023-01-01 00:00:00,2023-01-01 00:30:00,10,20,1.8\n'  # Duplicated PK
    )
    imerg_file = io.StringIO(duplicate_imerg_data)
    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        write_smap_db(smap_test_data, imerg_file, db_connection)
    assert 'UNIQUE constraint failed' in str(excinfo.value)
