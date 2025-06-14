"""Tests for smap_db"""

import numpy as np

import pytest

from smap_loss_functions.smap_db import (
    get_soil_moisture,
    get_precipitation,
    get_imerg_interpolant,
    create_piecewise_constant_interpolant,
    clip_to_timestamp_range,
)


def test_get_soil_moisture(in_memory_smap_db):
    """Test get_soil_moisture retrieves data correctly."""
    cursor = in_memory_smap_db.cursor()
    col, row = 10, 20
    smap_time, soil_moisture = get_soil_moisture(cursor, col, row)

    assert len(smap_time) == 4
    assert len(soil_moisture) == 4
    assert np.allclose(soil_moisture, [0.25, 0.28, 0.27, 0.30])
    # Check timestamps: 2023-01-01 12:00:00Z -> 1672574400.0 (middle of day)
    # Subsequent days are 86400 seconds later
    expected_times = np.array(
        [
            1672574400.0,
            1672574400.0 + 86400,
            1672574400.0 + 2 * 86400,
            1672574400.0 + 3 * 86400,
        ]
    )
    print([int(v) for v in smap_time])
    assert np.allclose(smap_time, expected_times)


def test_get_precipitation(in_memory_smap_db):
    """Test get_precipitation retrieves data correctly."""
    cursor = in_memory_smap_db.cursor()
    col, row = 10, 20
    imerg_start, precipitation = get_precipitation(cursor, col, row)

    assert len(imerg_start) == 6
    assert len(precipitation) == 6
    assert np.allclose(precipitation, [5.0, 10.0, 1.0, 0.0, 20.0, 2.0])
    # Check timestamps: 2022-12-31 00:00:00 -> 1672434000.0
    expected_times = np.array(
        [
            1672434000.0,
            1672434000.0 + 86400,
            1672434000.0 + 2 * 86400,
            1672434000.0 + 3 * 86400,
            1672434000.0 + 4 * 86400,
            1672434000.0 + 5 * 86400,
        ]
    )
    assert np.allclose(imerg_start, expected_times)


def test_get_imerg_interpolant(in_memory_smap_db):
    """Test get_imerg_interpolant"""
    cursor = in_memory_smap_db.cursor()
    col, row = 10, 20
    # Probe times around SMAP data interval (1 day before/after SMAP min/max)
    start_time_smap_min = 1672574400.0  # epoch seconds for 2023-01-01 12:00:00Z
    thru_time_smap_max = 1672833600.0  # epoch seconds for 2023-01-04 12:00:00Z
    # Add/subtract 129600 (2 days in seconds) as per get_imerg_interpolant's call
    query_start_time = start_time_smap_min - 129600
    query_thru_time = thru_time_smap_max + 129600

    interpolant = get_imerg_interpolant(
        cursor, col, row, query_start_time, query_thru_time
    )

    # Expected IMERG data for col 10, row 20, after clipping:
    # 2022-12-31 00:00:00Z (1672444800.0) -> 5.0
    # 2023-01-01 00:00:00Z (1672574400.0) -> 10.0
    # ...
    # 2023-01-05 00:00:00Z (1672876800.0) -> 2.0

    # Test at specific timestamps (hourly resolution is 3600s)
    # A time just after IMERG data starts (2022-12-31 00:00:00Z)
    assert interpolant(1672444800.0 + 100) == 5.0
    # A time just before next IMERG data point (2023-01-01 00:00:00Z)
    assert interpolant(1672531200.0 - 100) == 5.0
    # A time just after next IMERG data point
    assert interpolant(1672617600.0 + 100) == 1.0
    # At the exact end of the queried range (2023-01-05 12:00:00Z)
    assert interpolant(query_thru_time) == 2.0

    # Test extrapolation outside the *queried* range (should return edge values)
    assert interpolant(query_start_time - 1000) == 5.0
    assert interpolant(query_thru_time + 1000) == 2.0


def test_create_piecewise_constant_interpolant():
    """Test create_piecewise_constant_interpolant."""
    x = np.array([10, 20, 30])
    y = np.array([1, 5, 10])
    interpolant = create_piecewise_constant_interpolant(x, y)

    assert interpolant(5) == 1  # Less than x[0]
    assert interpolant(10) == 1  # At x[0]
    assert interpolant(15) == 1  # Between x[0] and x[1]
    assert interpolant(20) == 5  # At x[1]
    assert interpolant(25) == 5  # Between x[1] and x[2]
    assert interpolant(30) == 10  # At x[2]
    assert interpolant(35) == 10  # Greater than x[-1]

    # Test with single point
    x_single = np.array([100])
    y_single = np.array([50])
    interpolant_single = create_piecewise_constant_interpolant(x_single, y_single)
    assert interpolant_single(0) == 50
    assert interpolant_single(100) == 50
    assert interpolant_single(200) == 50

    # Test assertion for non-strictly increasing x
    with pytest.raises(AssertionError, match='not strictly increasing'):
        create_piecewise_constant_interpolant(np.array([10, 5]), np.array([1, 2]))


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
