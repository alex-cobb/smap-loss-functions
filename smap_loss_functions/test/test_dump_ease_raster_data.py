"""Tests for dump_ease_raster_data"""

import io
import os
import tempfile

import numpy as np

from osgeo import gdal

import pytest

from smap_loss_functions.dump_ease_raster_data import (
    dump_ease_raster_data,
    read_raster_data,
)

gdal.UseExceptions()


# Helper function to create a dummy GeoTIFF file
def create_dummy_geotiff(
    filepath, data, geotransform, nodata_value=None, metadata=None
):
    """Creates a single-band GeoTIFF file."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    if metadata:
        dataset.SetMetadata(metadata)
    band = dataset.GetRasterBand(1)
    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)
    band.WriteArray(data)
    dataset.FlushCache()  # Write to disk
    del dataset  # Close the dataset


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def dummy_col_row_files(temp_dir):
    """Creates dummy col and row GeoTIFF files."""
    geotransform = (
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
    )  # (ulx, xres, xskew, uly, yskew, yres)

    col_data = np.array([[0, 1], [0, 1]], dtype=np.float32)
    row_data = np.array([[0, 0], [1, 1]], dtype=np.float32)

    col_filepath = os.path.join(temp_dir, 'col_data.tif')
    row_filepath = os.path.join(temp_dir, 'row_data.tif')

    create_dummy_geotiff(col_filepath, col_data, geotransform)
    create_dummy_geotiff(row_filepath, row_data, geotransform)

    return col_filepath, row_filepath, geotransform, col_data.shape


# Test cases for read_raster_data
def test_read_raster_data_no_nodata(temp_dir):
    """Test read_raster_data with a file that has no NoDataValue set."""
    filepath = os.path.join(temp_dir, 'test_no_nodata.tif')
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    geotransform = (10.0, 1.0, 0.0, 20.0, 0.0, -1.0)
    metadata = {'TEST_META': 'value'}
    create_dummy_geotiff(filepath, data, geotransform, metadata=metadata)

    gt, arr, meta = read_raster_data(filepath)

    assert gt == geotransform
    assert np.array_equal(arr, data)
    assert not np.ma.is_masked(arr)
    assert meta == metadata


def test_read_raster_data_with_nodata_masked_equal(temp_dir):
    """Test read_raster_data with a file that has a specific NoDataValue."""
    filepath = os.path.join(temp_dir, 'test_with_nodata.tif')
    data = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
    geotransform = (0.0, 0.5, 0.0, 0.0, 0.0, -0.5)
    nodata_value = -9999.0
    create_dummy_geotiff(filepath, data, geotransform, nodata_value=nodata_value)

    gt, arr, _ = read_raster_data(filepath)

    assert gt == geotransform
    assert np.ma.is_masked(arr)
    assert arr[0, 1] is np.ma.masked
    assert arr[0, 0] == 1.0
    assert np.ma.getdata(arr).shape == data.shape
    assert np.ma.getmask(arr)[0, 1]
    assert not np.ma.getmask(arr)[0, 0]


def test_read_raster_data_with_nodata_masked_invalid(temp_dir):
    """Test read_raster_data with a file that has NaN as NoDataValue."""
    filepath = os.path.join(temp_dir, 'test_nan_nodata.tif')
    data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    geotransform = (0.0, 0.5, 0.0, 0.0, 0.0, -0.5)
    nodata_value = np.nan
    create_dummy_geotiff(filepath, data, geotransform, nodata_value=nodata_value)

    gt, arr, _ = read_raster_data(filepath)

    assert gt == geotransform
    assert np.ma.is_masked(arr)
    assert np.ma.getmask(arr)[0, 1]  # The NaN value should be masked


# Test cases for dump_ease_raster_data
def test_dump_ease_raster_data_basic(dummy_col_row_files, temp_dir):
    """Test dump_ease_raster_data with basic valid inputs."""
    col_filepath, row_filepath, geotransform, shape = dummy_col_row_files
    rows, cols = shape

    # Create dummy data input files
    infile_list = []
    expected_output = 'start_datetime,thru_datetime,column,row,value\n'

    # File 1
    data1 = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    metadata1 = {
        'TIME_BNDS_0': '2023-01-01T00:00:00Z',
        'TIME_BNDS_1': '2023-01-01T23:59:59Z',
    }
    infile1_path = os.path.join(temp_dir, 'data_1.tif')
    create_dummy_geotiff(infile1_path, data1, geotransform, metadata=metadata1)
    infile_list.append(infile1_path)
    # Expected output for file 1 (col, row, value)
    expected_output += '2023-01-01T00:00:00Z,2023-01-01T23:59:59Z,0.0,0.0,10.0\n'
    expected_output += '2023-01-01T00:00:00Z,2023-01-01T23:59:59Z,1.0,0.0,11.0\n'
    expected_output += '2023-01-01T00:00:00Z,2023-01-01T23:59:59Z,0.0,1.0,12.0\n'
    expected_output += '2023-01-01T00:00:00Z,2023-01-01T23:59:59Z,1.0,1.0,13.0\n'

    # File 2
    data2 = np.array([[20.0, 21.0], [22.0, 23.0]], dtype=np.float32)
    metadata2 = {
        'TIME_BNDS_0': '2023-01-02T00:00:00Z',
        'TIME_BNDS_1': '2023-01-02T23:59:59Z',
    }
    infile2_path = os.path.join(temp_dir, 'data_2.tif')
    create_dummy_geotiff(infile2_path, data2, geotransform, metadata=metadata2)
    infile_list.append(infile2_path)
    # Expected output for file 2
    expected_output += '2023-01-02T00:00:00Z,2023-01-02T23:59:59Z,0.0,0.0,20.0\n'
    expected_output += '2023-01-02T00:00:00Z,2023-01-02T23:59:59Z,1.0,0.0,21.0\n'
    expected_output += '2023-01-02T00:00:00Z,2023-01-02T23:59:59Z,0.0,1.0,22.0\n'
    expected_output += '2023-01-02T00:00:00Z,2023-01-02T23:59:59Z,1.0,1.0,23.0\n'

    output_buffer = io.StringIO()
    result = dump_ease_raster_data(
        col_filepath, row_filepath, infile_list, output_buffer
    )

    assert result == 0
    assert output_buffer.getvalue() == expected_output


def test_dump_ease_raster_data_with_nodata_in_input_file(dummy_col_row_files, temp_dir):
    """Test dump_ease_raster_data when an input file has NoDataValue."""
    col_filepath, row_filepath, geotransform, shape = dummy_col_row_files
    rows, cols = shape

    infile_list = []
    expected_output = 'start_datetime,thru_datetime,column,row,value\n'

    # File with a NoDataValue
    data_with_nodata = np.array([[10.0, -9999.0], [12.0, 13.0]], dtype=np.float32)
    nodata_value = -9999.0
    metadata = {
        'TIME_BNDS_0': '2023-03-01T00:00:00Z',
        'TIME_BNDS_1': '2023-03-01T23:59:59Z',
    }
    infile_path = os.path.join(temp_dir, 'data_with_nodata.tif')
    create_dummy_geotiff(
        infile_path,
        data_with_nodata,
        geotransform,
        nodata_value=nodata_value,
        metadata=metadata,
    )
    infile_list.append(infile_path)

    # Output should *not* include the masked value
    expected_output += '2023-03-01T00:00:00Z,2023-03-01T23:59:59Z,0.0,0.0,10.0\n'
    expected_output += '2023-03-01T00:00:00Z,2023-03-01T23:59:59Z,0.0,1.0,12.0\n'
    expected_output += '2023-03-01T00:00:00Z,2023-03-01T23:59:59Z,1.0,1.0,13.0\n'

    output_buffer = io.StringIO()
    result = dump_ease_raster_data(
        col_filepath, row_filepath, infile_list, output_buffer
    )

    assert result == 0
    assert output_buffer.getvalue() == expected_output


def test_dump_ease_raster_data_empty_infile_list(dummy_col_row_files, temp_dir):
    """Test dump_ease_raster_data with an empty list of input files."""
    col_filepath, row_filepath, geotransform, shape = dummy_col_row_files
    infile_list = []
    expected_output = 'start_datetime,thru_datetime,column,row,value\n'

    output_buffer = io.StringIO()
    result = dump_ease_raster_data(
        col_filepath, row_filepath, infile_list, output_buffer
    )

    assert result == 0
    assert output_buffer.getvalue() == expected_output


def test_dump_ease_raster_data_mismatched_geotransform(dummy_col_row_files, temp_dir):
    """Test dump_ease_raster_data with mismatched geotransform."""
    col_filepath, row_filepath, _, shape = dummy_col_row_files

    # Create a data file with a different geotransform
    mismatched_geotransform = (100.0, 1.0, 0.0, 100.0, 0.0, -1.0)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metadata = {
        'TIME_BNDS_0': '2023-01-01T00:00:00Z',
        'TIME_BNDS_1': '2023-01-01T23:59:59Z',
    }
    infile_path = os.path.join(temp_dir, 'mismatched_data.tif')
    create_dummy_geotiff(infile_path, data, mismatched_geotransform, metadata=metadata)
    infile_list = [infile_path]

    output_buffer = io.StringIO()
    with pytest.raises(AssertionError) as excinfo:
        dump_ease_raster_data(col_filepath, row_filepath, infile_list, output_buffer)

    assert 'not close to' in str(
        excinfo.value
    )  # Checks for the message from assert_close


def test_dump_ease_raster_data_mismatched_shape(dummy_col_row_files, temp_dir):
    """Test dump_ease_raster_data with mismatched shape in input file."""
    col_filepath, row_filepath, geotransform, _ = dummy_col_row_files

    # Create a data file with a different shape
    mismatched_data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )  # 2x3 instead of 2x2
    metadata = {
        'TIME_BNDS_0': '2023-01-01T00:00:00Z',
        'TIME_BNDS_1': '2023-01-01T23:59:59Z',
    }
    infile_path = os.path.join(temp_dir, 'mismatched_shape.tif')
    create_dummy_geotiff(infile_path, mismatched_data, geotransform, metadata=metadata)
    infile_list = [infile_path]

    output_buffer = io.StringIO()
    with pytest.raises(AssertionError) as excinfo:
        dump_ease_raster_data(col_filepath, row_filepath, infile_list, output_buffer)

    assert '!= (' in str(
        excinfo.value
    )  # Check for the message from assert_equal for shape
