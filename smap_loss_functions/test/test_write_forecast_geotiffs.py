"""Tests for write_forecast_geotiffs"""

import datetime
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from smap_loss_functions import write_forecast_geotiffs


@pytest.fixture
def in_memory_db():
    """Fixture for an in-memory SQLite database"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE smap_data (
            start_datetime TEXT,
            ease_col INTEGER,
            ease_row INTEGER,
            soil_moisture REAL
        )
    """)
    conn.commit()
    yield cursor
    conn.close()


@pytest.fixture
def temp_grid_tiffs():
    """Fixture for creating temporary GeoTIFF files for grid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define dummy grid dimensions
        rows, cols = 10, 10
        geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # Simple identity transform
        # Dummy WKT projection
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        projection_wkt = srs.ExportToWkt()

        # Dummy col and row data
        col_data = np.zeros((rows, cols), dtype=np.int32)
        row_data = np.zeros((rows, cols), dtype=np.int32)
        for i in range(rows):
            for j in range(cols):
                col_data[i, j] = j + 1
                row_data[i, j] = i + 1

        col_file_path = Path(tmpdir) / 'col.tif'
        row_file_path = Path(tmpdir) / 'row.tif'

        driver = gdal.GetDriverByName('GTiff')

        col_ds = driver.Create(str(col_file_path), cols, rows, 1, gdal.GDT_Int32)
        col_ds.SetGeoTransform(geotransform)
        col_ds.SetProjection(projection_wkt)
        col_ds.GetRasterBand(1).WriteArray(col_data)
        del col_ds

        row_ds = driver.Create(str(row_file_path), cols, rows, 1, gdal.GDT_Int32)
        row_ds.SetGeoTransform(geotransform)
        row_ds.SetProjection(projection_wkt)
        row_ds.GetRasterBand(1).WriteArray(row_data)
        del row_ds

        yield (
            col_file_path,
            row_file_path,
            geotransform,
            projection_wkt,
            col_data,
            row_data,
        )


def test_create_ease_index_lookup_table():
    """Test creation of EASE lookup table"""
    col_data = np.array([[1, 2], [1, 2]])
    row_data = np.array([[10, 10], [20, 20]])
    expected_lut = {
        (1, 10): (0, 0),
        (2, 10): (0, 1),
        (1, 20): (1, 0),
        (2, 20): (1, 1),
    }
    actual_lut = write_forecast_geotiffs.create_ease_index_lookup_table(
        col_data, row_data
    )
    assert actual_lut == expected_lut


def test_read_index_raster(tmp_path):
    """Test read_index_raster"""
    test_file = tmp_path / 'test_index.tif'
    rows, cols = 5, 5
    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    projection_wkt = srs.ExportToWkt()
    data = np.arange(25).reshape(rows, cols).astype(np.int32)

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(test_file), cols, rows, 1, gdal.GDT_Int32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)
    ds.GetRasterBand(1).WriteArray(data)
    del ds

    gt, arr, wkt = write_forecast_geotiffs.read_index_raster(test_file)
    assert gt == geotransform
    assert np.array_equal(arr, data)
    assert wkt == projection_wkt


def test_read_index_raster_non_integer_dtype_raises_error(tmp_path):
    """An index raster with non-integer dtype raises ValueError"""
    test_file = tmp_path / 'test_float.tif'
    rows, cols = 5, 5
    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    projection_wkt = srs.ExportToWkt()
    data = np.random.rand(rows, cols).astype(np.float32)

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(test_file), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)
    ds.GetRasterBand(1).WriteArray(data)
    del ds

    with pytest.raises(ValueError, match='Expected an integer dtype'):
        write_forecast_geotiffs.read_index_raster(test_file)


def test_get_grid_data(temp_grid_tiffs):
    """Test get_grid_data"""
    (
        col_file_path,
        row_file_path,
        expected_gt,
        expected_proj_wkt,
        expected_col_data,
        expected_row_data,
    ) = temp_grid_tiffs

    gt, proj_wkt, col_data, row_data = write_forecast_geotiffs.get_grid_data(
        col_file_path, row_file_path
    )

    assert gt == expected_gt
    assert proj_wkt == expected_proj_wkt
    assert np.array_equal(col_data, expected_col_data)
    assert np.array_equal(row_data, expected_row_data)


def test_write_geotiff(tmp_path):
    """Test write_geotiff"""
    output_filepath = tmp_path / 'output.tif'
    soil_moisture_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    projection_wkt = srs.ExportToWkt()

    write_forecast_geotiffs.write_geotiff(
        soil_moisture_data, dt, geotransform, output_filepath, projection_wkt
    )

    assert output_filepath.exists()

    # Verify contents
    ds = gdal.Open(str(output_filepath), gdal.GA_ReadOnly)
    assert ds is not None
    assert ds.RasterXSize == soil_moisture_data.shape[1]
    assert ds.RasterYSize == soil_moisture_data.shape[0]
    assert ds.GetGeoTransform() == geotransform
    assert ds.GetProjection() == projection_wkt

    band = ds.GetRasterBand(1)
    read_data = band.ReadAsArray()
    assert np.array_equal(read_data, soil_moisture_data)

    metadata = ds.GetMetadata()
    assert metadata['TB_MEAN_DATETIME'] == dt.isoformat(
        sep=' ', timespec='milliseconds'
    )
    assert metadata['units'] == 'm3/m3'
    assert (
        metadata['ImageDescription']
        == 'SMAP soil moisture forecast using loss functions. Units: m3/m3'
    )
    # Check DateTime format (YYYY:MM:DD HH:MM:SS)
    assert metadata['DateTime'] == dt.replace(tzinfo=None).isoformat(
        sep=' ', timespec='seconds'
    )
    del ds


def test_write_forecast_geotiffs_success(in_memory_db, tmp_path, temp_grid_tiffs):
    cursor = in_memory_db
    col_file_path, row_file_path, geotransform, projection_wkt, col_data, row_data = (
        temp_grid_tiffs
    )

    # Populate the in-memory database with dummy data
    # 5 days are required
    today = datetime.datetime(2023, 6, 14, 12, 0, 0)
    for i in range(5):
        dt = today - datetime.timedelta(days=i)
        for c_idx in range(5):  # Use a subset of the grid
            for r_idx in range(5):
                # Ensure col/row exist in grid data
                col = c_idx + 1
                row = r_idx + 1
                soil_moisture = 0.2 + (c_idx * 0.01) + (r_idx * 0.005) + (i * 0.001)
                cursor.execute(
                    'INSERT INTO smap_data '
                    '(start_datetime, ease_col, ease_row, soil_moisture) '
                    'VALUES (?, ?, ?, ?)',
                    (
                        dt.isoformat(sep=' ', timespec='milliseconds'),
                        col,
                        row,
                        soil_moisture,
                    ),
                )
    cursor.connection.commit()

    outfile_pattern = str(tmp_path / 'forecast_{}.tif')

    write_forecast_geotiffs.write_forecast_geotiffs(
        cursor, geotransform, projection_wkt, col_data, row_data, outfile_pattern
    )

    # Verify that 5 GeoTIFF files were created
    expected_files = []
    for i in range(5):
        dt = today - datetime.timedelta(days=i)
        expected_files.append(tmp_path / f'forecast_{dt.strftime("%Y%m%dT%H%M")}.tif')

    for f in expected_files:
        assert f.exists(), f'File {f} was not created'
        # Basic check on content
        ds = gdal.Open(str(f))
        assert ds is not None
        assert ds.RasterXSize == col_data.shape[1]
        assert ds.RasterYSize == col_data.shape[0]
        ds = None


def test_write_forecast_geotiffs_no_data_in_db_raises_error(
    in_memory_db, tmp_path, temp_grid_tiffs
):
    """ValueError is raised if there is no data in the db"""
    cursor = in_memory_db
    col_file_path, row_file_path, geotransform, projection_wkt, col_data, row_data = (
        temp_grid_tiffs
    )
    outfile_pattern = str(tmp_path / 'forecast_{}.tif')

    with pytest.raises(ValueError, match='No data in smap_data'):
        write_forecast_geotiffs.write_forecast_geotiffs(
            cursor, geotransform, projection_wkt, col_data, row_data, outfile_pattern
        )


def test_write_forecast_geotiffs_missing_expected_daily_data(
    in_memory_db, tmp_path, temp_grid_tiffs, caplog
):
    """Expect data in database are available on a daily grid for 5 days"""
    cursor = in_memory_db
    col_file_path, row_file_path, geotransform, projection_wkt, col_data, row_data = (
        temp_grid_tiffs
    )

    # Populate with only one day of data
    latest_dt = datetime.datetime(2023, 6, 14, 12, 0, 0)
    cursor.execute(
        'INSERT INTO smap_data (start_datetime, ease_col, ease_row, soil_moisture) '
        'VALUES (?, ?, ?, ?)',
        (
            latest_dt.isoformat(sep=' ', timespec='milliseconds'),
            1,
            1,
            0.3,
        ),  # col=1, row=1 is in the dummy grid
    )
    cursor.connection.commit()

    outfile_pattern = str(tmp_path / 'forecast_{}.tif')

    # The function expects 5 days; missing days will raise AssertionError
    with pytest.raises(AssertionError, match='No data at'):
        write_forecast_geotiffs.write_forecast_geotiffs(
            cursor, geotransform, projection_wkt, col_data, row_data, outfile_pattern
        )
