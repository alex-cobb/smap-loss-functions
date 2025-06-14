"""Tests for choose_ease_grid"""

from unittest import mock

import numpy as np

import pytest

from smap_loss_functions.choose_ease_grid import (
    choose_ease_grid,
    write_ease_column_raster,
    write_ease_row_raster,
    write_geotiff,
    gridspec_covers_bbox,
    assert_gridspec_covers_bbox,
    create_ease2_gridspec,
    transform_lonlat_bbox_to_ease2,
    transform_lonlat_to_ease2,
    EASE_GRID_EPSG,
)


# For fixtures:  pylint: disable=redefined-outer-name


@pytest.fixture
def mock_ease_grid():
    """ease_lonlat.EASE2GRID for consistent testing without actual grid calculations"""
    mock_supported_grid_params = {'cellsize': 36000, 'h_pixels': 200, 'v_pixels': 200}
    with mock.patch(
        'smap_loss_functions.choose_ease_grid.EASE2GRID'
    ) as MockEASE2GRID, mock.patch(
        'smap_loss_functions.choose_ease_grid.SUPPORTED_GRIDS',
        {'EASE2_G36km': mock_supported_grid_params},
    ):
        mock_grid_instance = MockEASE2GRID.return_value
        # Mock lonlat2rc to return consistent row/column for a given lat/lon
        mock_grid_instance.lonlat2rc.side_effect = [
            (100, 50),  # For bbox['w'], bbox['n'] (e.g., 0, 90)
            (110, 60),  # For bbox['e'], bbox['s'] (e.g., 10, 80)
        ]
        # Mock rc2lonlat to return consistent lat/lon for a given row/column
        mock_grid_instance.rc2lonlat.side_effect = [
            (0.0, 90.0),  # col=100, row=50
            (1.0, 90.0),  # col=101, row=50
            (0.0, 89.0),  # col=100, row=51
            (1.0, 89.0),  # col=101, row=51
        ]
        yield MockEASE2GRID, mock_supported_grid_params


@pytest.fixture
def sample_bbox_4326():
    """Fixture for a sample bounding box in EPSG:4326"""
    return {'w': -10.0, 's': 0.0, 'e': 10.0, 'n': 20.0, 'srs': 'EPSG:4326'}


@pytest.fixture
def sample_gridspec():
    """Fixture for a sample gridspec"""
    return {
        'w': -100000.0,
        'n': 200000.0,
        'e': 100000.0,
        's': 0.0,
        'ncols': 200,
        'nrows': 200,
        'xres': 1000.0,
        'yres': -1000.0,  # yres is typically negative for North-up
        'srs': f'EPSG:{EASE_GRID_EPSG}',
    }


def test_transform_lonlat_to_ease2():
    """Test transform_lonlat_to_ease2"""
    with mock.patch('pyproj.Transformer') as MockTransformer:
        mock_transformer_instance = MockTransformer.from_crs.return_value
        mock_transformer_instance.transform.return_value = (500000.0, 1000000.0)

        lon, lat = 10.0, 50.0
        easex, easey = transform_lonlat_to_ease2(lon, lat)

        MockTransformer.from_crs.assert_called_once()
        mock_transformer_instance.transform.assert_called_once_with(lon, lat)
        assert easex == 500000.0
        assert easey == 1000000.0


def test_transform_lonlat_bbox_to_ease2(sample_bbox_4326):
    """Test transform_lonlat_bbox_to_ease2"""
    with mock.patch(
        'smap_loss_functions.choose_ease_grid.transform_lonlat_to_ease2',
        side_effect=[
            (-100000.0, 200000.0),  # w, n
            (100000.0, 0.0),  # e, s
        ],
    ) as mock_transform_point:
        ease_bbox = transform_lonlat_bbox_to_ease2(sample_bbox_4326)

        assert mock_transform_point.call_count == 2
        assert ease_bbox['w'] == -100000.0
        assert ease_bbox['n'] == 200000.0
        assert ease_bbox['e'] == 100000.0
        assert ease_bbox['s'] == 0.0
        assert ease_bbox['srs'] == f'EPSG:{EASE_GRID_EPSG}'


def test_create_ease2_gridspec(mock_ease_grid):
    """Test create_ease2_gridspec"""
    mock_ease_grid_instance, _ = mock_ease_grid

    start_col, start_row = 100, 50
    thru_col, thru_row = 101, 51

    mock_lon_coords = np.array([0.0, 1000.0, 0.0, 1000.0])
    mock_lat_coords = np.array([100000.0, 100000.0, 99000.0, 99000.0])
    with mock.patch(
        'smap_loss_functions.choose_ease_grid.transform_lonlat_to_ease2',
        return_value=(mock_lon_coords, mock_lat_coords),
    ):
        gridspec = create_ease2_gridspec(
            mock_ease_grid_instance.return_value,
            start_col,
            start_row,
            thru_col,
            thru_row,
        )

        assert gridspec['ncols'] == 2
        assert gridspec['nrows'] == 2
        # Expected xres and yres based on the mock transform values
        assert pytest.approx(gridspec['xres']) == 1000.0
        assert (
            pytest.approx(gridspec['yres']) == -1000.0
        )  # Negative because rows increase South
        assert pytest.approx(gridspec['w']) == -500.0  # min(easex) - xres/2
        assert (
            pytest.approx(gridspec['n']) == 100500.0
        )  # max(easey) - yres/2 (note: yres is negative)
        assert pytest.approx(gridspec['e']) == 1500.0  # max(easex) + xres/2
        assert (
            pytest.approx(gridspec['s']) == 98500.0
        )  # min(easey) + yres/2 (note: yres is negative)
        assert gridspec['srs'] == f'EPSG:{EASE_GRID_EPSG}'


def test_assert_gridspec_covers_bbox_covers(sample_gridspec):
    """Tests for assert_gridspec_covers_bbox and gridspec_covers_bbox"""
    bbox_covered = {
        'w': -90000.0,
        'n': 190000.0,
        'e': 90000.0,
        's': 10000.0,
        'srs': f'EPSG:{EASE_GRID_EPSG}',
    }
    assert (
        assert_gridspec_covers_bbox(sample_gridspec, bbox_covered) is None
    )  # Should not raise error
    assert gridspec_covers_bbox(sample_gridspec, bbox_covered) is True


def test_assert_gridspec_covers_bbox_does_not_cover(sample_gridspec):
    """Negative tests for assert_gridspec_covers_bbox and gridspec_covers_bbox"""
    bbox_covered = {
        'w': -90000.0,
        'n': 190000.0,
        'e': 90000.0,
        's': 10000.0,
        'srs': f'EPSG:{EASE_GRID_EPSG}',
    }
    for direction, bad_value in [
        ('w', -110000.0),
        ('n', 210000.0),
        ('e', 110000.0),
        ('s', -10000.0),
    ]:
        bad_bbox = bbox_covered.copy()
        bad_bbox[direction] = bad_value
        with pytest.raises(AssertionError):
            assert_gridspec_covers_bbox(sample_gridspec, bad_bbox)
        assert gridspec_covers_bbox(sample_gridspec, bad_bbox) is False


def test_write_geotiff(tmp_path, sample_gridspec):
    """Tests for write_geotiff"""
    outfile_path = tmp_path / 'test.tif'
    data = np.zeros(
        (sample_gridspec['nrows'], sample_gridspec['ncols']), dtype=np.int16
    )

    with mock.patch(
        'smap_loss_functions.choose_ease_grid.gdal'
    ) as MockGDAL, mock.patch('smap_loss_functions.choose_ease_grid.osr') as MockOSR:
        mock_srs_instance = mock.Mock()
        mock_srs_instance.ExportToWkt.return_value = 'PROJCS["EASE2_Global_36km",...]'
        MockOSR.SpatialReference.return_value = mock_srs_instance

        mock_driver = mock.Mock()
        MockGDAL.GetDriverByName.return_value = mock_driver

        mock_dataset = mock.Mock()
        mock_driver.Create.return_value = mock_dataset

        mock_band = mock.Mock()
        mock_dataset.GetRasterBand.return_value = mock_band

        write_geotiff(sample_gridspec, data, outfile_path)

        # Verify gdal methods were called
        MockGDAL.GetDriverByName.assert_called_once_with('GTiff')
        mock_driver.Create.assert_called_once()
        mock_dataset.SetGeoTransform.assert_called_once()
        MockOSR.SpatialReference.assert_called_once()
        mock_srs_instance.ImportFromEPSG.assert_called_once_with(EASE_GRID_EPSG)
        mock_dataset.SetProjection.assert_called_once_with(
            mock_srs_instance.ExportToWkt.return_value
        )
        mock_dataset.GetRasterBand.assert_called_once_with(
            1
        )  # Ensure this is called before SetNoDataValue/WriteArray
        mock_band.WriteArray.assert_called_once_with(data)


def test_write_ease_column_raster(tmp_path, sample_gridspec):
    """Test write_ease_column_raster"""
    outfile_path = tmp_path / 'cols.tif'
    start_col = 100
    thru_col = 100 + sample_gridspec['ncols'] - 1
    with mock.patch(
        'smap_loss_functions.choose_ease_grid.write_geotiff'
    ) as mock_write_geotiff:
        write_ease_column_raster(sample_gridspec, start_col, thru_col, outfile_path)
        mock_write_geotiff.assert_called_once()
        args, _ = mock_write_geotiff.call_args
        assert args[0] == sample_gridspec
        # Check that the data array has correct shape and values
        data_arg = args[1]
        assert data_arg.shape == (sample_gridspec['nrows'], sample_gridspec['ncols'])
        assert (data_arg[0, :] == list(range(start_col, thru_col + 1))).all()
        assert (data_arg[:, 0] == start_col).all()
        assert args[2] == outfile_path


def test_write_ease_row_raster(tmp_path, sample_gridspec):
    """Test write_ease_row_raster"""
    outfile_path = tmp_path / 'rows.tif'
    start_row = 50
    thru_row = 50 + sample_gridspec['nrows'] - 1
    with mock.patch(
        'smap_loss_functions.choose_ease_grid.write_geotiff'
    ) as mock_write_geotiff:
        write_ease_row_raster(sample_gridspec, start_row, thru_row, outfile_path)
        mock_write_geotiff.assert_called_once()
        args, _ = mock_write_geotiff.call_args
        assert args[0] == sample_gridspec
        # Check that the data array has correct shape and values
        data_arg = args[1]
        assert data_arg.shape == (sample_gridspec['nrows'], sample_gridspec['ncols'])
        assert (data_arg[:, 0] == list(range(start_row, thru_row + 1))).all()
        assert (data_arg[0, :] == start_row).all()
        assert args[2] == outfile_path


def test_choose_ease_grid(tmp_path, mock_ease_grid, sample_bbox_4326):
    """Test choose_ease_grid"""
    grid_name = 'EASE2_G36km'
    col_outfile_path = tmp_path / 'cols.tif'
    row_outfile_path = tmp_path / 'rows.tif'

    # Unpack the mock_ease_grid fixture's return values
    mock_ease_grid_class, mock_supported_grid_params = mock_ease_grid

    mock_outfile = mock.mock_open()
    mock_col_file_path = str(col_outfile_path)
    mock_row_file_path = str(row_outfile_path)

    with mock.patch('json.dump') as mock_json_dump, mock.patch(
        'smap_loss_functions.choose_ease_grid.transform_lonlat_bbox_to_ease2'
    ) as mock_transform_bbox, mock.patch(
        'smap_loss_functions.choose_ease_grid.create_ease2_gridspec'
    ) as mock_create_gridspec, mock.patch(
        'smap_loss_functions.choose_ease_grid.assert_gridspec_covers_bbox'
    ) as mock_assert_covers, mock.patch(
        'smap_loss_functions.choose_ease_grid.write_ease_column_raster'
    ) as mock_write_cols, mock.patch(
        'smap_loss_functions.choose_ease_grid.write_ease_row_raster'
    ) as mock_write_rows:
        # Set up return values for mocks
        mock_transform_bbox.return_value = {
            'w': -100000.0,
            'n': 200000.0,
            'e': 100000.0,
            's': 0.0,
            'srs': f'EPSG:{EASE_GRID_EPSG}',
        }
        mock_create_gridspec.return_value = {
            'w': -100000.0,
            'n': 200000.0,
            'e': 100000.0,
            's': 0.0,
            'ncols': 10,
            'nrows': 10,
            'xres': 10000.0,
            'yres': -10000.0,
            'srs': f'EPSG:{EASE_GRID_EPSG}',
        }

        result = choose_ease_grid(
            sample_bbox_4326,
            grid_name,
            mock_outfile,
            mock_col_file_path,
            mock_row_file_path,
        )

        # Assertions
        assert result == 0
        mock_ease_grid_class.assert_called_once_with(
            name=grid_name, **mock_supported_grid_params
        )

        mock_transform_bbox.assert_called_once_with(sample_bbox_4326)
        mock_create_gridspec.assert_called_once()
        mock_assert_covers.assert_called_once()
        mock_json_dump.assert_called_once_with(
            mock_create_gridspec.return_value, mock_outfile
        )
        mock_write_cols.assert_called_once()
        mock_write_rows.assert_called_once()
