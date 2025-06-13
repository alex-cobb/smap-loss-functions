"""Test smap_to_raster_template"""

import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, DEFAULT

from smap_loss_functions import smap_to_raster_template


def compare_dictionaries(d1, d2, path=''):
    """
    Utility function to compare two dictionaries and print detailed differences.
    Useful for debugging AssertionErrors in dictionary comparisons.
    """
    errors = []

    # Check keys in d1 but not in d2
    for k in d1:
        if k not in d2:
            errors.append(f"Key '{path}{k}' present in first dict but not in second.")

    # Check keys in d2 but not in d1
    for k in d2:
        if k not in d1:
            errors.append(f"Key '{path}{k}' present in second dict but not in first.")

    # Compare common keys
    for k in d1:
        if k in d2:
            v1 = d1[k]
            v2 = d2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                errors.extend(compare_dictionaries(v1, v2, path=f'{path}{k}.'))
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2, equal_nan=True):
                    errors.append(f"Value mismatch for key '{path}{k}': {v1} != {v2}")
            elif v1 != v2:
                errors.append(f"Value mismatch for key '{path}{k}': {v1!r} != {v2!r}")
    return errors


@pytest.fixture
def mock_h5_dataset():
    """Fixture to mock an h5py dataset."""
    mock_dataset = MagicMock()
    mock_dataset.filename = '/path/to/mock_smap_file.h5'

    # Mock Metadata/DatasetIdentification and its attrs
    mock_metadata_id = MagicMock()
    mock_metadata_id.attrs = {
        'SMAPShortName': 'L3_SM_P_E'
    }  # Set attrs directly as a dict

    # Mock the SMAP data group (e.g., Soil_Moisture_Retrieval_Data_AM)
    mock_sm_group = MagicMock()

    # Define the actual numpy arrays for data (2D to match template dimensions indirectly)
    mock_ease_col_data = np.array([[100, 101], [102, 103]])  # Adjusted to 2x2
    mock_ease_row_data = np.array([[200, 201], [202, 203]])  # Adjusted to 2x2
    mock_tb_time_utc_data = np.array(  # Adjusted to 4 elements for 2x2
        [
            b'2023-01-01T12:00:00.000Z',
            b'2023-01-01T12:00:01.000Z',
            b'2023-01-01T12:00:02.000Z',
            b'2023-01-01T12:00:03.000Z',
        ]
    )
    raw_sm_data = np.array([[0.1, -9999.0], [0.3, 0.4]])  # Adjusted to 2x2

    # Mock EASE_column_index and EASE_row_index
    mock_sm_group.configure_mock(
        **{
            'EASE_column_index': MagicMock(
                __getitem__=lambda s, k: mock_ease_col_data
            ),  # Simulates [:] access
            'EASE_row_index': MagicMock(__getitem__=lambda s, k: mock_ease_row_data),
            'tb_time_utc': MagicMock(__getitem__=lambda s, k: mock_tb_time_utc_data),
        }
    )

    # Mock soil_moisture dataset
    mock_soil_moisture_h5_group = MagicMock()
    mock_soil_moisture_h5_group.attrs = {'_FillValue': -9999.0}
    mock_soil_moisture_h5_group.configure_mock(
        **{
            '__getitem__': lambda s, k: raw_sm_data,  # Simulates [:] access
            'filled': lambda val: np.ma.masked_array(
                raw_sm_data, mask=(raw_sm_data == -9999.0)
            ).filled(val),
        }
    )
    mock_sm_group.configure_mock(**{'soil_moisture': mock_soil_moisture_h5_group})

    # Configure the main dataset's __getitem__ to return appropriate mocks
    # This ensures nested dictionary-like access works correctly
    mock_dataset.__getitem__.side_effect = lambda key: {
        'Metadata/DatasetIdentification': mock_metadata_id,
        'Soil_Moisture_Retrieval_Data_AM': mock_sm_group,
        'L2_SM_P': mock_sm_group,  # Added for L2_SM_P short name
    }.get(key, DEFAULT)  # Use DEFAULT to raise AttributeError for unhandled keys

    return mock_dataset


@pytest.fixture
def mock_index_raster_data():
    """Fixture for mock index raster data (col_data, row_data, geotransform)."""
    mock_geotransform = (10.0, 0.5, 0.0, 20.0, 0.0, -0.5)
    mock_col_data = np.array(
        [[100, 101], [999, 102]], dtype=np.int16
    )  # 999 is a dummy index that won't match
    mock_row_data = np.array([[200, 201], [999, 202]], dtype=np.int16)
    return mock_geotransform, mock_col_data, mock_row_data


@pytest.fixture(autouse=True)
def patch_external_libs():
    """
    Patch external libraries like GDAL, OSR, pyproj, and datetime.
    autouse=True means this fixture runs automatically for all tests.
    """
    patches = []

    # Patch GDAL
    mock_gdal_patch = patch('smap_loss_functions.smap_to_raster_template.gdal')
    mock_gdal = mock_gdal_patch.start()  # Store the actual mock object
    patches.append(mock_gdal_patch)  # Append the patcher object
    # Configure the started mock using 'mock_gdal'
    mock_gdal.GetDriverByName.return_value = MagicMock()
    mock_gdal.GetDriverByName.return_value.Create.return_value = MagicMock()
    mock_gdal.Open.return_value = MagicMock()
    mock_gdal.Open.return_value.GetRasterBand.return_value = MagicMock()

    # Patch OSR
    mock_osr_patch = patch('smap_loss_functions.smap_to_raster_template.osr')
    mock_osr = mock_osr_patch.start()
    patches.append(mock_osr_patch)
    mock_osr.SpatialReference.return_value = MagicMock()

    # Patch numpy's nan related functions for compute_datetimes
    patch_np_nan = patch.object(
        smap_to_raster_template.np, 'nan', return_value=float('nan')
    )
    patch_np_nan.start()
    patches.append(patch_np_nan)

    patch_np_isnan = patch.object(smap_to_raster_template.np, 'isnan', wraps=np.isnan)
    patch_np_isnan.start()
    patches.append(patch_np_isnan)

    patch_np_nanmean = patch.object(
        smap_to_raster_template.np, 'nanmean', wraps=np.nanmean
    )
    patch_np_nanmean.start()
    patches.append(patch_np_nanmean)

    patch_np_nanmin = patch.object(
        smap_to_raster_template.np, 'nanmin', wraps=np.nanmin
    )
    patch_np_nanmin.start()
    patches.append(patch_np_nanmin)

    patch_np_nanmax = patch.object(
        smap_to_raster_template.np, 'nanmax', wraps=np.nanmax
    )
    patch_np_nanmax.start()
    patches.append(patch_np_nanmax)

    patch_np_empty = patch.object(smap_to_raster_template.np, 'empty', wraps=np.empty)
    patch_np_empty.start()
    patches.append(patch_np_empty)

    patch_np_masked_array = patch.object(
        smap_to_raster_template.np.ma, 'masked_array', wraps=np.ma.masked_array
    )
    patch_np_masked_array.start()
    patches.append(patch_np_masked_array)

    yield

    # Clean up patches after the test
    for p in reversed(patches):
        p.stop()


def test_timestamp_from_utc_isoformat_valid():
    """Test with a valid UTC ISO format string."""
    dt_str = '2023-01-01T12:30:00.000Z'
    result = smap_to_raster_template.timestamp_from_utc_isoformat(dt_str)
    assert result == 1672576200.0


def test_timestamp_from_utc_isoformat_invalid_format():
    """Test with an invalid ISO format string."""
    dt_str = 'invalid-datetime-string'
    # No patching of datetime.datetime needed here, using real datetime

    result = smap_to_raster_template.timestamp_from_utc_isoformat(dt_str)
    assert np.isnan(result)


def test_timestamp_from_utc_isoformat_non_utc():
    """Test with a non-UTC datetime string (should raise ValueError)."""
    dt_str = '2023-01-01T12:30:00.000+01:00'  # Not UTC
    # No patching of datetime.datetime needed here, using real datetime

    with pytest.raises(ValueError, match='does not specify that it is UTC'):
        smap_to_raster_template.timestamp_from_utc_isoformat(dt_str)


def test_compute_datetimes_valid_data():
    """Test compute_datetimes with valid datetime strings."""
    # We will use actual timestamps in the assertions, as the function calls real datetime now.
    expected_timestamps = np.array(
        [
            datetime.datetime.fromisoformat(b'2023-01-01T12:00:00.000Z'.decode('ascii'))
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp(),
            datetime.datetime.fromisoformat(b'2023-01-01T12:01:00.000Z'.decode('ascii'))
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp(),
            datetime.datetime.fromisoformat(b'2023-01-01T12:02:00.000Z'.decode('ascii'))
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp(),
        ]
    )

    # No patch for timestamp_from_utc_isoformat or np.array needed here, as the function calls real datetime
    datetimes_bytes = [
        b'2023-01-01T12:00:00.000Z',
        b'2023-01-01T12:01:00.000Z',
        b'2023-01-01T12:02:00.000Z',
    ]
    result = smap_to_raster_template.compute_datetimes(datetimes_bytes)

    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'start' in result
    assert 'thru' in result

    assert result['mean'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime.fromtimestamp(
        np.mean(expected_timestamps), tz=datetime.timezone.utc
    ).isoformat(sep=' ', timespec='milliseconds')
    assert result['start'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime.fromtimestamp(
        np.min(expected_timestamps), tz=datetime.timezone.utc
    ).isoformat(sep=' ', timespec='milliseconds')
    assert result['thru'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime.fromtimestamp(
        np.max(expected_timestamps), tz=datetime.timezone.utc
    ).isoformat(sep=' ', timespec='milliseconds')


def test_compute_datetimes_no_valid_data():
    """Test compute_datetimes when no valid datetime strings are found."""
    # No patch for timestamp_from_utc_isoformat needed here, using real datetime.
    datetimes_bytes = [b'invalid', b'another-invalid']
    result = smap_to_raster_template.compute_datetimes(datetimes_bytes)

    # Expect epoch if no valid datetimes
    assert result['mean'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).isoformat(
        sep=' ', timespec='milliseconds'
    )
    assert result['start'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).isoformat(
        sep=' ', timespec='milliseconds'
    )
    assert result['thru'].isoformat(
        sep=' ', timespec='milliseconds'
    ) == datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).isoformat(
        sep=' ', timespec='milliseconds'
    )


def test_get_masked_variable():
    """Test retrieving a variable as a masked array."""
    mock_h5_group = MagicMock()
    mock_h5_group.attrs = {'_FillValue': -9999.0}
    mock_h5_group.__getitem__.return_value = np.array(
        [0.1, 0.2, -9999.0, 0.4]
    )  # Simulate [:] access

    result = smap_to_raster_template.get_masked_variable(mock_h5_group)

    assert isinstance(result, np.ma.MaskedArray)
    assert result[0] == 0.1
    assert result[1] == 0.2
    assert result[3] == 0.4
    assert result.mask[2]  # Changed 'is True' to direct boolean check
    assert np.all(result.data == np.array([0.1, 0.2, -9999.0, 0.4]))


def test_read_index_raster():
    """Test reading geotransform and data from an index raster."""
    mock_geotransform = (10.0, 0.5, 0.0, 20.0, 0.0, -0.5)
    mock_band_data = np.array([[1, 2], [3, 4]], dtype=np.int16)

    with patch(
        'smap_loss_functions.smap_to_raster_template.gdal.Open'
    ) as mock_gdal_open:
        mock_dataset = mock_gdal_open.return_value
        mock_dataset.GetGeoTransform.return_value = mock_geotransform
        mock_band = mock_dataset.GetRasterBand.return_value
        mock_band.ReadAsArray.return_value = mock_band_data

        geotransform, data = smap_to_raster_template.read_index_raster(
            '/path/to/col_file.tif'
        )

        mock_gdal_open.assert_called_once_with(
            '/path/to/col_file.tif', smap_to_raster_template.gdal.GA_ReadOnly
        )
        mock_dataset.GetRasterBand.assert_called_once_with(1)
        mock_band.ReadAsArray.assert_called_once()
        assert geotransform == mock_geotransform
        assert np.array_equal(data, mock_band_data)
        assert data.dtype == np.dtype('int16')


def test_read_index_raster_wrong_dtype():
    """Test reading index raster with incorrect dtype (should fail assertion)."""
    mock_band_data_float = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    with patch(
        'smap_loss_functions.smap_to_raster_template.gdal.Open'
    ) as mock_gdal_open:
        mock_dataset = mock_gdal_open.return_value
        mock_dataset.GetGeoTransform.return_value = (10.0, 0.5, 0.0, 20.0, 0.0, -0.5)
        mock_band = mock_dataset.GetRasterBand.return_value
        mock_band.ReadAsArray.return_value = mock_band_data_float

        with pytest.raises(AssertionError):
            smap_to_raster_template.read_index_raster('/path/to/col_file.tif')


def test_write_geotiff():
    """Test writing soil moisture data to a GeoTIFF."""
    # Initialize without np.nan in literal to avoid potential ValueError
    soil_moisture = np.array([[0.1, 0.2], [0.3, 0.0]], dtype=np.float32)
    soil_moisture[1, 1] = float('nan')  # Explicitly set NaN

    geotransform = (10.0, 0.5, 0.0, 20.0, 0.0, -0.5)
    # These datetimes are actual datetime objects for this test
    datetimes = {
        'mean': datetime.datetime(2023, 1, 1, 12, 1, 0, tzinfo=datetime.timezone.utc),
        'start': datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        'thru': datetime.datetime(2023, 1, 1, 12, 2, 0, tzinfo=datetime.timezone.utc),
    }
    source_filename = 'mock_smap_file.h5'
    outfile_path = '/path/to/output.tif'

    with patch(
        'smap_loss_functions.smap_to_raster_template.gdal.GetDriverByName'
    ) as mock_get_driver_by_name, patch(
        'smap_loss_functions.smap_to_raster_template.osr.SpatialReference'
    ) as mock_osr_spatial_reference:
        mock_driver = mock_get_driver_by_name.return_value
        mock_dataset = mock_driver.Create.return_value
        mock_srs = mock_osr_spatial_reference.return_value
        mock_band = mock_dataset.GetRasterBand.return_value
        mock_dataset.GetMetadata.return_value = {}  # Ensure GetMetadata returns a mutable dict

        result = smap_to_raster_template.write_geotiff(
            soil_moisture, geotransform, datetimes, source_filename, outfile_path
        )

        mock_get_driver_by_name.assert_called_once_with('GTiff')
        mock_driver.Create.assert_called_once_with(
            outfile_path,
            soil_moisture.shape[1],
            soil_moisture.shape[0],
            1,
            smap_to_raster_template.gdal.GDT_Float32,
            options=['COMPRESS=DEFLATE', 'PREDICTOR=3'],
        )
        mock_dataset.SetGeoTransform.assert_called_once_with(geotransform)
        mock_srs.ImportFromEPSG.assert_called_once_with(
            smap_to_raster_template.EASE_GRID_EPSG
        )
        mock_dataset.SetProjection.assert_called_once_with(
            mock_srs.ExportToWkt.return_value
        )
        # Check that the argument passed to SetNoDataValue is a float and is NaN
        assert (
            len(mock_band.SetNoDataValue.call_args.args) > 0
            and isinstance(mock_band.SetNoDataValue.call_args.args[0], float)
            and np.isnan(mock_band.SetNoDataValue.call_args.args[0])
        )
        mock_band.WriteArray.assert_called_once()
        np.testing.assert_array_equal(
            mock_band.WriteArray.call_args[0][0], soil_moisture
        )

        # Check metadata - now it relies on the actual `datetimes` objects passed
        expected_metadata_dict = {
            'TB_MEAN_DATETIME': datetimes['mean'].isoformat(
                sep=' ', timespec='milliseconds'
            ),
            'TIME_BNDS_0': datetimes['start'].isoformat(
                sep=' ', timespec='milliseconds'
            ),
            'TIME_BNDS_1': datetimes['thru'].isoformat(
                sep=' ', timespec='milliseconds'
            ),
            'DateTime': datetimes['mean']
            .replace(tzinfo=None)
            .isoformat(sep=' ', timespec='seconds'),  # Use 'mean' for DateTime
            'ImageDescription': 'SMAP soil moisture data from mock_smap_file.h5. Units: m3/m3',
            'units': 'm3/m3',
        }
        # Use the utility function for detailed comparison
        diff_errors = compare_dictionaries(
            mock_dataset.SetMetadata.call_args[0][0], expected_metadata_dict
        )
        assert not diff_errors, '\n'.join(diff_errors)
        assert result == 0
