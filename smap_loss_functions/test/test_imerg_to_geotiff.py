"""Tests for imerg-to-geotiff"""

import datetime
import os

import netCDF4

import numpy as np

from osgeo import gdal, osr

import pytest

from smap_loss_functions.imerg_to_geotiff import imerg_to_geotiff


# For fixtures:  pylint: disable=redefined-outer-name


class MockNetCDF4Variable:
    """A simple class to mimic netCDF4.Variable behavior."""

    def __init__(self, data, units=None, dimensions=None):
        self._data = data
        self.units = units
        self.dimensions = dimensions

    def __getitem__(self, key):
        """Allows slicing like var[:] or var[0, :]."""
        return self._data[key]

    def __len__(self):
        """Allows len(var) for dimension checks."""
        return len(self._data)


class MockNetCDF4Dataset:
    """A simple class to mimic netCDF4.Dataset behavior."""

    def __init__(
        self,
        filename='dummy.nc',
        time_start_dt=datetime.datetime(2023, 1, 1, 0, 0, 0),
        time_end_dt=datetime.datetime(2023, 1, 1, 0, 30, 0),
        lat_data=None,
        lon_data=None,
        precip_data=None,
    ):
        self._filename = filename
        self._variables = {}

        # Default time_bnds data
        time_bnds_data = np.array(
            [
                netCDF4.date2num(  # pylint: disable=no-member
                    time_start_dt,
                    units='seconds since 1970-01-01 00:00:00',
                    calendar='standard',
                ),
                netCDF4.date2num(  # pylint: disable=no-member
                    time_end_dt,
                    units='seconds since 1970-01-01 00:00:00',
                    calendar='standard',
                ),
            ]
        )
        # Note: time_bnds usually has shape (1, 2) for a single time slice
        self._variables['time_bnds'] = MockNetCDF4Variable(
            data=time_bnds_data[np.newaxis, :],
            units='seconds since 1970-01-01 00:00:00',
            dimensions=('nv',),  # Common dimension name for boundaries
        )

        # Default lat/lon, or use provided custom data
        self.lat_len = 1800
        self.lon_len = 3600

        if lat_data is None:
            self._variables['lat'] = MockNetCDF4Variable(
                data=np.linspace(-89.95, 89.95, self.lat_len), dimensions=('lat',)
            )
        else:
            self._variables['lat'] = MockNetCDF4Variable(
                data=lat_data, dimensions=('lat',)
            )
            self.lat_len = len(
                lat_data
            )  # Update internal length if custom data is used

        if lon_data is None:
            self._variables['lon'] = MockNetCDF4Variable(
                data=np.linspace(-179.95, 179.95, self.lon_len), dimensions=('lon',)
            )
        else:
            self._variables['lon'] = MockNetCDF4Variable(
                data=lon_data, dimensions=('lon',)
            )
            self.lon_len = len(
                lon_data
            )  # Update internal length if custom data is used

        # Default precipitation data (random with some NaNs), or use provided
        if precip_data is None:
            # Precipitation dimension is ('time', 'lon', 'lat') as per the module's assertion
            # So, shape is (1, lon_len, lat_len)
            precip_data = np.random.rand(1, self.lon_len, self.lat_len) * 10
            # Introduce some NaNs to test masked_invalid and filled
            # We are putting NaNs in the original (time, lon, lat) order
            precip_data[0, self.lon_len // 2, self.lat_len // 2] = np.nan
        self._variables['precipitation'] = MockNetCDF4Variable(
            data=precip_data, units='mm/hr', dimensions=('time', 'lon', 'lat')
        )

    def __getitem__(self, key):
        """Allows access to variables like nc_dataset['variable_name']."""
        if key in self._variables:
            return self._variables[key]
        raise KeyError(f"Variable '{key}' not found in mock dataset.")

    def filepath(self):
        """Mimics the filepath() method for ImageDescription metadata."""
        return self._filename

    def close(self):
        """No-op for the mock dataset."""


@pytest.fixture
def dummy_netcdf_dataset():
    """Pytest fixture to provide a default dummy NetCDF4 dataset for tests."""
    return MockNetCDF4Dataset()


def test_imerg_to_geotiff_basic_conversion(tmp_path, dummy_netcdf_dataset):
    """
    Test basic conversion: ensures the function runs without error,
    creates a GeoTIFF, and verifies key properties of the output file.
    """
    output_filepath = tmp_path / 'test_output.tif'
    result = imerg_to_geotiff(dummy_netcdf_dataset, str(output_filepath))
    assert result == 0, 'imerg_to_geotiff should return 0 on success'

    # Verify that the file was created in the temporary directory
    assert output_filepath.exists(), 'Output GeoTIFF file was not created'

    # Open the created GeoTIFF using GDAL to inspect its properties
    dataset = gdal.Open(str(output_filepath))
    assert dataset is not None, 'Failed to open output GeoTIFF with GDAL'

    # Check raster dimensions
    assert dataset.RasterXSize == dummy_netcdf_dataset.lon_len
    assert dataset.RasterYSize == dummy_netcdf_dataset.lat_len
    assert dataset.RasterCount == 1  # Expecting a single band
    assert dataset.GetRasterBand(1).DataType == gdal.GDT_Float32  # Check data type

    # Check geotransform (spatial reference information)
    expected_geotransform = (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1)
    np.testing.assert_allclose(
        dataset.GetGeoTransform(), expected_geotransform, rtol=1e-6
    )

    # Check projection (should be EPSG:4326 - WGS 84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS 84
    assert dataset.GetProjection() == srs.ExportToWkt(), (
        'Projection does not match EPSG:4326'
    )

    # Check metadata written to the GeoTIFF
    metadata = dataset.GetMetadata()
    assert 'TIME_BNDS_0' in metadata
    assert 'TIME_BNDS_1' in metadata
    assert 'DateTime' in metadata
    assert 'ImageDescription' in metadata
    assert 'units' in metadata
    assert metadata['units'] == 'mm/d'
    # Verify the filename part in ImageDescription
    assert (
        dummy_netcdf_dataset.filepath().split(os.sep)[-1]
        in metadata['ImageDescription']
    )

    # Check data values and NaN handling
    band_data = dataset.GetRasterBand(1).ReadAsArray()
    # Retrieve original precipitation data, apply transpose and latitude flip
    # as done in function
    original_precipitation = np.ma.masked_invalid(
        dummy_netcdf_dataset['precipitation'][0, :].transpose()
    )
    expected_data = original_precipitation[::-1, :].filled(fill_value=float('nan'))
    np.testing.assert_allclose(band_data, expected_data, equal_nan=True, rtol=1e-5)

    # Check NoDataValue
    assert np.isnan(dataset.GetRasterBand(1).GetNoDataValue()), (
        'NoDataValue should be NaN'
    )

    del dataset


def test_imerg_to_geotiff_with_custom_times(tmp_path):
    """Test that custom time boundaries are correctly written to metadata."""
    start_dt = datetime.datetime(2022, 6, 15, 12, 0, 0)
    end_dt = datetime.datetime(2022, 6, 15, 12, 30, 0)
    custom_dataset = MockNetCDF4Dataset(time_start_dt=start_dt, time_end_dt=end_dt)

    output_filepath = tmp_path / 'custom_time_output.tif'
    imerg_to_geotiff(custom_dataset, str(output_filepath))

    dataset = gdal.Open(str(output_filepath))
    metadata = dataset.GetMetadata()

    assert metadata['TIME_BNDS_0'] == start_dt.isoformat(sep=' ') + '+00:00'
    assert metadata['TIME_BNDS_1'] == end_dt.isoformat(sep=' ') + '+00:00'
    mean_cftime = start_dt + (end_dt - start_dt) / 2
    assert metadata['DateTime'] == mean_cftime.isoformat(sep=' ', timespec='seconds')
    dataset = None


def test_imerg_to_geotiff_assert_failures():
    """
    Test scenarios that should trigger the internal `assert` statements
    within the `imerg_to_geotiff` function.
    """

    # Test case 1: Incorrect precipitation dimensions
    dataset_wrong_dims = MockNetCDF4Dataset(
        precip_data=np.random.rand(1, 1800, 3600)  # (time, lat, lon)
    )
    # pylint: disable=protected-access
    dataset_wrong_dims._variables['precipitation'].dimensions = (
        'time',
        'lat',
        'lon',
    )  # Force wrong dimensions
    with pytest.raises(AssertionError):
        imerg_to_geotiff(dataset_wrong_dims, 'temp_error.tif')

    # Test case 2: Latitude not strictly increasing
    bad_lat_data = np.array([-89.95, -89.85, -90.05, 89.95])  # Not strictly increasing
    # Adjust precip data size to match the bad_lat_data length for this specific test case
    precip_for_bad_lat = np.random.rand(1, 3600, len(bad_lat_data))
    with pytest.raises(AssertionError, match='Latitude not strictly increasing'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(lat_data=bad_lat_data, precip_data=precip_for_bad_lat),
            'temp_error.tif',
        )

    # Test case 3: Incorrect initial latitude value
    wrong_lat_start_data = np.linspace(-90.0, 89.95, 1800)  # Should be -89.95
    with pytest.raises(AssertionError, match='not close to -89.95'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(lat_data=wrong_lat_start_data), 'temp_error.tif'
        )

    # Test case 4: Incorrect final latitude value
    wrong_lat_end_data = np.linspace(-89.95, 90.0, 1800)  # Should be 89.95
    with pytest.raises(AssertionError, match='not close to 89.95'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(lat_data=wrong_lat_end_data), 'temp_error.tif'
        )

    # Test case 5: Incorrect initial longitude value
    wrong_lon_start_data = np.linspace(-180.0, 179.95, 3600)  # Should be -179.95
    with pytest.raises(AssertionError, match='not close to -179.95'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(lon_data=wrong_lon_start_data), 'temp_error.tif'
        )

    # Test case 6: Incorrect final longitude value
    wrong_lon_end_data = np.linspace(-179.95, 180.0, 3600)  # Should be 179.95
    with pytest.raises(AssertionError, match='not close to 179.95'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(lon_data=wrong_lon_end_data), 'temp_error.tif'
        )

    # Test case 7: Incorrect x_res (longitude difference)
    # Create lon data such that the step is not 0.1
    wrong_x_res_lon_data = np.linspace(
        -179.95, 179.95, 3601
    )  # One more point changes the resolution
    precip_for_wrong_x_res = np.random.rand(1, len(wrong_x_res_lon_data), 1800)
    with pytest.raises(AssertionError, match='not close to 0.1'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(
                lon_data=wrong_x_res_lon_data, precip_data=precip_for_wrong_x_res
            ),
            'temp_error.tif',
        )

    # Test case 8: Incorrect y_res (latitude difference)
    # Create lat data such that the step is not 0.1
    wrong_y_res_lat_data = np.linspace(
        -89.95, 89.95, 1801
    )  # One more point changes the resolution
    precip_for_wrong_y_res = np.random.rand(1, 3600, len(wrong_y_res_lat_data))
    with pytest.raises(AssertionError, match='not close to 0.1'):
        imerg_to_geotiff(
            MockNetCDF4Dataset(
                lat_data=wrong_y_res_lat_data, precip_data=precip_for_wrong_y_res
            ),
            'temp_error.tif',
        )
