"""Convert IMERG GPM data to GeoTIFF format

GPM IMERG data are transposed and flipped relative to GeoTIFF conventions.  It is
possible to do the coordinate manipulation with gdal_translate, but some operations on
the resulting file raise errors.  This module performs the transposition and flipping to
standardize the orientation of the data array.

"""

import pathlib

import netCDF4

import numpy as np

from osgeo import gdal, osr


gdal.UseExceptions()


def imerg_to_geotiff(nc_dataset, outfile_name):
    """Convert IMERG GPM data to GeoTIFF format"""
    time_bnds_var = nc_dataset['time_bnds']
    time_bnds = netCDF4.num2date(  # pylint: disable=no-member
        time_bnds_var[:],
        units=time_bnds_var.units,
        calendar='standard',
    )[0]
    del time_bnds_var
    # IMERG times are UTC
    start_datetime, thru_datetime = [
        dt.isoformat(sep=' ') + '+00:00' for dt in time_bnds
    ]
    mean_cftime = time_bnds[0] + (time_bnds[1] - time_bnds[0]) / 2
    del time_bnds

    lat = nc_dataset['lat'][:]
    lon = nc_dataset['lon'][:]
    assert_equal(nc_dataset['precipitation'].dimensions, ('time', 'lon', 'lat'))
    # Transpose to (lat, lon) for output
    precipitation = np.ma.masked_invalid(nc_dataset['precipitation'][0, :].transpose())
    assert_equal(precipitation.shape, (lat.shape[0], lon.shape[0]))
    assert (np.diff(lat) > 0).all(), f'Latitude not strictly increasing: {lat}'
    assert (np.diff(lon) > 0).all(), f'Longitude not strictly increasing: {lon}'
    assert_close(lat[0], -89.95)
    assert_close(lat[-1], 89.95)
    assert_close(lon[0], -179.95)
    assert_close(lon[-1], 179.95)
    x_res = 0.1
    y_res = 0.1
    assert_close(np.diff(lon), x_res, rtol=1e-4)
    assert_close(np.diff(lat), y_res, rtol=1e-4)
    assert len(lat) == 1800
    assert len(lon) == 3600

    # Flip latitude to match "north up" convention
    precipitation = precipitation[::-1, :]
    lat = lat[::-1]

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = precipitation.shape
    dataset = driver.Create(
        outfile_name,
        cols,
        rows,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=3'],
    )
    dataset.SetGeoTransform((-180, x_res, 0, 90, 0, -y_res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float('nan'))
    band.WriteArray(precipitation.filled(fill_value=float('nan')))
    metadata = dataset.GetMetadata() or {}
    metadata['TIME_BNDS_0'] = start_datetime
    metadata['TIME_BNDS_1'] = thru_datetime
    # DateTime is a standard TIFF tag and is meant to be written as "YYYY:MM:DD HH:MM:SS"
    metadata['DateTime'] = mean_cftime.isoformat(sep=' ', timespec='seconds')
    metadata['ImageDescription'] = (
        'IMERG precipitation data from '
        f'{pathlib.Path(nc_dataset.filepath()).name}. Units: mm/day'
    )
    metadata['units'] = 'mm/d'

    dataset.SetMetadata(metadata)
    return 0


def assert_equal(a, b):
    """Assert that a == b"""
    assert a == b, f'{a} != {b}'


def assert_close(a, b, rtol=1e-05, atol=1e-08):
    """Assert that floats a and b are approximately equal"""
    assert np.allclose(a, b, rtol, atol), f'{a} not close to {b}'
