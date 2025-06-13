"""Dump data from EASE 2.0 grid rasters"""

from osgeo import gdal, osr

import numpy as np


osr.UseExceptions()


def dump_ease_raster_data(colfile_path, rowfile_path, infile_list, outfile):
    """Dump data from EASE 2.0 grid rasters"""
    col_geotransform, col_data, _ = read_raster_data(colfile_path)
    row_geotransform, row_data, _ = read_raster_data(rowfile_path)
    assert_equal(row_geotransform, col_geotransform)
    assert_equal(col_data.shape, row_data.shape)
    geotransform = col_geotransform
    del row_geotransform, col_geotransform

    # Header
    outfile.write('start_datetime,thru_datetime,column,row,value\n')
    for infile_name in infile_list:
        data_geotransform, data, metadata = read_raster_data(infile_name)
        tb_0, tb_1 = (metadata[key] for key in ('TIME_BNDS_0', 'TIME_BNDS_1'))
        assert_close(data_geotransform, geotransform)
        del data_geotransform
        assert_equal(data.shape, col_data.shape)
        valid_mask = ~data.mask
        for col, row, value in zip(
            col_data[valid_mask].ravel(),
            row_data[valid_mask].ravel(),
            data[valid_mask].ravel(),
        ):
            outfile.write(f'{tb_0},{tb_1},{col},{row},{value}\n')
    return 0


def read_raster_data(path):
    """Read geotransform and data array from a single-band GeoTIFF

    Returns (geotransform, data array)

    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    band_data = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()

    if nodata_value is not None and not np.isnan(nodata_value):
        band_data = np.ma.masked_equal(band_data, nodata_value)
    else:
        band_data = np.ma.masked_invalid(band_data)
    geotransform = dataset.GetGeoTransform()
    metadata = dataset.GetMetadata()
    del dataset
    return (geotransform, band_data, metadata)


def assert_equal(a, b):
    """Assert that two objects are equal"""
    assert a == b, f'{a} != {b}'


def assert_close(a, b):
    """Assert that two floating-point arrays are approximately equal"""
    assert np.allclose(a, b), f'{a} not close to {b}'
