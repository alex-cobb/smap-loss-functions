"""Export soil moisture from a SMAP HDF5 file to a raster template"""

import datetime

import numpy as np

from osgeo import gdal, osr

from .choose_ease_grid import transform_lonlat_to_ease2


osr.UseExceptions()


EASE_GRID_EPSG = 6933


def smap_to_raster_template(h5_dataset, colfile_path, rowfile_path, outfile_path):
    """Export soil moisture from a SMAP file to a raster template"""
    col_geotransform, col_data = read_index_raster(colfile_path)
    row_geotransform, row_data = read_index_raster(rowfile_path)
    assert row_geotransform == col_geotransform
    assert col_data.shape == row_data.shape
    geotransform = col_geotransform
    del row_geotransform, col_geotransform

    short_name = h5_dataset['Metadata/DatasetIdentification'].attrs['SMAPShortName']
    sm_group = h5_dataset[
        {
            'L3_SM_P_E': 'Soil_Moisture_Retrieval_Data_AM',
            'L2_SM_P': 'Soil_Moisture_Retrieval_Data',
        }[short_name]
    ]

    col = sm_group['EASE_column_index'][:]
    row = sm_group['EASE_row_index'][:]

    datetimes = compute_datetimes(sm_group['tb_time_utc'][:].ravel())

    soil_moisture = get_masked_variable(sm_group['soil_moisture'])
    assert col.shape == row.shape
    assert row.shape == soil_moisture.shape

    sm_array = np.empty(col_data.shape, dtype=soil_moisture.dtype)
    sm_array[:] = float('nan')
    for c, r, sm in zip(
        col.ravel(), row.ravel(), soil_moisture.filled(float('nan')).ravel()
    ):
        if np.isnan(sm):
            continue
        assert r in row_data, (row_data, r, sm)
        assert c in col_data, (col_data, c, sm)
        matching_element = ((col_data == c) & (row_data == r)).nonzero()
        assert col_data[matching_element] == [c]
        assert row_data[matching_element] == [r]
        sm_array[matching_element] = sm

    write_geotiff(
        soil_moisture=sm_array,
        geotransform=geotransform,
        datetimes=datetimes,
        source_filename=h5_dataset.filename,
        outfile_path=outfile_path,
    )


def compute_datetimes(datetimes):
    """Compute datetime stats from a list of datetime strings

    Returns a dict with keys 'mean', 'start' and 'thru'.

    Invalid strings are ignored in the calculation.  If no valid datetimes
    are found, all values are set to the UNIX epoch (1970-01-01 00:00:00Z).

    """
    timestamps = np.array(
        [timestamp_from_utc_isoformat(s.decode('ascii')) for s in datetimes],
        dtype='float64',
    )
    if np.isnan(timestamps).all():
        timestamps = [0]
    return {
        key: datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
        for key, value in [
            ('mean', np.nanmean(timestamps)),
            ('start', np.nanmin(timestamps)),
            ('thru', np.nanmax(timestamps)),
        ]
    }


def timestamp_from_utc_isoformat(datetime_string):
    """Return a UNIX timestamp from a ISO 8601 UTC datetime string or NaN otherwise"""
    try:
        datetime_obj = datetime.datetime.fromisoformat(datetime_string)
    except ValueError:
        return float('NaN')
    if datetime_obj.tzinfo != datetime.timezone.utc:
        raise ValueError(
            f'Datetime string {datetime_string} does not specify that it is UTC'
        )
    return datetime_obj.timestamp()


def get_masked_variable(h5_group):
    """Retrieve a variable as a masked array"""
    nodata_value = h5_group.attrs['_FillValue']
    data = h5_group[:]
    return np.ma.masked_array(data, mask=(data == nodata_value))


def read_index_raster(path):
    """Read geotransform and data array from a row or column GeoTIFF

    Returns (geotransform, data array)

    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    band_data = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    del dataset
    assert band_data.dtype == np.dtype('int16')
    return (geotransform, band_data)


def write_geotiff(
    soil_moisture, geotransform, datetimes, source_filename, outfile_path
):
    """Write soil moisture to GeoTIFF file

    Start, thru and mean datetimes, units, and the source file name are written as
    metadata attributes.

    """
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        outfile_path,
        soil_moisture.shape[1],
        soil_moisture.shape[0],
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=3'],
    )
    dataset.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EASE_GRID_EPSG)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float('nan'))
    band.WriteArray(soil_moisture)
    metadata = dataset.GetMetadata() or {}
    metadata['TB_MEAN_DATETIME'] = datetimes['mean'].isoformat(
        sep=' ', timespec='milliseconds'
    )
    metadata['TIME_BNDS_0'], metadata['TIME_BNDS_1'] = [
        datetimes[key].isoformat(sep=' ', timespec='milliseconds')
        for key in ('start', 'thru')
    ]
    # DateTime is a standard TIFF tag and is meant to be written as "YYYY:MM:DD HH:MM:SS"
    metadata['DateTime'] = (
        datetimes['mean'].replace(tzinfo=None).isoformat(sep=' ', timespec='seconds')
    )
    metadata['ImageDescription'] = (
        f'SMAP soil moisture data from {source_filename}. Units: m3/m3'
    )
    metadata['units'] = 'm3/m3'
    dataset.SetMetadata(metadata)
    return 0
