"""Export SMAP forecasts from an SQLite database to GeoTIFFs

Reads hourly SMAP forecast data from an SQLite database and exports it as daily
GeoTIFFs. The geotransform and grid dimensions for the output GeoTIFFs are derived from
provided EASE grid column and row index GeoTIFFs.

"""

import datetime
import logging

import numpy as np

from osgeo import gdal, osr

osr.UseExceptions()

LOG = logging.getLogger('write_forecast_geotiffs')
LOG.setLevel(logging.INFO)


def write_forecast_geotiffs(
    cursor, geotransform, projection_wkt, col_data, row_data, outfile_pattern
):
    """Write SMAP forecast data from database into GeoTIFFs"""
    # Get the latest start_datetime from the database
    cursor.execute('SELECT MAX(start_datetime) FROM smap_data')
    latest_datetime_str = cursor.fetchone()[0]

    if not latest_datetime_str:
        raise ValueError('No data in smap_data')

    latest_datetime = datetime.datetime.strptime(
        latest_datetime_str, '%Y-%m-%d %H:%M:%S.%f'
    )

    # Determine the 5 consecutive daily intervals ending with the last timestamp
    target_datetimes = [
        latest_datetime - datetime.timedelta(days=i) for i in range(4, -1, -1)
    ]

    for dt in target_datetimes:
        cursor.execute(
            """
        SELECT ease_col, ease_row, soil_moisture
        FROM smap_data
        WHERE DATETIME(start_datetime) = DATETIME(?)""",
            (dt.isoformat(),),
        )
        records = cursor.fetchall()
        assert records, f'No data at {dt}'

        soil_moisture_array = np.full(col_data.shape, np.nan, dtype=np.float32)

        ease_index_lut = create_ease_index_lookup_table(col_data, row_data)

        for col, row, soil_moisture in records:
            try:
                soil_moisture_array[ease_index_lut[(col, row)]] = soil_moisture
            except KeyError:
                LOG.warn(
                    f'EASE grid ({col}, {row}) in DB but not in template files, skipping'
                )

        # Define output file path
        output_filename = outfile_pattern.format(dt.strftime('%Y%m%dT%H%M'))

        # Write the GeoTIFF, passing the projection WKT from the col_file
        write_geotiff(
            soil_moisture=soil_moisture_array,
            datetime=dt,
            geotransform=geotransform,
            outfile_path=output_filename,
            projection_wkt=projection_wkt,  # Pass the projection WKT
        )


def create_ease_index_lookup_table(col_data, row_data):
    """Create a mapping from (ease_col, ease_row) to (i, j) in the raster

    Assumes col_data and row_data are 2D arrays where each element
    corresponds to the EASE column and row at that pixel.

    """
    lut = {}
    for i in range(col_data.shape[0]):
        for j in range(col_data.shape[1]):
            lut[(col_data[i, j], row_data[i, j])] = (i, j)
    return lut


def get_grid_data(col_file_path, row_file_path):
    """Read geotransform, projection, and grid index data from template files"""
    col_geotransform, col_data, col_projection_wkt = read_index_raster(col_file_path)
    row_geotransform, row_data, row_projection_wkt = read_index_raster(row_file_path)
    assert np.array_equal(col_geotransform, row_geotransform)
    assert col_projection_wkt == row_projection_wkt
    assert col_data.shape == row_data.shape
    return col_geotransform, col_projection_wkt, col_data, row_data


def read_index_raster(path):
    """Read geotransform, data array, and projection (WKT) from a GeoTIFF

    Returns (geotransform, data array, projection WKT string).
    The data array is expected to contain the EASE grid indices.

    """
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    band_data = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection_wkt = dataset.GetProjection()

    if not np.issubdtype(band_data.dtype, np.integer):
        raise ValueError(f'Expected an integer dtype in {path}, got {band_data.dtype}')
    del dataset
    return (geotransform, band_data, projection_wkt)


def write_geotiff(soil_moisture, datetime, geotransform, outfile_path, projection_wkt):
    """Write soil moisture to GeoTIFF file"""
    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        str(outfile_path),
        soil_moisture.shape[1],
        soil_moisture.shape[0],
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=3'],
    )
    if dataset is None:
        raise IOError(f'Could not create output GeoTIFF file: {outfile_path}')

    dataset.SetGeoTransform(geotransform)

    assert projection_wkt, 'No projection provided'
    dataset.SetProjection(projection_wkt)

    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float('nan'))
    band.WriteArray(soil_moisture)
    # Write datetime into metadata in same format as rasters created directly
    # from SMAP data
    metadata = dataset.GetMetadata() or {}
    metadata['TB_MEAN_DATETIME'] = datetime.isoformat(sep=' ', timespec='milliseconds')
    metadata['TIME_BNDS_0'], metadata['TIME_BNDS_1'] = [
        datetime.isoformat(sep=' ', timespec='milliseconds')
        for key in ('start', 'thru')
    ]
    # DateTime is a standard TIFF tag and is meant to be written as "YYYY:MM:DD HH:MM:SS"
    metadata['DateTime'] = datetime.replace(tzinfo=None).isoformat(
        sep=' ', timespec='seconds'
    )
    metadata['ImageDescription'] = (
        'SMAP soil moisture forecast using loss functions. Units: m3/m3'
    )
    metadata['units'] = 'm3/m3'
    dataset.SetMetadata(metadata)
    del dataset
    return 0
