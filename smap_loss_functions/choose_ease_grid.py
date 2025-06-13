"""Choose EASE grid to cover a bounding box"""

import json

from ease_lonlat import EASE2GRID, SUPPORTED_GRIDS

import numpy as np

import pyproj

from osgeo import gdal, osr


osr.UseExceptions()


EASE_GRID_EPSG = 6933


def choose_ease_grid(bbox, grid_name, outfile, col_outfile_path, row_outfile_path):
    """Choose EASE grid to cover a bounding box"""
    ease_grid = EASE2GRID(name=grid_name, **SUPPORTED_GRIDS[grid_name])
    assert bbox['srs'] == 'EPSG:4326'
    # Initial guess
    # EASE Grid-2.0 rows increase from north to south
    start_col, start_row = ease_grid.lonlat2rc(lon=bbox['w'], lat=bbox['n'])
    thru_col, thru_row = ease_grid.lonlat2rc(lon=bbox['e'], lat=bbox['s'])
    ease_bbox = transform_lonlat_bbox_to_ease2(bbox)
    del bbox

    gridspec = create_ease2_gridspec(
        ease_grid, start_col, start_row, thru_col, thru_row
    )

    assert_gridspec_covers_bbox(gridspec, ease_bbox)
    json.dump(gridspec, outfile)

    if col_outfile_path is not None:
        write_ease_column_raster(gridspec, start_col, thru_col, col_outfile_path)

    if row_outfile_path is not None:
        write_ease_row_raster(gridspec, start_row, thru_row, row_outfile_path)

    return 0


def write_ease_column_raster(gridspec, start_col, thru_col, outfile_path):
    """Write ease column numbers to a GeoTIFF raster"""
    assert thru_col - start_col + 1 == gridspec['ncols']
    cols = [list(range(start_col, thru_col + 1))] * gridspec['nrows']
    write_geotiff(gridspec, np.array(cols, dtype='int16'), outfile_path)


def write_ease_row_raster(gridspec, start_row, thru_row, outfile_path):
    """Write ease column numbers to a GeoTIFF raster"""
    assert thru_row - start_row + 1 == gridspec['nrows']
    rows = [list(range(start_row, thru_row + 1))] * gridspec['ncols']
    write_geotiff(gridspec, np.array(rows, dtype='int16').T, outfile_path)


def write_geotiff(gridspec, data, outfile_path):
    """Write an int16 GeoTIFF containing data"""
    assert data.shape == (gridspec['nrows'], gridspec['ncols'])
    geotransform = (
        gridspec['w'],
        gridspec['xres'],
        0,
        gridspec['n'],
        0,
        gridspec['yres'],
    )

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        outfile_path,
        data.shape[1],
        data.shape[0],
        1,
        gdal.GDT_Int16,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=2'],
    )
    dataset.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EASE_GRID_EPSG)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float('nan'))
    band.WriteArray(data)
    return 0


def gridspec_covers_bbox(gridspec, bbox):
    """Return True if gridspec covers bbox, False otherwise"""
    try:
        assert_gridspec_covers_bbox(gridspec, bbox)
    except AssertionError:
        return False
    return True


def assert_gridspec_covers_bbox(gridspec, bbox):
    """Check that gridspec covers bbox"""
    assert gridspec['w'] <= bbox['w'], f'{gridspec["w"]} > {bbox["w"]}'
    assert gridspec['n'] >= bbox['n']
    assert gridspec['w'] + gridspec['xres'] * gridspec['ncols'] >= bbox['e']
    assert gridspec['n'] + gridspec['yres'] * gridspec['nrows'] <= bbox['s']


def create_ease2_gridspec(ease_grid, start_col, start_row, thru_col, thru_row):
    """ """
    cols = np.arange(start_col, thru_col + 1, dtype='int16')
    rows = np.arange(start_row, thru_row + 1, dtype='int16')
    assert cols[0] == start_col
    assert cols[-1] == thru_col
    assert rows[0] == start_row
    assert rows[-1] == thru_row
    col_cartesian, row_cartesian = np.meshgrid(cols, rows)
    ease_col = col_cartesian.flatten()
    ease_row = row_cartesian.flatten()
    lons = []
    lats = []
    for col, row in zip(ease_col, ease_row):
        lon, lat = ease_grid.rc2lonlat(col=col, row=row)
        lons.append(lon)
        lats.append(lat)
    easex, easey = transform_lonlat_to_ease2(np.array(lons), np.array(lats))
    del lons, lats
    easex_seq = [easex[ease_col == c].mean() for c in cols]
    easey_seq = [easey[ease_row == r].mean() for r in rows]
    assert (np.diff(easex_seq) > 0).all()
    assert (np.diff(easey_seq) < 0).all()
    xres = np.diff(easex_seq).mean()
    yres = np.diff(easey_seq).mean()
    del easex_seq, easey_seq
    gridspec = dict(
        w=min(easex) - xres / 2,
        n=max(easey) - yres / 2,
        e=max(easex) + xres / 2,
        s=min(easey) + yres / 2,
        ncols=len(cols),
        nrows=len(rows),
        xres=xres,
        yres=yres,
        srs=f'EPSG:{6933}',
    )
    del easex, easey
    return gridspec


def transform_lonlat_bbox_to_ease2(bbox):
    """Transform a bbox from EPSG:4326 to EPSG:6933"""
    assert bbox['srs'] == 'EPSG:4326'
    w, n = transform_lonlat_to_ease2(longitude=bbox['w'], latitude=bbox['n'])
    e, s = transform_lonlat_to_ease2(longitude=bbox['e'], latitude=bbox['s'])
    return dict(w=w, n=n, e=e, s=s, srs=f'EPSG:{6933}')


def transform_lonlat_to_ease2(longitude, latitude):
    """Transform points from EPSG:4326 to EPSG:6933"""
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS('EPSG:4326'), pyproj.CRS('EPSG:6933'), always_xy=True
    )
    return transformer.transform(longitude, latitude)
