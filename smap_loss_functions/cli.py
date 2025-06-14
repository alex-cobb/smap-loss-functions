#!/usr/bin/python3

"""Script dispatch for calculating loss functions"""

import argparse
import logging
import pathlib
import sys


logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format='%(levelname)s:%(name)s:%(message)s',
)
LOG = logging.getLogger('smap_loss_functions.cli')


def main():
    """Main CLI for smap-loss-functions"""
    parser = argparse.ArgumentParser(description='SMAP Loss Functions CLI')
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    add_choose_ease_grid_subparser(subparsers)
    add_smap_to_raster_template_subparser(subparsers)
    add_dump_ease_raster_data_subparser(subparsers)
    add_imerg_to_geotiff_subparser(subparsers)
    add_write_smap_db_subparser(subparsers)
    add_fit_loss_functions_subparser(subparsers)
    add_forecast_smap_subparser(subparsers)
    add_write_forecast_geotiff_subparser(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        return args.func(args)
    parser.print_help()
    return -1


def add_choose_ease_grid_subparser(subparsers):
    """Add subparser for choose_ease_grid"""
    parser = subparsers.add_parser(
        'choose-ease-grid', help='Choose EASE 2.0 grid to cover a bounding box'
    )
    parser.add_argument(
        'bbox',
        metavar='JSON-IN',
        help='Bounding box coordinates as JSON',
        type=argparse.FileType('rt', encoding='ascii'),
    )
    parser.add_argument(
        'grid',
        metavar='JSON-OUT',
        help='Outfile for selected EASE 2.0 grid as JSON',
        type=argparse.FileType('wt', encoding='ascii'),
    )
    parser.add_argument(
        '-g',
        '--grid-name',
        help='Which EASE grid type to use, default EASE2_G36km',
        default='EASE2_G36km',
    )
    parser.add_argument(
        '--cols',
        metavar='TIF',
        help='Optional output raster of EASE column numbers',
    )
    parser.add_argument(
        '--rows',
        metavar='TIF',
        help='Optional output raster of EASE row numbers',
    )
    parser.set_defaults(func=choose_ease_grid_cli)


def choose_ease_grid_cli(args):
    """Call choose_ease_grid with command-line arguments"""
    import json
    from .choose_ease_grid import choose_ease_grid

    with args.bbox as bbox_file, args.grid as outfile:
        bbox = json.load(bbox_file)
        return choose_ease_grid(
            bbox,
            grid_name=args.grid_name,
            outfile=outfile,
            col_outfile_path=args.cols,
            row_outfile_path=args.rows,
        )


def add_smap_to_raster_template_subparser(subparsers):
    """Add subparser for smap-to-raster-template"""
    parser = subparsers.add_parser(
        'smap-to-raster-template',
        help='Export soil moisture from a SMAP file to a raster template',
    )
    parser.add_argument('col_file', type=pathlib.Path, metavar='COLTIF')
    parser.add_argument('row_file', type=pathlib.Path, metavar='ROWTIF')
    parser.add_argument('infile', type=pathlib.Path, metavar='HDF5')
    parser.add_argument('outfile', type=pathlib.Path, metavar='OUTTIF')
    parser.set_defaults(func=smap_to_raster_template_cli)


def smap_to_raster_template_cli(args):
    """CLI to smap-to-raster-template"""
    import h5py
    from .smap_to_raster_template import smap_to_raster_template

    with h5py.File(args.infile, 'r') as h5_dataset:
        return smap_to_raster_template(
            h5_dataset=h5_dataset,
            colfile_path=args.col_file,
            rowfile_path=args.row_file,
            outfile_path=args.outfile,
        )


def add_dump_ease_raster_data_subparser(subparsers):
    """Add subparser for dump-ease-raster-data"""
    parser = subparsers.add_parser(
        'dump-ease-raster-data', help='Dump data from EASE 2.0 grid rasters'
    )
    parser.add_argument('col_file', type=pathlib.Path, metavar='COLTIF')
    parser.add_argument('row_file', type=pathlib.Path, metavar='ROWTIF')
    parser.add_argument(
        '-f',
        '--file-list',
        type=argparse.FileType('rt', encoding='utf-8'),
        metavar='HDF5',
        default=sys.stdin,
    )
    parser.add_argument(
        '-o',
        '--outfile',
        type=argparse.FileType('wt', encoding='ascii'),
        metavar='OUTTIF',
        default=sys.stdout,
    )
    parser.set_defaults(func=dump_ease_raster_data_cli)


def dump_ease_raster_data_cli(args):
    """CLI for dump-ease-raster-data"""
    from .dump_ease_raster_data import dump_ease_raster_data

    with args.file_list as infile, args.outfile as outfile:
        infile_list = [line.strip() for line in infile.readlines() if line.strip()]
        return dump_ease_raster_data(
            colfile_path=args.col_file,
            rowfile_path=args.row_file,
            infile_list=infile_list,
            outfile=outfile,
        )


def add_imerg_to_geotiff_subparser(subparsers):
    """Add subparser for imerg-to-geotiff"""
    parser = subparsers.add_parser(
        'imerg-to-geotiff', help='Convert IMERG GPM data to GeoTIFF format'
    )
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.set_defaults(func=imerg_to_geotiff_cli)


def imerg_to_geotiff_cli(args):
    """CLI for imerg-to-geotiff"""
    import netCDF4
    from .imerg_to_geotiff import imerg_to_geotiff

    with netCDF4.Dataset(args.infile) as nc_dataset:
        return imerg_to_geotiff(nc_dataset=nc_dataset, outfile_name=args.outfile)


def add_write_smap_db_subparser(subparsers):
    """Add subparser for write-smap-db"""
    parser = subparsers.add_parser(
        'write-smap-db', help='Write SMAP and IMERG data to an SQLite database'
    )
    parser.add_argument(
        'smap_data',
        metavar='SMAP',
        help='SMAP data as CSV',
        type=argparse.FileType('rt', encoding='ascii'),
    )
    parser.add_argument(
        'imerg_data',
        metavar='IMERG',
        help='IMERG data as CSV',
        type=argparse.FileType('rt', encoding='ascii'),
    )
    parser.add_argument('outfile', metavar='SQLITE', help='Output SQLite database')
    parser.set_defaults(func=write_smap_db_cli)


def write_smap_db_cli(args):
    """CLI for write-smap-db"""
    import sqlite3
    from .write_smap_db import write_smap_db

    with (
        args.smap_data as smap_infile,
        args.imerg_data as imerg_infile,
        sqlite3.connect(args.outfile) as connection,
    ):
        return write_smap_db(
            smap_infile=smap_infile, imerg_infile=imerg_infile, connection=connection
        )


def add_fit_loss_functions_subparser(subparsers):
    """Add subparser for fit-loss-functions"""
    parser = subparsers.add_parser('fit-loss-functions', help='Fit SMAP loss functions')
    parser.add_argument('in_db', metavar='DBIN', help='Path to SMAP and IMERG db')
    parser.add_argument(
        'out_db',
        metavar='DBOUT',
        help='Path to output db to store loss functions',
        type=pathlib.Path,
    )
    parser.set_defaults(func=fit_loss_functions_cli)


def fit_loss_functions_cli(args):
    """CLI for fit-loss-functions"""
    import sqlite3
    from .fit_loss_functions import set_up_loss_function_db, fit_loss_functions

    if args.out_db.exists():
        args.out_db.unlink()
    with (
        sqlite3.connect(f'file:{args.in_db}?mode=ro', uri=True) as in_connection,
        sqlite3.connect(args.out_db) as out_connection,
    ):
        set_up_loss_function_db(out_connection)
        return fit_loss_functions(in_connection, out_connection)


def add_forecast_smap_subparser(subparsers):
    """Add subparser for forecast-smap"""
    parser = subparsers.add_parser(
        'forecast-smap',
        help='Forecast SMAP soil moisture over 5 days with zero precipitation',
    )
    parser.add_argument(
        'loss_function_db',
        metavar='LOSSDB',
        help='Path to loss function database',
    )
    parser.add_argument(
        'smap_data_db',
        metavar='SMAPDB',
        help='Path to SMAP database with initial soil moisture',
    )
    parser.add_argument(
        'forecast_db',
        metavar='OUTPUTDB',
        help='Path to new SQLite database to which to write soil moisture forecasts',
    )
    parser.set_defaults(func=forecast_smap_cli)


def forecast_smap_cli(args):
    """CLI for forecast-smap"""
    import os
    from .forecast_smap import forecast_smap

    if os.path.exists(args.forecast_db):
        LOG.info('%s exists, removing', args.forecast_db)
        os.remove(args.forecast_db)
    return forecast_smap(
        loss_function_db_path=args.loss_function_db,
        smap_db_path=args.smap_data_db,
        forecast_db_path=args.forecast_db,
    )


def add_write_forecast_geotiff_subparser(subparsers):
    """Add subparser for forecast-smap"""
    parser = subparsers.add_parser(
        'write-forecast-geotiffs',
        description='Export SMAP forecasts from SQLite to daily GeoTIFFs. '
        'GeoTIFFs will be generated for 5 consecutive days up to '
        'the last datetime in the input database.',
    )
    parser.add_argument(
        'input_smap_db',
        metavar='SMAPDB',
        type=pathlib.Path,
        help='Path to SQLite database of SMAP forecasts',
    )
    parser.add_argument(
        'col_file',
        type=pathlib.Path,
        metavar='COLTIF',
        help='Path to GeoTIFF EASE grid column indices',
    )
    parser.add_argument(
        'row_file',
        type=pathlib.Path,
        metavar='ROWTIF',
        help='Path to GeoTIFF EASE grid row indices',
    )
    parser.add_argument(
        '-o',
        '--outfile-pattern',
        metavar='PATTERN',
        help=(
            'File path template for output GeoTIFFs as an f-string-style '
            'template with one field for the datetime'
        ),
        default='SMAP_forecast_{}.tif',
    )
    parser.set_defaults(func=write_forecast_geotiff_cli)


def write_forecast_geotiff_cli(args):
    """CLI to write forecast GeoTIFFs"""
    import sqlite3

    from .write_forecast_geotiffs import get_grid_data, write_forecast_geotiffs

    geotransform, projection_wkt, col_data, row_data = get_grid_data(
        col_file_path=args.col_file, row_file_path=args.row_file
    )

    with sqlite3.connect(f'file:{args.input_smap_db}?mode=ro', uri=True) as connection:
        write_forecast_geotiffs(
            cursor=connection.cursor(),
            geotransform=geotransform,
            projection_wkt=projection_wkt,
            col_data=col_data,
            row_data=row_data,
            outfile_pattern=args.outfile_pattern,
        )
    return 0
