#!/usr/bin/python3

"""Script dispatch for calculating loss functions"""

import argparse
import pathlib
import sys


def main():
    """Main CLI for smap-loss-functions"""
    parser = argparse.ArgumentParser(description='SMAP Loss Functions CLI')
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    add_choose_ease_grid_subparser(subparsers)
    add_smap_to_raster_template_subparser(subparsers)
    add_dump_ease_raster_data_subparser(subparsers)
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
