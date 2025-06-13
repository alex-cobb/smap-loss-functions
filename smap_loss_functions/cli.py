#!/usr/bin/python3

"""Script dispatch for calculating loss functions"""

import argparse


def main():
    """Main CLI for smap-loss-functions"""
    parser = argparse.ArgumentParser(description='SMAP Loss Functions CLI')
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    add_choose_ease_grid_subparser(subparsers)
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
