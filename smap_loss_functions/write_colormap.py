"""Write a Matplotlib colormap to a text file for GDAL"""

import matplotlib as mpl

import numpy as np


def write_colormap(colormap_name, outfile, min_value, max_value, n_colors=256):
    """Write a matplotlib colormap to a text file for GDAL"""

    colormap = mpl.colormaps.get_cmap(colormap_name)
    values = np.linspace(min_value, max_value, n_colors)

    # Format: value R G B [A]
    # NoData is expressed as nv
    outfile.write('nv 0 0 0 0\n')
    for i in range(n_colors):
        # Convert RGBA values (0-1) to bytes
        r, g, b, a = [int(round(v * 255)) for v in colormap(i / (n_colors - 1))]
        outfile.write(f'{values[i]} {r} {g} {b} {a}\n')
    return 0
