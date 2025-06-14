"""Tests for write_colormap"""

import io
import pytest
import matplotlib as mpl
import numpy as np

from smap_loss_functions.write_colormap import write_colormap


def test_write_colormap_basic_functionality():
    """
    Tests the basic functionality of write_colormap:
    - Writes to the outfile object.
    - Includes the 'nv' (NoData) line.
    - Correct number of color lines.
    - Correct format of output lines.
    """
    output_buffer = io.StringIO()
    colormap_name = 'viridis'
    min_val = 0
    max_val = 100
    n_colors = 10

    result = write_colormap(colormap_name, output_buffer, min_val, max_val, n_colors)
    assert result == 0

    output_content = output_buffer.getvalue()
    lines = output_content.strip().split('\n')

    assert lines[0] == 'nv 0 0 0 0', 'Bad first line'
    assert len(lines) == n_colors + 1, 'Wrong number of lines written'

    cmap = mpl.colormaps.get_cmap(colormap_name)
    expected_values = np.linspace(min_val, max_val, n_colors)

    # Check each color entry
    for i in range(n_colors):
        line_num = i + 1  # +1 because of the 'nv' line
        line_parts = lines[line_num].split()

        assert len(line_parts) == 5, 'Wrong number of parts (value R G B A)'

        assert float(line_parts[0]) == pytest.approx(expected_values[i])

        r, g, b, a = [int(p) for p in line_parts[1:]]
        assert 0 <= r <= 255, 'Expected integer 0-255'
        assert 0 <= g <= 255, 'Expected integer 0-255'
        assert 0 <= b <= 255, 'Expected integer 0-255'
        assert 0 <= a <= 255, 'Expected integer 0-255'

        expected_rgba_float = cmap(i / (n_colors - 1))
        expected_r, expected_g, expected_b, expected_a = [
            int(round(v * 255)) for v in expected_rgba_float
        ]
        assert r == expected_r
        assert g == expected_g
        assert b == expected_b
        assert a == expected_a


def test_write_colormap_edge_cases_n_colors():
    """Tests with different n_colors values, including minimum."""
    output_buffer = io.StringIO()
    min_val = -10
    max_val = 10

    n_colors_small = 2
    write_colormap('gray', output_buffer, min_val, max_val, n_colors_small)
    lines = output_buffer.getvalue().strip().split('\n')
    assert len(lines) == n_colors_small + 1
    output_buffer.seek(0)  # Reset buffer for next test
    output_buffer.truncate(0)

    # Try a larger number of colors
    n_colors_large = 512
    write_colormap('magma', output_buffer, min_val, max_val, n_colors_large)
    lines = output_buffer.getvalue().strip().split('\n')
    assert len(lines) == n_colors_large + 1


def test_write_colormap_different_min_max_values():
    """Tests with different min_value and max_value combinations."""
    output_buffer = io.StringIO()
    n_colors = 5

    # min_value > max_value (should still produce sorted values)
    write_colormap('plasma', output_buffer, 100, 0, n_colors)
    lines = output_buffer.getvalue().strip().split('\n')
    # Check if values are correctly interpolated, even if min > max
    assert float(lines[1].split()[0]) == pytest.approx(100.0)
    assert float(lines[-1].split()[0]) == pytest.approx(0.0)
    output_buffer.seek(0)
    output_buffer.truncate(0)

    # Negative values
    write_colormap('jet', output_buffer, -50, -20, n_colors)
    lines = output_buffer.getvalue().strip().split('\n')
    assert float(lines[1].split()[0]) == pytest.approx(-50.0)
    assert float(lines[-1].split()[0]) == pytest.approx(-20.0)
    output_buffer.seek(0)
    output_buffer.truncate(0)

    # Min and max are the same (should produce n_colors lines with the same value)
    write_colormap('viridis', output_buffer, 50, 50, n_colors)
    lines = output_buffer.getvalue().strip().split('\n')
    for i in range(1, len(lines)):
        assert float(lines[i].split()[0]) == pytest.approx(50.0)


def test_write_colormap_invalid_colormap_name():
    """
    Tests that an invalid colormap name raises a ValueError,
    as expected from matplotlib's get_cmap.
    """
    output_buffer = io.StringIO()
    with pytest.raises(
        ValueError, match="'non_existent_colormap' is not a valid value for cmap."
    ):
        write_colormap('non_existent_colormap', output_buffer, 0, 100)
