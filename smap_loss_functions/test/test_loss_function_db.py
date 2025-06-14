"""Tests for loss_function_db"""

import numpy as np

import pytest

from smap_loss_functions.loss_function import LossFunction
from smap_loss_functions.loss_function_db import (
    set_up_loss_function_db,
    get_loss_function_from_db,
)


def test_set_up_loss_function_db(in_memory_output_db):
    """Test set_up_loss_function_db creates the table correctly."""
    set_up_loss_function_db(in_memory_output_db)
    cursor = in_memory_output_db.cursor()
    cursor.execute('PRAGMA table_info(loss_function)')
    columns = [col[1] for col in cursor.fetchall()]
    expected_columns = [
        'ease_col',
        'ease_row',
        'Wmin',
        'Wmax',
        'LA',
        'LB',
        'LC',
        'rmse',
    ]
    assert sorted(columns) == sorted(expected_columns)
    cursor.close()


def test_get_loss_function_from_db(in_memory_loss_function_db):
    """
    Test get_loss_function_from_db with valid and invalid inputs.
    """
    conn = in_memory_loss_function_db

    # Test valid case
    lf = get_loss_function_from_db(conn, 1, 1)
    assert isinstance(lf, LossFunction)
    assert np.isclose(lf.W[0], 0.1)
    assert np.isclose(lf.W[-1], 0.9)

    # Test invalid case (non-existent ease_col, ease_row)
    with pytest.raises(ValueError, match='EASE col=99 row=99 not in ranges: '):
        get_loss_function_from_db(conn, 99, 99)
