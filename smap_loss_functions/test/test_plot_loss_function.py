"""Tests for plot_loss_function"""

import matplotlib.pyplot as plt
import pytest


from smap_loss_functions.loss_function import LossFunction
from smap_loss_functions.plot_loss_function import (
    plot_loss_function,
    plot_loss_function_simulation,
)


@pytest.mark.mpl_image_compare
def test_plot_loss_function_execution():
    """
    Test that plot_loss_function executes and returns a matplotlib Figure.
    """
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)

    fig = plot_loss_function(loss_func)
    assert isinstance(fig, plt.Figure)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_loss_function_simulation_execution(in_memory_loss_function_db):
    """
    Test that plot_loss_function_simulation executes and returns a matplotlib Figure.
    """
    conn = in_memory_loss_function_db
    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)
    ease_col, ease_row = 1, 1

    fig = plot_loss_function_simulation(loss_func, conn, ease_col, ease_row)
    assert isinstance(fig, plt.Figure)
    return fig


def test_plot_loss_function_simulation_insufficient_smap_data(
    in_memory_loss_function_db,
):
    """
    Test plot_loss_function_simulation with insufficient SMAP data.
    """
    conn = in_memory_loss_function_db
    cursor = conn.cursor()
    # Delete all but one SMAP data entry for col=1, row=1
    cursor.execute(
        'DELETE FROM smap_data '
        'WHERE ease_col = ? AND ease_row = ? '
        "AND start_datetime != '2023-01-01 00:00:00'",
        (1, 1),
    )
    conn.commit()

    Wmax, Wmin, LA, LB, LC = 0.9, 0.1, 0.01, 0.05, 0.1
    loss_func = LossFunction(Wmax, Wmin, LA, LB, LC)
    ease_col, ease_row = 1, 1

    with pytest.raises(
        ValueError, match='Only 1 SMAP values in col 1 row 1; nothing to do'
    ):
        plot_loss_function_simulation(loss_func, conn, ease_col, ease_row)
