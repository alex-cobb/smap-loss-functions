"""Plot loss functions sensu Koster et al (2017)

Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A data-driven approach for
  daily real-time estimates and forecasts of near-surface soil moisture. Journal of
  Hydrometeorology, 18(3), 837–843. https://doi.org/10.1175/jhm-d-16-0285.1

"""

import datetime

import matplotlib.pyplot as plt

import numpy as np

from .smap_db import (
    get_soil_moisture,
    get_precipitation,
    create_piecewise_constant_interpolant,
    clip_to_timestamp_range,
)


def plot_loss_function(loss_function):
    """Plot a loss function"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    W_grid = np.linspace(loss_function.W[0], loss_function.W[-1], 100)
    ax.plot(
        W_grid,
        [loss_function(W) for W in W_grid],
        'b-',
    )
    ax.set_xlabel('SMAP soil moisture')
    ax.set_ylabel('Loss rate, 1 / h')
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_loss_function_simulation(loss_function, smap_connection, ease_col, ease_row):
    """Plot loss function simulations against SMAP data

    Given a connection to a database with SMAP and IMERG data, simulates SMAP dynamics
    given their initial value driven by the IMERG precipitation and plots it together
    with the SMAP soil moisture in the database.

    """
    cursor = smap_connection.cursor()
    smap_time, soil_moisture = get_soil_moisture(cursor, ease_col, ease_row)
    if not len(smap_time) >= 2:
        raise ValueError(
            f'Only {len(smap_time)} SMAP values in col {ease_col} row {ease_row}; '
            'nothing to do'
        )
    imerg_start, precipitation = get_precipitation(cursor, ease_col, ease_row)
    # Clip IMERG data to a day before and after SMAP datetime range
    imerg_start, precipitation = clip_to_timestamp_range(
        time=imerg_start,
        value=precipitation,
        start_time=smap_time.min() - 86400,
        thru_time=smap_time.max() + 86400,
    )
    interpolate_imerg = create_piecewise_constant_interpolant(
        imerg_start, precipitation
    )
    tsim, Wsim = loss_function.simulate_soil_moisture(
        Winit=soil_moisture[0],
        Wmax=loss_function.W[-1],
        P_of_t_mm_d=interpolate_imerg,
        ts_start=smap_time[0],
        ts_thru=smap_time[-1],
    )

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(
        [datetime.datetime.fromtimestamp(ts) for ts in smap_time],
        soil_moisture,
        'bo-',
    )
    axs[0].plot([datetime.datetime.fromtimestamp(ts) for ts in tsim], Wsim, 'r.-')
    time_grid = np.linspace(imerg_start.min(), imerg_start.max(), len(imerg_start) * 3)
    imerg_on_grid = interpolate_imerg(time_grid)
    # Piecewise-constant (stair) plot, imerg
    axs[1].step(
        [datetime.datetime.fromtimestamp(ts) for ts in imerg_start],
        precipitation,
        '-',
        where='post',
    )
    # Points, imerg_on_grid
    axs[1].plot(
        [datetime.datetime.fromtimestamp(ts) for ts in time_grid],
        imerg_on_grid,
        'k.',
    )
    axs[0].set_ylabel('SMAP soil moisture')
    axs[1].set_ylabel('GPM IMERG precipitation')
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig
