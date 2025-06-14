"""Loss functions"""

import datetime

import numpy as np


class LossFunction:  # pylint: disable=too-few-public-methods
    """A loss function sensu Koster et al (2017)

    A function L from equation 1 in Koster et al (2017) that accepts a SMAP
    retrieval W and returns a rate of decrease in soil moisture by evaporation and
    drainage.  Units of the returned loss function are 1 / h.

    As described in section 2.c of the paper, the loss function is specified by:
    1. A maximum soil moisture, Wmax.  Wmax = Whigh + 0.1 (Whigh - Wlow) where
       Whigh and Wlow are the highest and lowest soil moistures retrieved in the
       grid.  L(Wmax) = Wmax / (24 h).
    2. A minimum soil moisture, Wmin = Wlow.  L(Wmin) = 0.
    3. Parameters LA, LB, LC specifying the loss rate at three intermediate
       soil moisture values WA, WB, WC that divide the interval [Wmin, Wmax] into
       four equal segments:
       WA, WB, WC = [Wmin + f x (Wmax - Wmin) for f in (0.25, 0.5, 0.75)]
    4. L at intermediate soil moistures is commputed by linear interpolation.

    The loss function extrapolates values L(Wmin) for W < Wmin and L(Wmax)
    for W > WMax.

    """

    def __init__(self, Wmax, Wmin, LA, LB, LC):
        WA, WB, WC = [Wmin + f * (Wmax - Wmin) for f in (0.25, 0.5, 0.75)]
        W_values = np.array([Wmin, WA, WB, WC, Wmax], dtype='float64')
        if not (np.diff(W_values) >= 0).all():
            raise ValueError(f'W values not non-decreasing: {W_values}')
        # "We set the value of the loss function at the low end L(Wmin) to 0"
        Lmin = 0.0
        Lmax = Wmax / 24.0  # "L(Wmax) = Wmax volumetric units per day"
        self.W = W_values
        self.L = np.array([Lmin, LA, LB, LC, Lmax], dtype='float64')

    def __call__(self, soil_moisture):
        """Compute loss function L(w) after Koster et al (2017) equation 1

        Accepts a SMAP retrieval W and returns a rate of decrease in soil moisture by
        evaporation and drainage.

        """
        return np.interp(soil_moisture, self.W, self.L)

    def simulate_soil_moisture(
        self, Winit, Wmax, P_of_t_mm_d, ts_start, ts_thru, max_infiltration_h=1.0
    ):
        """Simulate soil moisture given precipitation and a loss function

        Simulates soil moisture following Koster et al (2017) section 2,
        equations 1--3.
        W[i + 1] = W[i] - L(W[i]) Δt + Wadd (eqn 1)
        Wadd = I Δt / D (eqn 2)
        I = min(P, (Wmax - W[i]) D / nd) (eqn 3)

        Δt = 1 because calculations are done on an hourly grid in time units of hours.
        D = 50 mm is the effective thickness of SMAP soil moisture observations.
        nd is a time scale limiting the infiltration rate.

        In Koster et al (2017), nd is set to one day (24 h) such that the maximum
        infiltration rate is "the rate [that] if it were to be applied over a full
        day... would exceed the current soil water deficit".  Because the time step is 1
        h, this implies that soil moisture cannot reach full saturation (Wmax).  In
        some settings, this assumption seems unrealistic and so the default for this
        parameter is 1.0, which simply limits infiltration to prevent the soil moisture
        from exceeding Wmax.  The original behavior from Koster et al (2017) can be
        recovered by setting max_infiltration_h to 24.

        Inputs:
          1. Winit: Initial soil moisture Winit(ts_start).
          2. Wmax: Maximum soil moisture in the pixel.
          3. P_of_t_mm_d: Function returning precipitation (mm / d)
                          at time t (seconds since the epoch).
          4. loss_function: Loss function L in volumetric units per hour.
          5. ts_start: Timestamp at which to start simulations, in seconds
             since the UNIX epoch (1970-01-01 00:00:00Z).
          6. ts_thru: Timestamp through which to simulate.
          7. max_infiltration_h: Time scale limiting the infiltration rate (nd in eqn 3).
        Returns:
          (time, W)
        where time is in seconds since the epoch on an hourly grid and W is the simulated
        soil moisture.
        time[0] is the smallest round hour >= ts_start, and time[-1] is the largest
        round hour <= ts_thru.

        """
        # Calculations are done on an hourly grid, then converted back to seconds since
        # the epoch at exit.
        delta_t = 1
        D = 50  # mm
        L = self
        nd = max_infiltration_h

        start_hour = int(np.ceil(ts_start / 3600.0))
        thru_hour = int(np.floor(ts_thru / 3600.0))
        if thru_hour <= start_hour:
            raise ValueError(
                'Zero-length simulation '
                f'from {datetime.datetime.fromtimestamp(ts_start, datetime.UTC)} '
                f'to {datetime.datetime.fromtimestamp(ts_thru, datetime.UTC)}'
            )
        t = np.arange(start_hour, thru_hour + 1, dtype='int').astype('float64')
        P = P_of_t_mm_d(t * 3600) / 24
        assert P.shape == t.shape
        W = np.empty(shape=t.shape, dtype='float64')
        W[:] = float('NaN')
        W[0] = Winit
        for i in range(len(t) - 1):
            I = min((P[i], (Wmax - W[i]) * D / nd))  # noqa: E741
            Wadd = I * delta_t / D
            W[i + 1] = W[i] - L(W[i]) * delta_t + Wadd
        assert not np.isnan(W).any(), W
        return (t * 3600, W)
