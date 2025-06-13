import pytest
import sqlite3
import numpy as np

from smap_loss_functions import fit_loss_functions as flf


@pytest.fixture
def in_memory_input_db():
    """Fixture for an in-memory SQLite input database with smap_data and imerg_data."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create smap_data table
    cursor.execute("""
        CREATE TABLE smap_data (
            ease_col INTEGER,
            ease_row INTEGER,
            start_datetime TEXT,
            thru_datetime TEXT,
            soil_moisture REAL
        )
    """)
    # Create imerg_data table
    cursor.execute("""
        CREATE TABLE imerg_data (
            ease_col INTEGER,
            ease_row INTEGER,
            start_datetime TEXT,
            precipitation REAL
        )
    """)

    # Populate with some dummy data for testing
    # SMAP data (daily, for simplicity, using 12:00:00 for start and thru)
    # These timestamps are for 2023-01-01 12:00:00 UTC onwards
    smap_data_entries = [
        (10, 20, '2023-01-01 12:00:00', '2023-01-01 12:00:00', 0.25),
        (10, 20, '2023-01-02 12:00:00', '2023-01-02 12:00:00', 0.28),
        (10, 20, '2023-01-03 12:00:00', '2023-01-03 12:00:00', 0.27),
        (10, 20, '2023-01-04 12:00:00', '2023-01-04 12:00:00', 0.30),
        # Another cell with less than 2 data points to test the 'skip' condition
        (11, 21, '2023-01-01 12:00:00', '2023-01-01 12:00:00', 0.15),
    ]
    cursor.executemany(
        'INSERT INTO smap_data VALUES (?, ?, ?, ?, ?)', smap_data_entries
    )

    # IMERG data (daily, precipitation in mm/d)
    # Timestamps are for 2022-12-31 00:00:00 UTC onwards (daily values)
    imerg_data_entries = [
        (10, 20, '2022-12-31 00:00:00', 5.0),  # Day before SMAP data starts
        (10, 20, '2023-01-01 00:00:00', 10.0),
        (10, 20, '2023-01-02 00:00:00', 1.0),
        (10, 20, '2023-01-03 00:00:00', 0.0),
        (10, 20, '2023-01-04 00:00:00', 20.0),
        (10, 20, '2023-01-05 00:00:00', 2.0),  # Day after SMAP data ends
        # For the other cell (11, 21)
        (11, 21, '2022-12-31 00:00:00', 3.0),
        (11, 21, '2023-01-01 00:00:00', 7.0),
        (11, 21, '2023-01-02 00:00:00', 0.5),
    ]
    cursor.executemany('INSERT INTO imerg_data VALUES (?, ?, ?, ?)', imerg_data_entries)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def in_memory_output_db():
    """Fixture for an in-memory SQLite output database."""
    conn = sqlite3.connect(':memory:')
    # The set_up_loss_function_db function will create the table,
    # so we just yield the connection here.
    yield conn
    conn.close()


# --- Unit Tests ---


def test_set_up_loss_function_db(in_memory_output_db):
    """Test set_up_loss_function_db creates the table correctly."""
    flf.set_up_loss_function_db(in_memory_output_db)
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


def test_loss_function_init():
    """Test LossFunction initialization."""
    Wmax = 0.5
    Wmin = 0.1
    LA, LB, LC = 0.01, 0.02, 0.03
    loss_func = flf.LossFunction(Wmax, Wmin, LA, LB, LC)

    expected_W_values = np.array(
        [
            Wmin,
            Wmin + 0.25 * (Wmax - Wmin),
            Wmin + 0.5 * (Wmax - Wmin),
            Wmin + 0.75 * (Wmax - Wmin),
            Wmax,
        ]
    )
    assert np.allclose(loss_func.W, expected_W_values)
    assert loss_func.L[0] == 0.0
    assert np.allclose(loss_func.L[1:4], [LA, LB, LC])
    assert np.isclose(loss_func.L[-1], Wmax / 24.0)


def test_loss_function_call():
    """Test LossFunction __call__ (interpolation)."""
    Wmax = 0.5
    Wmin = 0.1
    LA, LB, LC = 0.01, 0.02, 0.03
    loss_func = flf.LossFunction(Wmax, Wmin, LA, LB, LC)

    # Test at Wmin
    assert np.isclose(loss_func(Wmin), 0.0)
    # Test at Wmax
    assert np.isclose(loss_func(Wmax), Wmax / 24.0)
    # Test at an intermediate point (e.g., WA)
    WA = Wmin + 0.25 * (Wmax - Wmin)
    assert np.isclose(loss_func(WA), LA)
    # Test outside bounds (extrapolation should return edge values)
    assert np.isclose(loss_func(Wmin - 0.1), 0.0)
    assert np.isclose(loss_func(Wmax + 0.1), Wmax / 24.0)
    # Test interpolation between Wmin and WA
    test_w = Wmin + 0.1 * (WA - Wmin)
    expected_l = np.interp(test_w, loss_func.W, loss_func.L)
    assert np.isclose(loss_func(test_w), expected_l)


def test_get_soil_moisture(in_memory_input_db):
    """Test get_soil_moisture retrieves data correctly."""
    cursor = in_memory_input_db.cursor()
    col, row = 10, 20
    smap_time, soil_moisture = flf.get_soil_moisture(cursor, col, row)

    assert len(smap_time) == 4
    assert len(soil_moisture) == 4
    assert np.allclose(soil_moisture, [0.25, 0.28, 0.27, 0.30])
    # Check timestamps: 2023-01-01 12:00:00Z -> 1672574400.0 (middle of day)
    # Subsequent days are 86400 seconds later
    expected_times = np.array(
        [
            1672574400.0,
            1672574400.0 + 86400,
            1672574400.0 + 2 * 86400,
            1672574400.0 + 3 * 86400,
        ]
    )
    print([int(v) for v in smap_time])
    assert np.allclose(smap_time, expected_times)


def test_get_precipitation(in_memory_input_db):
    """Test get_precipitation retrieves data correctly."""
    cursor = in_memory_input_db.cursor()
    col, row = 10, 20
    imerg_start, precipitation = flf.get_precipitation(cursor, col, row)

    assert len(imerg_start) == 6
    assert len(precipitation) == 6
    assert np.allclose(precipitation, [5.0, 10.0, 1.0, 0.0, 20.0, 2.0])
    # Check timestamps: 2022-12-31 00:00:00 -> 1672434000.0
    expected_times = np.array(
        [
            1672434000.0,
            1672434000.0 + 86400,
            1672434000.0 + 2 * 86400,
            1672434000.0 + 3 * 86400,
            1672434000.0 + 4 * 86400,
            1672434000.0 + 5 * 86400,
        ]
    )
    assert np.allclose(imerg_start, expected_times)


def test_create_piecewise_constant_interpolant():
    """Test create_piecewise_constant_interpolant."""
    x = np.array([10, 20, 30])
    y = np.array([1, 5, 10])
    interpolant = flf.create_piecewise_constant_interpolant(x, y)

    assert interpolant(5) == 1  # Less than x[0]
    assert interpolant(10) == 1  # At x[0]
    assert interpolant(15) == 1  # Between x[0] and x[1]
    assert interpolant(20) == 5  # At x[1]
    assert interpolant(25) == 5  # Between x[1] and x[2]
    assert interpolant(30) == 10  # At x[2]
    assert interpolant(35) == 10  # Greater than x[-1]

    # Test with single point
    x_single = np.array([100])
    y_single = np.array([50])
    interpolant_single = flf.create_piecewise_constant_interpolant(x_single, y_single)
    assert interpolant_single(0) == 50
    assert interpolant_single(100) == 50
    assert interpolant_single(200) == 50

    # Test assertion for non-strictly increasing x
    with pytest.raises(AssertionError, match='not strictly increasing'):
        flf.create_piecewise_constant_interpolant(np.array([10, 5]), np.array([1, 2]))


def test_clip_to_timestamp_range():
    """Test clip_to_timestamp_range."""
    time = np.array([100, 200, 300, 400, 500])
    value = np.array([1, 2, 3, 4, 5])

    # Clip within range
    clipped_time, clipped_value = flf.clip_to_timestamp_range(time, value, 150, 450)
    assert np.array_equal(clipped_time, np.array([200, 300, 400]))
    assert np.array_equal(clipped_value, np.array([2, 3, 4]))

    # Clip from start
    clipped_time, clipped_value = flf.clip_to_timestamp_range(time, value, 0, 350)
    assert np.array_equal(clipped_time, np.array([100, 200, 300]))
    assert np.array_equal(clipped_value, np.array([1, 2, 3]))

    # Clip to end
    clipped_time, clipped_value = flf.clip_to_timestamp_range(time, value, 350, 600)
    assert np.array_equal(clipped_time, np.array([400, 500]))
    assert np.array_equal(clipped_value, np.array([4, 5]))

    # Empty result
    clipped_time, clipped_value = flf.clip_to_timestamp_range(time, value, 600, 700)
    assert len(clipped_time) == 0
    assert len(clipped_value) == 0


def test_get_imerg_interpolant(in_memory_input_db):
    """Test get_imerg_interpolant"""
    cursor = in_memory_input_db.cursor()
    col, row = 10, 20
    # Probe times around SMAP data interval (1 day before/after SMAP min/max)
    start_time_smap_min = 1672574400.0  # epoch seconds for 2023-01-01 12:00:00Z
    thru_time_smap_max = 1672833600.0  # epoch seconds for 2023-01-04 12:00:00Z
    # Add/subtract 129600 (2 days in seconds) as per get_imerg_interpolant's call
    query_start_time = start_time_smap_min - 129600
    query_thru_time = thru_time_smap_max + 129600

    interpolant = flf.get_imerg_interpolant(
        cursor, col, row, query_start_time, query_thru_time
    )

    # Expected IMERG data for col 10, row 20, after clipping:
    # 2022-12-31 00:00:00Z (1672444800.0) -> 5.0
    # 2023-01-01 00:00:00Z (1672574400.0) -> 10.0
    # ...
    # 2023-01-05 00:00:00Z (1672876800.0) -> 2.0

    # Test at specific timestamps (hourly resolution is 3600s)
    # A time just after IMERG data starts (2022-12-31 00:00:00Z)
    assert interpolant(1672444800.0 + 100) == 5.0
    # A time just before next IMERG data point (2023-01-01 00:00:00Z)
    assert interpolant(1672531200.0 - 100) == 5.0
    # A time just after next IMERG data point
    assert interpolant(1672617600.0 + 100) == 1.0
    # At the exact end of the queried range (2023-01-05 12:00:00Z)
    assert interpolant(query_thru_time) == 2.0

    # Test extrapolation outside the *queried* range (should return edge values)
    assert interpolant(query_start_time - 1000) == 5.0
    assert interpolant(query_thru_time + 1000) == 2.0


def test_simulate_soil_moisture():
    """Test simulate_soil_moisture for a basic scenario."""
    Winit = 0.25
    Wmax = 0.5
    # Simple LossFunction (LA, LB, LC chosen for a plausible test)
    loss_func = flf.LossFunction(Wmax=Wmax, Wmin=0.1, LA=0.01, LB=0.015, LC=0.02)
    # Constant precipitation of 24 mm/day (1 mm/hr)
    P_of_t_mm_d = lambda t: np.full_like(t, 24.0)

    # Simulate for 3 hours (total 4 hourly points: t0, t1, t2, t3)
    ts_start = 1672555200.0  # Jan 1 2023 12:00:00 (epoch seconds)
    ts_thru = ts_start + 3 * 3600  # 3 hours later

    times, W = flf.simulate_soil_moisture(
        Winit, Wmax, loss_func, P_of_t_mm_d, ts_start, ts_thru
    )

    # XXX Check me
    # assert len(times) == 4
    assert np.isclose(times[0], ts_start)  # Should be exactly the start hour
    assert np.isclose(times[-1], ts_thru)  # Should be exactly the thru hour

    assert np.isclose(W[0], Winit)

    # Manual calculation for the first step:
    # W[i + 1] = W[i] - L(W[i]) * Δt + Wadd
    # Δt = 1 hour, D = 50 mm, nd = 1.0 (default max_infiltration_h)
    # P_hourly = P_of_t_mm_d / 24 = 24.0 / 24 = 1.0 mm/hr
    # I = min(P_hourly, (Wmax - W[i]) * D / nd)
    # Wadd = I * Δt / D

    # For W[0] = 0.25:
    L_W0 = loss_func(
        Winit
    )  # Loss function value at Winit (interpolation from init test)
    # For Winit=0.25, it's between WA=0.2 and WB=0.3
    # L_values are [0.0, 0.01, 0.015, 0.02, 0.5/24]
    # L(0.25) is halfway between L(0.2)=0.01 and L(0.3)=0.015, so 0.0125
    assert np.isclose(L_W0, 0.0125)

    I_0 = min(1.0, (Wmax - Winit) * 50 / 1.0)  # min(1.0, (0.5 - 0.25) * 50 / 1.0)
    # min(1.0, 0.25 * 50) = min(1.0, 12.5) = 1.0
    Wadd_0 = I_0 * 1 / 50  # 1.0 / 50 = 0.02

    expected_W1 = Winit - L_W0 * 1 + Wadd_0  # 0.25 - 0.0125 + 0.02 = 0.2575
    assert np.isclose(W[1], expected_W1)
    assert not np.isnan(W).any()

    # Test zero-length simulation
    with pytest.raises(ValueError, match='Zero-length simulation'):
        flf.simulate_soil_moisture(
            Winit, Wmax, loss_func, P_of_t_mm_d, ts_start, ts_start + 100
        )  # Less than one full hour


def test_optimize_loss_function_parameters():
    """Test optimize_loss_function_parameters with a simple objective."""
    Lmax = 1.0

    # A simple objective function where the true minimum satisfies constraints
    # Minimum is at LA=0.1, LB=0.2, LC=0.3
    def simple_objective(params):
        LA, LB, LC = params
        return (LA - 0.1) ** 2 + (LB - 0.2) ** 2 + (LC - 0.3) ** 2

    result = flf.optimize_loss_function_parameters(
        Lmax, simple_objective, n=10, iters=1
    )

    assert result.success is True
    # shgo might not find the *exact* minimum in limited iterations, but it should be close
    assert np.allclose(result.x, [0.1, 0.2, 0.3], atol=1e-3)
    assert np.isclose(result.fun, 0.0, atol=1e-6)

    # Test with an objective that initially violates constraints but should still converge
    # e.g., unconstrained min would be LA=0.3, LB=0.1, LC=0.2 (violates LA <= LB)
    # The optimizer should find the best constrained solution.
    def constrained_violating_objective(params):
        LA, LB, LC = params
        return (LA - 0.3) ** 2 + (LB - 0.1) ** 2 + (LC - 0.2) ** 2

    result_constrained = flf.optimize_loss_function_parameters(
        Lmax, constrained_violating_objective, n=10, iters=1
    )
    assert result_constrained.success is True
    # Verify that the constraints are approximately respected
    assert round(result_constrained.x[0], 12) <= round(result_constrained.x[1], 12)
    assert round(result_constrained.x[1], 12) <= round(result_constrained.x[2], 12)
    assert 0 <= result_constrained.x[0] <= Lmax
    assert 0 <= result_constrained.x[1] <= Lmax
    assert 0 <= result_constrained.x[2] <= Lmax


@pytest.mark.filterwarnings(
    'ignore:Values in x were outside bounds during a minimize step, '
    'clipping to bounds:RuntimeWarning'
)
def test_get_optimized_loss_function():
    """Test get_optimized_loss_function with simplified inputs."""
    # Test case 1: Constant soil moisture (should result in Wmin=Wmax, LA=LB=LC=0)
    smap_time_const = np.array([1, 2, 3]) * 86400.0  # Daily timestamps
    soil_moisture_const = np.full_like(smap_time_const, 0.3)

    # Dummy interpolant for precipitation
    interpolate_imerg_dummy = lambda t: np.full_like(
        t, 10.0
    )  # Constant 10 mm/day precip

    L_const, rmse_const = flf.get_optimized_loss_function(
        smap_time_const, soil_moisture_const, interpolate_imerg_dummy
    )
    assert isinstance(L_const, flf.LossFunction)
    assert np.isclose(L_const.W[0], L_const.W[-1])  # Wmin should be close to Wmax
    assert (
        L_const.L[1] == 0 and L_const.L[2] == 0 and L_const.L[3] == 0
    )  # LA, LB, LC should be 0
    assert rmse_const is None  # For constant SMAP, RMSE is not computed meaningfully

    # Test case 2: Increasing soil moisture (requires actual optimization)
    smap_time_inc = np.array([1, 2, 3, 4]) * 86400.0
    soil_moisture_inc = np.array([0.2, 0.25, 0.3, 0.35])

    L_inc, rmse_inc = flf.get_optimized_loss_function(
        smap_time_inc, soil_moisture_inc, interpolate_imerg_dummy
    )

    assert isinstance(L_inc, flf.LossFunction)
    assert rmse_inc is not None
    assert rmse_inc >= 0  # RMSE should be non-negative

    # Check that the fitted parameters adhere to constraints and general sanity
    # Wmin = 0.2, Wmax = 0.35 + 0.1*(0.35-0.2) = 0.35 + 0.015 = 0.365
    # Lmax for this scenario = Wmax / 24 = 0.365 / 24
    expected_Lmax = (
        soil_moisture_inc.max()
        + 0.1 * (soil_moisture_inc.max() - soil_moisture_inc.min())
    ) / 24.0
    assert 0 <= L_inc.L[1] <= expected_Lmax  # LA
    assert 0 <= L_inc.L[2] <= expected_Lmax  # LB
    assert 0 <= L_inc.L[3] <= expected_Lmax  # LC
    # Test that inequality constraints are approximately satisfied
    assert round(L_inc.L[1], 12) <= round(L_inc.L[2], 12)  # LA <= LB
    assert round(L_inc.L[2], 12) <= round(L_inc.L[3], 12)  # LB <= LC


def test_fit_loss_functions(in_memory_input_db, in_memory_output_db):
    """Test the main fit_loss_functions workflow."""
    flf.set_up_loss_function_db(in_memory_output_db)  # Ensure output table exists

    # Run the main function
    result_code = flf.fit_loss_functions(in_memory_input_db, in_memory_output_db)
    assert result_code == 0  # Should return 0 on success

    # Verify data in the output database
    out_cursor = in_memory_output_db.cursor()
    out_cursor.execute(
        'SELECT ease_col, ease_row, Wmin, Wmax, LA, LB, LC, rmse FROM loss_function'
    )
    fitted_data = out_cursor.fetchall()
    out_cursor.close()

    # From the fixture, we inserted two distinct cells: (10, 20) and (11, 21).
    # Cell (11, 21) had only 1 SMAP data point, so it should be skipped by the 'len(soil_moisture) < 2' check.
    # Therefore, only cell (10, 20) should have a fitted entry.
    assert len(fitted_data) == 1
    col, row, Wmin, Wmax, LA, LB, LC, rmse = fitted_data[0]

    assert col == 10
    assert row == 20
    # Assert Wmin and Wmax are derived from the input SMAP data for cell (10, 20)
    assert np.isclose(Wmin, 0.25)  # Minimum soil_moisture for (10,20)
    assert np.isclose(
        Wmax, 0.30 + 0.1 * (0.30 - 0.25)
    )  # Wmax formula: max + 0.1 * (max - min)
    assert LA >= 0 and LB >= 0 and LC >= 0
    assert LA <= LB <= LC  # Verify optimization constraints are respected
    assert rmse >= 0  # RMSE should be non-negative and finite
