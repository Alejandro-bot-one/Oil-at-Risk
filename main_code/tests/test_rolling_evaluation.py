"""Tests for rolling-window direct forecasting evaluation."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from auxi.diagnostics.direct_forecasting import compute_rolling_pinball


@pytest.fixture
def rolling_panel():
    """Small synthetic panel for rolling-window tests.
    200 rows, enough for a window of 100 + 50 test origins + max horizon of ~20.
    """
    rng = np.random.default_rng(99)
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    gpr = rng.normal(0.0, 1.0, n)
    brent = 0.5 * gpr + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"Brent_Return": brent, "gpr": gpr}, index=idx)


def test_compute_rolling_pinball_returns_correct_shape(rolling_panel):
    """One horizon, one quantile → one row."""
    result = compute_rolling_pinball(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        taus=[0.5],
        max_h=1,
        window_size=100,
        test_start_date="2020-07-01",
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # 1 horizon × 1 tau
    assert set(result.columns) >= {"Horizon", "Tau", "Avg_Pinball_Loss", "N_Forecasts"}


def test_compute_rolling_pinball_multiple_taus_horizons(rolling_panel):
    """3 horizons × 2 quantiles → 6 rows."""
    result = compute_rolling_pinball(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        taus=[0.05, 0.95],
        max_h=3,
        window_size=100,
        test_start_date="2020-07-01",
    )
    assert len(result) == 6  # 3 × 2
    assert (result["N_Forecasts"] > 0).all()
    assert (result["Avg_Pinball_Loss"] >= 0).all()


def test_compute_rolling_pinball_with_controls(rolling_panel):
    """Controls column should be accepted without error."""
    rolling_panel = rolling_panel.copy()
    rng = np.random.default_rng(77)
    rolling_panel["control1"] = rng.normal(0, 1, len(rolling_panel))

    result = compute_rolling_pinball(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        taus=[0.5],
        max_h=1,
        window_size=100,
        test_start_date="2020-07-01",
        controls=["control1"],
    )
    assert len(result) == 1
    assert result["N_Forecasts"].iloc[0] > 0


def test_compute_rolling_pinball_does_not_mutate_input(rolling_panel):
    """Input df must not be mutated."""
    original_cols = list(rolling_panel.columns)
    original_len = len(rolling_panel)
    compute_rolling_pinball(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        taus=[0.5],
        max_h=1,
        window_size=100,
        test_start_date="2020-07-01",
    )
    assert list(rolling_panel.columns) == original_cols
    assert len(rolling_panel) == original_len


from auxi.diagnostics.direct_forecasting import get_oos_predictions_rolling


def test_get_oos_predictions_rolling_returns_aligned_series(rolling_panel):
    """Returns two aligned Series with DatetimeIndex."""
    realized, forecasted = get_oos_predictions_rolling(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        tau=0.5,
        h=1,
        window_size=100,
        test_start_date="2020-07-01",
    )
    assert isinstance(realized, pd.Series)
    assert isinstance(forecasted, pd.Series)
    assert len(realized) == len(forecasted)
    assert len(realized) > 0
    assert realized.index.equals(forecasted.index)


def test_get_oos_predictions_rolling_does_not_mutate(rolling_panel):
    """Input df must not be mutated."""
    original_cols = list(rolling_panel.columns)
    get_oos_predictions_rolling(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        tau=0.5,
        h=1,
        window_size=100,
        test_start_date="2020-07-01",
    )
    assert list(rolling_panel.columns) == original_cols


from auxi.diagnostics.direct_forecasting import plot_rolling_pinball


def test_plot_rolling_pinball_smoke(rolling_panel):
    """Smoke test: plot renders without error under Agg backend."""
    result_df = compute_rolling_pinball(
        df=rolling_panel,
        x="gpr",
        y="Brent_Return",
        taus=[0.05, 0.50, 0.95],
        max_h=2,
        window_size=100,
        test_start_date="2020-07-01",
    )
    fig = plot_rolling_pinball(result_df)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_new_functions_resolvable_via_diagnostics():
    """All new functions must be importable from auxi.diagnostics."""
    import auxi.diagnostics as diags
    assert callable(diags.compute_rolling_pinball)
    assert callable(diags.plot_rolling_pinball)
    assert callable(diags.get_oos_predictions_rolling)
    # Old function still accessible under both names
    assert callable(diags.evaluate_direct_forecasting)
    assert callable(diags.evaluate_direct_forecasting_single)
