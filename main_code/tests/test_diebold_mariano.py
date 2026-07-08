"""Tests for the Diebold-Mariano test functions in diagnostics/direct_forecasting.py."""
import numpy as np
import pandas as pd
import pytest

from auxi.diagnostics.direct_forecasting import (
    tick_loss_series,
    diebold_mariano_test,
    compute_dm_comparison,
)
from auxi.qreg import pinball_loss


class TestTickLossSeries:
    def test_known_values(self):
        tau = 0.05
        realized = np.array([1.0, -1.0, 0.5])
        forecasted = np.array([0.0, 0.0, 0.0])
        losses = tick_loss_series(tau, realized, forecasted)
        expected = np.array([0.05, 0.95, 0.025])
        np.testing.assert_allclose(losses, expected)

    def test_symmetric_at_median(self):
        tau = 0.5
        realized = np.array([2.0, -2.0])
        forecasted = np.array([0.0, 0.0])
        losses = tick_loss_series(tau, realized, forecasted)
        np.testing.assert_allclose(losses, [1.0, 1.0])

    def test_mean_matches_pinball(self):
        rng = np.random.default_rng(42)
        for tau in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
            realized = rng.normal(0, 1, 500)
            forecasted = rng.normal(0, 1, 500)
            expected = pinball_loss(tau, realized, forecasted)
            actual = np.mean(tick_loss_series(tau, realized, forecasted))
            np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        for tau in [0.01, 0.05, 0.50, 0.95, 0.99]:
            losses = tick_loss_series(
                tau, rng.normal(0, 1, 100), rng.normal(0, 1, 100)
            )
            assert np.all(losses >= 0)


class TestDieboldMarianoTest:
    def test_identical_losses_not_rejected(self):
        rng = np.random.default_rng(42)
        losses = rng.normal(1, 0.5, 200)
        result = diebold_mariano_test(losses, losses, h=1)
        assert result["alpha"] == 0.0
        assert result["p_value"] == pytest.approx(1.0)

    def test_clearly_different_losses(self):
        rng = np.random.default_rng(42)
        loss_1 = rng.normal(1.0, 0.3, 500)
        loss_2 = rng.normal(2.0, 0.3, 500)
        result = diebold_mariano_test(loss_1, loss_2, h=1)
        assert result["alpha"] < 0
        assert result["p_value"] < 0.01

    def test_horizon_changes_variance(self):
        rng = np.random.default_rng(42)
        loss_1 = rng.normal(1.0, 0.5, 200)
        loss_2 = rng.normal(1.5, 0.5, 200)
        r1 = diebold_mariano_test(loss_1, loss_2, h=1)
        r5 = diebold_mariano_test(loss_1, loss_2, h=5)
        assert r1["t_stat"] != r5["t_stat"]

    def test_output_keys(self):
        result = diebold_mariano_test(np.ones(50), np.ones(50) * 2, h=1)
        assert set(result.keys()) == {"alpha", "t_stat", "p_value", "P"}
        assert result["P"] == 50

    def test_h1_no_lag_correction(self):
        rng = np.random.default_rng(42)
        d = rng.normal(-0.5, 1.0, 300)
        loss_1 = np.abs(d)
        loss_2 = np.abs(d) + 0.5
        result = diebold_mariano_test(loss_1, loss_2, h=1)
        assert result["alpha"] < 0
        assert result["P"] == 300


class TestComputeDmComparison:
    def test_output_shapes_3_models(self):
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        realized = pd.Series(rng.normal(0, 1, n), index=dates)
        models = {
            "A": pd.Series(rng.normal(0.1, 1, n), index=dates),
            "B": pd.Series(rng.normal(0, 1, n), index=dates),
            "C": pd.Series(rng.normal(-0.1, 1, n), index=dates),
        }
        error_df, dm_df = compute_dm_comparison(models, realized, tau=0.05, h=2)
        assert len(error_df) == 3
        assert len(dm_df) == 3  # C(3,2) = 3 pairs
        assert set(error_df.columns) >= {"Model", "RMSE", "MAPE", "Avg_Tick_Loss"}
        assert set(dm_df.columns) >= {
            "Model_1", "Model_2", "Alpha", "t_stat", "p_value", "Significance",
        }

    def test_output_shapes_2_models(self):
        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        realized = pd.Series(rng.normal(0, 1, n), index=dates)
        models = {
            "X": pd.Series(rng.normal(0, 1, n), index=dates),
            "Y": pd.Series(rng.normal(0, 1.5, n), index=dates),
        }
        error_df, dm_df = compute_dm_comparison(models, realized, tau=0.50, h=1)
        assert len(error_df) == 2
        assert len(dm_df) == 1

    def test_index_alignment(self):
        rng = np.random.default_rng(42)
        dates_full = pd.date_range("2020-01-01", periods=100, freq="D")
        realized = pd.Series(rng.normal(0, 1, 100), index=dates_full)
        models = {
            "A": pd.Series(rng.normal(0, 1, 100), index=dates_full),
            "B": pd.Series(rng.normal(0, 1, 80), index=dates_full[:80]),
        }
        error_df, dm_df = compute_dm_comparison(models, realized, tau=0.5, h=1)
        assert len(error_df) == 2
        assert len(dm_df) == 1

    def test_error_metrics_positive(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        realized = pd.Series(rng.normal(0, 2, n), index=dates)
        models = {
            "M1": pd.Series(rng.normal(0, 2, n), index=dates),
        }
        error_df, _ = compute_dm_comparison(models, realized, tau=0.05, h=1)
        assert error_df["RMSE"].iloc[0] > 0
        assert error_df["Avg_Tick_Loss"].iloc[0] > 0
