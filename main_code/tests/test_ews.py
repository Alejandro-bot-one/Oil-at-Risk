"""Tests for early warning system diagnostics (auxi/diagnostics/ews.py)."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from auxi.diagnostics.ews import (
    compute_ccf,
    granger_causality_test,
    compute_anticipation_test,
    compute_ews_battery,
    compute_coherence_test,
    plot_ccf,
    plot_ews_battery,
    plot_coherence_dashboard,
)


class TestComputeCCF:
    def test_known_lag(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        ccf_df, meta = compute_ccf(leader, follower, max_lag=12)
        assert meta["h_star"] == 5
        assert meta["r_at_hstar"] > 0.5
        assert isinstance(ccf_df, pd.DataFrame)
        assert set(ccf_df.columns) == {"lag", "r", "significant"}
        assert len(ccf_df) == 25  # -12..+12

    def test_self_correlation(self):
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 100))
        ccf_df, meta = compute_ccf(x, x, max_lag=5)
        row_zero = ccf_df.loc[ccf_df["lag"] == 0, "r"].values[0]
        np.testing.assert_allclose(row_zero, 1.0, atol=1e-10)

    def test_constant_series_raises(self):
        x = pd.Series(np.ones(50))
        y = pd.Series(np.ones(50))
        with pytest.raises(ValueError, match="zero variance"):
            compute_ccf(x, y)

    def test_short_series_reduces_lags(self):
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 1, 10))
        y = pd.Series(rng.normal(0, 1, 10))
        ccf_df, meta = compute_ccf(x, y, max_lag=24)
        max_lag_used = ccf_df["lag"].abs().max()
        assert max_lag_used < 24
        assert meta["n_obs"] == 10

    def test_ci95_correct(self):
        rng = np.random.default_rng(42)
        n = 100
        x = pd.Series(rng.normal(0, 1, n))
        y = pd.Series(rng.normal(0, 1, n))
        _, meta = compute_ccf(x, y, max_lag=5)
        np.testing.assert_allclose(meta["ci95"], 1.96 / np.sqrt(n), atol=1e-10)


class TestGrangerCausalityTest:
    def test_rejects_for_causal_pair(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        result = granger_causality_test(follower, leader, max_lag=12)
        assert result["p_value"] < 0.05
        assert result["significant"] is True
        assert result["F"] > 0
        assert 1 <= result["selected_lag"] <= 12

    def test_does_not_reject_independent(self, independent_pair):
        x, y = independent_pair
        result = granger_causality_test(y, x, max_lag=6)
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_bic_option(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        result = granger_causality_test(follower, leader, max_lag=8, criterion="bic")
        assert 1 <= result["selected_lag"] <= 8
        assert "F" in result
        assert "p_value" in result
        assert len(result["criterion_values"]) == 8

    def test_output_keys(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        result = granger_causality_test(follower, leader, max_lag=4)
        expected_keys = {"F", "p_value", "selected_lag", "criterion_values",
                         "significant", "stars"}
        assert set(result.keys()) == expected_keys

    def test_invalid_criterion_raises(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        with pytest.raises(ValueError, match="criterion"):
            granger_causality_test(follower, leader, criterion="hqic")


class TestComputeAnticipationTest:
    def test_returns_expected_keys(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        result = compute_anticipation_test(leader, follower)
        expected_keys = {"h_star", "r_at_hstar", "ccf_significant", "granger_F",
                         "granger_p", "granger_lag", "granger_stars", "ci95",
                         "n_obs", "ccf_df"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["ccf_df"], pd.DataFrame)

    def test_known_lag_propagates(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        result = compute_anticipation_test(leader, follower, max_lag_ccf=12)
        assert result["h_star"] == 5


class TestComputeEWSBattery:
    def test_output_shape_and_columns(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        rng = np.random.default_rng(99)
        indicators = {
            "leader": leader,
            "noise": pd.Series(rng.normal(0, 1, len(follower)), index=follower.index),
        }
        battery_df = compute_ews_battery(indicators, follower)
        assert isinstance(battery_df, pd.DataFrame)
        assert len(battery_df) == 2
        expected_cols = {"Indicator", "h_star", "r_at_hstar", "CCF_Significant",
                         "Granger_F", "Granger_p", "Granger_Lag", "Granger_Stars"}
        assert set(battery_df.columns) == expected_cols

    def test_leader_detected(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        indicators = {"leader": leader}
        battery_df = compute_ews_battery(indicators, follower, max_lag_ccf=12)
        assert battery_df.iloc[0]["h_star"] == 5


class TestComputeCoherenceTest:
    def test_pairwise_count(self):
        rng = np.random.default_rng(42)
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        indicators = {
            "A": pd.Series(rng.normal(0, 1, n), index=idx),
            "B": pd.Series(rng.normal(0, 1, n), index=idx),
            "C": pd.Series(rng.normal(0, 1, n), index=idx),
        }
        coherence_df = compute_coherence_test(indicators, max_lag=10)
        assert isinstance(coherence_df, pd.DataFrame)
        assert len(coherence_df) == 3  # C(3,2) = 3 pairs
        expected_cols = {"Series_X", "Series_Y", "h_star", "r_at_hstar", "Significant"}
        assert set(coherence_df.columns) == expected_cols

    def test_identical_series_coherent(self):
        rng = np.random.default_rng(42)
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        s = pd.Series(rng.normal(0, 1, n), index=idx)
        indicators = {"A": s, "B": s.copy()}
        coherence_df = compute_coherence_test(indicators, max_lag=5)
        assert coherence_df.iloc[0]["h_star"] == 0


class TestPlotCCF:
    def test_smoke(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        ccf_df, meta = compute_ccf(leader, follower, max_lag=12)
        fig, ax = plt.subplots()
        returned_ax = plot_ccf(ccf_df, meta, ax)
        assert returned_ax is ax
        plt.close(fig)

    def test_custom_color_and_title(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        ccf_df, meta = compute_ccf(leader, follower, max_lag=8)
        fig, ax = plt.subplots()
        returned_ax = plot_ccf(ccf_df, meta, ax, title="Custom", color="#C0392B")
        assert returned_ax is ax
        assert ax.get_title() == "Custom"
        plt.close(fig)


class TestPlotEWSBattery:
    def test_smoke(self, synthetic_ews_pair):
        leader, follower = synthetic_ews_pair
        rng = np.random.default_rng(99)
        indicators = {
            "leader": leader,
            "noise": pd.Series(rng.normal(0, 1, len(follower)), index=follower.index),
        }
        battery_df = compute_ews_battery(indicators, follower, max_lag_ccf=10)
        fig = plot_ews_battery(battery_df, indicators, follower, max_lag_ccf=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotCoherenceDashboard:
    def test_smoke(self):
        rng = np.random.default_rng(42)
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        indicators = {
            "A": pd.Series(rng.normal(0, 1, n), index=idx),
            "B": pd.Series(rng.normal(0, 1, n), index=idx),
            "C": pd.Series(rng.normal(0, 1, n), index=idx),
        }
        coherence_df = compute_coherence_test(indicators, max_lag=8)
        fig = plot_coherence_dashboard(coherence_df, indicators, max_lag=8)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestInitReexports:
    def test_all_public_names_resolve(self):
        import auxi.diagnostics as diags
        public_names = [
            "compute_ccf",
            "granger_causality_test",
            "compute_anticipation_test",
            "compute_ews_battery",
            "compute_coherence_test",
            "plot_ccf",
            "plot_ews_battery",
            "plot_coherence_dashboard",
        ]
        for name in public_names:
            assert hasattr(diags, name), f"diags.{name} not found"
