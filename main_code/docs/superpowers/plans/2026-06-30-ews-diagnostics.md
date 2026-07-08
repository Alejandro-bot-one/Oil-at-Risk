# Early Warning System Diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `auxi/diagnostics/ews.py` — CCF, Granger causality, and anticipation tests that evaluate whether tail entropy measures are early warning systems for oil price changes.

**Architecture:** Compute/plot/orchestrator pattern matching the existing diagnostics modules. Five compute functions (`compute_ccf`, `granger_causality_test`, `compute_anticipation_test`, `compute_ews_battery`, `compute_coherence_test`) produce numeric results; three plot functions (`plot_ccf`, `plot_ews_battery`, `plot_coherence_dashboard`) render them. All functions take generic aligned series — the notebook decides what to pass.

**Tech Stack:** numpy, pandas, scipy.stats, matplotlib, pytest

---

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `tests/conftest.py` | Modify | Add `synthetic_ews_pair` and `independent_pair` fixtures |
| `tests/test_ews.py` | Create | All EWS tests |
| `auxi/diagnostics/ews.py` | Create | All EWS compute + plot functions |
| `auxi/diagnostics/__init__.py` | Modify | Re-export EWS public names |
| `context/architecture.md` | Modify | Document new module |
| `context/decisions.md` | Modify | Record EWS design decisions |
| `context/glossary.md` | Modify | Add CCF, Granger, EWS terms |

---

### Task 1: Test fixtures for EWS

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add `synthetic_ews_pair` fixture**

```python
@pytest.fixture
def synthetic_ews_pair():
    """Leader/follower pair with known lag for EWS tests.

    - leader: sine wave (period 50) + small noise.
    - follower: same sine shifted forward by 5 periods + noise.
    - Ground truth: h* = 5 (leader leads follower by 5).
    """
    rng = np.random.default_rng(123)
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n, dtype=float)
    signal = np.sin(2 * np.pi * t / 50)
    leader = pd.Series(signal + rng.normal(0, 0.1, n), index=idx, name="leader")
    follower = pd.Series(
        np.roll(signal, 5) + rng.normal(0, 0.1, n), index=idx, name="follower"
    )
    return leader, follower
```

- [ ] **Step 2: Add `independent_pair` fixture**

```python
@pytest.fixture
def independent_pair():
    """Two independent Gaussian noise series (no causal relationship)."""
    rng = np.random.default_rng(456)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
    y = pd.Series(rng.normal(0, 1, n), index=idx, name="y")
    return x, y
```

- [ ] **Step 3: Run existing tests to confirm no regression**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic_ews_pair and independent_pair fixtures for EWS diagnostics"
```

---

### Task 2: `compute_ccf` — cross-correlation function

**Files:**
- Create: `tests/test_ews.py`
- Create: `auxi/diagnostics/ews.py`

- [ ] **Step 1: Write the failing tests for `compute_ccf`**

```python
# tests/test_ews.py
"""Tests for early warning system diagnostics (auxi/diagnostics/ews.py)."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from auxi.diagnostics.ews import compute_ccf


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ews.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (module does not exist yet).

- [ ] **Step 3: Implement `compute_ccf`**

```python
# auxi/diagnostics/ews.py
"""Early Warning System diagnostics — CCF, Granger causality, anticipation tests.

Tests whether indicator series (e.g. tail entropy) anticipate a target series
(e.g. Brent returns). Adapted from the CLI/CCI composite-indicator diagnostic
framework (Bujosa, García-Ferrer & de Juan, 2013).

SECTION 1 — Cross-Correlation Function (CCF)
SECTION 2 — Granger Causality Test
SECTION 3 — Anticipation Test (CCF + Granger combined)
SECTION 4 — Battery and Coherence orchestrators
SECTION 5 — Plot layer
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _get_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# =============================================================================
# SECTION 1 — CROSS-CORRELATION FUNCTION (CCF)
# =============================================================================

def compute_ccf(x, y, max_lag=24):
    """
    Cross-correlation function between two series at lags -max_lag..+max_lag.

    At lag h >= 0: r(h) = cor(X[0..N-h-1], Y[h..N-1])
    At lag h <  0: r(h) = cor(X[-h..N-1],  Y[0..N+h-1])

    Convention: h > 0 means X leads Y by h periods.

    Both series should typically be differenced before calling this function
    when the levels are non-stationary.

    Parameters
    ----------
    x : array-like, candidate leading series.
    y : array-like, target series. Same length as x.
    max_lag : int, default 24.

    Returns
    -------
    ccf_df : pd.DataFrame with columns 'lag', 'r', 'significant'.
    meta : dict with 'h_star', 'r_at_hstar', 'ci95', 'n_obs'.

    Raises
    ------
    ValueError : if either series has zero variance.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)

    if np.std(x_arr) < 1e-15 or np.std(y_arr) < 1e-15:
        raise ValueError("Cannot compute CCF: one or both series have zero variance.")

    effective_max_lag = min(max_lag, n - 2)

    lags = np.arange(-effective_max_lag, effective_max_lag + 1)
    correlations = np.empty(len(lags))

    for i, h in enumerate(lags):
        if h >= 0:
            correlations[i] = np.corrcoef(x_arr[:n - h], y_arr[h:])[0, 1]
        else:
            correlations[i] = np.corrcoef(x_arr[-h:], y_arr[:n + h])[0, 1]

    ci95 = 1.96 / np.sqrt(n)
    significant = np.abs(correlations) > ci95

    best_idx = np.argmax(np.abs(correlations))
    h_star = int(lags[best_idx])
    r_at_hstar = float(correlations[best_idx])

    ccf_df = pd.DataFrame({
        "lag": lags,
        "r": correlations,
        "significant": significant,
    })

    meta = {
        "h_star": h_star,
        "r_at_hstar": r_at_hstar,
        "ci95": ci95,
        "n_obs": n,
    }

    return ccf_df, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ews.py::TestComputeCCF -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Run full suite for regression check**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/ews.py tests/test_ews.py
git commit -m "feat: add compute_ccf — cross-correlation function for EWS diagnostics"
```

---

### Task 3: `granger_causality_test`

**Files:**
- Modify: `tests/test_ews.py`
- Modify: `auxi/diagnostics/ews.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ews.py`:

```python
from auxi.diagnostics.ews import granger_causality_test


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ews.py::TestGrangerCausalityTest -v`
Expected: FAIL with `ImportError` (function not defined yet).

- [ ] **Step 3: Implement `granger_causality_test`**

Add to `auxi/diagnostics/ews.py`:

```python
# =============================================================================
# SECTION 2 — GRANGER CAUSALITY TEST
# =============================================================================

def granger_causality_test(y, x, max_lag=12, criterion="aic"):
    """
    Granger F-test: does X Granger-cause Y?

    Fits restricted (Y ~ own lags) and unrestricted (Y ~ own lags + X lags)
    models by OLS for each candidate lag order p in 1..max_lag, selects the
    optimal p by AIC or BIC, then runs the F-test at the selected p.

    Parameters
    ----------
    y : array-like, the dependent variable (target to predict).
    x : array-like, the candidate Granger-cause. Same length as y.
    max_lag : int, default 12.
    criterion : 'aic' or 'bic', default 'aic'.

    Returns
    -------
    dict with keys: F, p_value, selected_lag, criterion_values (list),
    significant (bool at 5%), stars (str).

    Raises
    ------
    ValueError : if criterion is not 'aic' or 'bic'.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'")

    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)

    safe_max_lag = min(max_lag, max(1, (n - 2) // 3))

    criterion_values = []
    for p in range(1, safe_max_lag + 1):
        t_eff = n - p
        Y_dep = y_arr[p:]

        # Build unrestricted design matrix: intercept + p Y-lags + p X-lags
        X_ur = np.ones((t_eff, 2 * p + 1))
        for lag in range(1, p + 1):
            X_ur[:, lag] = y_arr[p - lag: n - lag]
            X_ur[:, p + lag] = x_arr[p - lag: n - lag]

        res_ur, _, _, _ = np.linalg.lstsq(X_ur, Y_dep, rcond=None)
        rss_ur = np.sum((Y_dep - X_ur @ res_ur) ** 2)

        k = 2 * p + 1
        if criterion == "aic":
            ic = np.log(rss_ur / t_eff) + 2 * k / t_eff
        else:
            ic = np.log(rss_ur / t_eff) + np.log(t_eff) * k / t_eff

        criterion_values.append(ic)

    selected_lag = int(np.argmin(criterion_values)) + 1

    # Run the F-test at the selected lag
    p = selected_lag
    t_eff = n - p
    Y_dep = y_arr[p:]

    # Restricted: intercept + p Y-lags
    X_r = np.ones((t_eff, p + 1))
    for lag in range(1, p + 1):
        X_r[:, lag] = y_arr[p - lag: n - lag]

    # Unrestricted: intercept + p Y-lags + p X-lags
    X_ur = np.ones((t_eff, 2 * p + 1))
    for lag in range(1, p + 1):
        X_ur[:, lag] = y_arr[p - lag: n - lag]
        X_ur[:, p + lag] = x_arr[p - lag: n - lag]

    res_r, _, _, _ = np.linalg.lstsq(X_r, Y_dep, rcond=None)
    rss_r = np.sum((Y_dep - X_r @ res_r) ** 2)

    res_ur, _, _, _ = np.linalg.lstsq(X_ur, Y_dep, rcond=None)
    rss_ur = np.sum((Y_dep - X_ur @ res_ur) ** 2)

    df_num = p
    df_den = t_eff - (2 * p + 1)
    f_stat = ((rss_r - rss_ur) / df_num) / (rss_ur / df_den)
    p_value = float(stats.f.sf(f_stat, df_num, df_den))

    return {
        "F": float(f_stat),
        "p_value": p_value,
        "selected_lag": selected_lag,
        "criterion_values": criterion_values,
        "significant": p_value < 0.05,
        "stars": _get_stars(p_value),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ews.py::TestGrangerCausalityTest -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Run full suite for regression check**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/ews.py tests/test_ews.py
git commit -m "feat: add granger_causality_test with AIC/BIC lag selection"
```

---

### Task 4: `compute_anticipation_test`, `compute_ews_battery`, `compute_coherence_test`

**Files:**
- Modify: `tests/test_ews.py`
- Modify: `auxi/diagnostics/ews.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ews.py`:

```python
from auxi.diagnostics.ews import (
    compute_anticipation_test,
    compute_ews_battery,
    compute_coherence_test,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ews.py::TestComputeAnticipationTest tests/test_ews.py::TestComputeEWSBattery tests/test_ews.py::TestComputeCoherenceTest -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the three orchestrator functions**

Add to `auxi/diagnostics/ews.py`:

```python
# =============================================================================
# SECTION 3 — ANTICIPATION TEST (CCF + GRANGER COMBINED)
# =============================================================================

def compute_anticipation_test(x, y, max_lag_ccf=24, max_lag_granger=12,
                              criterion="aic"):
    """
    Combined CCF + Granger causality test for one (x, y) pair.

    Parameters
    ----------
    x : array-like, candidate leading indicator.
    y : array-like, target series.
    max_lag_ccf : int, default 24.
    max_lag_granger : int, default 12.
    criterion : 'aic' or 'bic', default 'aic'.

    Returns
    -------
    dict with keys: h_star, r_at_hstar, ccf_significant, granger_F,
    granger_p, granger_lag, granger_stars, ci95, n_obs, ccf_df.
    """
    ccf_df, ccf_meta = compute_ccf(x, y, max_lag=max_lag_ccf)
    granger = granger_causality_test(y, x, max_lag=max_lag_granger,
                                     criterion=criterion)
    return {
        "h_star": ccf_meta["h_star"],
        "r_at_hstar": ccf_meta["r_at_hstar"],
        "ccf_significant": bool(abs(ccf_meta["r_at_hstar"]) > ccf_meta["ci95"]),
        "granger_F": granger["F"],
        "granger_p": granger["p_value"],
        "granger_lag": granger["selected_lag"],
        "granger_stars": granger["stars"],
        "ci95": ccf_meta["ci95"],
        "n_obs": ccf_meta["n_obs"],
        "ccf_df": ccf_df,
    }


# =============================================================================
# SECTION 4 — BATTERY AND COHERENCE ORCHESTRATORS
# =============================================================================

def compute_ews_battery(indicators, target, max_lag_ccf=24, max_lag_granger=12,
                        criterion="aic"):
    """
    Run the anticipation test for each indicator against a single target.

    Parameters
    ----------
    indicators : dict[str, array-like], named indicator series.
    target : array-like, the target series.
    max_lag_ccf : int, default 24.
    max_lag_granger : int, default 12.
    criterion : 'aic' or 'bic', default 'aic'.

    Returns
    -------
    pd.DataFrame with one row per indicator. Columns: Indicator, h_star,
    r_at_hstar, CCF_Significant, Granger_F, Granger_p, Granger_Lag,
    Granger_Stars.
    """
    records = []
    for name, series in indicators.items():
        result = compute_anticipation_test(
            series, target,
            max_lag_ccf=max_lag_ccf,
            max_lag_granger=max_lag_granger,
            criterion=criterion,
        )
        records.append({
            "Indicator": name,
            "h_star": result["h_star"],
            "r_at_hstar": round(result["r_at_hstar"], 4),
            "CCF_Significant": result["ccf_significant"],
            "Granger_F": round(result["granger_F"], 4),
            "Granger_p": round(result["granger_p"], 6),
            "Granger_Lag": result["granger_lag"],
            "Granger_Stars": result["granger_stars"],
        })
    return pd.DataFrame(records)


def compute_coherence_test(indicators, max_lag=24):
    """
    Pairwise CCF among all indicator series.

    Tests internal coherence: all pairs should have h* approx 0 if the
    indicators move together.

    Parameters
    ----------
    indicators : dict[str, array-like], named indicator series.
    max_lag : int, default 24.

    Returns
    -------
    pd.DataFrame with one row per pair. Columns: Series_X, Series_Y,
    h_star, r_at_hstar, Significant.
    """
    from itertools import combinations

    names = list(indicators.keys())
    records = []
    for name_x, name_y in combinations(names, 2):
        ccf_df, meta = compute_ccf(indicators[name_x], indicators[name_y],
                                   max_lag=max_lag)
        records.append({
            "Series_X": name_x,
            "Series_Y": name_y,
            "h_star": meta["h_star"],
            "r_at_hstar": round(meta["r_at_hstar"], 4),
            "Significant": bool(abs(meta["r_at_hstar"]) > meta["ci95"]),
        })
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ews.py::TestComputeAnticipationTest tests/test_ews.py::TestComputeEWSBattery tests/test_ews.py::TestComputeCoherenceTest -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Run full suite for regression check**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/ews.py tests/test_ews.py
git commit -m "feat: add compute_anticipation_test, compute_ews_battery, compute_coherence_test"
```

---

### Task 5: `plot_ccf` — atomic CCF renderer

**Files:**
- Modify: `tests/test_ews.py`
- Modify: `auxi/diagnostics/ews.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ews.py`:

```python
from auxi.diagnostics.ews import plot_ccf
import matplotlib.pyplot as plt


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ews.py::TestPlotCCF -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `plot_ccf`**

Add to `auxi/diagnostics/ews.py`:

```python
# =============================================================================
# SECTION 5 — PLOT LAYER
# =============================================================================

def plot_ccf(ccf_df, meta, ax, title=None, color="#5D6D7E"):
    """
    Atomic CCF bar chart renderer.

    Parameters
    ----------
    ccf_df : pd.DataFrame from compute_ccf (columns: lag, r, significant).
    meta : dict from compute_ccf (h_star, r_at_hstar, ci95).
    ax : matplotlib Axes to draw on.
    title : str or None. If None, auto-generates from meta.
    color : bar color, default '#5D6D7E'.

    Returns
    -------
    ax : the same Axes object.
    """
    ax.bar(ccf_df["lag"], ccf_df["r"], color=color, width=0.7)
    ax.axhline(y=meta["ci95"], linestyle="--", color="red", linewidth=0.8)
    ax.axhline(y=-meta["ci95"], linestyle="--", color="red", linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, linestyle=":", color="grey", linewidth=0.5)

    ax.plot(meta["h_star"], meta["r_at_hstar"], marker="v", color="black",
            markersize=7, zorder=5)

    if title is None:
        title = f"CCF  (h*={meta['h_star']:+d}, r={meta['r_at_hstar']:.3f})"
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Lag h  (h>0: X leads Y)")
    ax.set_ylabel("CCF")

    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ews.py::TestPlotCCF -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add auxi/diagnostics/ews.py tests/test_ews.py
git commit -m "feat: add plot_ccf atomic renderer for EWS bar charts"
```

---

### Task 6: `plot_ews_battery` and `plot_coherence_dashboard` — figure orchestrators

**Files:**
- Modify: `tests/test_ews.py`
- Modify: `auxi/diagnostics/ews.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ews.py`:

```python
from auxi.diagnostics.ews import plot_ews_battery, plot_coherence_dashboard
import matplotlib


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ews.py::TestPlotEWSBattery tests/test_ews.py::TestPlotCoherenceDashboard -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `plot_ews_battery`**

Add to `auxi/diagnostics/ews.py`:

```python
def plot_ews_battery(battery_df, indicators, target, max_lag_ccf=24,
                     figsize=None):
    """
    Multi-panel CCF figure: one subplot per indicator vs target.

    Parameters
    ----------
    battery_df : pd.DataFrame from compute_ews_battery.
    indicators : dict[str, array-like], the original indicator series.
    target : array-like, the target series.
    max_lag_ccf : int, default 24.
    figsize : tuple or None.

    Returns
    -------
    matplotlib.Figure
    """
    n_indicators = len(indicators)
    nr = math.ceil(math.sqrt(n_indicators))
    nc = math.ceil(n_indicators / nr)

    if figsize is None:
        figsize = (7 * nc, 5 * nr)

    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, (name, series) in enumerate(indicators.items()):
        ccf_df, meta = compute_ccf(series, target, max_lag=max_lag_ccf)
        row = battery_df.loc[battery_df["Indicator"] == name]
        granger_info = ""
        if len(row) > 0:
            r = row.iloc[0]
            granger_info = (f"\nGranger: F={r['Granger_F']:.2f}, "
                           f"p={r['Granger_p']:.4f} {r['Granger_Stars']}")
        plot_ccf(ccf_df, meta, axes_flat[i],
                 title=f"{name} vs Target  (h*={meta['h_star']:+d}, "
                       f"r={meta['r_at_hstar']:.3f}){granger_info}")

    for j in range(n_indicators, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Early Warning System — CCF: Indicators vs Target",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Implement `plot_coherence_dashboard`**

Add to `auxi/diagnostics/ews.py`:

```python
def plot_coherence_dashboard(coherence_df, indicators, max_lag=24,
                             figsize=None):
    """
    Multi-panel CCF figure: one subplot per indicator pair.

    Parameters
    ----------
    coherence_df : pd.DataFrame from compute_coherence_test.
    indicators : dict[str, array-like], the original indicator series.
    max_lag : int, default 24.
    figsize : tuple or None.

    Returns
    -------
    matplotlib.Figure
    """
    n_pairs = len(coherence_df)
    nr = math.ceil(math.sqrt(n_pairs))
    nc = math.ceil(n_pairs / nr)

    if figsize is None:
        figsize = (7 * nc, 5 * nr)

    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, (_, row) in enumerate(coherence_df.iterrows()):
        name_x, name_y = row["Series_X"], row["Series_Y"]
        ccf_df, meta = compute_ccf(indicators[name_x], indicators[name_y],
                                   max_lag=max_lag)
        plot_ccf(ccf_df, meta, axes_flat[i],
                 title=f"{name_x} vs {name_y}  (h*={meta['h_star']:+d}, "
                       f"r={meta['r_at_hstar']:.3f})")

    for j in range(n_pairs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Internal Coherence — Pairwise CCF",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_ews.py::TestPlotEWSBattery tests/test_ews.py::TestPlotCoherenceDashboard -v`
Expected: All 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/ews.py tests/test_ews.py
git commit -m "feat: add plot_ews_battery and plot_coherence_dashboard orchestrators"
```

---

### Task 7: `__init__.py` re-exports and integration test

**Files:**
- Modify: `auxi/diagnostics/__init__.py`
- Modify: `tests/test_ews.py`

- [ ] **Step 1: Write the failing test for re-exports**

Add to `tests/test_ews.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ews.py::TestInitReexports -v`
Expected: FAIL with `AssertionError` (names not in `__init__.py` yet).

- [ ] **Step 3: Update `__init__.py`**

Add this import block at the end of `auxi/diagnostics/__init__.py`:

```python
from .ews import (
    compute_ccf,
    granger_causality_test,
    compute_anticipation_test,
    compute_ews_battery,
    compute_coherence_test,
    plot_ccf,
    plot_ews_battery,
    plot_coherence_dashboard,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ews.py::TestInitReexports -v`
Expected: PASS.

- [ ] **Step 5: Run full suite for final regression check**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/__init__.py tests/test_ews.py
git commit -m "feat: re-export EWS functions from diagnostics __init__.py"
```

---

### Task 8: Update context files

**Files:**
- Modify: `context/architecture.md`
- Modify: `context/decisions.md`
- Modify: `context/glossary.md`

- [ ] **Step 1: Update `architecture.md`**

In the `diagnostics/` subpackage section of the file structure diagram, add:

```
│   │   ├── ews.py                 ← early warning system: CCF, Granger causality, anticipation tests
```

In the `diagnostics/__init__.py` paragraph, add a sentence:

> The **EWS layer** (`diagnostics/ews.py`) tests whether indicator series (tail entropy) anticipate a target (Brent returns) via cross-correlation functions, Granger causality with AIC/BIC lag selection, and combined anticipation tests. `compute_ews_battery` runs the full test across multiple indicators; `compute_coherence_test` checks pairwise coherence among indicators.

- [ ] **Step 2: Update `decisions.md`**

Add a new section:

```markdown
## Early warning system diagnostics (2026-06-30)

**Generic series interface for EWS tests.** All functions in `diagnostics/ews.py` take
two aligned `pd.Series` or `np.ndarray` — nothing is hardcoded to entropy or Brent
returns. The notebook decides what to pass and whether to difference the series first.

**No composite construction.** Unlike the R reference (Bujosa et al. 2013), there is no
NS-DFM eigenvector step. The entropy series are pre-computed by `vulnerability_metrics.py`.
Bartlett sphericity and eigenvalue-share tests are out of scope.

**Granger uses AIC by default, BIC available.** Lag search range `1..max_lag` (default 12).
F-test degrees of freedom: `F(p, T_eff - 2p - 1)` where `T_eff = N - p`.

**CCF convention: h > 0 means X leads Y.** Matches the R script and standard econometrics.
```

- [ ] **Step 3: Update `glossary.md`**

Add under "Methodology & domain terms":

```markdown
- **CCF (Cross-Correlation Function)** — correlation between two time series at different
  lags. In `diagnostics/ews.py`, `h > 0` means X leads Y. `h*` is the optimal lag with
  the highest absolute correlation.
- **Granger causality** — does X help predict Y beyond Y's own past? Tested via an F-test
  comparing restricted (Y ~ own lags) vs unrestricted (Y ~ own lags + X lags) models.
  `granger_causality_test` selects lags by AIC or BIC.
- **Anticipation test** — combined CCF + Granger test for whether an indicator (e.g.
  entropy) leads a target (e.g. returns). `h* > 0` and significant Granger = the indicator
  is an early warning system.
- **EWS battery** — running the anticipation test across multiple indicator series against
  a single target. `compute_ews_battery` orchestrates this.
- **Coherence test** — pairwise CCF among indicator series; expects `h* ≈ 0` (they move
  together). `compute_coherence_test`.
```

- [ ] **Step 4: Commit**

```bash
git add context/architecture.md context/decisions.md context/glossary.md
git commit -m "docs: update context files for EWS diagnostics module"
```

---

### Task 9: Final verification

- [ ] **Step 1: Smoke-import all modules**

Run:
```bash
python -c "import auxi.diagnostics as diags; print('OK:', [n for n in dir(diags) if not n.startswith('_')])"
```
Expected: output includes all 8 EWS function names alongside existing names.

- [ ] **Step 2: Full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS, including all `test_ews.py` tests.

- [ ] **Step 3: Verify public surface**

Run:
```bash
python -c "from auxi.diagnostics.ews import compute_ccf, granger_causality_test, compute_anticipation_test, compute_ews_battery, compute_coherence_test, plot_ccf, plot_ews_battery, plot_coherence_dashboard; print('All 8 public names resolve OK')"
```
Expected: prints "All 8 public names resolve OK".

- [ ] **Step 4: Tag the milestone**

```bash
git tag post-ews-2026-06-30
```
