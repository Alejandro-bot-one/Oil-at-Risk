# Rolling-Window Direct Forecasting Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-split evaluation in `evaluate_direct_forecasting` with a rolling-window scheme that produces many h-step-ahead forecasts per horizon, evaluate across multiple quantiles (tail + median), and plot them together.

**Architecture:** Two new functions replace the old `evaluate_direct_forecasting`:
- `compute_rolling_pinball` — pure numeric core. For each horizon h and each quantile tau, rolls a fixed-size training window across the test set, fits the QR model at each step, makes one h-step-ahead forecast, computes the pinball loss, and averages. Returns a tidy DataFrame of `(h, tau, avg_pinball_loss)`.
- `plot_rolling_pinball` — takes the DataFrame from `compute_rolling_pinball` and plots all quantiles on the same axes (one line per tau) across horizons.

The old `evaluate_direct_forecasting` is kept (renamed to `evaluate_direct_forecasting_single`) as a backward-compat alias but the notebook switches to the new functions. `get_oos_predictions` is also replaced with a rolling-window version `get_oos_predictions_rolling` that returns all rolling forecasts instead of a single train/test split.

**Tech Stack:** pandas, numpy, statsmodels (`smf.quantreg`), matplotlib, tqdm

---

## Problem summary

### Problem 1: Single quantile evaluation
`evaluate_direct_forecasting` takes a single `tau` and plots IS vs OOS loss across horizons. We want to evaluate **multiple quantiles** (e.g. `[0.05, 0.50, 0.95]`) and overlay them on the same plot so the user sees tail vs median forecast quality simultaneously.

### Problem 2: Single-split OOS is unreliable
The current code trains once on a fixed training window and evaluates on the remaining test data. With ~896 test obs and an h-step shift, the OOS pinball loss is a single number per horizon — a noisy estimate that could be misleading due to the particular regime in the test window.

**Fix (Rolling Window):** For each forecast origin `t` in the test window, train on the most recent `W` observations (`[t-W, t)`), predict `y_{t+h}`, and record the pinball loss. Then average across all origins. This gives ~`(896 - h)` loss samples per horizon instead of one.

### Data dimensions (for sizing the window)
- Usable rows (all key columns non-null): ~5330 (2006-01-03 to 2026-06-08)
- Train rows (before 2023-01-01): ~4434
- Test rows (from 2023-01-01): ~896
- **Rolling window `W` = 1000 observations** (Alejandro's choice). The window keeps a fixed
  size of 1000 rows: as each new origin advances, the oldest row drops off the back so the
  model "forgets" old regimes. The notebook passes `window_size=1000` explicitly. The function
  signature still defaults `window_size=None` (= use full pre-test training slice) so the
  synthetic-panel tests, which only have a few hundred rows, can pass a smaller window.

---

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `auxi/diagnostics/direct_forecasting.py` | Modify | Add `compute_rolling_pinball`, `plot_rolling_pinball`, `get_oos_predictions_rolling`; rename old `evaluate_direct_forecasting` → `evaluate_direct_forecasting_single` |
| `auxi/diagnostics/__init__.py` | Modify | Re-export new public names |
| `tests/test_rolling_evaluation.py` | Create | Tests for the new rolling functions |
| `direct_forecasting.ipynb` | Modify | Switch calls to the new rolling API |
| `context/architecture.md` | Modify | Document new functions |
| `context/decisions.md` | Modify | Record rolling-window decision |
| `context/known_errors.md` | Modify | Record the single-split mistake and the fix |
| `context/glossary.md` | Modify | Add "rolling window" / "forecast origin" terms |

---

### Task 1: `compute_rolling_pinball` — numeric core

**Files:**
- Create: `tests/test_rolling_evaluation.py`
- Modify: `auxi/diagnostics/direct_forecasting.py`

- [ ] **Step 1: Write the failing test for `compute_rolling_pinball`**

```python
# tests/test_rolling_evaluation.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rolling_evaluation.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_rolling_pinball'`

- [ ] **Step 3: Implement `compute_rolling_pinball`**

Add to `auxi/diagnostics/direct_forecasting.py` (after the existing imports):

```python
def compute_rolling_pinball(df: pd.DataFrame,
                            x: str,
                            y: str,
                            taus: list[float],
                            max_h: int = 30,
                            window_size: int = None,
                            test_start_date: str = None,
                            train_fraction: float = 0.8,
                            controls: list[str] = None) -> pd.DataFrame:
    """
    Rolling-window h-step-ahead pinball loss for multiple quantiles.

    For each forecast origin t in the test window and each horizon h:
      1. Train on [t - window_size, t) (a fixed-width rolling window).
      2. Predict y_{t+h} at each quantile in taus.
      3. Record pinball loss.
    Average across all origins to get one loss per (h, tau).

    Parameters
    ----------
    df : DataFrame with DatetimeIndex. Not mutated.
    x : main regressor column name.
    y : target column name.
    taus : list of quantiles to evaluate.
    max_h : maximum forecast horizon (evaluates h = 1 … max_h).
    window_size : number of observations in the rolling training window.
        If None, defaults to the number of rows before test_start_date
        (i.e. the first window spans all available training data, then
        rolls forward keeping that size fixed).
    test_start_date : str 'YYYY-MM-DD'. Origins run from here onward.
        Mutually exclusive with train_fraction.
    train_fraction : fallback fraction-based split.
    controls : optional list of control variable column names.

    Returns
    -------
    DataFrame with columns: Horizon, Tau, Avg_Pinball_Loss, N_Forecasts.
    """
    if controls is None:
        controls = []

    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].dropna().copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)

    # Determine the test start index
    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        split_idx = df_work.index.searchsorted(test_start_dt)
    else:
        split_idx = int(len(df_work) * train_fraction)

    if window_size is None:
        window_size = split_idx

    n = len(df_work)
    records = []

    for h in tqdm(range(1, max_h + 1), desc="Rolling evaluation"):
        # Build the shifted target once for this h
        target_col = f"{y}_target_h{h}"
        df_h = df_work.copy()
        df_h[target_col] = df_h[y].shift(-h)

        # Build formula once
        if controls:
            control_str = " + ".join([f"Q('{c}')" for c in controls])
            equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
        else:
            equation = f"Q('{target_col}') ~ Q('{x}')"

        # Collect per-origin losses for each tau
        tau_losses = {tau: [] for tau in taus}

        # Roll through test origins
        for t in range(split_idx, n - h):
            train_start = max(0, t - window_size)
            df_train = df_h.iloc[train_start:t].dropna(subset=[target_col])

            if len(df_train) < 30:
                continue

            # The row at position t has features at time t and target y_{t+h}
            row_t = df_h.iloc[[t]]
            realized = row_t[target_col].values[0]
            if np.isnan(realized):
                continue

            for tau in taus:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = smf.quantreg(data=df_train, formula=equation)
                    try:
                        reg = mod.fit(q=tau, vcov="robust", max_iter=2000)
                    except (ValueError, np.linalg.LinAlgError):
                        try:
                            reg = mod.fit(q=tau, vcov="iid", max_iter=2000)
                        except Exception:
                            continue

                    forecast = reg.predict(exog=row_t).values[0]
                    loss = pinball_loss(tau, np.array([realized]),
                                       np.array([forecast]))
                    tau_losses[tau].append(loss)

        for tau in taus:
            losses = tau_losses[tau]
            records.append({
                "Horizon": h,
                "Tau": tau,
                "Avg_Pinball_Loss": np.mean(losses) if losses else np.nan,
                "N_Forecasts": len(losses),
            })

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rolling_evaluation.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Run existing test suite for regression check**

Run: `python -m pytest tests/ -v`
Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_rolling_evaluation.py auxi/diagnostics/direct_forecasting.py
git commit -m "feat: add compute_rolling_pinball for rolling-window OOS evaluation"
```

---

### Task 2: `plot_rolling_pinball` — multi-quantile visualization

**Files:**
- Modify: `tests/test_rolling_evaluation.py` (add smoke test)
- Modify: `auxi/diagnostics/direct_forecasting.py`

- [ ] **Step 1: Write the failing test for `plot_rolling_pinball`**

Append to `tests/test_rolling_evaluation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rolling_evaluation.py::test_plot_rolling_pinball_smoke -v`
Expected: FAIL with `ImportError: cannot import name 'plot_rolling_pinball'`

- [ ] **Step 3: Implement `plot_rolling_pinball`**

Add to `auxi/diagnostics/direct_forecasting.py`:

```python
def plot_rolling_pinball(results_df: pd.DataFrame,
                         title: str = None) -> plt.Figure:
    """
    Plots average rolling-window pinball loss across horizons, one line per tau.

    Parameters
    ----------
    results_df : DataFrame from compute_rolling_pinball with columns
        Horizon, Tau, Avg_Pinball_Loss, N_Forecasts.
    title : optional custom title.

    Returns
    -------
    matplotlib Figure.
    """
    taus = sorted(results_df["Tau"].unique())
    cmap = plt.get_cmap("coolwarm")
    colors = [cmap(i) for i in np.linspace(0, 1, len(taus))]

    fig, ax = plt.subplots(figsize=(12, 6))

    for tau, color in zip(taus, colors):
        sub = results_df[results_df["Tau"] == tau].sort_values("Horizon")
        ax.plot(sub["Horizon"], sub["Avg_Pinball_Loss"],
                marker="o", markersize=3, linewidth=2, color=color,
                label=f"$\\tau$ = {tau}")

    if title is None:
        title = (f"Rolling-Window Average Pinball Loss\n"
                 f"Quantiles: {[round(t, 2) for t in taus]}")
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Forecast Horizon (h days)", fontsize=12)
    ax.set_ylabel("Average Pinball Loss", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    fig.tight_layout()

    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rolling_evaluation.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add auxi/diagnostics/direct_forecasting.py tests/test_rolling_evaluation.py
git commit -m "feat: add plot_rolling_pinball for multi-quantile loss visualization"
```

---

### Task 3: `get_oos_predictions_rolling` — rolling OOS prediction series

**Files:**
- Modify: `tests/test_rolling_evaluation.py`
- Modify: `auxi/diagnostics/direct_forecasting.py`

The current `get_oos_predictions` in `qreg.py` trains once and predicts on the entire test set. We need a rolling-window version that returns the same `(realized, forecasted)` tuple but where each forecast was made from a separate rolling-window model. This lives in `diagnostics/direct_forecasting.py` since it's an evaluation concern, not a model engine concern.

- [ ] **Step 1: Write the failing test for `get_oos_predictions_rolling`**

Append to `tests/test_rolling_evaluation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rolling_evaluation.py::test_get_oos_predictions_rolling_returns_aligned_series -v`
Expected: FAIL with `ImportError: cannot import name 'get_oos_predictions_rolling'`

- [ ] **Step 3: Implement `get_oos_predictions_rolling`**

Add to `auxi/diagnostics/direct_forecasting.py`:

```python
def get_oos_predictions_rolling(df: pd.DataFrame,
                                x: str,
                                y: str,
                                tau: float,
                                h: int = 1,
                                window_size: int = None,
                                test_start_date: str = None,
                                train_fraction: float = 0.8,
                                controls: list[str] = None) -> tuple[pd.Series, pd.Series]:
    """
    Rolling-window OOS predictions for a single quantile and horizon.

    For each forecast origin t in the test window:
      1. Train on [t - window_size, t).
      2. Predict y_{t+h}.
    Returns (realized, forecasted) as aligned pd.Series with DatetimeIndex,
    indexed by the origin date t (the date at which the forecast is made).

    Parameters
    ----------
    df : DataFrame with DatetimeIndex. Not mutated.
    x : main regressor.
    y : target.
    tau : quantile level.
    h : forecast horizon.
    window_size : rolling training window size. Defaults to split_idx.
    test_start_date : 'YYYY-MM-DD'.
    train_fraction : fallback if test_start_date not given.
    controls : optional list of control columns.

    Returns
    -------
    (realized, forecasted) : tuple of pd.Series.
    """
    if controls is None:
        controls = []

    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].dropna().copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)

    # Determine split
    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        split_idx = df_work.index.searchsorted(test_start_dt)
    else:
        split_idx = int(len(df_work) * train_fraction)

    if window_size is None:
        window_size = split_idx

    # Build shifted target
    target_col = f"{y}_target_h{h}"
    df_h = df_work.copy()
    df_h[target_col] = df_h[y].shift(-h)

    # Build formula
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{target_col}') ~ Q('{x}')"

    n = len(df_h)
    dates = []
    realized_vals = []
    forecast_vals = []

    for t in tqdm(range(split_idx, n - h), desc=f"Rolling OOS (tau={tau}, h={h})"):
        train_start = max(0, t - window_size)
        df_train = df_h.iloc[train_start:t].dropna(subset=[target_col])

        if len(df_train) < 30:
            continue

        row_t = df_h.iloc[[t]]
        realized = row_t[target_col].values[0]
        if np.isnan(realized):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=df_train, formula=equation)
            try:
                reg = mod.fit(q=tau, vcov="robust", max_iter=2000)
            except (ValueError, np.linalg.LinAlgError):
                try:
                    reg = mod.fit(q=tau, vcov="iid", max_iter=2000)
                except Exception:
                    continue

            forecast = reg.predict(exog=row_t).values[0]

        dates.append(df_h.index[t])
        realized_vals.append(realized)
        forecast_vals.append(forecast)

    realized_series = pd.Series(realized_vals, index=pd.DatetimeIndex(dates),
                                name="Realized")
    forecast_series = pd.Series(forecast_vals, index=pd.DatetimeIndex(dates),
                                name="Forecasted")
    return realized_series, forecast_series
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rolling_evaluation.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add auxi/diagnostics/direct_forecasting.py tests/test_rolling_evaluation.py
git commit -m "feat: add get_oos_predictions_rolling for rolling-window OOS forecast series"
```

---

### Task 4: Update `__init__.py` re-exports and rename old function

**Files:**
- Modify: `auxi/diagnostics/direct_forecasting.py` (rename old function)
- Modify: `auxi/diagnostics/__init__.py`
- Modify: `tests/test_rolling_evaluation.py` (add surface test)

- [ ] **Step 1: Write the failing test for the public surface**

Append to `tests/test_rolling_evaluation.py`:

```python
def test_new_functions_resolvable_via_diagnostics():
    """All new functions must be importable from auxi.diagnostics."""
    import auxi.diagnostics as diags
    assert callable(diags.compute_rolling_pinball)
    assert callable(diags.plot_rolling_pinball)
    assert callable(diags.get_oos_predictions_rolling)
    # Old function still accessible under both names
    assert callable(diags.evaluate_direct_forecasting)
    assert callable(diags.evaluate_direct_forecasting_single)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rolling_evaluation.py::test_new_functions_resolvable_via_diagnostics -v`
Expected: FAIL with `AttributeError: module 'auxi.diagnostics' has no attribute 'compute_rolling_pinball'`

- [ ] **Step 3: Rename old function and update `__init__.py`**

In `auxi/diagnostics/direct_forecasting.py`, rename the old function:

```python
# Keep the old name as an alias for backward compat
def evaluate_direct_forecasting_single(df, x, y, controls, tau=0.05, max_h=90,
                                       train_fraction=0.8, test_start_date=None):
    # ... (existing body, unchanged)

# Backward compat alias
evaluate_direct_forecasting = evaluate_direct_forecasting_single
```

In `auxi/diagnostics/__init__.py`, update the `direct_forecasting` import block to:

```python
from .direct_forecasting import (
    evaluate_direct_forecasting,
    evaluate_direct_forecasting_single,
    compute_rolling_pinball,
    plot_rolling_pinball,
    get_oos_predictions_rolling,
    diagnose_residual_acf,
    compute_fallout_errors,
    plot_fallout_errors,
    evaluate_cumulative_fallout,
    compute_unconditional_coverage_unified,
    plot_unconditional_coverage_unified,
    compute_conditional_coverage,
    plot_conditional_coverage,
    plot_unconditional_coverage,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rolling_evaluation.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run full test suite for regression check**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (including old tests that use `evaluate_direct_forecasting`)

- [ ] **Step 6: Commit**

```bash
git add auxi/diagnostics/direct_forecasting.py auxi/diagnostics/__init__.py tests/test_rolling_evaluation.py
git commit -m "refactor: rename old evaluate_direct_forecasting, re-export new rolling functions"
```

---

### Task 5: Update the notebook to use the new rolling API

**Files:**
- Modify: `direct_forecasting.ipynb`

The notebook currently calls `diags.evaluate_direct_forecasting(...)` three times (once for vanilla QR, once for CAViaR-i, once for CAViaR-s) with a single `tau`. Switch each to `diags.compute_rolling_pinball(...)` with `taus=[0.05, 0.50, 0.95]`, `window_size=1000`, and `diags.plot_rolling_pinball(...)`.

Similarly, the notebook calls `fc.get_oos_predictions(...)` three times with a single tau. Switch each to `diags.get_oos_predictions_rolling(...)` with `window_size=1000`.

**First, add `window_size = 1000` to the shared-settings cell (Cell 3)**, next to `h`, `tau_eval`, `test_start`, `max_h`, so every call references one variable:

```python
window_size = 1000   # fixed rolling-window length (Alejandro's choice)
```

Then have every call below pass `window_size=window_size`.

- [ ] **Step 1: Update Cell 13 (vanilla QR evaluation)**

Replace:
```python
evaluation_results = diags.evaluate_direct_forecasting(
    df              = data,
    x               = x_var,
    y               = y_var,
    controls        = controls,
    tau             = tau_eval,
    max_h           = max_h,
    test_start_date = test_start,
)
```

With:
```python
evaluation_results = diags.compute_rolling_pinball(
    df              = data,
    x               = x_var,
    y               = y_var,
    taus            = [0.05, 0.50, 0.95],
    max_h           = max_h,
    window_size     = window_size,
    test_start_date = test_start,
    controls        = controls,
)
fig = diags.plot_rolling_pinball(evaluation_results)
plt.show()
```

- [ ] **Step 2: Update Cell 18 (vanilla QR OOS predictions)**

Replace `fc.get_oos_predictions(...)` with `diags.get_oos_predictions_rolling(...)`:
```python
y_oos_actual, y_oos_pred = diags.get_oos_predictions_rolling(
    df              = data,
    x               = x_var,
    y               = y_var,
    tau             = tau_eval,
    h               = h,
    window_size     = window_size,
    test_start_date = test_start,
    controls        = controls,
)
```

- [ ] **Step 3: Update Cell 26 (CAViaR-i evaluation)**

Same pattern as Step 1 but with `data_caviar` and `caviar_controls`.

- [ ] **Step 4: Update Cell 30 (CAViaR-i OOS predictions)**

Same pattern as Step 2 but with `data_caviar` and `caviar_controls`.

- [ ] **Step 5: Update Cell 38 (CAViaR-s evaluation)**

Same pattern as Step 1 but with `data_caviar_s` and `caviar_s_controls`.

- [ ] **Step 6: Update Cell 42 (CAViaR-s OOS predictions)**

Same pattern as Step 2 but with `data_caviar_s` and `caviar_s_controls`.

- [ ] **Step 7: Commit**

```bash
git add direct_forecasting.ipynb
git commit -m "feat: switch notebook to rolling-window evaluation with multi-quantile plots"
```

---

### Task 6: Update context files

**Files:**
- Modify: `context/architecture.md`
- Modify: `context/decisions.md`
- Modify: `context/known_errors.md`
- Modify: `context/glossary.md`

> Worker note: re-verify each context file against the final code before editing — paraphrase
> the prose below to match each file's existing voice rather than pasting verbatim.

- [ ] **Step 1: Update `architecture.md`**

In the `diagnostics/direct_forecasting.py` description, add:

> `compute_rolling_pinball` — rolling-window average pinball loss for multiple quantiles across horizons. `plot_rolling_pinball` — plots all quantiles on one axis. `get_oos_predictions_rolling` — rolling-window OOS predictions for Kupiec/Christoffersen evaluation.

- [ ] **Step 2: Update `decisions.md`**

Add a new entry:

```markdown
## Direct forecasting evaluation

**Rolling-window evaluation replaces single-split OOS** (2026-06-29). The old
`evaluate_direct_forecasting` (now aliased as `evaluate_direct_forecasting_single`) trained
once and computed OOS loss on a single test window — a noisy single-number estimate. The new
`compute_rolling_pinball` uses a fixed-width rolling window: for each origin in the test set,
train on the most recent W observations, predict h-step ahead, record pinball loss, and average.
This yields ~(N_test - h) loss samples per (horizon, quantile) instead of one.

**Evaluate multiple quantiles simultaneously.** `compute_rolling_pinball` takes a list of taus
(e.g. `[0.05, 0.50, 0.95]`), and `plot_rolling_pinball` overlays them on one chart. This shows
tail vs median forecast quality in a single figure.

**OOS predictions for coverage tests use rolling windows too.** `get_oos_predictions_rolling`
replaces `get_oos_predictions` for the Kupiec/Christoffersen evaluation path. Each forecast in
the returned series was produced from its own rolling-window fit, making the coverage test more
honest.

**Window length is 1000 observations** (fixed-width, not expanding) so the model adapts to the
most recent regime and forgets old ones. The notebook passes `window_size=1000`; the function
default `None` means "use the full pre-test training slice".
```

- [ ] **Step 3: Update `known_errors.md`**

Add a new numbered entry recording the methodological mistake:

```markdown
## N. Single-split OOS evaluation gave a noisy, unfair error metric (FIXED)

**What happened.** `evaluate_direct_forecasting` trained once on a fixed training window and
computed the OOS pinball loss on the *single* remaining test window — one number per horizon.
That single realization could be dominated by the particular regime in the test window, so the
error metric was not a fair measure of the model's h-step skill. It also only evaluated *one*
quantile at a time.

**The fix (2026-06-29).** `compute_rolling_pinball` rolls a fixed-width (1000-obs) training
window across the test set: for every forecast origin it refits, makes one h-step-ahead
forecast, and records the pinball loss, then averages over ~(N_test − h) origins per
(horizon, quantile). It evaluates a list of quantiles at once and `plot_rolling_pinball`
overlays them. `get_oos_predictions_rolling` does the same for the coverage-test path. The old
function survives as `evaluate_direct_forecasting_single` (aliased to the old name).

**How to avoid.** A backtest error metric should be an *average over many forecasts*, not a
single split. When evaluating an h-step forecaster, generate many h-step forecasts (one per
rolling origin) and aggregate; never report a single test-window number as "the" OOS loss.
```

- [ ] **Step 4: Update `glossary.md`**

Add to the domain/code vocabulary section:

```markdown
- **Rolling window** — a fixed-width training window (here 1000 observations) that slides
  forward one origin at a time, dropping the oldest row as it adds a new one. Lets the model
  forget old regimes. Contrast with an *expanding* window that never drops old data.
- **Forecast origin** — the date `t` at which a forecast is made; the model is trained on data
  up to (but not including) `t` and predicts `y_{t+h}`.
- **Rolling-window pinball loss** — the average pinball (tick) loss over all rolling forecast
  origins for a given horizon and quantile; the fair OOS error metric used by
  `compute_rolling_pinball`.
```

- [ ] **Step 5: Commit**

```bash
git add context/architecture.md context/decisions.md context/known_errors.md context/glossary.md
git commit -m "docs: update context files with rolling-window evaluation"
```

---

### Task 7: Verification

- [ ] **Step 1: Run the full pytest suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Smoke-import all public names**

Run:
```python
python -c "
import auxi.diagnostics as diags
for name in ['compute_rolling_pinball', 'plot_rolling_pinball',
             'get_oos_predictions_rolling', 'evaluate_direct_forecasting',
             'evaluate_direct_forecasting_single']:
    assert callable(getattr(diags, name)), f'{name} not callable'
print('All public names resolve OK')
"
```

- [ ] **Step 3: Quick numerical sanity check on synthetic data**

Run:
```python
python -c "
import numpy as np, pandas as pd
from auxi.diagnostics.direct_forecasting import compute_rolling_pinball

rng = np.random.default_rng(42)
n = 300
idx = pd.bdate_range('2020-01-01', periods=n)
gpr = rng.normal(0, 1, n)
brent = 0.5*gpr + rng.normal(0, 1, n)
df = pd.DataFrame({'Brent_Return': brent, 'gpr': gpr}, index=idx)

result = compute_rolling_pinball(df, 'gpr', 'Brent_Return',
                                  taus=[0.05, 0.50, 0.95], max_h=3,
                                  window_size=150, test_start_date='2020-10-01')
print(result.to_string(index=False))
# Check: median pinball should be roughly symmetric
# Check: N_Forecasts should be > 0 for all rows
assert (result['N_Forecasts'] > 0).all()
assert (result['Avg_Pinball_Loss'] > 0).all()
print('Sanity check PASSED')
"
```
