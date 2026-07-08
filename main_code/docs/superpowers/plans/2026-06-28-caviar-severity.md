# CAViaR Severity (`caviar_s`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a severity-based CAViaR variant (`caviar_s`) to `auxi/caviar.py` that uses continuous breach distances instead of binary indicators, and add a full diagnostic section in `direct_forecasting.ipynb`.

**Architecture:** Mirror the existing `caviar_i` three-layer design. Layer 1 adds `_compute_breach_severity` and `compute_breach_severity_indicators` (h-aware, lookahead-free). Layer 2 adds `caviar_s` and `multiple_caviar_s`. Layer 3 adds `plot_caviar_s_results`. The notebook section reuses the shared diagnostic settings and runs the same evaluation pipeline (in-sample fit, pinball loss, fallout, Kupiec, Christoffersen).

**Tech Stack:** Python 3.12, statsmodels (`smf.quantreg`), pandas, numpy, matplotlib, pytest

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `auxi/caviar.py` | Modify (append) | Add severity helpers + estimators + plotter |
| `tests/test_caviar.py` | Modify (append) | Add tests for all new public & private functions |
| `direct_forecasting.ipynb` | Modify (append section) | Add "CAViaR (severity)" section after CAViaR I |
| `context/architecture.md` | Modify | Document new functions |
| `context/conventions.md` | Modify | Add `_s` suffix convention |
| `context/decisions.md` | Modify | Record decision, remove YAGNI entry |
| `context/glossary.md` | Modify | Add severity term |

---

### Task 1: `_compute_breach_severity` — Layer 1 private helper

**Files:**
- Modify: `auxi/caviar.py` (after `_compute_breaches`, ~line 113)
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_caviar.py` after the existing `_compute_breaches` tests:

```python
from auxi.caviar import _compute_breach_severity


def test_compute_breach_severity_values():
    realized = pd.Series([1.0, 5.0, -5.0, 0.5], index=range(4))
    bounds = pd.DataFrame(
        {"Bound_Low": [-2.0, -2.0, -2.0, -2.0],
         "Bound_High": [2.0, 2.0, 2.0, 2.0]},
        index=range(4),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    # No breach -> 0; upside breach -> positive distance; downside -> negative.
    assert list(upside.values) == [0.0, 3.0, 0.0, 0.0]
    assert list(downside.values) == [0.0, 0.0, -3.0, 0.0]


def test_compute_breach_severity_nan_bound_gives_nan():
    realized = pd.Series([1.0, 5.0], index=range(2))
    bounds = pd.DataFrame(
        {"Bound_Low": [np.nan, -2.0],
         "Bound_High": [np.nan, 2.0]},
        index=range(2),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    assert np.isnan(upside.iloc[0])
    assert np.isnan(downside.iloc[0])
    # Row 1: upside breach of 3.0
    assert upside.iloc[1] == 3.0
    assert downside.iloc[1] == 0.0


def test_compute_breach_severity_sign_convention():
    """upside_severity >= 0 always; downside_severity <= 0 always."""
    realized = pd.Series([10.0, -10.0, 0.0], index=range(3))
    bounds = pd.DataFrame(
        {"Bound_Low": [-2.0, -2.0, -2.0],
         "Bound_High": [2.0, 2.0, 2.0]},
        index=range(3),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    assert (upside >= 0).all()
    assert (downside <= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_caviar.py -k "breach_severity" -v`
Expected: FAIL — `_compute_breach_severity` not defined.

- [ ] **Step 3: Implement `_compute_breach_severity`**

Add to `auxi/caviar.py` immediately after `_compute_breaches` (after line 113):

```python
def _compute_breach_severity(realized, bounds):
    """
    Severity of quantile-boundary breaches: signed distance from the
    realized value to the violated boundary.

      upside_severity   = max(0, realized - Bound_High)   >= 0
      downside_severity = min(0, realized - Bound_Low)    <= 0

    Where the bound is NaN the severity is NaN (not 0).

    Parameters
    ----------
    realized : Series of the realized target.
    bounds   : DataFrame with columns ["Bound_Low", "Bound_High"].

    Returns
    -------
    (upside_severity, downside_severity) : tuple of two Series.
    """
    realized = pd.Series(realized).astype(float)
    low = bounds["Bound_Low"]
    high = bounds["Bound_High"]

    diff_high = realized - high
    diff_low = realized - low

    upside_severity = diff_high.clip(lower=0.0)
    downside_severity = diff_low.clip(upper=0.0)

    upside_severity[high.isna()] = np.nan
    downside_severity[low.isna()] = np.nan

    upside_severity.name = "upside_severity"
    downside_severity.name = "downside_severity"
    return upside_severity, downside_severity
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_caviar.py -k "breach_severity" -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add _compute_breach_severity (Layer 1 severity helper)"
```

---

### Task 2: `compute_breach_severity_indicators` — Layer 1 h-aware helper

**Files:**
- Modify: `auxi/caviar.py` (after `compute_breach_indicators`, ~line 209)
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_caviar.py`:

```python
from auxi.caviar import compute_breach_severity_indicators


def test_compute_breach_severity_indicators_shape(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert list(indicators.columns) == ["upside_severity", "downside_severity"]
    assert len(indicators) == len(synthetic_panel)


def test_compute_breach_severity_indicators_first_h_are_nan(synthetic_panel):
    for h_val in [1, 3, 5]:
        indicators = compute_breach_severity_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=h_val, breach_quantiles=[0.05, 0.95],
        )
        assert indicators["upside_severity"].iloc[:h_val].isna().all()
        assert indicators["downside_severity"].iloc[:h_val].isna().all()


def test_compute_breach_severity_indicators_sign_convention(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    valid = indicators.dropna()
    assert (valid["upside_severity"] >= 0).all()
    assert (valid["downside_severity"] <= 0).all()


def test_compute_breach_severity_indicators_continuous_values(synthetic_panel):
    """Severity is continuous, not binary — some values should differ from 0/1."""
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    valid = indicators.dropna()
    up_nonzero = valid["upside_severity"][valid["upside_severity"] > 0]
    if len(up_nonzero) > 0:
        assert not set(up_nonzero.values).issubset({0.0, 1.0})


def test_compute_breach_severity_indicators_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_compute_breach_severity_indicators_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        compute_breach_severity_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=1, breach_quantiles=[0.5, 0.5],
        )


def test_compute_breach_severity_indicators_test_start_date(synthetic_panel):
    mid = synthetic_panel.index[200]
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
        test_start_date=str(mid.date()),
    )
    assert len(indicators) == len(synthetic_panel)
    valid = indicators.dropna()
    assert (valid["upside_severity"] >= 0).all()
    assert (valid["downside_severity"] <= 0).all()


def test_compute_breach_severity_indicators_detects_outliers(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert (indicators["upside_severity"] > 0).any()
    assert (indicators["downside_severity"] < 0).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_caviar.py -k "severity_indicators" -v`
Expected: FAIL — `compute_breach_severity_indicators` not defined.

- [ ] **Step 3: Implement `compute_breach_severity_indicators`**

Add to `auxi/caviar.py` after `compute_breach_indicators` (after line 209). This mirrors `compute_breach_indicators` exactly, but calls `_compute_breach_severity` instead of the binary comparison at the end:

```python
def compute_breach_severity_indicators(df, vars_x, vars_y, h,
                                       breach_quantiles=None,
                                       train_fraction=0.8,
                                       test_start_date=None):
    """
    Severity-based breach indicators, h-aware and lookahead-free.

    Same as compute_breach_indicators but returns the signed distance from
    the realized value to the violated boundary instead of a binary flag:

      upside_severity_t   = max(0, y_t - Q_high(y_t | x_{t-h}))   >= 0
      downside_severity_t = min(0, y_t - Q_low(y_t  | x_{t-h}))   <= 0

    The first h rows are NaN (no prior prediction available).

    Parameters
    ----------
    df               : DataFrame with features and target. Never mutated.
    vars_x           : list[str] | str, specification regressors.
    vars_y           : str, target.
    h                : int, forecast horizon (steps ahead).
    breach_quantiles : list[float], default [0.05, 0.95].
    train_fraction   : float, default 0.8 (ignored if test_start_date given).
    test_start_date  : str 'YYYY-MM-DD', optional.

    Returns
    -------
    DataFrame indexed like df with columns
    ['upside_severity', 'downside_severity'].
    """
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]
    if isinstance(vars_x, str):
        vars_x = [vars_x]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    cols = [vars_y] + list(vars_x)
    work = df[cols].copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    target_col = f"{vars_y}_target_h{h}"
    work[target_col] = work[vars_y].shift(-h)

    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        train = work[work.index < test_start_dt]
    else:
        complete = work.dropna()
        split_idx = int(len(complete) * train_fraction)
        split_date = complete.index[split_idx]
        train = work[work.index < split_date]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="iid")
        try:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="iid")

    boundary_low = pd.Series(reg_low.predict(work), index=work.index)
    boundary_high = pd.Series(reg_high.predict(work), index=work.index)

    boundary_low_lagged = boundary_low.shift(h)
    boundary_high_lagged = boundary_high.shift(h)

    realized = df[vars_y]
    bounds_lagged = pd.DataFrame({
        "Bound_Low": boundary_low_lagged,
        "Bound_High": boundary_high_lagged,
    }, index=df.index)

    upside_sev, downside_sev = _compute_breach_severity(realized, bounds_lagged)

    return pd.DataFrame({
        "upside_severity": upside_sev,
        "downside_severity": downside_sev,
    }, index=df.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_caviar.py -k "severity_indicators" -v`
Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add compute_breach_severity_indicators (h-aware, no lookahead)"
```

---

### Task 3: `caviar_s` and `multiple_caviar_s` — Layer 2 in-sample estimators

**Files:**
- Modify: `auxi/caviar.py` (after `multiple_caviar_i`, ~line 377)
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_caviar.py`:

```python
from auxi.caviar import caviar_s, multiple_caviar_s


def test_caviar_s_returns_reg_and_indicators(synthetic_panel):
    reg, indicators = caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    assert hasattr(reg, "params")
    param_names = list(reg.params.index)
    assert any("upside_severity" in p for p in param_names)
    assert any("downside_severity" in p for p in param_names)
    assert "upside_severity" in indicators.columns
    assert "downside_severity" in indicators.columns


def test_caviar_s_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_caviar_s_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        caviar_s(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            tau=0.5, breach_quantiles=[0.5, 0.5],
        )


def test_caviar_s_default_breach_quantiles(synthetic_panel):
    reg, indicators = caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5,
    )
    assert hasattr(reg, "params")


def test_multiple_caviar_s_schema(synthetic_panel):
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    expected_cols = {
        "Dependent Variable", "Regressor", "Tau",
        "Coefficient", "Significance", "Pseudo R-Squared",
    }
    assert expected_cols.issubset(set(master.columns))


def test_multiple_caviar_s_includes_severity(synthetic_panel):
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    regressors = set(master["Regressor"].unique())
    assert {"gpr", "vix", "upside_severity", "downside_severity"}.issubset(regressors)


def test_multiple_caviar_s_one_row_per_regressor_per_tau(synthetic_panel):
    quantiles = [0.25, 0.5, 0.75]
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=quantiles, breach_quantiles=[0.05, 0.95],
    )
    # 4 regressors (gpr, vix, upside_severity, downside_severity) x 3 taus = 12
    assert len(master) == 4 * len(quantiles)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_caviar.py -k "caviar_s" -v`
Expected: FAIL — `caviar_s` not defined.

- [ ] **Step 3: Implement `caviar_s` and `multiple_caviar_s`**

Add to `auxi/caviar.py` after `multiple_caviar_i`:

```python
def caviar_s(df, vars_x, vars_y, tau, breach_quantiles=None,
             errors="robust", **kwargs):
    """
    CAViaR with severity (signed distance) regressors instead of binary
    indicators.

    Same as caviar_i but uses:
      upside_severity   = max(0, y_t - Q_high)   >= 0
      downside_severity = min(0, y_t - Q_low)    <= 0

    Parameters
    ----------
    df              : DataFrame. Never mutated.
    vars_x          : list[str] | str, specification regressors.
    vars_y          : str, target.
    tau             : float, quantile for the CAViaR regression.
    breach_quantiles: list[float], default [0.05, 0.95].
    errors          : str, vcov (default "robust").
    **kwargs        : forwarded to q_reg.

    Returns
    -------
    (reg, indicators) : tuple.
        reg        -> fitted regression result (statsmodels).
        indicators -> DataFrame with upside_severity, downside_severity,
                      Bound_Low, Bound_High.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    bounds = _compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high)
    upside_sev, downside_sev = _compute_breach_severity(df[vars_y], bounds)

    indicators = pd.DataFrame(
        {"upside_severity": upside_sev,
         "downside_severity": downside_sev,
         "Bound_Low": bounds["Bound_Low"],
         "Bound_High": bounds["Bound_High"]},
        index=df.index,
    )

    work = df.copy()
    work["upside_severity"] = upside_sev
    work["downside_severity"] = downside_sev

    all_x = list(vars_x) + ["upside_severity", "downside_severity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov=errors, **kwargs)
        except ValueError:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov="iid", **kwargs)

    return reg, indicators


def multiple_caviar_s(data, vars_x, vars_y, quantiles=None,
                      breach_quantiles=None, errors="robust"):
    """
    Run caviar_s across a quantile grid and return a master_df with the
    same schema as multiple_q_regs / multiple_caviar_i.

    Severity indicators are computed once and reused across all taus.

    Parameters
    ----------
    data             : DataFrame. Never mutated.
    vars_x           : list[str] | str, specification.
    vars_y           : str, target.
    quantiles        : list[float], default 21-point grid.
    breach_quantiles : list[float], default [0.05, 0.95].
    errors           : str, vcov (default "robust").

    Returns
    -------
    master_df : DataFrame sorted by (Regressor, Tau) with columns
        Dependent Variable, Regressor, Tau, Coefficient, Significance,
        Pseudo R-Squared.
    """
    import statsmodels.formula.api as smf

    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    def get_stars(p_value):
        if p_value < 0.01: return '***'
        elif p_value < 0.05: return '**'
        elif p_value < 0.10: return '*'
        else: return ''

    bounds = _compute_quantile_bounds(data, vars_x, vars_y, tau_low, tau_high)
    upside_sev, downside_sev = _compute_breach_severity(data[vars_y], bounds)

    work = data.copy()
    work["upside_severity"] = upside_sev
    work["downside_severity"] = downside_sev

    all_indep_vars = list(vars_x) + ["upside_severity", "downside_severity"]
    rhs = " + ".join([f"Q('{v}')" for v in all_indep_vars])
    equation = f"Q('{vars_y}') ~ {rhs}"

    res_dict = {
        "Dependent Variable": [], "Regressor": [], "Tau": [],
        "Coefficient": [], "Significance": [], "Pseudo R-Squared": [],
    }

    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=work, formula=equation)
            try:
                reg = mod.fit(q=q, vcov=errors)
            except ValueError:
                reg = mod.fit(q=q, vcov="iid")

            pseudo_r2 = getattr(reg, 'prsquared', np.nan)
            for var in all_indep_vars:
                param_name = f"Q('{var}')"
                res_dict["Dependent Variable"].append(vars_y)
                res_dict["Regressor"].append(var)
                res_dict["Tau"].append(q)
                res_dict["Coefficient"].append(reg.params[param_name])
                res_dict["Significance"].append(get_stars(reg.pvalues[param_name]))
                res_dict["Pseudo R-Squared"].append(pseudo_r2)

    master_df = pd.DataFrame(res_dict)
    master_df = master_df.sort_values(by=["Regressor", "Tau"]).reset_index(drop=True)
    return master_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_caviar.py -k "caviar_s" -v`
Expected: 8 PASSED.

- [ ] **Step 5: Run full test suite to check nothing broke**

Run: `python -m pytest tests/test_caviar.py -v`
Expected: ALL PASSED (existing + new).

- [ ] **Step 6: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add caviar_s and multiple_caviar_s (Layer 2 severity estimators)"
```

---

### Task 4: `plot_caviar_s_results` — Layer 3 visualization

**Files:**
- Modify: `auxi/caviar.py` (after `plot_caviar_i_results`)
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_caviar.py`:

```python
from auxi.caviar import plot_caviar_s_results


def test_plot_caviar_s_results_returns_master_df(synthetic_panel):
    master = plot_caviar_s_results(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        breach_quantiles=[0.05, 0.95], quantiles=[0.25, 0.5, 0.75],
    )
    assert "Coefficient" in master.columns
    regressors = set(master["Regressor"].unique())
    assert {"gpr", "vix", "upside_severity", "downside_severity"}.issubset(regressors)
    plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_caviar.py::test_plot_caviar_s_results_returns_master_df -v`
Expected: FAIL — `plot_caviar_s_results` not defined.

- [ ] **Step 3: Implement `plot_caviar_s_results`**

Add to `auxi/caviar.py` after `plot_caviar_i_results`:

```python
def plot_caviar_s_results(data, vars_x, vars_y, breach_quantiles=None,
                          quantiles=None, errors="robust"):
    """
    2x2 dashboard for the CAViaR severity model — mirrors plot_caviar_i_results.

    Panels:
      [0,0] Coefficients of the specification regressors (plot_quantile_coefs).
      [0,1] Coefficients of upside_severity / downside_severity by tau.
      [1,0] Pseudo R^2 (plot_pseudo_r2).
      [1,1] Breach diagnostics (bounds + breaches timeline).

    Returns
    -------
    master_df : the numeric output of multiple_caviar_s.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    results_df = multiple_caviar_s(
        data=data, vars_x=vars_x, vars_y=vars_y, quantiles=quantiles,
        breach_quantiles=breach_quantiles, errors=errors,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"CAViaR (severity) Dashboard: {vars_y} ~ {vars_x}",
                 fontsize=14, fontweight="bold")

    plot_quantile_coefs(axes[0, 0], results_df, list(vars_x),
                        title=f"Specification regressors: {vars_x}")
    plot_quantile_coefs(axes[0, 1], results_df,
                        ["upside_severity", "downside_severity"],
                        title="Severity regressors (breach distance)")
    plot_pseudo_r2(axes[1, 0], results_df)
    plot_breach_diagnostics(axes[1, 1], data, vars_x=vars_x, vars_y=vars_y,
                            breach_quantiles=breach_quantiles)

    plt.tight_layout()
    plt.show()

    return results_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_caviar.py::test_plot_caviar_s_results_returns_master_df -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/test_caviar.py -v`
Expected: ALL PASSED.

- [ ] **Step 6: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add plot_caviar_s_results (Layer 3 severity dashboard)"
```

---

### Task 5: Update module docstring and exports

**Files:**
- Modify: `auxi/caviar.py` (module docstring at top)

- [ ] **Step 1: Update the module docstring**

Update the module docstring (lines 1-28) to include the severity variant. Replace the layer map and add the new functions:

```python
"""
CAViaR con variables indicador / severidad
===========================================
Regresión cuantílica tipo CAViaR (Engle & Manganelli, 2004) con dos variantes
de variables adicionales que capturan las rupturas de las fronteras cuantílicas:

  Variante _i (indicador binario):
    upside_breach   = 1{y_t > Q_high(y_t | x_{t-h})}
    downside_breach = 1{y_t < Q_low(y_t  | x_{t-h})}

  Variante _s (severidad — distancia con signo):
    upside_severity   = max(0, y_t - Q_high(y_t | x_{t-h}))   >= 0
    downside_severity = min(0, y_t - Q_low(y_t  | x_{t-h}))   <= 0

Los indicadores se computan con conciencia del horizonte h: comparan el valor
realizado en t con la frontera predicha h pasos antes, evitando lookahead.
El panel de entrada NUNCA se muta.

Arquitectura por capas (SoC):
  Capa 1 (helpers puros):    _compute_quantile_bounds, _compute_breaches,
                             _compute_breach_severity,
                             compute_breach_indicators (h-aware, sin lookahead),
                             compute_breach_severity_indicators (h-aware, sin lookahead)
  Capa 2 (estimación in-s.): caviar_i, multiple_caviar_i,
                             caviar_s, multiple_caviar_s
  Capa 3 (visualización):    plot_breach_diagnostics,
                             plot_caviar_i_results, plot_caviar_s_results

Para direct forecasting con indicadores CAViaR, usar compute_breach_indicators
(binario) o compute_breach_severity_indicators (severidad) para generar las
columnas y pasarlas como controls a las funciones estándar de auxi/qreg.py.

Referencia:
  Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional Autoregressive
  Value at Risk by Regression Quantiles. JBES, 22(4), 367-381.
"""
```

- [ ] **Step 2: Run full test suite to confirm nothing broke**

Run: `python -m pytest tests/test_caviar.py -v`
Expected: ALL PASSED.

- [ ] **Step 3: Commit**

```bash
git add auxi/caviar.py
git commit -m "docs: update caviar.py module docstring with severity variant"
```

---

### Task 6: Notebook section — CAViaR (severity) in `direct_forecasting.ipynb`

**Files:**
- Modify: `direct_forecasting.ipynb` (append after the CAViaR I section, after cell `c98c5f63`)

This task adds cells to the notebook. The section mirrors the CAViaR I section exactly but uses severity indicators. All shared settings (`h`, `tau_eval`, `test_start`, `max_h`, `insample_date`, `breach_q`) come from the existing settings cell `df-settings`.

- [ ] **Step 1: Add the markdown header cell**

```markdown
# CAViaR (severity)
Variante de severidad de los indicadores de breach CAViaR (Engle & Manganelli, 2004). En lugar de variables binarias (0/1), se usan las distancias con signo entre el valor realizado y la frontera cuantílica violada:

  $\text{upside\_severity}_t   = \max(0,\; y_t - Q_{high}(y_t \mid x_{t-h})) \geq 0$
  $\text{downside\_severity}_t = \min(0,\; y_t - Q_{low}(y_t  \mid x_{t-h})) \leq 0$

Las fronteras se ajustan **sin lookahead** (misma mecánica que CAViaR indicador).
```

- [ ] **Step 2: Add the import + indicators cell**

```python
from auxi.caviar import compute_breach_severity_indicators

# 1. Severity indicators (h-aware, no lookahead)
severity_spec = [x_var] + controls
severity_indicators = compute_breach_severity_indicators(
    df               = data,
    vars_x           = severity_spec,
    vars_y           = y_var,
    h                = h,
    breach_quantiles = breach_q,
    test_start_date  = test_start,
)

print(f"Severity indicators — first {h} rows are NaN (no prior prediction):")
print(severity_indicators.dropna().describe())
print(f"\nNon-zero upside severities:   {(severity_indicators['upside_severity'] > 0).sum()}")
print(f"Non-zero downside severities: {(severity_indicators['downside_severity'] < 0).sum()}")
```

- [ ] **Step 3: Add the augment panel cell**

```python
# 2. Augment panel
data_caviar_s = data.copy(deep=True)
data_caviar_s["upside_severity"]   = severity_indicators["upside_severity"]
data_caviar_s["downside_severity"] = severity_indicators["downside_severity"]
```

- [ ] **Step 4: Add the controls definition cell**

```python
caviar_s_controls = controls + ["upside_severity", "downside_severity"]
```

- [ ] **Step 5: Add the in-sample fit section**

Markdown cell:
```markdown
### In-sample fit (CAViaR severity)
```

Code cell:
```python
df_caviar_s_insample = fc.insample_direct_forecasting(
    df             = data_caviar_s,
    x              = x_var,
    y              = y_var,
    quantiles      = quantiles,
    controls       = caviar_s_controls,
    train_end_date = insample_date,
    h              = h,
)
```

- [ ] **Step 6: Add the pinball loss section**

Markdown cell:
```markdown
### Pinball loss across horizons (CAViaR severity)
```

Code cell:
```python
eval_caviar_s = diags.evaluate_direct_forecasting(
    df              = data_caviar_s,
    x               = x_var,
    y               = y_var,
    controls        = caviar_s_controls,
    tau             = tau_eval,
    max_h           = max_h,
    test_start_date = test_start,
)

print(f"Total accumulated loss: {eval_caviar_s['OOS_Loss'].sum():.4f}")
print(f"OOS loss minimised at horizon: {eval_caviar_s['OOS_Loss'].idxmin()}")
```

- [ ] **Step 7: Add the fallout errors section**

Markdown cell:
```markdown
### Breach / fallout errors (CAViaR severity)
```

Code cell:
```python
fallout_caviar_s = diags.plot_fallout_errors(
    df        = data_caviar_s,
    x         = x_var,
    y         = y_var,
    controls  = caviar_s_controls,
    h         = h,
    lower_tau = 0.01,
    upper_tau = 0.99,
)

total_breaches_caviar_s = len(fallout_caviar_s[
    (fallout_caviar_s["Realized"] > fallout_caviar_s["Upper_Bound"]) |
    (fallout_caviar_s["Realized"] < fallout_caviar_s["Lower_Bound"])
])
within_ratio_caviar_s = (len(fallout_caviar_s) - total_breaches_caviar_s) / len(fallout_caviar_s)
print(f"{round(within_ratio_caviar_s*100, 2)}% of the data falls within the predicted quantiles")
```

- [ ] **Step 8: Add the coverage tests section**

Markdown cell:
```markdown
### Kupiec & Christoffersen coverage tests (CAViaR severity)
```

Code cell:
```python
y_caviar_s_actual, y_caviar_s_pred = fc.get_oos_predictions(
    df              = data_caviar_s,
    x               = x_var,
    y               = y_var,
    controls        = caviar_s_controls,
    tau             = tau_eval,
    h               = h,
    test_start_date = test_start,
)

print(f"Test window: {y_caviar_s_actual.index[0].date()} -> {y_caviar_s_actual.index[-1].date()}")
print(f"Observations in test set: {len(y_caviar_s_actual)}")

# Unconditional coverage (Kupiec POF)
coverage_caviar_s = diags.compute_unconditional_coverage_unified(
    realized=y_caviar_s_actual, forecasted=y_caviar_s_pred, tau=tau_eval)
diags.plot_unconditional_coverage_unified(
    realized=y_caviar_s_actual, forecasted=y_caviar_s_pred, tau=tau_eval)

# Conditional coverage (Christoffersen)
cond_coverage_caviar_s = diags.compute_conditional_coverage(
    realized=y_caviar_s_actual, forecasted=y_caviar_s_pred, tau=tau_eval)
diags.plot_conditional_coverage(
    realized=y_caviar_s_actual, forecasted=y_caviar_s_pred, tau=tau_eval)
```

- [ ] **Step 9: Commit**

```bash
git add direct_forecasting.ipynb
git commit -m "feat: add CAViaR severity section to direct_forecasting notebook"
```

---

### Task 7: Update context files

**Files:**
- Modify: `context/architecture.md`
- Modify: `context/conventions.md`
- Modify: `context/decisions.md`
- Modify: `context/glossary.md`

- [ ] **Step 1: Update `context/architecture.md`**

In the "Estimation engines" paragraph (~line 57-63), after the sentence about `caviar.py`, add mention of the severity variant. Update to:

> `caviar.py` is the CAViaR layer (Engle & Manganelli, 2004): it computes binary breach
> indicators (`_i` variant) and severity-based breach distances (`_s` variant) and adds them
> as regressors. Its key public entry points are `compute_breach_indicators` (binary, h-aware)
> and `compute_breach_severity_indicators` (severity, h-aware), which fit the quantile bounds
> on the training slice only, predict over the panel, lag by `h`, and compare — **h-aware and
> lookahead-free**.

- [ ] **Step 2: Update `context/conventions.md`**

In the Naming section (~line 58-59), update the `_i`/`_s` suffix entry to:

> - **Function-name suffix `_i`** = "indicator variant" (binary breach), as in `caviar_i`.
>   **Suffix `_s`** = "severity variant" (signed distance), as in `caviar_s`. The two share
>   the same three-layer architecture and `master_df` schema.

- [ ] **Step 3: Update `context/decisions.md`**

In the "Explicitly out of scope (YAGNI)" section (~line 105), **remove** the line:
> - A `caviar_s` severity variant.

In the "CAViaR breach modelling" section (~line 29), add a new paragraph after the existing content:

> **`caviar_s` (severity) built on 2026-06-28.** Uses continuous signed distances instead of
> binary indicators: `upside_severity = max(0, y_t - Q_high) >= 0`, `downside_severity =
> min(0, y_t - Q_low) <= 0`. Zero when no breach (not a continuous proximity measure). Same
> three-layer architecture, same h-aware forecasting path, same `master_df` schema. The sign
> convention (upside positive, downside negative) was chosen by Alejandro to encode direction
> in the value itself.

- [ ] **Step 4: Update `context/glossary.md`**

In the Methodology & domain terms section, after the "Breach" entry (~line 43-45), add:

> - **Severity (breach severity)** — the signed distance from the realized value to the
>   violated quantile boundary. `upside_severity = max(0, y_t - Q_high)` (>= 0),
>   `downside_severity = min(0, y_t - Q_low)` (<= 0). Zero when no breach occurs. Used in the
>   `caviar_s` variant. Contrast with the binary breach in `caviar_i`.

In the Code-specific terms section, update the `_i` suffix entry (~line 112):

> - **`_i` suffix** — indicator (binary) variant. **`_s` suffix** — severity (signed distance)
>   variant. Both follow the three-layer architecture.

- [ ] **Step 5: Commit**

```bash
git add context/architecture.md context/conventions.md context/decisions.md context/glossary.md
git commit -m "docs: update context files with caviar_s severity variant"
```

---

## Verification checklist

After all tasks:

- [ ] `python -m pytest tests/test_caviar.py -v` — all tests pass (existing + new)
- [ ] `python -m pytest tests/ -v` — full suite passes
- [ ] Notebook Restart Kernel + Run All on `direct_forecasting.ipynb` — no errors
- [ ] Severity columns are never written to the caller's DataFrame
- [ ] First `h` rows of severity indicators are NaN
- [ ] upside_severity >= 0 always; downside_severity <= 0 always
- [ ] No lookahead in the h-aware path (bounds fitted on training slice only)
- [ ] Context files match the code
