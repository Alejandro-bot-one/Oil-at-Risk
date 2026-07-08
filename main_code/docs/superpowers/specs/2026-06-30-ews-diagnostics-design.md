# Early Warning System Diagnostics (`ews.py`) — Design

**Date:** 2026-06-30
**New file:** `auxi/diagnostics/ews.py`
**Status:** Approved, pending implementation

## Context and motivation

The Oil-at-Risk framework produces time-varying tail entropy measures (Full / Left /
Right KL divergence) from the fitted conditional JSU distributions. The final research
question is whether these entropy signals act as **early warning systems** for future
oil price changes — i.e., whether entropy movements today anticipate Brent return
movements tomorrow.

Three diagnostic families, adapted from the CLI/CCI composite-indicator literature
(Bujosa, García-Ferrer & de Juan, 2013), test this:

1. **Anticipation test** — does a given entropy series lead Brent returns?
2. **Predictive power test** — does each entropy component individually lead returns?
3. **Coherence test** — do the entropy components (Full/Left/Right) move together?

Tests 1 and 2 use the same functions applied to different series pairs; the notebook
decides what to pass. Test 3 runs pairwise CCF among the indicator series.

### Design decisions

1. **Generic series interface.** All functions take two aligned `pd.Series` or
   `np.ndarray` inputs. Nothing is hardcoded to entropy or Brent returns. The same
   functions can test GPR vs returns, VIX vs entropy, etc.

2. **First-difference convention.** The R reference script differenced both series
   before computing CCF and Granger tests. The EWS functions here operate on whatever
   the user passes — the notebook is responsible for differencing if needed. The
   docstrings note that first differences are the standard practice for non-stationary
   series.

3. **CCF sign convention: `h > 0` means X leads Y.** At lag `h`, the function
   correlates `X_t` with `Y_{t+h}`. A positive `h*` (optimal lag) means X anticipates
   Y by `h*` periods. This matches the R script and the standard econometrics
   convention.

4. **Granger causality tests the direction X → Y.** The restricted model regresses Y
   on its own lags; the unrestricted adds lags of X. "X Granger-causes Y" if the
   unrestricted model has significantly lower RSS.

5. **AIC lag selection by default, BIC available.** The `criterion` parameter accepts
   `'aic'` or `'bic'`. Lag search range is `1..max_lag` (default 12).

6. **Module lives in `diagnostics/`.** It is an evaluation diagnostic (does the entropy
   signal have predictive power?), not an estimator. It belongs alongside the other
   evaluation tests in the diagnostics subpackage.

7. **No composite construction.** Unlike the R reference, there is no NS-DFM
   eigenvector composition step. The entropy series are pre-computed by
   `vulnerability_metrics.py`. The Bartlett sphericity test and eigenvalue-share
   analysis from the R script are out of scope.

## Architecture — compute / plot / orchestrator (SoC)

Follows the existing three-layer pattern from `conventions.md`:

```
compute_ccf()                    ← pure numeric, returns DataFrame + metadata dict
granger_causality_test()         ← pure numeric, returns dict
compute_anticipation_test()      ← combines CCF + Granger for one (x,y) pair
compute_ews_battery()            ← orchestrator: runs anticipation across many indicators
compute_coherence_test()         ← orchestrator: pairwise CCF among indicators

plot_ccf()                       ← atomic renderer, takes ax
plot_ews_battery()               ← figure-level orchestrator
plot_coherence_dashboard()       ← figure-level orchestrator
```

### Dependencies

`ews.py` imports only from the standard library, NumPy, pandas, SciPy (`scipy.stats`),
and Matplotlib. No imports from other `auxi` modules — it is self-contained.

## Function inventory

### Compute layer

#### `compute_ccf(x, y, max_lag=24)`

Cross-correlation function between two series at lags `-max_lag..+max_lag`.

**Formula (per-lag Pearson correlation on overlapping segments):**

At lag `h ≥ 0`:  `r(h) = cor(X[1..N-h], Y[h+1..N])`
At lag `h < 0`:  `r(h) = cor(X[1-h..N], Y[1..N+h])`

This is equivalent to the normalized cross-covariance formula from the notes for
stationary series.

**Parameters:**
- `x` : array-like, the candidate leading series.
- `y` : array-like, the target series. Same length as x.
- `max_lag` : int, default 24.

**Returns:**
- `ccf_df` : `pd.DataFrame` with columns `lag` (int), `r` (float), `significant` (bool).
- `meta` : dict with `h_star` (int), `r_at_hstar` (float), `ci95` (float),
  `n_obs` (int).

Significance: `|r| > 1.96 / √N`. The `h_star` is the lag (across all lags, positive
and negative) with the largest absolute correlation.

#### `granger_causality_test(y, x, max_lag=12, criterion='aic')`

Granger F-test: does X Granger-cause Y?

1. For each `p` in `1..max_lag`, fit restricted (`Y ~ own lags`) and unrestricted
   (`Y ~ own lags + X lags`) by OLS via `np.linalg.lstsq`.
2. Compute the information criterion (AIC or BIC) for each `p`.
3. Select optimal `p` by minimum criterion.
4. Run the F-test at the selected `p`.

**F-statistic:** `((RSS_r - RSS_ur) / p) / (RSS_ur / (T_eff - 2p - 1))`

where `T_eff = N - p` (effective sample after losing `p` observations to lags) and the
denominator df is `T_eff - ncol(X_ur) = T_eff - (2p + 1)`.

**P-value:** from `F(p, T_eff - 2p - 1)` distribution.

**Parameters:**
- `y` : array-like, the dependent variable (target to predict).
- `x` : array-like, the candidate cause.
- `max_lag` : int, default 12.
- `criterion` : `'aic'` or `'bic'`, default `'aic'`.

**Returns:**
- dict with keys: `F`, `p_value`, `selected_lag`, `criterion_values` (list),
  `significant` (bool at 5%), `stars` (str).

#### `compute_anticipation_test(x, y, max_lag_ccf=24, max_lag_granger=12, criterion='aic')`

Combined CCF + Granger for one (x, y) pair.

**Returns:**
- dict with keys: `h_star`, `r_at_hstar`, `ccf_significant`, `granger_F`,
  `granger_p`, `granger_lag`, `granger_stars`, `ci95`, `n_obs`, `ccf_df`.

#### `compute_ews_battery(indicators, target, max_lag_ccf=24, max_lag_granger=12, criterion='aic')`

Runs `compute_anticipation_test` for each indicator in the dict against a single
target series.

**Parameters:**
- `indicators` : `dict[str, array-like]` — e.g. `{"Full entropy": full, "Left tail": left, "Right tail": right}`.
- `target` : array-like — e.g. differenced Brent returns.
- Other params forwarded to `compute_anticipation_test`.

**Returns:**
- `battery_df` : `pd.DataFrame` with one row per indicator. Columns: `Indicator`,
  `h_star`, `r_at_hstar`, `CCF_Significant`, `Granger_F`, `Granger_p`,
  `Granger_Lag`, `Granger_Stars`.

#### `compute_coherence_test(indicators, max_lag=24)`

Pairwise CCF among all indicators in the dict.

**Returns:**
- `coherence_df` : `pd.DataFrame` with one row per pair. Columns: `Series_X`,
  `Series_Y`, `h_star`, `r_at_hstar`, `Significant`.

### Plot layer

#### `plot_ccf(ccf_df, meta, ax, title=None, color='#5D6D7E')`

Atomic renderer: CCF bar chart on a given `ax`.

- Bars colored by `color`.
- Dashed red horizontal lines at `±ci95`.
- Black reference line at `r=0`.
- Dotted vertical line at `lag=0`.
- Triangle marker at `(h_star, r_at_hstar)`.
- Title shows `h*` and `r`.

Returns the `ax`.

#### `plot_ews_battery(battery_df, indicators, target, max_lag_ccf=24, figsize=None)`

Figure-level orchestrator: one CCF subplot per indicator.

- Calls `compute_ccf` internally for each indicator to get the per-lag data.
- Arranges subplots in a `ceil(√n) × ceil(n/ceil(√n))` grid.
- Suptitle: "Early Warning System — CCF: Indicators vs Target".
- Returns the `matplotlib.Figure`.

#### `plot_coherence_dashboard(coherence_df, indicators, max_lag=24, figsize=None)`

Figure-level orchestrator: one CCF subplot per pair.

- Same layout logic as `plot_ews_battery`.
- Suptitle: "Internal Coherence — Pairwise CCF".
- Returns the `matplotlib.Figure`.

## Testing strategy

### Synthetic test data

Extend `conftest.py` with a new fixture `synthetic_ews_pair`:

- `leader` : sine wave with period 50, plus small Gaussian noise.
- `follower` : same sine wave shifted forward by 5 periods, plus noise.
- Both as `pd.Series` with `DatetimeIndex` (200 business days).
- The known ground truth is `h* = 5` (leader leads follower by 5).

Also a fixture `independent_pair`: two independent Gaussian noise series (no causal
relationship). Ground truth: Granger should not reject H₀.

### Test cases

1. **`test_compute_ccf_known_lag`** — CCF of leader/follower returns `h_star == 5`.
2. **`test_compute_ccf_symmetric`** — CCF at lag 0 of a series with itself is 1.0.
3. **`test_granger_rejects_for_causal_pair`** — Granger p-value < 0.05 for
   leader → follower.
4. **`test_granger_does_not_reject_independent`** — Granger p-value > 0.05 for
   independent series.
5. **`test_granger_bic_option`** — `criterion='bic'` runs without error and returns a
   valid `selected_lag`.
6. **`test_compute_anticipation_test`** — returns dict with all expected keys.
7. **`test_compute_ews_battery`** — returns DataFrame with correct shape and columns.
8. **`test_compute_coherence_test`** — returns DataFrame with n*(n-1)/2 rows.
9. **`test_plot_ccf_smoke`** — Agg backend, no exception, returns `ax`.
10. **`test_plot_ews_battery_smoke`** — Agg backend, no exception, returns `Figure`.
11. **`test_plot_coherence_dashboard_smoke`** — Agg backend, no exception, returns
    `Figure`.
12. **`test_edge_constant_series`** — constant input raises `ValueError` (zero
    variance).
13. **`test_edge_short_series`** — series shorter than `max_lag` handled gracefully
    (reduced lags or `ValueError`).
14. **`test_init_reexports`** — all public names resolve via
    `import auxi.diagnostics as diags`.

## Out of scope (YAGNI)

- NS-DFM composite construction (eigenvector weighting).
- Bartlett sphericity test.
- Eigenvalue share analysis.
- Excel/IO export from the module (notebook concern).
- Impulse-response functions.
- Vector autoregression (VAR) — Granger via OLS is sufficient for this thesis.
