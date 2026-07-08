# OOS Loss ACF/PACF Diagnostic (`direct_forecasting.py`) — Design

**Date:** 2026-07-08
**Modified file:** `auxi/diagnostics/direct_forecasting.py` (add functions), `auxi/diagnostics/__init__.py` (re-export), `direct_forecasting.ipynb` (apply to standard model)
**Status:** Approved, pending implementation

## Context and motivation

The direct-forecasting notebook evaluates each model's out-of-sample skill with a
**rolling-window pinball loss per horizon** (`compute_rolling_pinball` /
`plot_rolling_pinball`), which *averages* the per-origin loss into one number per
`(horizon, tau)`. That average hides the **time structure** of the loss.

For an h-step direct forecast the errors follow an MA(h−1) process (this is exactly why
`diebold_mariano_test` uses a rectangular-kernel HAC variance at bandwidth h−1). The
per-origin tick loss

```
L_t = ρ_τ(y_{t+h} − ŷ_t)      (the pinball loss at origin t)
```

inherits serial dependence. Examining the **autocorrelogram (ACF)** and **partial
autocorrelogram (PACF)** of this OOS loss series at a selected quantile τ answers two
questions:

1. Is the empirical serial correlation of the loss consistent with the theoretical
   MA(h−1) cutoff (validating the DM HAC bandwidth choice)?
2. Does the forecaster's loss cluster in time (predictable loss ⇒ exploitable structure
   ⇒ scope for a better model)?

This is the **out-of-sample** analogue of the existing `diagnose_residual_acf`, which
plots the ACF of the *in-sample* residuals and marks the same MA(h−1) cutoff.

### Design decisions

1. **The series is the tick loss `L_t`, not the error `e_t`.** "OOS loss function"
   is read literally: the per-origin pinball loss at τ. Computed via the existing
   `tick_loss_series`, whose mean equals the rolling pinball loss already reported above
   in the notebook. (The signed error variant is out of scope.)

2. **Model-agnostic signature `(realized, forecasted, tau, …)`.** The functions take a
   pre-computed OOS forecast as two aligned `pd.Series` and never refit internally. This
   matches the coverage-test and Diebold-Mariano convention and honours
   `known_errors.md` #5 ("New diagnostics should take `(realized, forecasted)` and stay
   model-agnostic"). The notebook produces the series with the existing
   `get_oos_predictions_rolling`. Deliberately *not* the self-contained refit style of
   `diagnose_residual_acf`, which #5 flags as debt not to extend.

3. **Layered compute / plot separation** per `conventions.md`: a pure `compute_*` that
   returns the dated loss series, and a `plot_*` that renders ACF + PACF. The compute
   output is inspectable/saveable on its own.

4. **Reuse over duplication.** `compute_oos_loss_series` wraps `tick_loss_series` (no new
   loss math); `plot_oos_loss_acf` uses statsmodels `plot_acf` / `plot_pacf` and reuses
   the MA(h−1) cutoff visual idiom (crimson vline + shaded span) already established in
   `diagnose_residual_acf`.

5. **Lives in `diagnostics/direct_forecasting.py`.** It is a forecast-evaluation
   diagnostic, alongside the rolling-pinball, coverage, and DM functions. New banner
   section placed after the DM block (which defines `tick_loss_series`).

6. **`lags` capped for PACF validity.** statsmodels `plot_pacf` requires
   `lags < n_obs / 2`. The plotter caps the requested `lags` to
   `min(lags, n_obs // 2 − 1)` so short OOS windows do not raise.

## Architecture — compute / plot (SoC)

```
compute_oos_loss_series()   ← pure numeric, returns the dated tick-loss pd.Series
plot_oos_loss_acf()         ← figure-level orchestrator: ACF + PACF on a 1×2 grid
```

### Dependencies

Reuses `tick_loss_series` (same module). Adds `plot_pacf` to the existing
`from statsmodels.graphics.tsaplots import plot_acf` import. No new third-party
dependency. `pinball_loss` is already imported from `auxi.qreg`.

## Function inventory

### Compute layer

#### `compute_oos_loss_series(realized, forecasted, tau)`

The per-origin OOS tick loss at quantile τ.

1. Align `realized` and `forecasted` on their common index via a DataFrame + `dropna`
   (the same idiom as the coverage functions). Input series are not mutated.
2. Compute `L = tick_loss_series(tau, realized_aligned, forecasted_aligned)`.
3. Return `pd.Series(L, index=common_index, name="Tick_Loss")`.

**Parameters:**
- `realized` : `pd.Series` of realized `y_{t+h}`, indexed by forecast origin `t`.
- `forecasted` : `pd.Series` of forecasts `ŷ_t`, same index.
- `tau` : quantile level in (0, 1).

**Returns:**
- `pd.Series` of non-negative losses, `name="Tick_Loss"`, indexed by origin date.

**Invariant:** `series.mean()` equals `pinball_loss(tau, realized, forecasted)` on the
aligned sample.

### Plot layer

#### `plot_oos_loss_acf(realized, forecasted, tau, h=1, lags=20, title=None)`

Figure-level orchestrator. Builds the loss series with `compute_oos_loss_series`, then
draws a **1×2 figure**:

- **Left — ACF:** `plot_acf(loss, lags=lags_eff, alpha=0.05, ax=…)`.
- **Right — PACF:** `plot_pacf(loss, lags=lags_eff, alpha=0.05, ax=…)`.
- Both panels mark the **theoretical MA(h−1) cutoff**: a crimson dashed vertical line at
  lag `h−1` plus a light shaded span over `[0, h−1]` (mirroring `diagnose_residual_acf`).
- `lags_eff = min(lags, n_obs // 2 − 1)` to keep PACF valid.
- Suptitle: `"OOS Loss Serial Dependence (τ={tau}, h={h})"`; per-axis titles
  "Autocorrelation (ACF)" / "Partial Autocorrelation (PACF)"; grid on the y-axis
  (`linestyle="--", alpha≈0.4`), consistent with the module's plotting idioms.

**Parameters:**
- `realized`, `forecasted` : the OOS series (as above).
- `tau` : quantile level.
- `h` : forecast horizon; sets the MA(h−1) cutoff marker.
- `lags` : max lags to display (default 20).
- `title` : optional custom suptitle.

**Returns:**
- `matplotlib.Figure`. Does **not** call `plt.show()` (matches `plot_rolling_pinball`;
  the notebook calls `plt.show()`).

## Notebook integration

In `direct_forecasting.ipynb`, insert a new markdown + code cell **immediately below**
the `### Pinball loss across horizons` block (standard model, cell `7acd0567`) and above
`### Breach / fallout errors`.

Markdown: `### Serial dependence of the OOS loss (ACF & PACF)` with a one-line note on the
MA(h−1) motivation.

Code:
```python
# OOS per-origin loss series at the selected quantile, then its ACF / PACF
loss_actual, loss_pred = diags.get_oos_predictions_rolling(
    df=data, x=x_var, y=y_var, tau=tau_eval, h=h,
    window_size=window_size, test_start_date=test_start_coverage, controls=controls,
)
fig = diags.plot_oos_loss_acf(loss_actual, loss_pred, tau=tau_eval, h=h)
plt.show()
```

- **Selected τ:** `tau_eval` (0.95), consistent with the coverage-test and DM blocks.
- **Test window:** `test_start_coverage` (2023-01-01) rather than the block-above's
  `test_start` (2025-01-01), because the sample ACF/PACF needs a longer OOS series
  (~500+ origins vs ~130) to be reliable. Trivially switchable to `test_start`.

## Re-export

Add `compute_oos_loss_series` and `plot_oos_loss_acf` to the
`from .direct_forecasting import (…)` block in `auxi/diagnostics/__init__.py`.

## Testing strategy

New file `tests/test_oos_loss_acf.py`. Model-agnostic functions make the numeric tests
independent of any rolling fit — synthetic `pd.Series` are constructed directly.

1. **`test_loss_series_mean_matches_pinball`** — for several τ, the mean of
   `compute_oos_loss_series` equals `pinball_loss(τ, …)` (rtol 1e-10).
2. **`test_loss_series_non_negative`** — all losses ≥ 0.
3. **`test_loss_series_index_alignment`** — misaligned `realized`/`forecasted` indices
   are aligned to the intersection; output length equals the overlap; NaNs dropped.
4. **`test_inputs_not_mutated`** — the passed series are unchanged after the call.
5. **`test_plot_oos_loss_acf_smoke`** — Agg backend, ~250-point synthetic series, returns
   a `matplotlib.Figure` with 2 axes, no exception.
6. **`test_plot_caps_lags_short_series`** — a short series (e.g. 40 points) with
   `lags=20` does not raise (PACF lag cap works) and returns a `Figure`.
7. **`test_init_reexports`** — `diags.compute_oos_loss_series` and
   `diags.plot_oos_loss_acf` resolve via `import auxi.diagnostics as diags`.

Run: `python -m pytest tests/test_oos_loss_acf.py -v`, then the full suite
`python -m pytest tests/ -v` as a regression check.

## Out of scope (YAGNI)

- A signed-error (`e_t`) variant of the correlogram.
- A multi-τ grid (one τ per call; the notebook can loop if ever needed).
- New rolling-forecast machinery (reuses `get_oos_predictions_rolling`).
- Normalizing `diagnose_residual_acf` to the model-agnostic signature (separate,
  pre-existing debt — see `known_errors.md` #5).
- Ljung–Box or other formal serial-correlation test statistics (visual diagnostic only).
