# Known Errors & Pitfalls — Oil-at-Risk TFM

> Context file for Claude. Mistakes already made on this project, why they happened, and how
> to avoid repeating them. When a new bug is found and fixed, add it here with the date and the
> commit. Treat every entry as a standing check before shipping related code.

## 1. Lookahead bias in CAViaR breach indicators (FIXED — keep it fixed)

**What happened.** The first CAViaR direct-forecasting design ("Approach C", commit `a940dc8`)
fit the quantile **bounds on the full panel** and then used them to build breach indicators for
both training and test rows. The direct-forecasting shift only cured the *trivial* leak
(predicting `y_t` from a function of `y_t`); the in-sample fit leak survived, so out-of-sample
pinball loss looked better than honest. The 2026-06-26 reorg spec explicitly flagged this as
"lookahead (B), deferred".

**The fix.** It was *not* left deferred. Commit `df8b32f` replaced the wrappers with
`compute_breach_indicators(df, vars_x, vars_y, h, train_fraction/test_start_date)`, which fits
bounds on the **training slice only**, predicts over the panel, **lags by `h`**, and compares
realized-at-`t` to the boundary predicted at `t-h`. The first `h` rows are `NaN` by design.

**How to avoid.** Never fit bounds, distribution parameters, or any quantity used in an OOS
backtest on data that includes the test window. Always fit on a strict training slice, predict
forward, and respect the horizon lag. If you touch any backtest, prove the cutoff date and that
no future row informs a past prediction.

## 2. Hardcoded absolute paths that no longer match the repo location

**What happened.** The project folders were renamed twice (`Msc Analisis Economico Cuantitativo
UAM` → `MQuEA`, and `main code` → `main_code`), but absolute paths were hardcoded in several
places and went stale: `auxi/data.py` (`_BASE`), `data/aux.py` (`WD`), and the `os.chdir(...)` /
`wd = ...` cells of all eight notebooks. They pointed at a location that no longer existed, so
the code read the wrong files or failed.

**Status.** Fixed on 2026-06-28 — every active path now points at
`C:\Users\Alejandro\Documents\MQuEA\TFM\...` (the FRED `API_PATH`, which lives in a different
unchanged folder, was deliberately left as-is). But the **underlying fragility remains**: the
paths are still hardcoded absolute strings, so the next rename/move breaks them again.

**How to avoid.** Before running anything in `data.py` (`import_data`, `update_*`,
`generate_panel`) or a notebook's setup cell, check `_BASE` / `WD` / `os.chdir(...)` against the
*current* machine. Prefer deriving paths relative to the module/repo root
(e.g. `pathlib.Path(__file__).resolve().parents[…]`) over absolute strings. If you must keep an
absolute path, flag it to Alejandro rather than assuming it is correct. Do not introduce new
hardcoded machine paths.

## 3. Pre-regression `dropna` that misaligned the panel (FIXED)

**What happened.** `compute_breach_indicators` originally dropped NaN rows *before* the
quantile regression. Commit `da0984d` ("fix: remove pre-regression dropna") removed it because
the drop happened at the wrong stage and corrupted index alignment between the boundary series,
its `h`-lag, and the realized series.

**How to avoid.** Let `statsmodels` handle missing rows inside the fit; keep the panel's index
intact so that `.shift(h)` and the realized-vs-boundary comparison stay aligned. Decide *very*
deliberately where (and whether) to `dropna`, and never between fitting bounds and lagging them.

## 4. Quantile crossing

**What happened/risk.** Separately fit quantile regressions are not guaranteed monotone, so
`Bound_Low` can exceed `Bound_High` on some rows. The tests tolerate *occasional* crossing
(`frac_ok > 0.95`) but treat *massive* crossing as a real failure.

**How to avoid.** When adding bound/quantile logic, assert that low < high on the large
majority of rows; if crossing becomes frequent, the specification or sample is the problem, not
a tolerance to widen. Do not silently sort or clip quantiles without saying so.

## 5. Diagnostic signature inconsistency (KNOWN DEBT — not yet fixed)

**What it is.** Most forecast-evaluation diagnostics take `(realized, forecasted, tau)` and are
model-agnostic (Kupiec, Christoffersen). But `compute_fallout_errors` and
`diagnose_residual_acf` **refit a `q_reg` internally** instead. This is inconsistent and means
those two cannot be reused on an arbitrary forecast. It was deliberately left as-is in the
reorg (it works, it is not load-bearing).

**How to avoid making it worse.** New diagnostics should take `(realized, forecasted)` and stay
model-agnostic. If you ever normalize these two functions, do it as its own change with tests,
not as a side effect.

## 6. Deleting a function still called by a notebook

**What happened.** `select_horizon_rolling_origin` was removed in the reorg. A cell in
`direct_forecasting.ipynb` still called it and broke on *Run All*.

**How to avoid.** Before deleting a public function, grep the notebooks *and* `auxi/` for every
call site and handle each (rewrite, comment with a dated `# DEPRECATED …` note, or replace).
After any backend deletion, run *Restart Kernel → Run All* on every affected notebook.

## 7. Defaults drifting out of sync across modules

**What happened.** CAViaR's default quantiles diverged from `qreg`'s grid until commit `b5d72ff`
("align caviar quantile defaults with qreg's 21-point grid").

**How to avoid.** When two modules share a conceptual default (quantile grid, breach quantiles,
train fraction), keep them aligned and ideally reference a single source. If you change a grid
in one place, search for the same default elsewhere.

## 8. Trusting the specs over the code

**What it is.** `docs/superpowers/specs/` and `plans/` are point-in-time design records. The
2026-06-26 reorg spec describes the Approach-C wrappers and an *uncured* lookahead — both of
which the code has since superseded (see #1). The rolling-entropy spec references
`auxi/jsu_dist.py`, a filename that no longer exists (it is now `distribution_analysis.py` /
`vulnerability_metrics.py`).

**How to avoid.** The **code is ground truth**; specs explain intent and history. Verify any
symbol, filename, or signature against the current source before relying on it. If a spec and
the code disagree, the code wins — and note the drift here or in `decisions.md`.

## 9. Single-split OOS gave a noisy, unfair error metric (FIXED)

**What happened.** `evaluate_direct_forecasting` trained once on a fixed training window and
computed the OOS pinball loss on the *single* remaining test window — one number per horizon.
That single realization could be dominated by the particular regime in the test window, so the
error metric was not a fair measure of the model's h-step skill. It also evaluated only *one*
quantile at a time.

**The fix (2026-06-29).** `compute_rolling_pinball` rolls a fixed-width (1000-obs) training window
across the test set: for every forecast origin it refits, makes one h-step-ahead forecast, and
records the pinball loss, then averages over ~(N_test − h) origins per (horizon, quantile). It
takes a *list* of quantiles and `plot_rolling_pinball` overlays them. `get_oos_predictions_rolling`
does the same for the coverage-test path. The old function survives as
`evaluate_direct_forecasting_single` (aliased to the old name) but the notebook no longer uses it.

**How to avoid.** A backtest error metric must be an *average over many forecasts*, not a single
split. When evaluating an h-step forecaster, generate many h-step forecasts (one per rolling
origin) and aggregate; never report a single test-window number as "the" OOS loss. (Note the
documented multi-step convention: training targets may run to `y_{t+h-1}` for `h>1`, but the
scored `y_{t+h}` is never in the training set — see `decisions.md`.)

## 10. Missing imports in `diagnostics/distribution_fitting.py` after refactor (FIXED)

**What happened.** When distribution-fitting diagnostics were moved from
`distribution_analysis.py` to `diagnostics/distribution_fitting.py` (commit `8703c75`), the
import block only carried over the fitter/PDF/CDF/sampler functions. Five names used inside the
moved functions — `johnsonsu`, `kstest` (from `scipy.stats`), `jsu_summary`, `jsu_plot`,
`skewt_summary`, and `scipy_skewt` (from `auxi.distribution_analysis`) — were not imported. The
module imported fine (no top-level use), but every runtime call (`fit_and_diagnose_jsu`,
`evaluate_oos_pit`, `fit_and_diagnose_skewt`, `evaluate_oos_pit_skewt`) hit `NameError`.

**The fix (2026-06-30).** Added all six missing names to the import block.

**How to avoid.** When moving functions between modules, grep for *every* unqualified name used
inside the function bodies — not just the functions they call from the origin module. Names from
third-party packages (`scipy.stats`) and module-level singletons (`scipy_skewt`) are easy to
miss because they don't show up as direct-call imports.

## Pre-flight checklist (run mentally before shipping)

- No fit touches the test window; horizon lag respected; cutoff provable.
- Input `df` not mutated; ephemeral columns live on a `.copy()`.
- Paths in `data.py` match the current machine.
- Index alignment intact across shift/lag/compare; `NaN` ≠ 0.
- Every deleted/renamed symbol's call sites (notebooks + `auxi/`) handled.
- Shared defaults still consistent across modules.
- Symbols referenced actually exist in the current code, not just in a spec.
