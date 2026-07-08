# Decisions — Oil-at-Risk TFM

> Context file for Claude. Settled design decisions, so we don't relitigate or rebuild what is
> already done. Each entry: the decision, the rationale, and its current status. Append new
> decisions with a date; mark superseded ones rather than deleting them (the history is useful).

## Backend organization

**Organize `auxi/` by purpose, not by historical category.** Direct-forecasting estimators
live *with the model engine that fits them* (`qreg.py`, `caviar.py`); evaluation diagnostics
are *shared* and live in `diagnostics/`. (2026-06-26 reorg, tag `post-reorg-2026-06-26`.)

**Single file for `qreg.py` and `caviar.py`; a real subpackage only for `diagnostics/`.** Each
of `qreg`/`caviar` covers exactly one concern, so they stay single files with section banners
even as they grow (~800 / ~470 lines). `diagnostics/` became a package because its contents are
genuinely heterogeneous (model tests vs forecast evals vs distribution GoF vs series utils).
Rationale: the heterogeneity argument that justified splitting diagnostics does **not** apply to
the two engines. Do not convert `qreg`/`caviar` into subpackages.

**`diagnostics/__init__.py` re-exports every public name explicitly.** So
`import auxi.diagnostics as diags; diags.dq_test(...)` keeps working and notebooks did not have
to change their access pattern. Keep the flat surface; add new public functions to the relevant
`__init__` import block.

**Deleted on purpose:** `auxi/forecasting.py` (contents redistributed to `qreg.py` +
`diagnostics/direct_forecasting.py`), `auxi/diagnostics.py` (became the subpackage), and
`select_horizon_rolling_origin` (out of scope). Don't resurrect these.

## CAViaR breach modelling (the most-revised area — read carefully)

**Two distinct tools, kept separate by design:**

- `caviar_i` / `multiple_caviar_i` — **in-sample, contemporaneous** breach indicators for
  *specification analysis*. Bounds from `Q(y_t | x_t)`, indicator anchored to row `t`, no
  shift. Returns `(reg, indicators)` or a `master_df`.
- `compute_breach_indicators(df, vars_x, vars_y, h, …)` — **h-aware, lookahead-free** breach
  indicators for *direct forecasting*. Fit bounds on the training slice, predict over the
  panel, lag by `h`, compare. This is the production forecasting path.

**SUPERSEDED:** the original "Approach C" caviar DF wrappers (`caviar_direct_forecasting`,
`caviar_get_oos_predictions`, etc., commit `a940dc8`) that augmented the full panel and
delegated to `qreg`. They carried the lookahead-B leak and were **replaced** by
`compute_breach_indicators` (commit `df8b32f`). Do **not** rebuild the augment-and-delegate
wrappers; the 2026-06-26 reorg spec that describes them is outdated on this point.

**Breach indicators are ephemeral.** They depend on the specification (`vars_x`, quantiles), so
writing them to the panel would couple the panel to one spec. They live inside the call and are
discarded; the input panel is never mutated.

**Bounds use the same specification (`vars_x`) as the CAViaR regression.** No separate regressor
set for defining the breach vs. running the regression — rejected as needless complexity.

**Breach cuts come from `breach_quantiles` extremes** (`tau_low = min`, `tau_high = max`),
default `[0.05, 0.95]`. Degenerate cuts (`min == max`) raise `ValueError`. Changing the spec
recomputes breaches automatically.

**Naming `_i` (indicator) and `_s` (severity)** distinguish the two breach-regressor variants.
`_i` uses binary indicators (`{0, 1}`); `_s` uses absolute distances (both >= 0).

**`caviar_s` (severity) built on 2026-06-28.** Uses continuous absolute distances instead of
binary indicators: `upside_severity = max(0, y_t - Q_high) >= 0`, `downside_severity =
max(0, Q_low - y_t) >= 0`. Both are non-negative; zero when no breach. Same three-layer
architecture, same h-aware forecasting path, same `master_df` schema.

## Direct forecasting evaluation

**Rolling-window evaluation replaces single-split OOS (2026-06-29).** The old
`evaluate_direct_forecasting` trained once and computed the OOS pinball loss on a *single* test
window — one number per horizon, noisy and unfair to the model (the realization could be
dominated by the particular regime in the test window). It is now renamed
`evaluate_direct_forecasting_single` (with `evaluate_direct_forecasting` kept as a backward-compat
alias) and superseded by `compute_rolling_pinball`. The new function uses a **fixed-width rolling
window**: for each forecast origin in the test set it refits on the most recent `window_size`
observations, makes one h-step-ahead forecast, and records the pinball loss, then averages the
losses per (horizon, quantile). Many h-step forecasts averaged, not one split.

**Window length is 1000 observations** (Alejandro's choice), fixed-width rather than expanding, so
the model adapts to the recent regime and forgets old ones. The notebook passes `window_size=1000`;
the function default `window_size=None` means "use the full pre-test training slice" (used by the
small synthetic-panel tests).

**Evaluate multiple quantiles simultaneously.** `compute_rolling_pinball` takes a *list* of taus
(the notebook uses `eval_taus = [0.05, 0.50, 0.95]` — tail / median / tail), and
`plot_rolling_pinball` overlays them on one chart so tail vs median forecast quality is comparable
in a single figure. This is why the old IS-vs-OOS single-tau plot was dropped.

**Coverage tests use rolling predictions too.** `get_oos_predictions_rolling` replaces
`get_oos_predictions` (from `qreg.py`) in the Kupiec/Christoffersen path; each forecast in the
returned `(realized, forecasted)` series came from its own rolling-window fit. The function lives
in `diagnostics/direct_forecasting.py` (evaluation concern), not `qreg.py` (engine concern); the
original `get_oos_predictions` is left in place, unused by the notebook.

**Multi-step convention, stated explicitly.** The scored realized value `y_{t+h}` is never in the
training window (lookahead-free in the load-bearing sense), but for `h>1` training rows near the
window end carry targets up to `y_{t+h-1}` — the standard direct-forecasting convention, inherited
verbatim from `evaluate_direct_forecasting_single`, a mild optimistic bias documented in the
docstring rather than "fixed".

**Diebold-Mariano test uses the rectangular kernel, not Bartlett (2026-06-30).**
`diebold_mariano_test` implements the standard DM (1995) test with a rectangular-kernel
HAC variance at bandwidth h-1, matching the known MA(h-1) autocorrelation structure of
h-step direct forecast errors. The rectangular kernel assigns weight 1 to all lags up to
h-1 (no tapering). P-values use the Student-t(P-1) distribution for small-sample
robustness. `compute_dm_comparison` is the orchestrator: takes a dict of pre-computed
forecast series, aligns on the common index, and produces an error-metrics table (RMSE,
MAPE, average tick loss) and a pairwise DM table (alpha, t-stat, p-value, significance
stars). The three DM functions live in `diagnostics/direct_forecasting.py` (evaluation
concern) alongside the rolling-pinball and coverage-test functions.

## Distributions

**Exactly two candidate return densities:** Johnson SU (Johnson, 1949) and Azzalini &
Capitanio (2003) skew-t. Each follows the same workflow (pdf/cdf/sample, MLE, MDE,
fit-and-diagnose, comparison plot, OOS backtest). Fit smooth densities to the *forecasted
quantiles* via **Minimum Distance Estimation (MDE)**, following the Adrian et al. (2019)
approach of turning a quantile fan into a density.

**Fitters stay in `distribution_analysis.py`; their goodness-of-fit diagnostics live in
`diagnostics/distribution_fitting.py`.** The diagnostics import the fitters. (2026-06-26 reorg.)

## Risk & vulnerability metrics

**VaR and CVaR are independent metrics**, each with its own compute/OOS/plot functions, sharing
only a private renderer (`_plot_tail_risk`) to avoid plot duplication without coupling the
pipelines. **Both tails** are always covered: left = Oil-at-Risk (downside), right =
Growth-at-Risk (upside). **Two flavours per date:** conditional (JSU) and unconditional
(historical simulation).

**Keep `risk_metrics.py` and `risk_metrics_boosted.py` as separate modules.** `_boosted` is a
drop-in vectorized replacement with the *same public API* and numbers matching the readable
version to ~1e-11. Merging them was explicitly declined (different concern: optimization).
Decision rule: the readable module is the reference; the boosted one must always be proven to
match before use.

**Tail vulnerability = KL divergence vs a Normal baseline, decomposed Full / Left / Right**,
per Adrian et al. Both **expanding-** and **rolling-window** OOS backtests are provided, sharing
one output schema so the same `plot_tail_entropy` works on either.

## Data

**Returns are in percent (`pct_change() * 100`)** to match the controls, which are `*100`
log-differences. Keep the scale consistent across target and regressors.

**Daily and monthly panels** are both generated and cached as CSVs; notebooks load via
`import_data`.

## Process

**Git initialized at the reorg** with rollback tags (`pre-reorg-2026-06-26`,
`post-reorg-2026-06-26`); one commit per plan task. **TDD** (failing test first) is the working
method. See `workflow.md`.

## Explicitly out of scope (YAGNI — don't build unless asked)

- A distinct regressor set for bounds vs. the CAViaR regression.
- An `h_breach` / double-horizon parameter (the h-aware design makes it unnecessary).
- Merging `risk_metrics_boosted.py` into `risk_metrics.py`.
- Normalizing the `vars_x`-vs-`x` signature convention across modules (high notebook risk, low
  benefit).
- Converting `qreg.py` or `caviar.py` into subpackages.
- An empty `diagnostics/risk_metrics.py` (create it only when real risk-metric diagnostics
  exist).
- The slice-aware bounds refactor for the old caviar wrappers — moot, since
  `compute_breach_indicators` solved the lookahead a different way.

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
