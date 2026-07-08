# Conventions — Oil-at-Risk TFM

> Context file for Claude. These are the patterns the codebase already follows. Match them;
> do not invent new ones. When a convention is academic (it exists because a paper does it
> a certain way), the paper is named so the choice is auditable.

## Guiding principle

This is an **academic** codebase. Every modelling choice should be traceable to a reference
in `references/` and named in the relevant module docstring. Reproducibility and honesty of
the empirical exercise outrank cleverness. A result you cannot defend in the thesis defense is
worse than no result. Prefer transparent, slightly slower code in `risk_metrics.py` and keep
the optimized path quarantined in `risk_metrics_boosted.py` with a proven numerical match.

## File and module layout

Each module owns **one concern** and opens with a docstring that states the concept, names
the paper(s), and lists the internal parts. Two header styles coexist and both are fine —
match the one already in the file you are editing:

- Triple-quoted module docstring with a `===` underline and a `PART 1 / PART 2` or layer map
  (e.g. `distribution_analysis.py`, `caviar.py`, `risk_metrics.py`).
- Banner-comment header listing numbered sections (e.g. `data.py`).

Within a file, separate sections with a full-width banner so long files stay navigable:

```python
# =============================================================================
# SECTION 3 — DIRECT FORECASTING HELPERS
# =============================================================================
```

Spanish module docstrings use `CAPA 1/2/3` (layer); English ones use `SECTION` or `PART`.

## Layered Separation of Concerns (the dominant pattern)

Numeric producers are separated from renderers, with a one-call orchestrator on top. This is
the single most important structural convention — replicate it for any new analysis:

1. **`compute_*` / estimator** — pure numeric core. Returns a DataFrame/Series/tuple. No
   plotting, no `plt.show()`. This output feeds *both* tables and plots.
2. **`plot_*` sub-functions** — atomic renderers that take an existing Matplotlib `ax` as
   their first argument and draw onto it. They do not create figures.
3. **Orchestrator** (e.g. `plot_quantile_results`, `plot_caviar_i_results`) — builds the
   figure/grid, calls the atomic plotters, and returns the underlying `master_df`.

`caviar.py` deliberately *reuses* `qreg.py`'s `plot_quantile_coefs` / `plot_pseudo_r2` rather
than duplicating them, because its `master_df` schema is identical. Reuse over duplication.

## Naming

- **Specification variables:** `vars_x` (list, the regressors), `vars_y` / `y` (target),
  `controls` (list, extra regressors), `tau` (a single quantile), `quantiles` (the grid),
  `breach_quantiles` (the tail cuts that define a breach), `h` (forecast horizon).
- A function that takes a single regressor names it `x`; one that takes the whole
  specification names it `vars_x`. Helpers that accept either coerce a `str` to a 1-element
  list at the top: `if isinstance(vars_x, str): vars_x = [vars_x]`.
- **Function-name suffix `_i`** = "indicator variant" (binary breach), as in `caviar_i`.
  **Suffix `_s`** = "severity variant" (signed distance), as in `caviar_s`. The two share
  the same three-layer architecture and `master_df` schema.
- **Private helpers** are prefixed `_` (`_compute_quantile_bounds`, `_compute_breaches`,
  `_plot_coef_panel`) and are not re-exported.
- **`master_df` schema** (canonical, do not rename columns): `Dependent Variable`,
  `Regressor`, `Tau`, `Coefficient`, `Significance`, `Pseudo R-Squared`.
- **Panel column names** follow the data layer's style: `Brent_Return`, `Realized_Volatility`,
  lags as `Brent_Return (t-1)`, moving averages as `Realized_Volatility_MA7`.
- **OOS result folders** are named by spec, e.g. `results/oos/h22_gprd_ma7`.
- **Dated backups** use an ISO suffix, e.g. `data/_pre_pct_backup_2026-06-24/`.

## Docstrings

Write docstrings in **English** (the code surface is English even though planning docs are in
Spanish). Style is NumPy-ish but pragmatic: a one-line summary, then `Parameters` /
`Returns`, and — crucially — a note on **invariants and limitations**. Examples already in
the code:

- State when the input is not mutated: *"No muta df."* / "df is never mutated."
- State lookahead behaviour explicitly (every forecasting function says whether and how it
  avoids lookahead).
- Name the paper when the function implements a published method.

## Statistical / numerical conventions

- **Quantile engine:** always `statsmodels` `smf.quantreg` via `q_reg`. Formulas wrap every
  variable in `Q('name')` so column names with spaces survive (`Q('Brent_Return (t-1)')`).
- **Default quantile grid:** the codebase uses two defaults and they are *not* uniform, so
  check the function you are near. The low-level `multiple_q_regs` still defaults to the sparse
  `[0.05, 0.25, 0.50, 0.75, 0.95]`, but the orchestrators/plotters and the caviar table
  functions default to a denser **21-point grid**, defined exactly as
  `[0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]` (i.e. `0.01, 0.05, 0.10,
  …, 0.95, 0.99`). `breach_quantiles` defaults to `[0.05, 0.95]`. When adding a new quantile
  consumer, match the grid the neighbouring functions use rather than hardcoding five points.
- **Covariance:** request robust standard errors and fall back gracefully:
  ```python
  try:    reg = mod.fit(q=tau, vcov="robust")
  except ValueError:  reg = mod.fit(q=tau, vcov="iid")
  ```
- **Significance stars** via a local `get_stars`: `***` <0.01, `**` <0.05, `*` <0.10.
- **Suppress optimizer chatter** around fits with `warnings.catch_warnings()` +
  `simplefilter("ignore")` — but never suppress an error you should handle.
- **NaN discipline:** an indicator on a row with no computable bound is `NaN`, **not 0** — do
  not let "no breach" and "not computable" collapse. The first `h` rows of an h-lagged series
  are `NaN` by construction.
- **Distributions:** the two sanctioned families are Johnson SU (Johnson, 1949) and Azzalini
  & Capitanio (2003) skew-t. Each exposes the same workflow (pdf/cdf/sample, MLE, MDE,
  fit-and-diagnose, comparison plot, OOS backtest). Add new densities the same way or not
  at all.
- **Tail entropy** decomposes KL divergence into Full / Left / Right following the Adrian et
  al. convention.

## Plotting

- Atomic plotters receive `ax` and draw on it; orchestrators own `plt.subplots(...)`,
  `suptitle`, `tight_layout`, and the single `plt.show()`.
- Use the existing palette/idioms (`tab:` colors, `linestyle="--"`, `alpha≈0.4–0.6` grids,
  `axhline(0)` reference lines) so figures look consistent across the thesis.
- Tests run Matplotlib with the `Agg` backend — never assume a display.

## Tests

- Framework: **pytest**, run from `main_code/` as `python -m pytest tests/ -v` so `auxi` and
  `tests` are importable.
- `tests/conftest.py` holds a **reproducible synthetic panel** fixture (`np.random.default_rng(seed)`,
  injected outliers at known positions to force deterministic breaches). Reuse it; add new
  fixtures there rather than per-file.
- **TDD is the norm** (see `workflow.md`): write the failing test first, then the code.
- Test the numeric layers with asserts; test the plotting layers with a **smoke test** under
  `Agg` that asserts "no exception raised and the expected object returned".
- Standing invariants every relevant test guards: *input panel is never mutated*, *bounds
  don't cross en masse*, *degenerate `breach_quantiles` raise `ValueError`*, *the public
  surface resolves* (post-refactor `dir()` checks).

## What NOT to do

- Don't add `upside_breach` / `downside_breach` (or any ephemeral column) to the caller's
  `df`. Work on a `.copy()`.
- Don't duplicate `qreg` logic into other modules — import it.
- Don't hardcode absolute machine paths in new code (the existing ones in `data.py` are a
  known liability — see `known_errors.md`).
- Don't introduce lookahead into any forecasting/backtest path. If unsure, prove the cutoff.
