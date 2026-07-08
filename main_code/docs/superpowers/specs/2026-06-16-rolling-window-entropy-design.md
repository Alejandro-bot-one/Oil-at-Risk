# Rolling-Window OOS Entropy Backtest — Design

## Context

[`auxi/jsu_dist.py`](../../../auxi/jsu_dist.py) contains `generate_oos_entropy_normal`, which runs an
out-of-sample backtest comparing a conditional Johnson SU (JSU) forecast (fit via quantile
regression + MDE) against a historical Normal baseline, computing KL divergence (relative
entropy) for the full distribution and each tail at every date.

Both the conditional model's training data (`df_train`) and the baseline statistics
(`hist_mean`, `hist_std`, unconditional `fit_jsu`) use an **expanding window**: anchored at the
start of the dataset and growing by one observation each date. This means the cost of every
fit (quantile regressions + `fit_jsu`) grows over the backtest, making the full run slow on long
samples.

## Goal

Add a **rolling/moving-window** analog so the backtest runs in roughly constant time per
iteration instead of growing, while producing output compatible with the existing
`plot_tail_entropy` plotting function (no changes to plotting required).

## Design

### New function: `generate_oos_entropy_normal_rolling`

Added to `auxi/jsu_dist.py`, alongside `generate_oos_entropy_normal`.

```python
def generate_oos_entropy_normal_rolling(df: pd.DataFrame, x_var: str, y_var: str,
                                         controls: list, quantiles: list, h: int,
                                         window: int) -> pd.DataFrame:
```

Signature mirrors `generate_oos_entropy_normal` plus one new required parameter: `window` (int),
the number of most-recent observations to use for every fit.

### Behavior

For each `current_date` in the evaluable date range:

1. **Look-ahead bias prevention** — unchanged: training data may only use observations up to
   `current_date - h`.
2. **Rolling slice** — instead of `df_eval.loc[:cutoff_date]` (all history), take the last
   `window` rows ending at the look-ahead-safe cutoff:
   `df_eval.iloc[cutoff_idx - window + 1 : cutoff_idx + 1]`.
3. **Skip condition** — if fewer than `window` observations are available yet (early dates),
   skip the date. This replaces the `len(df_train) < 30` minimum-history check in the expanding
   version — every fit in the rolling version uses a full, fixed-size window or is skipped.
4. **Historical baseline (`hist_mean`, `hist_std`, unconditional `fit_jsu`)** — also computed on
   the last `window` observations of `y_var` up to `current_date` (same `window` parameter,
   single knob — no separate baseline window), instead of all history from the start.
5. **Conditional forecast** — same per-quantile `q_reg` + `mde_jsu_weighted` logic as the
   expanding version, just fit on the rolling `df_train` slice.
6. **Entropy computation** — unchanged, reuses `compute_kl_divergence_normal`.

Everything else (per-date try/except fallbacks that skip a date on optimizer failure, the result
dict fields collected per date) is identical to `generate_oos_entropy_normal`.

### Output schema (unchanged from the expanding version)

A `pandas.DataFrame` indexed by `Date` with columns:
`Full_Entropy`, `Left_Entropy`, `Right_Entropy`, `Uncond_Full_Entropy`, `Uncond_Left_Entropy`,
`Uncond_Right_Entropy`, `hist_mean`, `hist_std`, `cond_a`, `cond_b`, `cond_loc`, `cond_scale`.

Because the schema and index name match exactly, **`plot_tail_entropy(price_series, df_entropy, h)`
works unmodified** on the output of either function — no changes to the plotting function are
needed.

### Non-goals

- No changes to `generate_oos_entropy_normal` (expanding version) — it stays as-is for comparison.
- No changes to `plot_tail_entropy`.
- No calendar-based (date-offset) windowing — `window` is a fixed observation count.
