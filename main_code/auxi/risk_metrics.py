"""
Risk Metrics — VaR & CVaR (Oil-at-Risk / Growth-at-Risk)
=========================================================
VaR and CVaR are treated as **independent metrics** with their own compute
functions, OOS drivers, and plot functions.  A shared private renderer
``_plot_tail_risk`` avoids code duplication in the plots without coupling the
two pipelines together.

Both metrics cover **both tails**:
  Left tail  -> Oil-at-Risk (OaR)     downside  (negative values)
  Right tail -> Growth-at-Risk (GaR)  upside    (positive values)

Two flavours per date:
  ① Unconditional (Historical Simulation) — rolling window of 1-day returns
    scaled to horizon via sqrt-of-time rule (Basel III §OIS20.5).
  ② Conditional (GaR / JSU) — from JSU params stored in df_entropy output
    of vulnerability_metrics.generate_oos_entropy_normal_rolling.
    No time-scaling needed (quantile regression already targets y_{t+h}).
    Methodology: Adrian, Boyarchenko & Giannone (2019, AER).

Sign convention
---------------
  *_Left  < 0   (losses  — Oil-at-Risk direction)
  *_Right > 0   (gains   — Growth-at-Risk direction)

Basel III defaults
------------------
  confidence = 0.975  (97.5%)
  horizon    = 10     (10-day holding period)
  window     = 1_008  (4 trading years x 252)

Public API
----------
  VaR pipeline:
      compute_conditional_var(cond_params, confidence)
      compute_historical_var(returns_window, confidence, horizon)
      generate_oos_var(df, y_var, df_entropy, window, confidence, horizon)
      plot_var(price_series, df_var, h, confidence, horizon, window)

  CVaR / ES pipeline:
      compute_conditional_cvar(cond_params, confidence)
      compute_historical_cvar(returns_window, confidence, horizon)
      generate_oos_cvar(df, y_var, df_entropy, window, confidence, horizon)
      plot_cvar(price_series, df_cvar, h, confidence, horizon, window)

References
----------
Basel Committee on Banking Supervision (2019). Minimum capital requirements
    for market risk (Basel III, §OIS20).
Adrian, T., Boyarchenko, N., & Giannone, D. (2019). Vulnerable Growth.
    American Economic Review, 109(4), 1263-1289.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu
from tqdm import tqdm


# =============================================================================
# VaR PIPELINE
# =============================================================================

def compute_conditional_var(
    cond_params,
    confidence: float = 0.975,
) -> tuple:
    """
    Value-at-Risk from a fitted Johnson SU distribution — both tails.

    No time-scaling applied. The JSU is already fitted to the h-step-ahead
    return distribution (quantile regression targets y_{t+h}).

    Parameters
    ----------
    cond_params : array-like of length 4
        [a, b, loc, scale] — scipy johnsonsu parameterisation.
    confidence : float, default 0.975 (Basel III).

    Returns
    -------
    (var_left, var_right) : tuple[float, float]
        var_left  < 0   Oil-at-Risk downside threshold.
        var_right > 0   Growth-at-Risk upside threshold.
    """
    a, b, loc, scale = cond_params
    alpha = 1.0 - confidence
    var_left  = float(johnsonsu.ppf(alpha,      a, b, loc=loc, scale=scale))
    var_right = float(johnsonsu.ppf(confidence, a, b, loc=loc, scale=scale))
    return var_left, var_right


def compute_historical_var(
    returns_window: np.ndarray,
    confidence: float = 0.975,
    horizon: int = 10,
) -> tuple:
    """
    Historical Simulation VaR scaled to a multi-day holding period.

    Procedure (Basel III §OIS20):
      1. (1-confidence) empirical quantile of 1-day returns -> VaR1_left.
         confidence      empirical quantile                 -> VaR1_right.
      2. VaR_h = VaR_1 x sqrt(horizon)  (square-root-of-time rule).

    Parameters
    ----------
    returns_window : 1-D array-like — 1-day returns (NaNs dropped).
    confidence : float, default 0.975
    horizon    : int,   default 10

    Returns
    -------
    (var_left, var_right) : tuple[float, float]
        Both scaled to horizon days.
        var_left  < 0   downside VaR.
        var_right > 0   upside VaR.
    """
    r = np.asarray(returns_window, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return np.nan, np.nan
    alpha  = 1.0 - confidence
    sqrt_h = np.sqrt(horizon)
    var1_left  = float(np.nanpercentile(r, alpha * 100))
    var1_right = float(np.nanpercentile(r, confidence * 100))
    return var1_left * sqrt_h, var1_right * sqrt_h


def generate_oos_var(
    df: pd.DataFrame,
    y_var: str,
    df_entropy: pd.DataFrame,
    window: int = 1_008,
    confidence: float = 0.975,
    horizon: int = 10,
    retrain_after: int = 30,
) -> pd.DataFrame:
    """
    Out-of-Sample rolling Value-at-Risk — unconditional and conditional.

    Conditional VaR
        Reads the JSU parameters stored in df_entropy (cond_a, cond_b,
        cond_loc, cond_scale). These are produced by
        vulnerability_metrics.generate_oos_entropy_normal_rolling at no
        extra cost — the expensive quantile regression + MDE fitting that
        computed entropy already produced the per-date conditional distribution.
        This function reads those stored parameters; it does NOT refit anything.
        Its retrain cadence is therefore inherited from the entropy run that
        produced df_entropy.

    Unconditional (Historical Simulation) VaR
        At each date, slices the last `window` 1-day returns strictly before
        current_date (no look-ahead) and calls compute_historical_var.
        ``retrain_after`` gates this recomputation: the historical percentile is
        recomputed only once every ``retrain_after`` observations (default 30 ≈
        one trading month) and carried forward unchanged in between. Set
        ``retrain_after=1`` to recover the original recompute-every-day behaviour.

    Parameters
    ----------
    df         : DataFrame with DatetimeIndex containing 1-day returns y_var.
    y_var      : Column of 1-day returns (e.g. "Brent_Return").
    df_entropy : Output of generate_oos_entropy_normal_rolling.
                 Only cond_a/b/loc/scale columns are used; entropy values ignored.
    window     : int, default 1_008 (4 trading years x 252, Basel III).
    confidence : float, default 0.975.
    horizon    : int,   default 10  (Basel III 10-day holding period).

    Returns
    -------
    pd.DataFrame indexed by Date:
        Cond_VaR_Left, Cond_VaR_Right, Hist_VaR_Left, Hist_VaR_Right.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = pd.to_datetime(df.index)
    results = []
    _hist_cache = None       # last valid (h_vl, h_vr); forward-filled between retrains
    _since_hist = 0          # observations since last historical recompute
    for current_date in tqdm(df_entropy.index, desc="OOS VaR"):
        row = df_entropy.loc[current_date]
        cond_params = [row["cond_a"], row["cond_b"],
                       row["cond_loc"], row["cond_scale"]]
        if any(np.isnan(p) for p in cond_params):
            c_vl = c_vr = np.nan
        else:
            try:
                c_vl, c_vr = compute_conditional_var(cond_params,
                                                      confidence=confidence)
            except Exception:
                c_vl = c_vr = np.nan
        # Historical Simulation — recompute only every `retrain_after` obs.
        if (_hist_cache is None) or (_since_hist >= retrain_after):
            past = df.loc[df.index < current_date, y_var].dropna()
            if len(past) < window:
                h_vl = h_vr = np.nan        # not enough data yet — keep retrying daily
                _hist_cache = None
            else:
                h_vl, h_vr = compute_historical_var(past.iloc[-window:].values,
                                                     confidence=confidence,
                                                     horizon=horizon)
                _hist_cache = (h_vl, h_vr)
            _since_hist = 0
        else:
            h_vl, h_vr = _hist_cache
        _since_hist += 1
        results.append({"Date": current_date,
                        "Cond_VaR_Left":  c_vl, "Cond_VaR_Right": c_vr,
                        "Hist_VaR_Left":  h_vl, "Hist_VaR_Right": h_vr})
    df_results = pd.DataFrame(results).set_index("Date")
    return df_results


# =============================================================================
# CVaR / EXPECTED SHORTFALL PIPELINE
# =============================================================================

def compute_conditional_cvar(
    cond_params,
    confidence: float = 0.975,
) -> tuple:
    """
    CVaR (Expected Shortfall) from a fitted Johnson SU distribution — both tails,
    via numerical integration.

    CVaR_left  = E[X | X <= VaR_left]  = integral x f(x) dx / P(X <= VaR)
    CVaR_right = E[X | X >= VaR_right] = integral x f(x) dx / P(X >= VaR)

    No closed form exists for the JSU Expected Shortfall, so a Riemann sum over
    a 5,000-point grid is used (same approach as compute_kl_divergence_normal).

    This function is fully independent of compute_conditional_var — it
    recomputes the quantile thresholds internally.

    No time-scaling — JSU already represents the h-step-ahead distribution.

    Parameters
    ----------
    cond_params : array-like of length 4 — [a, b, loc, scale].
    confidence  : float, default 0.975

    Returns
    -------
    (cvar_left, cvar_right) : tuple[float, float]
        cvar_left  < var_left   (more extreme, negative).
        cvar_right > var_right  (more extreme, positive).
    """
    a, b, loc, scale = cond_params
    alpha = 1.0 - confidence
    # Thresholds (recomputed independently)
    var_left  = float(johnsonsu.ppf(alpha,      a, b, loc=loc, scale=scale))
    var_right = float(johnsonsu.ppf(confidence, a, b, loc=loc, scale=scale))
    # Dense integration grid
    x_lo = float(johnsonsu.ppf(1e-6,   a, b, loc=loc, scale=scale))
    x_hi = float(johnsonsu.ppf(1-1e-6, a, b, loc=loc, scale=scale))
    x_grid = np.linspace(x_lo, x_hi, 5_000)
    dx     = x_grid[1] - x_grid[0]
    pdf    = johnsonsu.pdf(x_grid, a, b, loc=loc, scale=scale)
    # Left CVaR
    mask_l = x_grid <= var_left
    prob_l = np.sum(pdf[mask_l]) * dx
    cvar_left = float(np.sum(x_grid[mask_l] * pdf[mask_l]) * dx / max(prob_l, 1e-12))
    # Right CVaR
    mask_r = x_grid >= var_right
    prob_r = np.sum(pdf[mask_r]) * dx
    cvar_right = float(np.sum(x_grid[mask_r] * pdf[mask_r]) * dx / max(prob_r, 1e-12))
    return cvar_left, cvar_right


def compute_historical_cvar(
    returns_window: np.ndarray,
    confidence: float = 0.975,
    horizon: int = 10,
) -> tuple:
    """
    Historical Simulation CVaR (Expected Shortfall) scaled to a multi-day
    holding period.

    Procedure:
      1. Recompute (1-confidence) quantile threshold internally (independent).
      2. CVaR1 = mean of all returns strictly in the tail beyond that threshold.
      3. CVaR_h = CVaR_1 x sqrt(horizon).

    Parameters
    ----------
    returns_window : 1-D array-like — 1-day returns (NaNs dropped).
    confidence : float, default 0.975
    horizon    : int,   default 10

    Returns
    -------
    (cvar_left, cvar_right) : tuple[float, float]
        Both scaled to horizon days.
        cvar_left  < 0   (more extreme than var_left).
        cvar_right > 0   (more extreme than var_right).
    """
    r = np.asarray(returns_window, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return np.nan, np.nan
    alpha  = 1.0 - confidence
    sqrt_h = np.sqrt(horizon)
    # Thresholds (recomputed independently)
    var1_left  = float(np.nanpercentile(r, alpha * 100))
    var1_right = float(np.nanpercentile(r, confidence * 100))
    tail_l = r[r <= var1_left]
    tail_r = r[r >= var1_right]
    cvar1_left  = float(np.mean(tail_l)) if len(tail_l) > 0 else var1_left
    cvar1_right = float(np.mean(tail_r)) if len(tail_r) > 0 else var1_right
    return cvar1_left * sqrt_h, cvar1_right * sqrt_h


def generate_oos_cvar(
    df: pd.DataFrame,
    y_var: str,
    df_entropy: pd.DataFrame,
    window: int = 1_008,
    confidence: float = 0.975,
    horizon: int = 10,
    retrain_after: int = 30,
) -> pd.DataFrame:
    """
    Out-of-Sample rolling CVaR (Expected Shortfall) — unconditional and conditional.

    Mirrors generate_oos_var exactly in structure; uses
    compute_conditional_cvar and compute_historical_cvar instead.

    The conditional CVaR reads the stored df_entropy params (cadence inherited
    from the entropy run). ``retrain_after`` gates the Historical Simulation
    CVaR: recomputed once every ``retrain_after`` observations (default 30) and
    carried forward in between. Set ``retrain_after=1`` for the original
    recompute-every-day behaviour.

    Parameters
    ----------
    Same as generate_oos_var.

    Returns
    -------
    pd.DataFrame indexed by Date:
        Cond_CVaR_Left, Cond_CVaR_Right, Hist_CVaR_Left, Hist_CVaR_Right.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = pd.to_datetime(df.index)
    results = []
    _hist_cache = None       # last valid (h_cl, h_cr); forward-filled between retrains
    _since_hist = 0          # observations since last historical recompute
    for current_date in tqdm(df_entropy.index, desc="OOS CVaR"):
        row = df_entropy.loc[current_date]
        cond_params = [row["cond_a"], row["cond_b"],
                       row["cond_loc"], row["cond_scale"]]
        if any(np.isnan(p) for p in cond_params):
            c_cl = c_cr = np.nan
        else:
            try:
                c_cl, c_cr = compute_conditional_cvar(cond_params,
                                                       confidence=confidence)
            except Exception:
                c_cl = c_cr = np.nan
        # Historical Simulation — recompute only every `retrain_after` obs.
        if (_hist_cache is None) or (_since_hist >= retrain_after):
            past = df.loc[df.index < current_date, y_var].dropna()
            if len(past) < window:
                h_cl = h_cr = np.nan        # not enough data yet — keep retrying daily
                _hist_cache = None
            else:
                h_cl, h_cr = compute_historical_cvar(past.iloc[-window:].values,
                                                      confidence=confidence,
                                                      horizon=horizon)
                _hist_cache = (h_cl, h_cr)
            _since_hist = 0
        else:
            h_cl, h_cr = _hist_cache
        _since_hist += 1
        results.append({"Date": current_date,
                        "Cond_CVaR_Left":  c_cl, "Cond_CVaR_Right": c_cr,
                        "Hist_CVaR_Left":  h_cl, "Hist_CVaR_Right": h_cr})
    df_results = pd.DataFrame(results).set_index("Date")
    return df_results


# =============================================================================
# SHARED PLOTTING
# =============================================================================

def _plot_tail_risk(
    price_series: pd.Series,
    df_metric: pd.DataFrame,
    col_cond_left: str,
    col_cond_right: str,
    col_hist_left: str,
    col_hist_right: str,
    metric_name: str,
    metric_label: str,
    h: int,
    confidence: float,
    horizon: int,
    window: int,
) -> None:
    """
    Shared 3-panel renderer for any tail-risk metric (VaR or CVaR).

    Panel 1 : Brent spot price.
    Panel 2 : Conditional (GaR) metric — left tail (red), right tail (green).
    Panel 3 : Unconditional (Historical) metric — same layout.

    All series are smoothed with a 5-day rolling average (same convention as
    plot_tail_entropy in vulnerability_metrics).
    """
    conf_pct  = int(confidence * 100)
    alpha_pct = round((1.0 - confidence) * 100, 1)
    yrs       = window // 252
    sm        = df_metric.rolling(5, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"Oil-at-Risk & Growth-at-Risk — {metric_name} (Basel III)\n"
        f"h = {h}-day forecast  |  {conf_pct}% confidence  |  "
        f"{horizon}-day holding period  |  {yrs}-year historical window",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Panel 1: Brent Price
    ax0 = axes[0]
    aligned = price_series.reindex(df_metric.index)
    ax0.plot(aligned.index, aligned.values, color="black", linewidth=1.5)
    ax0.set_title("Brent Crude Spot Price", fontweight="bold")
    ax0.set_ylabel("USD / bbl")
    ax0.grid(True, alpha=0.3)

    # Panel 2: Conditional
    ax1 = axes[1]
    ax1.plot(sm.index, sm[col_cond_left],  color="crimson",  linewidth=2.2,
             label=f"OaR — {metric_label} ({alpha_pct}%)")
    ax1.plot(sm.index, sm[col_cond_right], color="seagreen", linewidth=2.2,
             label=f"GaR — {metric_label} ({conf_pct}%)")
    ax1.fill_between(sm.index, sm[col_cond_left],  0, color="darkred",   alpha=0.15)
    ax1.fill_between(sm.index, sm[col_cond_right], 0, color="darkgreen", alpha=0.15)
    ax1.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_title(
        f"Conditional (GaR) {metric_name} — JSU from Quantile Regression"
        f"  [h = {h} days, no time-scaling]", fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 3: Unconditional
    ax2 = axes[2]
    ax2.plot(sm.index, sm[col_hist_left],  color="crimson",  linewidth=2.2,
             label=f"OaR — {metric_label} ({alpha_pct}%)")
    ax2.plot(sm.index, sm[col_hist_right], color="seagreen", linewidth=2.2,
             label=f"GaR — {metric_label} ({conf_pct}%)")
    ax2.fill_between(sm.index, sm[col_hist_left],  0, color="darkred",   alpha=0.15)
    ax2.fill_between(sm.index, sm[col_hist_right], 0, color="darkgreen", alpha=0.15)
    ax2.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_title(
        f"Unconditional (Historical Simulation) {metric_name}"
        f"  [{yrs}-yr window  x  sqrt({horizon}) time-scaling]", fontweight="bold")
    ax2.set_ylabel(f"{horizon}-day Return (%)")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date", fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_var(
    price_series: pd.Series,
    df_var: pd.DataFrame,
    h: int,
    confidence: float = 0.975,
    horizon: int = 10,
    window: int = 1_008,
) -> None:
    """
    3-panel VaR dashboard. Wraps _plot_tail_risk with VaR column names.

    Parameters
    ----------
    price_series : Brent spot price series (DatetimeIndex).
    df_var       : Output of generate_oos_var.
    h            : Forecast horizon used when generating df_var.
    confidence, horizon, window : Basel III parameters (for labels).
    """
    _plot_tail_risk(
        price_series=price_series, df_metric=df_var,
        col_cond_left="Cond_VaR_Left",  col_cond_right="Cond_VaR_Right",
        col_hist_left="Hist_VaR_Left",  col_hist_right="Hist_VaR_Right",
        metric_name="Value-at-Risk (VaR)", metric_label="VaR",
        h=h, confidence=confidence, horizon=horizon, window=window,
    )


def plot_cvar(
    price_series: pd.Series,
    df_cvar: pd.DataFrame,
    h: int,
    confidence: float = 0.975,
    horizon: int = 10,
    window: int = 1_008,
) -> None:
    """
    3-panel CVaR / Expected Shortfall dashboard. Wraps _plot_tail_risk.

    Parameters
    ----------
    price_series : Brent spot price series (DatetimeIndex).
    df_cvar      : Output of generate_oos_cvar.
    h            : Forecast horizon used when generating df_cvar.
    confidence, horizon, window : Basel III parameters (for labels).
    """
    _plot_tail_risk(
        price_series=price_series, df_metric=df_cvar,
        col_cond_left="Cond_CVaR_Left",  col_cond_right="Cond_CVaR_Right",
        col_hist_left="Hist_CVaR_Left",  col_hist_right="Hist_CVaR_Right",
        metric_name="CVaR / Expected Shortfall", metric_label="CVaR",
        h=h, confidence=confidence, horizon=horizon, window=window,
    )
