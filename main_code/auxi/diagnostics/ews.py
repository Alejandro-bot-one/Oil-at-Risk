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


# =============================================================================
# SECTION 2 — GRANGER CAUSALITY TEST
# =============================================================================

def granger_causality_test(y, x, max_lag=12, criterion="aic"):
    """
    Granger F-test: does X Granger-cause Y?

    Fits restricted (Y ~ own lags) and unrestricted (Y ~ own lags + X lags)
    models by OLS for each candidate lag order p in 1..max_lag, selects the
    optimal p by AIC or BIC, then runs the F-test at the selected p.

    F = ((RSS_r - RSS_ur) / p) / (RSS_ur / (T_eff - 2p - 1))
    where T_eff = N - p.

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

    p = selected_lag
    t_eff = n - p
    Y_dep = y_arr[p:]

    X_r = np.ones((t_eff, p + 1))
    for lag in range(1, p + 1):
        X_r[:, lag] = y_arr[p - lag: n - lag]

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
