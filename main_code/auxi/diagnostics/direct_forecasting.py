"""Direct-forecasting evaluation diagnostics (shared by qreg-DF and caviar-DF).

These functions take a pre-fitted forecast as (realized, forecasted) series
OR refit a quantile-regression model internally on the dataframe.
Either way, the model used is opaque to the caller - the same diagnostic
applies to a qreg forecast and to a caviar forecast.

Moved from auxi/forecasting.py during the 2026-06-26 backend reorg.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf

from auxi.qreg import pinball_loss


def evaluate_direct_forecasting_single(df, x, y, controls, tau=0.05, max_h=90,
                                       train_fraction=0.8, test_start_date=None):
    """
    Evalúa el modelo GaR para horizontes desde h=1 hasta max_h,
    comparando la pérdida Pinball In-Sample vs Out-of-Sample.

    The train/test split can be specified in two ways (mutually exclusive):
      - test_start_date (str, 'YYYY-MM-DD'): train on data strictly before
        that date, test from that date onward.
      - train_fraction (float, default 0.8): fallback fraction-based split
        when test_start_date is not provided.
    """

    if controls is None:
        controls = []

    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].dropna().copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)

    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        split_idx = df_work.index.searchsorted(test_start_dt)
    else:
        split_idx = int(len(df_work) * train_fraction)
    
    results = []
    
    # Usamos tqdm para tener una barra de progreso
    for h in tqdm(range(1, max_h + 1), desc="Evaluando horizontes (h)"):
        
        # 1. Crear el target desplazado h períodos para este bucle
        target_col = f"{y}_target_h{h}"
        df_step = df_work.copy()
        df_step[target_col] = df_step[y].shift(-h)
        
        # 2. Dividir la muestra cronológicamente
        # train: desde el principio hasta el punto de corte
        # test: desde el punto de corte en adelante
        df_train = df_step.iloc[:split_idx].dropna().copy()
        df_test = df_step.iloc[split_idx:].dropna().copy()
        
        # 3. Formular la ecuación
        if controls:
            control_str = " + ".join([f"Q('{c}')" for c in controls])
            equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
        else:
            equation = f"Q('{target_col}') ~ Q('{x}')"
            
        # 4. Ajustar el modelo (In-Sample)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=df_train, formula=equation)
            try:
                reg = mod.fit(q=tau, vcov="robust", max_iter=2000)
            except ValueError:
                reg = mod.fit(q=tau, vcov="iid", max_iter=2000)
                
            # 5. Predecir In-Sample y calcular pérdida
            # statsmodels no necesita pasar exog si predecimos sobre el mismo df_train
            pred_train = reg.predict(exog=df_train)
            loss_is = pinball_loss(tau, df_train[target_col], pred_train)
            
            # 6. Predecir Out-of-Sample y calcular pérdida
            pred_test = reg.predict(exog=df_test)
            loss_oos = pinball_loss(tau, df_test[target_col], pred_test)
            
            results.append({
                "Horizon (h)": h,
                "IS_Loss": loss_is,
                "OOS_Loss": loss_oos
            })
            
    df_results = pd.DataFrame(results)

    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(12, 6))

    plt.plot(df_results["Horizon (h)"], df_results["IS_Loss"],
             label="In-Sample Loss (Train)", color="steelblue", linewidth=2.5)

    plt.plot(df_results["Horizon (h)"], df_results["OOS_Loss"],
             label="Out-of-Sample Loss (Test)", color="crimson", linewidth=2.5, linestyle="--")

    # Rellenar el área entre ambas líneas (la brecha de generalización o "Generalization Gap")
    plt.fill_between(df_results["Horizon (h)"], df_results["IS_Loss"], df_results["OOS_Loss"],
                     color="gray", alpha=0.15, label="Generalization Gap")

    plt.title(f"Pinball Loss Function ($\\tau$={tau})\nIS vs OOS (Horizons 1 to {max_h})", fontsize=14, pad=15)
    plt.xlabel("Forecast Horizon (h days)", fontsize=12)
    plt.ylabel("Pinball Loss (Lower is Better)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_h)
    plt.tight_layout()
    plt.show()

    return df_results


# Backward compat alias for the old single-split evaluator
evaluate_direct_forecasting = evaluate_direct_forecasting_single


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
    max_h : maximum forecast horizon (evaluates h = 1 ... max_h).
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

    Notes
    -----
    Multi-step convention. The realized target y_{t+h} that is scored at
    each origin t is never part of the training window [t - window_size, t),
    so there is no lookahead in the load-bearing sense. However, for h > 1
    the training rows near the window end carry shifted targets up to
    y_{t+h-1} (i.e. observations dated at or after t). This is the standard
    direct-forecasting convention and matches the existing
    `evaluate_direct_forecasting`. It introduces a mild optimistic bias for
    h > 1, documented deliberately rather than corrected.
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

            # Build the model once per origin; the design matrix does not
            # depend on tau, only the fit's q= does. Reused across all taus.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = smf.quantreg(data=df_train, formula=equation)

                for tau in taus:
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


def diagnose_residual_acf(df: pd.DataFrame,
                          x: str, 
                          y: str, 
                          tau: float = 0.5, 
                          h: int = 1, 
                          controls: list[str] = None, 
                          lags: int = 20):
    """
    Diagnoses the MA(h-1) structure of direct multi-step forecast residuals.
    Fits the quantile regression and plots the Autocorrelation Function (ACF) 
    of the residuals, highlighting the theoretical h-1 cutoff point.
    """
    if controls is None:
        controls = []
        
    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].copy()
    
    # 1. Create the shifted target
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    df_train = df_work.dropna().copy()
    
    # 2. Formulate the equation
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{target_col}') ~ Q('{x}')"
        
    # 3. Fit the model and extract In-Sample residuals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.quantreg(data=df_train, formula=equation)
        try:
            reg = mod.fit(q=tau, vcov="robust", max_iter=2000)
        except ValueError:
            reg = mod.fit(q=tau, vcov="iid", max_iter=2000)
            
    residuals = reg.resid
    
    # 4. Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot ACF
    plot_acf(residuals, lags=lags, ax=ax, alpha=0.05, title="")
    
    # Highlight the theoretical MA(h-1) cutoff
    theoretical_cutoff = h - 1
    ax.axvline(x=theoretical_cutoff, color="crimson", linestyle="--", linewidth=2.5, 
               label=f"Theoretical MA Cutoff (Lag {theoretical_cutoff})")
    ax.axvspan(0, theoretical_cutoff, color="crimson", alpha=0.1)
    
    ax.set_title(f"Residual Autocorrelation Function (ACF)\nQuantile $\\tau={tau}$, Horizon $h={h}$", 
                 fontsize=14, pad=15)
    ax.set_xlabel("Lags (Periods)", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def compute_fallout_errors(df: pd.DataFrame, 
                           x: str, 
                           y: str, 
                           lower_tau: float = 0.05, 
                           upper_tau: float = 0.95, 
                           h: int = 1, 
                           controls: list[str] = None):
    """
    Calculates the 'Fallout Error' (Magnitude of Exceedance).
    The error is the absolute distance from the realized value to the closest 
    predicted quantile bound if it falls outside the range, and 0 otherwise.
    
    Returns a DataFrame with the time series of the bounds and errors.
    """
    if controls is None:
        controls = []
        
    # 1. Prepare data and shift target by h
    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].copy()
    
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    df_work = df_work.dropna()
    
    # 2. Formulate the equation
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{target_col}') ~ Q('{x}')"
        
    # 3. Fit Models (In-Sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.quantreg(data=df_work, formula=equation)
        
        # Fit Lower Bound
        try:
            reg_low = mod.fit(q=lower_tau, vcov="robust", max_iter=2000)
        except ValueError:
            reg_low = mod.fit(q=lower_tau, vcov="iid", max_iter=2000)
            
        # Fit Upper Bound
        try:
            reg_high = mod.fit(q=upper_tau, vcov="robust", max_iter=2000)
        except ValueError:
            reg_high = mod.fit(q=upper_tau, vcov="iid", max_iter=2000)
            
    # 4. Generate Time Series Predictions
    pred_low = reg_low.predict(exog=df_work)
    pred_high = reg_high.predict(exog=df_work)
    realized = df_work[target_col]
    
    # 5. Calculate Fallout Error
    fallout = np.zeros(len(df_work))
    
    # Breach below the lower bound
    mask_low = realized < pred_low
    fallout[mask_low] = pred_low[mask_low] - realized[mask_low]
    
    # Breach above the upper bound
    mask_high = realized > pred_high
    fallout[mask_high] = realized[mask_high] - pred_high[mask_high]
    
    df_results = pd.DataFrame({
        "Realized": realized,
        "Lower_Bound": pred_low,
        "Upper_Bound": pred_high,
        "Fallout_Error": fallout
    }, index=df_work.index)
    
    return df_results

def plot_fallout_errors(df: pd.DataFrame, 
                        x: str, 
                        y: str, 
                        lower_tau: float = 0.05, 
                        upper_tau: float = 0.95, 
                        h: int = 1, 
                        controls: list[str] = None):
    """
    Plots the 'Fallout Error' (Magnitude of Exceedance).
    """
    df_results = compute_fallout_errors(df, x, y, lower_tau, upper_tau, h, controls)
    
    # ---------------------------------------------------------
    # 6. VISUALIZATION
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    
    # --- Panel 1: The Tunnel and Breaches ---
    axes[0].plot(df_results.index, df_results["Realized"], color="black", linewidth=1.2, alpha=0.8, label=f"Realized {y} (t+{h})")
    axes[0].plot(df_results.index, df_results["Lower_Bound"], color="crimson", linestyle="--", linewidth=2, label=f"Lower Risk Bound ($\\tau$={lower_tau})")
    axes[0].plot(df_results.index, df_results["Upper_Bound"], color="steelblue", linestyle="--", linewidth=2, label=f"Upper Potential Bound ($\\tau$={upper_tau})")
    
    # Fill the "Safe Zone"
    axes[0].fill_between(df_results.index, df_results["Lower_Bound"], df_results["Upper_Bound"], color="gray", alpha=0.1)
    
    # Highlight the exact points of breach
    breaches_low = df_results[df_results["Realized"] < df_results["Lower_Bound"]]
    breaches_high = df_results[df_results["Realized"] > df_results["Upper_Bound"]]
    axes[0].scatter(breaches_low.index, breaches_low["Realized"], color="red", zorder=5, s=40, label="Downside Breach")
    axes[0].scatter(breaches_high.index, breaches_high["Realized"], color="blue", zorder=5, s=40, label="Upside Breach")
    
    axes[0].set_title(f"In-Sample Quantile Bounds & Breaches (Horizon $h={h}$)", fontsize=14, pad=10)
    axes[0].set_ylabel(f"Value of {y}", fontsize=12)
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    # --- Panel 2: The Fallout Error Series ---
    # Using a bar plot makes the isolated nature of the errors visually pop
    axes[1].bar(df_results.index, df_results["Fallout_Error"], color="darkorange", width=2.0)
    
    # Calculate summary stat for text box
    total_breaches = len(breaches_low) + len(breaches_high)
    avg_fallout_when_breached = df_results[df_results["Fallout_Error"] > 0]["Fallout_Error"].mean()
    if pd.isna(avg_fallout_when_breached):
        avg_fallout_when_breached = 0.0
        
    stats_text = (
        f"Total Breaches: {total_breaches}\n"
        f"Avg Severity (when breached): {avg_fallout_when_breached:.2f}"
    )
    axes[1].text(0.02, 0.85, stats_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"))

    axes[1].set_title("Fallout Error (Magnitude of Exceedance)", fontsize=14, pad=10)
    axes[1].set_ylabel("Error Magnitude", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    return df_results

def evaluate_cumulative_fallout(df: pd.DataFrame, 
                                x: str, 
                                y: str, 
                                lower_tau: float = 0.05, 
                                upper_tau: float = 0.95, 
                                max_h: int = 90, 
                                controls: list[str] = None):
    """
    Computes the summed magnitude of exceedance (Fallout Error) for horizons from 1 to max_h.
    
    Returns a DataFrame with the total fallout error for each horizon.
    """
    results = []
    
    for h in tqdm(range(1, max_h + 1), desc="Evaluating fallout across horizons"):
        df_results = compute_fallout_errors(df, x, y, lower_tau, upper_tau, h, controls)
        sum_fallout = df_results["Fallout_Error"].sum()
        
        results.append({
            "Horizon (h)": h,
            "Total_Fallout_Error": sum_fallout
        })
        
    df_fallout_summary = pd.DataFrame(results)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_fallout_summary["Horizon (h)"], df_fallout_summary["Total_Fallout_Error"], 
             marker='o', color="darkorange", linewidth=2)
             
    plt.title(f"Cumulative Exceedance Magnitude (Fallout Error) up to h={max_h}", fontsize=14, pad=15)
    plt.xlabel("Horizon (h)", fontsize=12)
    plt.ylabel("Summed Magnitude of Exceedance", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    return df_fallout_summary

def compute_unconditional_coverage_unified(realized: pd.Series, forecasted: pd.Series, tau: float) -> pd.DataFrame:
    """
    Computes the Kupiec POF Likelihood Ratio test using unified strict quantile logic.
    Always tests if the realized value falls BELOW the forecast exactly tau% of the time.
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()
    
    # UNIFIED LOGIC: A "hit" is ALWAYS when reality is less than the forecast.
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int)
        
    T = len(hits)
    x = hits.sum()
    p_empirical = x / T
    
    # Kupiec LR Statistic
    if x == 0:
        lr_stat = -2 * (T * np.log(1 - tau))
    elif x == T:
        lr_stat = -2 * (T * np.log(tau))
    else:
        lr_num = (T - x) * np.log(1 - tau) + x * np.log(tau)
        lr_den = (T - x) * np.log(1 - p_empirical) + x * np.log(p_empirical)
        lr_stat = -2 * (lr_num - lr_den)
        
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    results = {
        "Total_Observations": int(T),
        "Expected_Below": tau * T,
        "Actual_Below": int(x),
        "Target_Rate": tau,
        "Empirical_Rate": p_empirical,
        "LR_Statistic": lr_stat,
        "P_Value": p_value,
        "Reject_Null_5pct": p_value < 0.05
    }
    
    print(f"--- Unified Kupiec POF Test (tau={tau}) ---")
    print(f"Target P(Y < Forecast): {tau:.4f} ({results['Expected_Below']:.1f} expected days)")
    print(f"Actual P(Y < Forecast): {p_empirical:.4f} ({x} actual days)")
    print(f"LR Statistic:           {lr_stat:.4f}")
    print(f"P-Value:                {p_value:.4f}")
    if p_value < 0.05:
        print("Result: REJECT Null. The model's unconditional coverage is inaccurate.\n")
    else:
        print("Result: FAIL TO REJECT Null. The model accurately captures the quantile.\n")
        
    return pd.DataFrame([results])

def plot_unconditional_coverage_unified(realized: pd.Series, forecasted: pd.Series, tau: float):
    """
    Visualizes the Unconditional Coverage using unified directional logic.
    Plots the Cumulative Probability (Y < Forecast) over time against the theoretical tau.
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()
    
    # UNIFIED LOGIC: A hit is always when reality falls below the predicted quantile line.
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int)
        
    cumulative_hits = hits.cumsum()
    observations = np.arange(1, len(hits) + 1)
    cumulative_rate = cumulative_hits / observations
    
    # Confidence Intervals built strictly around tau
    z_score = 1.96 
    std_error = np.sqrt((tau * (1 - tau)) / observations)
    upper_bound = np.minimum(1.0, tau + z_score * std_error)
    lower_bound = np.maximum(0.0, tau - z_score * std_error) 
    
    plt.figure(figsize=(12, 6))
    
    # Plot Confidence Funnel
    plt.fill_between(df_eval.index, lower_bound, upper_bound, color="gray", alpha=0.15, 
                     label="95% Confidence Interval (Null Hypothesis)")
    
    # Plot Target Line
    plt.axhline(y=tau, color="black", linestyle="--", linewidth=2, 
                label=f"Target P(Y < Forecast) = {tau}")
    
    # Plot Empirical Cumulative Rate
    color = "crimson" if tau < 0.5 else "steelblue"
    plt.plot(df_eval.index, cumulative_rate, color=color, linewidth=2.5, 
             label="Empirical Cumulative Rate")
    
    final_rate = cumulative_rate.iloc[-1]
    plt.scatter(df_eval.index[-1], final_rate, color=color, s=80, zorder=5)
    plt.annotate(f"{final_rate:.3f}", 
                 (df_eval.index[-1], final_rate), 
                 xytext=(10, 0), textcoords='offset points', 
                 va='center', fontweight='bold', color=color)

    plt.title(f"Unified Kupiec Coverage Convergence ($\tau$={tau})\nEvaluating $P(Y_t < \hat{{y}}_t) = \tau$", 
              fontsize=14, pad=15)
    plt.ylabel("Cumulative Probability Rate", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in vals])

def compute_conditional_coverage(realized: pd.Series, forecasted: pd.Series, tau: float) -> pd.DataFrame:
    """
    Computes the Christoffersen Conditional Coverage Test.
    Evaluates both Unconditional Coverage (Kupiec) and Independence (Clustering) 
    using the unified directional logic.
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()

    # UNIFIED LOGIC: A "hit" is ALWAYS when reality falls below the forecast.
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int).values

    T = len(hits)
    x = hits.sum()

    if T == 0:
        return pd.DataFrame()

    # 1. Unconditional Coverage (LR_UC)
    p_empirical = x / T
    if x == 0:
        lr_uc = -2 * (T * np.log(1 - tau))
    elif x == T:
        lr_uc = -2 * (T * np.log(tau))
    else:
        lr_num = (T - x) * np.log(1 - tau) + x * np.log(tau)
        lr_den = (T - x) * np.log(1 - p_empirical) + x * np.log(p_empirical)
        lr_uc = max(0.0, -2 * (lr_num - lr_den))

    # 2. Christoffersen Independence Test (LR_IND) ~ Chi^2(1)
    # Calculate Markov transition counts
    T00 = T01 = T10 = T11 = 0
    for i in range(1, T):
        if hits[i-1] == 0 and hits[i] == 0: T00 += 1
        elif hits[i-1] == 0 and hits[i] == 1: T01 += 1
        elif hits[i-1] == 1 and hits[i] == 0: T10 += 1
        elif hits[i-1] == 1 and hits[i] == 1: T11 += 1
        
    # Empirical Transition Probabilities
    pi_01 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0.0
    pi_11 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0.0
    pi_total = (T01 + T11) / (T00 + T01 + T10 + T11) if (T00 + T01 + T10 + T11) > 0 else 0.0

    # Safe log function to handle log(0) which evaluates to 0 in likelihoods
    def safe_term(count, prob):
        return count * np.log(prob) if (count > 0 and prob > 0) else 0.0
        
    # Log-Likelihoods
    ll_indep = safe_term(T00 + T10, 1 - pi_total) + safe_term(T01 + T11, pi_total)
    ll_dep = (safe_term(T00, 1 - pi_01) + safe_term(T01, pi_01) + 
                safe_term(T10, 1 - pi_11) + safe_term(T11, pi_11))
                
    # Avoid negative zeros from float precision limits
    lr_ind = max(0.0, -2 * (ll_indep - ll_dep))

    # 3. Conditional Coverage (LR_CC) = LR_UC + LR_IND ~ Chi^2(2)
    lr_cc = lr_uc + lr_ind
    pval_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

    print("p_value for the conditional test is: ", pval_cc)

    return pd.DataFrame([{
        "LR_CC": lr_cc,
        "P_Value": pval_cc,
        "Reject_Null_5pct": pval_cc < 0.05
    }])

def plot_conditional_coverage(realized: pd.Series, forecasted: pd.Series, tau: float):
    """
    Visualizes Conditional Coverage by plotting the cumulative transition probabilities
    (Probability of a Hit given a previous Hit, and given no previous Hit) over time.
    Both should theoretically converge to tau if the model has correct conditional coverage.
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int).values
    
    T = len(hits)
    
    t00 = t01 = t10 = t11 = 0
    pi_01_series = np.full(T, np.nan)
    pi_11_series = np.full(T, np.nan)
    
    for i in range(1, T):
        if hits[i-1] == 0 and hits[i] == 0: t00 += 1
        elif hits[i-1] == 0 and hits[i] == 1: t01 += 1
        elif hits[i-1] == 1 and hits[i] == 0: t10 += 1
        elif hits[i-1] == 1 and hits[i] == 1: t11 += 1
        
        if (t00 + t01) > 0:
            pi_01_series[i] = t01 / (t00 + t01)
        if (t10 + t11) > 0:
            pi_11_series[i] = t11 / (t10 + t11)
            
    plt.figure(figsize=(12, 6))
    
    # Plot Target Line
    plt.axhline(y=tau, color="black", linestyle="--", linewidth=2, label=f"Target $\\tau$={tau}")
    
    # Plot Empirical Cumulative Rates
    plt.plot(df_eval.index, pi_01_series, color="steelblue", linewidth=2.5, label="P(Hit | No Hit yesterday)")
    plt.plot(df_eval.index, pi_11_series, color="crimson", linewidth=2.5, label="P(Hit | Hit yesterday)")
    
    # Plot final markers
    if not np.isnan(pi_01_series[-1]):
        plt.scatter(df_eval.index[-1], pi_01_series[-1], color="steelblue", s=80, zorder=5)
        plt.annotate(f"{pi_01_series[-1]:.3f}", 
                     (df_eval.index[-1], pi_01_series[-1]), 
                     xytext=(10, 5), textcoords='offset points', 
                     va='center', fontweight='bold', color="steelblue")
                     
    if not np.isnan(pi_11_series[-1]):
        plt.scatter(df_eval.index[-1], pi_11_series[-1], color="crimson", s=80, zorder=5)
        plt.annotate(f"{pi_11_series[-1]:.3f}", 
                     (df_eval.index[-1], pi_11_series[-1]), 
                     xytext=(10, -5), textcoords='offset points', 
                     va='center', fontweight='bold', color="crimson")

    plt.title(f"Conditional Coverage Convergence ($\\tau$={tau})\nEvaluating Independence of Hits", 
              fontsize=14, pad=15)
    plt.ylabel("Cumulative Transition Probability", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.ylim(-0.05, 1.05)
    
    vals = plt.gca().get_yticks()
    vals = [v for v in vals if 0 <= v <= 1]
    plt.gca().set_yticks(vals)
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
    plt.tight_layout()
    plt.show()


def plot_coverage_dashboard(realized: pd.Series, forecasted: pd.Series, tau: float):
    """
    Master 1x2 panel combining unconditional and conditional coverage plots.
    Left: Kupiec unconditional coverage convergence.
    Right: Christoffersen conditional coverage (transition probabilities).
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # ── LEFT PANEL: Unconditional Coverage ──
    ax = axes[0]
    cumulative_hits = hits.cumsum()
    observations = np.arange(1, len(hits) + 1)
    cumulative_rate = cumulative_hits.values / observations

    z_score = 1.96
    std_error = np.sqrt((tau * (1 - tau)) / observations)
    upper_bound = np.minimum(1.0, tau + z_score * std_error)
    lower_bound = np.maximum(0.0, tau - z_score * std_error)

    ax.fill_between(df_eval.index, lower_bound, upper_bound, color="gray", alpha=0.15,
                    label="95% Confidence Interval")
    ax.axhline(y=tau, color="black", linestyle="--", linewidth=2,
               label=f"Target P(Y < Forecast) = {tau}")

    color_uc = "crimson" if tau < 0.5 else "steelblue"
    ax.plot(df_eval.index, cumulative_rate, color=color_uc, linewidth=2.5,
            label="Empirical Cumulative Rate")

    final_rate = cumulative_rate[-1]
    ax.scatter(df_eval.index[-1], final_rate, color=color_uc, s=80, zorder=5)
    ax.annotate(f"{final_rate:.3f}",
                (df_eval.index[-1], final_rate),
                xytext=(10, 0), textcoords='offset points',
                va='center', fontweight='bold', color=color_uc)

    ax.set_title(f"Unconditional Coverage ($\\tau$={tau})", fontsize=14, pad=15)
    ax.set_ylabel("Cumulative Probability Rate", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(v) for v in vals])

    # ── RIGHT PANEL: Conditional Coverage ──
    ax = axes[1]
    hits_arr = hits.values

    T = len(hits_arr)
    t00 = t01 = t10 = t11 = 0
    pi_01_series = np.full(T, np.nan)
    pi_11_series = np.full(T, np.nan)

    for i in range(1, T):
        if hits_arr[i-1] == 0 and hits_arr[i] == 0: t00 += 1
        elif hits_arr[i-1] == 0 and hits_arr[i] == 1: t01 += 1
        elif hits_arr[i-1] == 1 and hits_arr[i] == 0: t10 += 1
        elif hits_arr[i-1] == 1 and hits_arr[i] == 1: t11 += 1

        if (t00 + t01) > 0:
            pi_01_series[i] = t01 / (t00 + t01)
        if (t10 + t11) > 0:
            pi_11_series[i] = t11 / (t10 + t11)

    ax.axhline(y=tau, color="black", linestyle="--", linewidth=2, label=f"Target $\\tau$={tau}")
    ax.plot(df_eval.index, pi_01_series, color="steelblue", linewidth=2.5, label="P(Hit | No Hit)")
    ax.plot(df_eval.index, pi_11_series, color="crimson", linewidth=2.5, label="P(Hit | Hit)")

    if not np.isnan(pi_01_series[-1]):
        ax.scatter(df_eval.index[-1], pi_01_series[-1], color="steelblue", s=80, zorder=5)
        ax.annotate(f"{pi_01_series[-1]:.3f}",
                    (df_eval.index[-1], pi_01_series[-1]),
                    xytext=(10, 5), textcoords='offset points',
                    va='center', fontweight='bold', color="steelblue")

    if not np.isnan(pi_11_series[-1]):
        ax.scatter(df_eval.index[-1], pi_11_series[-1], color="crimson", s=80, zorder=5)
        ax.annotate(f"{pi_11_series[-1]:.3f}",
                    (df_eval.index[-1], pi_11_series[-1]),
                    xytext=(10, -5), textcoords='offset points',
                    va='center', fontweight='bold', color="crimson")

    ax.set_title(f"Conditional Coverage ($\\tau$={tau})", fontsize=14, pad=15)
    ax.set_ylabel("Cumulative Transition Probability", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.05, 1.05)

    vals = ax.get_yticks()
    vals = [v for v in vals if 0 <= v <= 1]
    ax.set_yticks(vals)
    ax.set_yticklabels(['{:,.1%}'.format(v) for v in vals])

    fig.suptitle(f"Coverage Diagnostics ($\\tau$={tau})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


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


def plot_unconditional_coverage(realized: pd.Series, forecasted: pd.Series, tau: float):
    """
    Visualizes the Unconditional Coverage by plotting the Cumulative Hit Rate 
    over time against the theoretical tau, including 95% binomial confidence bands.
    """
    df_eval = pd.DataFrame({"Realized": realized, "Forecasted": forecasted}).dropna()
    
    # UNIFIED LOGIC: P(Y < ŷ(τ)) = τ for ALL quantiles.
    # For τ=0.10: 10% of realized falls below the 10th-pct forecast.
    # For τ=0.90: 90% of realized falls below the 90th-pct forecast.
    # The old branch for τ≥0.5 measured P(Y > ŷ(τ)) = 1-τ but kept τ as the
    # target line and CI center — a systematic mismatch, now removed.
    hits = (df_eval["Realized"] < df_eval["Forecasted"]).astype(int)
    tail_name = "Downside" if tau < 0.5 else "Upside"

    # Calculate cumulative metrics
    cumulative_hits = hits.cumsum()
    observations = np.arange(1, len(hits) + 1)
    cumulative_rate = cumulative_hits / observations

    # Calculate 95% Confidence Intervals centered on the null τ.
    # Var(p̂) = τ(1-τ)/n under H₀: p = τ.
    z_score = 1.96
    std_error = np.sqrt((tau * (1 - tau)) / observations)
    upper_bound = tau + z_score * std_error
    lower_bound = np.maximum(0, tau - z_score * std_error)
    
    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Plot the expanding confidence funnel
    plt.fill_between(df_eval.index, lower_bound, upper_bound, color="gray", alpha=0.15, label="95% Confidence Interval (Null Hypothesis)")

    # Plot Target Line
    plt.axhline(y=tau, color="black", linestyle="--", linewidth=2,
                label=f"Target $\\tau$ = {tau}")

    # Plot Empirical Cumulative Rate
    color = "crimson" if tau < 0.5 else "steelblue"
    plt.plot(df_eval.index, cumulative_rate, color=color, linewidth=2.5,
             label=f"{tail_name} Cumulative Hit Rate")

    final_rate = cumulative_rate.iloc[-1]
    plt.scatter(df_eval.index[-1], final_rate, color=color, s=80, zorder=5)
    plt.annotate(f"{final_rate:.3f}",
                 (df_eval.index[-1], final_rate),
                 xytext=(10, 0), textcoords="offset points",
                 va="center", fontweight="bold", color=color)

    plt.title(f"Unconditional Coverage: {tail_name} Risk ($\\tau$={tau})\n"
              f"Target P(Y < Forecast) = {tau}", fontsize=14, pad=15)
    plt.ylabel("Cumulative Hit Rate", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION — DIEBOLD-MARIANO TEST FOR EQUAL PREDICTIVE ACCURACY
# =============================================================================

def tick_loss_series(tau: float,
                     realized: np.ndarray,
                     forecasted: np.ndarray) -> np.ndarray:
    """
    Element-wise asymmetric tick loss (quantile loss).

    L_t = tau * e_t        if e_t >= 0
    L_t = (tau - 1) * e_t  if e_t < 0

    where e_t = realized_t - forecasted_t. Always non-negative.
    Mean of this series equals pinball_loss(tau, realized, forecasted).

    Parameters
    ----------
    tau : quantile level in (0, 1).
    realized : array of realized values.
    forecasted : array of forecasted quantile values.

    Returns
    -------
    np.ndarray of non-negative losses, same length as inputs.
    """
    error = np.asarray(realized) - np.asarray(forecasted)
    return np.where(error >= 0, tau * error, (tau - 1) * error)


def diebold_mariano_test(loss_1: np.ndarray,
                         loss_2: np.ndarray,
                         h: int) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H_0: E[d_t] = 0 where d_t = L_{1,t} - L_{2,t}.
    Uses the rectangular-kernel HAC variance with bandwidth h-1,
    matching the known MA(h-1) autocorrelation structure of h-step
    direct forecast errors (Diebold & Mariano, 1995).
    P-value from the Student-t(P-1) distribution.

    Parameters
    ----------
    loss_1, loss_2 : element-wise loss vectors (same length P).
    h : forecast horizon (HAC bandwidth = h - 1).

    Returns
    -------
    dict with keys: alpha (mean loss differential), t_stat, p_value,
    P (number of observations).

    Interpretation: alpha < 0 means Model 1 has lower average loss.
    """
    d = np.asarray(loss_1) - np.asarray(loss_2)
    P = len(d)
    d_bar = np.mean(d)

    d_centered = d - d_bar
    max_lag = min(h, P)
    gamma = np.array([
        np.dot(d_centered[k:], d_centered[:P - k]) / P
        for k in range(max_lag)
    ])

    lrv = gamma[0] + 2 * np.sum(gamma[1:]) if len(gamma) > 1 else gamma[0]
    lrv = max(lrv, 1e-15)

    t_stat = d_bar / np.sqrt(lrv / P)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=P - 1)

    return {"alpha": d_bar, "t_stat": t_stat, "p_value": p_value, "P": P}


def compute_dm_comparison(models: dict,
                          realized: pd.Series,
                          tau: float,
                          h: int) -> tuple:
    """
    Error metrics and pairwise Diebold-Mariano tests across models.

    Aligns all forecast series on their common index, computes per-model
    error metrics (RMSE, MAPE, average tick loss), and runs pairwise DM
    tests using the asymmetric tick loss as the loss function.

    Parameters
    ----------
    models : dict mapping model name (str) to forecasted pd.Series.
    realized : pd.Series of realized values, same index range as forecasts.
    tau : quantile level for the tick loss.
    h : forecast horizon (sets DM HAC bandwidth to h - 1).

    Returns
    -------
    (error_df, dm_df) : tuple of DataFrames.
        error_df: one row per model with RMSE, MAPE (%), Avg_Tick_Loss.
        dm_df: one row per model pair with Alpha, t_stat, p_value, Significance.
    """
    def _get_stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    common_idx = realized.index.copy()
    for fcst in models.values():
        common_idx = common_idx.intersection(fcst.index)

    r = realized.loc[common_idx].values
    aligned = {name: fcst.loc[common_idx].values
               for name, fcst in models.items()}

    error_records = []
    for name, f in aligned.items():
        errors = r - f
        rmse = np.sqrt(np.mean(errors ** 2))
        mask = np.abs(r) > 1e-8
        mape = (np.mean(np.abs(errors[mask]) / np.abs(r[mask])) * 100
                if mask.sum() > 0 else np.nan)
        avg_tick = np.mean(tick_loss_series(tau, r, f))
        error_records.append({
            "Model": name,
            "RMSE": round(rmse, 6),
            "MAPE": round(mape, 4),
            "Avg_Tick_Loss": round(avg_tick, 6),
        })
    error_df = pd.DataFrame(error_records)

    names = list(aligned.keys())
    dm_records = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            l1 = tick_loss_series(tau, r, aligned[names[i]])
            l2 = tick_loss_series(tau, r, aligned[names[j]])
            res = diebold_mariano_test(l1, l2, h)
            dm_records.append({
                "Model_1": names[i],
                "Model_2": names[j],
                "Alpha": round(res["alpha"], 6),
                "t_stat": round(res["t_stat"], 4),
                "p_value": round(res["p_value"], 4),
                "Significance": _get_stars(res["p_value"]),
            })
    dm_df = pd.DataFrame(dm_records)

    return error_df, dm_df
