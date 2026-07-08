import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def optimize_gprd_window(df: pd.DataFrame, 
                         y: str, 
                         tau: float, 
                         h: int = 1, 
                         max_window: int = 30, 
                         controls: list = None, 
                         train_fraction: float = 0.8) -> pd.DataFrame:
    """
    Optimizes the moving average window for GPRD in a direct forecasting 
    quantile regression by minimizing Out-of-Sample Pinball Loss and RMSE.
    """
    if controls is None:
        controls = []
        
    df_work = df.copy()
    
    # 1. Generate all Moving Average columns iteratively
    ma_cols = []
    for w in range(1, max_window + 1):
        col_name = f"GPRD_MA{w}"
        df_work[col_name] = df_work["GPRD"].rolling(window=w).mean()
        ma_cols.append(col_name)
        
    # 2. Generate the Direct Forecasting Target (Y_{t+h})
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    
    # 3. Strict Data Cleaning (Crucial Step)
    # We drop NaNs across ALL generated columns simultaneously. 
    # This ensures that window=1 and window=30 are evaluated on the exact same 
    # out-of-sample dates, allowing for an apples-to-apples comparison.
    cols_to_keep = [target_col, "GPRD"] + ma_cols + controls
    df_clean = df_work[cols_to_keep].dropna().copy()
    
    # 4. Chronological Train/Test Split (No random shuffling in time series!)
    split_idx = int(len(df_clean) * train_fraction)
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]
    
    print(f"Optimization setup: {len(df_train)} Train days, {len(df_test)} Test days.")
    
    results = []
    
    # 5. The Optimization Loop
    for w in range(1, max_window + 1):
        x_col = f"GPRD_MA{w}"
        
        # Build formula
        control_str = (" + " + " + ".join([f"Q('{c}')" for c in controls])) if controls else ""
        formula = f"Q('{target_col}') ~ Q('{x_col}')" + control_str
        
        # Train the model on the Training Set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(formula=formula, data=df_train)
            try:
                reg = mod.fit(q=tau, max_iter=2000)
            except ValueError:
                reg = mod.fit(q=tau, max_iter=2000, vcov="iid")
        
        # 6. Forecast on the Test Set
        # statsmodels' predict handles the Q() wrapping automatically
        preds = reg.predict(exog=df_test)
        actuals = df_test[target_col]
        
        # 7. Calculate Out-of-Sample Errors
        errors = actuals - preds
        
        # Metric A: RMSE (Included for reference, but biased towards the mean)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Metric B: Pinball Loss (The true objective function of QR)
        # Formula: L(e) = (tau - 1_{e < 0}) * e
        pinball_loss = np.mean(np.where(errors >= 0, tau * errors, (tau - 1) * errors))
        
        results.append({
            "Window": w,
            "RMSE": rmse,
            "Pinball_Loss": pinball_loss
        })
        
    return pd.DataFrame(results)


from scipy.signal import find_peaks


def identify_brent_turning_points(prices: pd.Series, min_distance_days: int = 90, min_prominence_pct: float = 0.15):
    """
    Identifies structural peaks (turning points into bear markets) and 
    troughs (turning points into bull markets) in Brent crude prices.
    
    Parameters:
    - min_distance_days: Minimum duration of a cycle phase.
    - min_prominence_pct: Minimum percentage drop/gain required to qualify as a structural turn.
    """
    # 1. To use percentage-based prominence, we work with log prices
    log_prices = np.log(prices.dropna())
    
    # 2. Find Peaks (Turning point before a Crash)
    peaks_idx, peak_props = find_peaks(
        log_prices.values, 
        distance=min_distance_days, 
        prominence=min_prominence_pct
    )
    
    # 3. Find Troughs (Turning point before a Rally)
    # We invert the series to find minimums using the same peak-finding math
    troughs_idx, trough_props = find_peaks(
        -log_prices.values, 
        distance=min_distance_days, 
        prominence=min_prominence_pct
    )
    
    # 4. Extract the actual dates
    peak_dates = log_prices.iloc[peaks_idx].index
    trough_dates = log_prices.iloc[troughs_idx].index
    
    # 5. Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(prices.index, prices.values, color='steelblue', label='Brent Crude Price', alpha=0.8)
    
    # Mark Peaks (Red Triangles)
    plt.scatter(peak_dates, prices.loc[peak_dates], color='crimson', 
                marker='v', s=100, zorder=5, label='Peak (Start of Crash)')
    
    # Mark Troughs (Green Triangles)
    plt.scatter(trough_dates, prices.loc[trough_dates], color='forestgreen', 
                marker='^', s=100, zorder=5, label='Trough (Start of Rally)')
    
    plt.title("Structural Turning Points in Brent Crude", fontsize=14, fontweight='bold')
    plt.ylabel("Price (USD)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return peak_dates, trough_dates
