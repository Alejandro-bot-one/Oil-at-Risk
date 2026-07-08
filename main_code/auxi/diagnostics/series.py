"""Series/data utilities (stationarity tests, trend filters).

Moved from auxi/diagnostics.py during the 2026-06-26 backend reorg.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def adf_test_all(df, signif=0.05):
    """Augmented Dickey-Fuller Test on all columns of a dataframe."""
    results = {}
    for col in df.columns:
        series = df[col].dropna()
        adf_result = adfuller(series)
        results[col] = {"ADF Statistic": adf_result[0], "p-value": adf_result[1], "Stationary": adf_result[1] < signif}
    return pd.DataFrame(results).T

def hamilton_filter(series, h=24, p=12):
    """Removes long-term trend from an economic series."""
    df = pd.DataFrame({"y": series})
    lag_cols = []
    for i in range(p):
        lag_name = f"lag_{h + i}"
        df[lag_name] = df["y"].shift(h + i)
        lag_cols.append(lag_name)

    df_clean = df.dropna()
    if df_clean.empty:
        return pd.Series(index=series.index, dtype=float)

    X = sm.add_constant(df_clean[lag_cols])
    model = sm.OLS(df_clean["y"], X).fit()

    cycle = pd.Series(index=series.index, dtype=float)
    cycle.loc[df_clean.index] = model.resid
    return cycle
