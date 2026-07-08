# data.py
# =============================================================================
# DATA MODULE  -  Oil-at-Risk TFM
# =============================================================================
#
#  SECTION 1 : Imports & Configuration
#  SECTION 2 : Update Functions  (update_brent, update_gpr, update_controls)
#  SECTION 3 : Panel Generation  (generate_panel)
#  SECTION 4 : Data Import       (import_data)
#  SECTION 5 : Master Update     (update_all)
#
# =============================================================================

# =============================================================================
# SECTION 1 - IMPORTS & CONFIGURATION
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Paths
_BASE   = r"C:\Users\Alejandro\Documents\MQuEA\TFM"
RAW_DIR = os.path.join(_BASE, "data", "raw")   # raw source xlsx files
PAN_DIR = os.path.join(_BASE, "data")          # generated panel CSVs

API_PATH = r"C:\Users\Alejandro\Documents\Documentos personales\Claves de recuperacion de cuentas y APIs\Fred API.txt"

# Panel CSV filenames
_PANEL_FILES = {
    "Daily":   "daily_panel.csv",
    "Monthly": "monthly_panel.csv",
}


# =============================================================================
# SECTION 2 - UPDATE FUNCTIONS
# =============================================================================

def update_brent() -> None:
    """
    Downloads Brent oil, VOIL, and VIX data from FRED, calculates returns,
    squared returns (Realized Volatility), rolling windows, and lagged features.
    Saves to data/raw/brent.xlsx with 'Daily' and 'Monthly' sheets.
    """
    with open(API_PATH, "r") as f:
        api_key = f.read().strip()

    fred = Fred(api_key=api_key)

    try:
        print("Downloading brent/voil/vix from FRED...")
        brent_raw = fred.get_series("DCOILBRENTEU")
        voil_raw  = fred.get_series("OVXCLS")
        vix_raw   = fred.get_series("VIXCLS")
    except Exception as e:
        raise RuntimeError(f"Could not download from FRED. Exception: {e}")

    # Daily
    df_daily = brent_raw.to_frame(name="Brent_Price")
    df_daily = df_daily.join(voil_raw.rename("VOIL_Price"), how="left")
    df_daily = df_daily.join(vix_raw.rename("VIX_Price"),  how="left")

    df_daily["Brent_Return"]        = df_daily["Brent_Price"].pct_change() * 100  # percent (matches control *100 log-diffs)
    df_daily["VOIL_Return"]         = df_daily["VOIL_Price"].diff()
    df_daily["VIX_Return"]          = df_daily["VIX_Price"].diff()
    df_daily["Realized_Volatility"] = df_daily["Brent_Return"] ** 2
    df_daily["Brent_Abs_Return"]    = df_daily["Brent_Return"].abs()

    for i in range(1, 4):
        df_daily[f"Brent_Abs_Return (t-{i})"]    = df_daily["Brent_Abs_Return"].shift(i)
        df_daily[f"Brent_Return (t-{i})"]        = df_daily["Brent_Return"].shift(i)
        df_daily[f"Realized_Volatility (t-{i})"] = df_daily["Realized_Volatility"].shift(i)
        df_daily[f"VOIL_Return (t-{i})"]         = df_daily["VOIL_Return"].shift(i)
        df_daily[f"VIX_Return (t-{i})"]          = df_daily["VIX_Return"].shift(i)

    for window in (2, 5, 7):
        df_daily[f"Realized_Volatility_MA{window}"] = df_daily["Realized_Volatility"].rolling(window).mean()
        df_daily[f"VOIL_Return_MA{window}"]          = df_daily["VOIL_Return"].rolling(window).mean()
        df_daily[f"VIX_Return_MA{window}"]           = df_daily["VIX_Return"].rolling(window).mean()

    # Monthly
    df_monthly = df_daily[["Brent_Price", "VOIL_Price", "VIX_Price"]].resample("MS").mean()
    df_monthly["Brent_Return"]        = df_monthly["Brent_Price"].pct_change() * 100  # percent (matches control *100 log-diffs)
    df_monthly["VOIL_Return"]         = df_monthly["VOIL_Price"].diff()
    df_monthly["VIX_Return"]          = df_monthly["VIX_Price"].diff()
    df_monthly["Realized_Volatility"] = df_monthly["Brent_Return"] ** 2
    df_monthly["Brent_Abs_Return"]    = df_monthly["Brent_Return"].abs()

    for i in range(1, 4):
        df_monthly[f"Brent_Abs_Return (t-{i})"]    = df_monthly["Brent_Abs_Return"].shift(i)
        df_monthly[f"Brent_Return (t-{i})"]        = df_monthly["Brent_Return"].shift(i)
        df_monthly[f"Realized_Volatility (t-{i})"] = df_monthly["Realized_Volatility"].shift(i)
        df_monthly[f"VOIL_Return (t-{i})"]         = df_monthly["VOIL_Return"].shift(i)
        df_monthly[f"VIX_Return (t-{i})"]          = df_monthly["VIX_Return"].shift(i)

    for window in (2, 5, 7):
        df_monthly[f"Realized_Volatility_MA{window}"] = df_monthly["Realized_Volatility"].rolling(window).mean()
        df_monthly[f"VOIL_Return_MA{window}"]          = df_monthly["VOIL_Return"].rolling(window).mean()
        df_monthly[f"VIX_Return_MA{window}"]           = df_monthly["VIX_Return"].rolling(window).mean()

    out = os.path.join(RAW_DIR, "brent.xlsx")
    with pd.ExcelWriter(out) as writer:
        df_daily.to_excel(writer,   sheet_name="Daily",   index_label="Date")
        df_monthly.to_excel(writer, sheet_name="Monthly", index_label="Date")

    print(f" brent.xlsx updated  (daily: {len(df_daily):,} | monthly: {len(df_monthly):,})")


def update_gpr() -> None:
    """
    Downloads monthly and daily GPR data from Matteo Iacoviello's website,
    computes regional common factors via PCA, and saves to data/raw/gpr.xlsx.
    """
    print("Downloading GPR data from Iacoviello...")

    try:
        url_m = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
        url_d = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

        gpr_m = pd.read_excel(url_m)
        gpr_m["month"] = pd.to_datetime(gpr_m["month"])
        gpr_m.set_index("month", inplace=True)

        gpr_d = pd.read_excel(url_d)
        gpr_d["date"] = pd.to_datetime(gpr_d["date"])
        gpr_d.set_index("date", inplace=True)

    except Exception as e:
        raise RuntimeError(f"Could not download GPR data. Exception: {e}")

    factor_dict = {
        "Europe_Factor": ["GPRC_GBR","GPRC_DEU","GPRC_FRA","GPRC_ITA","GPRC_ESP",
                          "GPRC_NLD","GPRC_CHE","GPRC_SWE","GPRC_BEL","GPRC_POL",
                          "GPRC_PRT","GPRC_GRC","GPRC_NOR"],
        "Oil_Factor":    ["GPR_SAU","GPR_USA","GPR_RUS","GPR_CAN","GPR_IRQ","GPR_IRN",
                          "GPR_ARE","GPR_KWT","GPR_VEN","GPR_NGA","GPR_NOR","GPR_MEX",
                          "GPR_DZA","GPR_AGO"],
        "SA_Factor":     ["GPR_BRA","GPR_ARG","GPR_COL","GPR_CHL","GPR_PER","GPR_VEN"],
        "BRICS_Factor":  ["GPR_BRA","GPR_RUS","GPR_IND","GPR_CHN","GPR_ZAF"],
    }

    def _add_factors(df):
        for name, cols in factor_dict.items():
            valid = [c for c in cols if c in df.columns]
            if not valid:
                continue
            data = df[valid].ffill().dropna()
            if data.empty:
                continue
            data = data.replace(0, 0.001)
            scaled = StandardScaler().fit_transform(np.log(data))
            pc1 = PCA(n_components=1).fit_transform(scaled)
            df[name] = pd.Series(pc1[:, 0], index=data.index)
        return df

    print("Computing PCA factors...")
    gpr_m = _add_factors(gpr_m)
    gpr_d = _add_factors(gpr_d)

    out = os.path.join(RAW_DIR, "gpr.xlsx")
    with pd.ExcelWriter(out) as writer:
        gpr_m.to_excel(writer, sheet_name="Monthly GPR", index_label="Date")
        gpr_d.to_excel(writer, sheet_name="Daily GPR",   index_label="Date")

    print(f" gpr.xlsx updated  (monthly: {len(gpr_m):,} | daily: {len(gpr_d):,})")


def update_controls() -> None:
    """
    Downloads macro/energy control variables from online sources and FRED,
    builds Daily (interpolated) and Monthly panels, applies log-differences
    (or first-difference for REIA), and saves to data/raw/controls.xlsx.

    Sources
    -------
    - USD index, nat-gas spot price : FRED API
    - US weekly crude stocks        : EIA (WCRSTUS1w.xls)
    - US total product supplied     : EIA (MTTUPUS1m.xls)
    - US crude oil production       : EIA (MCRFPUS1m.xls)
    - Real Economic Activity Index  : Dallas Fed (igrea.xlsx)
    - BADI                          : local data/raw/badi.xlsx
    """
    print("Processing control variables...")

    with open(API_PATH, "r") as f:
        api_key = f.read().strip()

    fred = Fred(api_key=api_key)

    try:
        print("Downloading USD index and nat-gas from FRED...")
        usd    = fred.get_series("DTWEXBGS").to_frame(name="usd_index")
        natgas = fred.get_series("DHHNGSP").to_frame(name="natgas_spotprice")
    except Exception as e:
        raise RuntimeError(f"Could not download from FRED. Exception: {e}")

    print("Downloading EIA and Dallas Fed data...")
    stocks = pd.read_excel(
        "https://www.eia.gov/dnav/pet/hist_xls/WCRSTUS1w.xls",
        sheet_name="Data 1", skiprows=2, index_col="Date", parse_dates=True)
    stocks.columns = ["us_weekly_stocks"]

    supply = pd.read_excel(
        "https://www.eia.gov/dnav/pet/hist_xls/MTTUPUS1m.xls",
        sheet_name="Data 1", skiprows=2, index_col="Date", parse_dates=True)
    supply.columns = ["weekly_us_supply"]

    prod = pd.read_excel(
        "https://www.eia.gov/dnav/pet/hist_xls/MCRFPUS1m.xls",
        sheet_name="Data 1", skiprows=2, index_col="Date", parse_dates=True)
    prod.columns = ["us_oil_production"]

    reia = pd.read_excel(
        "https://www.dallasfed.org/-/media/Documents/research/igrea/igrea.xlsx",
        index_col="Date", parse_dates=True)
    reia.columns = ["REIA"]

    # BADI is not available online - read from local raw file
    badi = pd.read_excel(
        os.path.join(RAW_DIR, "badi.xlsx"), index_col="Date", parse_dates=True)
    badi.columns = ["badi"]

    # Daily
    df_raw   = pd.concat([usd, natgas, stocks, supply, prod, reia, badi], axis=1)
    df_daily = df_raw.resample("D").mean().interpolate(method="linear")

    for col in df_daily.columns:
        if col == "REIA":
            df_daily[f"{col}_fd"] = df_daily[col].diff()
        else:
            df_daily[f"{col}_ld"] = np.log(df_daily[col]).diff() * 100

    # Monthly
    df_monthly = pd.DataFrame({
        "usd_index":         usd["usd_index"].resample("MS").mean(),
        "natgas_spotprice":  natgas["natgas_spotprice"].resample("MS").mean(),
        "weekly_us_stocks":  stocks["us_weekly_stocks"].resample("MS").last(),
        "weekly_us_supply":  supply["weekly_us_supply"].resample("MS").mean(),
        "us_oil_production": prod["us_oil_production"].resample("MS").mean(),
        "REIA":              reia["REIA"].resample("MS").mean(),
        "badi":              badi["badi"].resample("MS").mean(),
    })

    for col in df_monthly.columns:
        if col == "REIA":
            df_monthly[f"{col}_fd"] = df_monthly[col].diff()
        else:
            df_monthly[f"{col}_ld"] = np.log(df_monthly[col]).diff() * 100

    df_daily.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_monthly.replace([np.inf, -np.inf], np.nan, inplace=True)

    out = os.path.join(RAW_DIR, "controls.xlsx")
    with pd.ExcelWriter(out) as writer:
        df_daily.to_excel(writer,   sheet_name="Daily",   index_label="Date")
        df_monthly.to_excel(writer, sheet_name="Monthly", index_label="Date")

    print(f" controls.xlsx updated  (daily: {len(df_daily):,} | monthly: {len(df_monthly):,})")


# =============================================================================
# SECTION 3 - PANEL GENERATION
# =============================================================================

def generate_panel(freq: str = "Monthly", date_range: list = None) -> pd.DataFrame:
    """
    Assembles the full analysis panel from the raw xlsx source files and
    saves it to a CSV in data/.

    Parameters
    ----------
    freq : str
        "Daily" or "Monthly".
    date_range : list, optional
        Two-element list [start_date, end_date] (strings or datetime-like).
        If None, the full available history is saved.

    Returns
    -------
    pd.DataFrame  - the assembled panel (also written to CSV).
    """
    if freq not in _PANEL_FILES:
        raise ValueError("freq must be 'Daily' or 'Monthly'.")

    raw = RAW_DIR

    if freq == "Daily":
        brent    = pd.read_excel(os.path.join(raw, "brent.xlsx"),
                                 sheet_name="Daily",     index_col=0, parse_dates=True)
        gpr      = pd.read_excel(os.path.join(raw, "gpr.xlsx"),
                                 sheet_name="Daily GPR", index_col=0, parse_dates=True)
        controls = pd.read_excel(os.path.join(raw, "controls.xlsx"),
                                 sheet_name="Daily",     index_col=0, parse_dates=True)
    else:  # Monthly
        brent    = pd.read_excel(os.path.join(raw, "brent.xlsx"),
                                 sheet_name="Monthly",     index_col=0, parse_dates=True)
        gpr      = pd.read_excel(os.path.join(raw, "gpr.xlsx"),
                                 sheet_name="Monthly GPR", index_col=0, parse_dates=True)
        controls = pd.read_excel(os.path.join(raw, "controls.xlsx"),
                                 sheet_name="Monthly",     index_col=0, parse_dates=True)

    panel = pd.concat([brent, gpr, controls], axis=1)

    if date_range is not None:
        if len(date_range) != 2:
            raise ValueError("date_range must be a two-element list [start, end].")
        panel = panel.loc[date_range[0]:date_range[1]]

    out_name = _PANEL_FILES[freq]
    out_path = os.path.join(PAN_DIR, out_name)
    panel.to_csv(out_path, index_label="Date")

    print(f" {out_name} written  ({len(panel):,} obs, {len(panel.columns):,} columns)")
    return panel


# =============================================================================
# SECTION 4 - DATA IMPORT
# =============================================================================

def import_data(freq: str = "Monthly", date_range: list = None) -> pd.DataFrame:
    """
    Reads the pre-generated panel CSV and returns the requested date slice.

    Parameters
    ----------
    freq : str
        "Daily" or "Monthly".
    date_range : list, optional
        Two-element list [start_date, end_date] (strings or datetime-like).
        Defaults to ['1990-01-01', '2026-04-30'] if not provided.

    Returns
    -------
    pd.DataFrame
    """
    if freq not in _PANEL_FILES:
        raise ValueError("freq must be 'Daily' or 'Monthly'.")

    csv_path = os.path.join(PAN_DIR, _PANEL_FILES[freq])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{_PANEL_FILES[freq]} not found in {PAN_DIR}. "
            f"Run generate_panel(freq='{freq}') first."
        )

    panel = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if date_range is None:
        date_range = ["1990-01-01", "2026-04-30"]

    if len(date_range) != 2:
        raise ValueError("date_range must be a two-element list [start, end].")

    return panel.loc[date_range[0]:date_range[1]]


# =============================================================================
# SECTION 5 - MASTER UPDATE
# =============================================================================

def update_all() -> None:
    """
    Runs all update functions in sequence, then rebuilds both panel CSVs.

      1. update_brent()    - FRED: Brent, VOIL, VIX
      2. update_gpr()      - Iacoviello GPR website
      3. update_controls() - FRED + EIA + Dallas Fed + local badi.xlsx
      4. generate_panel()  - rebuilds daily_panel.csv and monthly_panel.csv
    """
    print("=" * 60)
    print("  DATA UPDATE - Oil-at-Risk TFM")
    print("=" * 60)
    update_brent()
    update_gpr()
    update_controls()
    print("-" * 60)
    print("  Rebuilding panel CSVs...")
    generate_panel(freq="Daily")
    generate_panel(freq="Monthly")
    print("=" * 60)
    print("  Done. Panels are ready to import.")
    print("=" * 60)
