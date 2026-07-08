import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf

def q_reg(df, x, y, tau: float, controls: list[str] = None, **kwargs):
    """
    The engine room. This runs a single quantile regression for a specific tau.
    It builds the math formula for statsmodels to process.
    """
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{y}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{y}') ~ Q('{x}')"

    mod = smf.quantreg(data=df, formula=equation)
    reg = mod.fit(q=tau, **kwargs)
    
    return reg

def multiple_q_regs(data, vars_x, vars_y, quantiles=None, controls=None, errors="robust"):
    """
    Performs quantile regressions across specified quantiles.
    Returns a single, neatly sorted DataFrame containing all regressors.

    `quantiles` defaults to [0.05, 0.25, 0.50, 0.75, 0.95] when not supplied,
    so callers such as plot_quantile_results / plot_contemporaneous_vs_predictive_coefs
    that don't pass it explicitly work out of the box.
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if controls is None:
        controls = []
        
    def get_stars(p_value: float) -> str:
        if p_value < 0.01: return '***'
        elif p_value < 0.05: return '**'
        elif p_value < 0.10: return '*'
        else: return ''

    all_indep_vars = vars_x + controls
    rhs = " + ".join([f"Q('{v}')" for v in all_indep_vars])
    equation = f"Q('{vars_y}') ~ {rhs}"
    
    res_dict = {
        "Dependent Variable": [],
        "Regressor": [],
        "Tau": [],
        "Coefficient": [],
        "Significance": [],
        "Pseudo R-Squared": []
    }
    
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=data, formula=equation)
            
            try:
                reg = mod.fit(q=q, vcov=errors)
            except ValueError:
                reg = mod.fit(q=q, vcov="iid")
            
            pseudo_r2 = getattr(reg, 'prsquared', np.nan)
            
            for var in all_indep_vars:
                param_name = f"Q('{var}')"
                coef = reg.params[param_name]
                pval = reg.pvalues[param_name]
                
                res_dict["Dependent Variable"].append(vars_y)
                res_dict["Regressor"].append(var)
                res_dict["Tau"].append(q)
                res_dict["Coefficient"].append(coef)
                res_dict["Significance"].append(get_stars(pval))
                res_dict["Pseudo R-Squared"].append(pseudo_r2)
                
    master_df = pd.DataFrame(res_dict)
    master_df = master_df.sort_values(by=["Regressor", "Tau"]).reset_index(drop=True)

    return master_df

def plot_quantile_coefs(ax, results_df: pd.DataFrame, vars_to_plot: list, title: str) -> None:
    """Sub-function: Plots quantile regression coefficients and significance stars."""
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i, var in enumerate(vars_to_plot):
        plot_df = results_df[results_df["Regressor"] == var].sort_values(by="Tau")
        color = colors[i % len(colors)]
        
        ax.plot(plot_df["Tau"], plot_df["Coefficient"], marker='o', linestyle='-', color=color, linewidth=2, label=var)
        
        for idx, row in plot_df.iterrows():
            stars = row["Significance"]
            if pd.notna(stars) and stars != "":
                ax.annotate(stars, 
                             (row["Tau"], row["Coefficient"]), 
                             xytext=(5, 5), textcoords='offset points',
                             color=color, fontsize=14, fontweight='bold')
                
    ax.set_title(title)
    ax.set_xlabel("Quantile (Tau)")
    ax.set_ylabel("Coefficient")
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if len(vars_to_plot) > 1:
        ax.legend(loc="best")

def plot_pseudo_r2(ax, results_df: pd.DataFrame) -> None:
    """Sub-function: Extracts the Pseudo R-Squared for each quantile and plots them."""
    r2_df = results_df[["Tau", "Pseudo R-Squared"]].drop_duplicates().sort_values(by="Tau")
    tau_labels = [f"{t:.2f}" for t in r2_df["Tau"]]
    
    ax.bar(tau_labels, r2_df["Pseudo R-Squared"], color='tab:cyan', edgecolor='black', alpha=0.7)
    ax.set_title("Model Fit: Pseudo R-Squared by Quantile")
    ax.set_xlabel("Quantile (Tau)")
    ax.set_ylabel("Pseudo R-Squared")
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

def plot_residuals(ax, data: pd.DataFrame, vars_x: str, vars_y: str, controls: list = None, errors: str = "robust", tau: float = 0.5) -> None:
    """Sub-function: Runs one regression, plots residuals, and checks stationarity."""
    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = q_reg(df=data, x=vars_x, y=vars_y, tau=tau, controls=controls, vcov=errors)
    
    residuals = reg.resid
    
    # Calculate Pinball Loss instead of RMSE
    pinball_loss = np.mean(np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals))
    
    adf_result = adfuller(residuals.dropna())
    adf_pval = adf_result[1] 
    
    ax.plot(residuals.index, residuals, color='tab:gray', linewidth=1)
    ax.set_title(f"Regression Residuals ($\\tau = {tau}$) | ADF p-val = {adf_pval:.3f} | Pinball Loss = {pinball_loss:.4f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.axhline(0, color='tab:red', linestyle='--', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_quantile_results(data: pd.DataFrame, vars_x: str, vars_y: str, controls: list = None, errors: str = "robust", tau_resid: float = 0.5, quantiles: list = None) -> pd.DataFrame:
    """
    Main orchestrator function: Creates the 2x2 dashboard for Quantile Regression outputs.

    `quantiles` defaults to 0.01, then 0.05 to 0.95 (step 0.05), then 0.99 when not supplied.
    """
    if controls is None:
        controls = []
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]

    results_df = multiple_q_regs(
        data=data, vars_x=vars_x, vars_y=vars_y, quantiles=quantiles, controls=controls, errors=errors
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Quantile Regression Dashboard: {vars_y} ~ {vars_x}", fontsize=14, fontweight="bold")

    all_vars = vars_x if isinstance(vars_x, list) else [vars_x]
    plot_quantile_coefs(axes[0, 0], results_df, all_vars, title=f"Main Regressor(s): {vars_x}")

    if controls:
        plot_quantile_coefs(axes[0, 1], results_df, controls, title="Control Variables")
    else:
        axes[0, 1].set_visible(False)

    plot_pseudo_r2(axes[1, 0], results_df)
    plot_residuals(axes[1, 1], data=data, vars_x=vars_x if isinstance(vars_x, str) else vars_x[0],
                   vars_y=vars_y, controls=controls, errors=errors, tau=tau_resid)

    plt.tight_layout()
    plt.show()

    return results_df


# =============================================================================
# SECTION 3 - DIRECT FORECASTING HELPERS
# =============================================================================

def pinball_loss(tau, y_true, y_pred):
    """
    Calcula la pérdida asimétrica (Tick Loss) para un cuantil específico.
    """
    error = y_true - y_pred
    loss = np.where(error < 0, (1 - tau) * np.abs(error), tau * np.abs(error))
    return np.mean(loss)


# =============================================================================
# SECTION 4 - DIRECT FORECASTING ESTIMATORS
# =============================================================================

def direct_forecasting(df: pd.DataFrame, 
                       x: str, 
                       y: str, 
                       quantiles: list[float], 
                       h: int = 1, 
                       controls: list[str] = None, 
                       **kwargs) -> pd.DataFrame:
    """
    Performs h-step ahead direct forecasting using Quantile Regression.
    Returns ONLY the predictions DataFrame (no plots).
    """
    if controls is None:
        controls = []
        
    # 1. Crear el Target desplazado
    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].copy()
    
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    
    # 2. Aislar features de "Hoy" y el dataset de entrenamiento
    latest_features = df_work.iloc[[-1]].copy()
    df_train = df_work.dropna().copy()
    
    predictions = []
    
    # 3. Bucle de predicción usando tu función q_reg
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Utilizamos tu función importada. 
                # Nota: 'y' para q_reg debe ser el target_col desplazado
                reg = q_reg(df=df_train, x=x, y=target_col, tau=q, controls=controls, vcov="robust", max_iter=2000)
            except ValueError:
                # Fallback en caso de que la matriz robusta no sea invertible
                reg = q_reg(df=df_train, x=x, y=target_col, tau=q, controls=controls, vcov="iid", max_iter=2000)
                
            pred_value = reg.predict(exog=latest_features).values[0]
            
            predictions.append({
                "Quantile": q,
                "Forecast": pred_value
            })
            
    return pd.DataFrame(predictions)


def insample_direct_forecasting(df: pd.DataFrame, 
                                x: str, 
                                y: str, 
                                quantiles: list[float], 
                                train_end_date: str,
                                h: int = 1, 
                                controls: list[str] = None, 
                                **kwargs) -> pd.DataFrame:
    """
    Performs In-Sample h-step ahead direct forecasting.
    Trains the model up to 'train_end_date', predicts the target for h days forward 
    from that specific date, and compares quantiles vs OLS mean vs Realized Value.
    
    Parameters:
    - df: The master dataframe (must have a DatetimeIndex)
    - x: The main independent variable (e.g., 'GPRD_MA7')
    - y: The dependent variable to forecast (e.g., 'Brent_Return')
    - quantiles: List of quantiles to predict
    - train_end_date: The specific date (T) up to which the model is trained.
    - h: Forecast horizon (number of days forward)
    - controls: List of control variables
    """
    if controls is None:
        controls = []
        
    # 1. Asegurar que el índice es formato Datetime para evitar KeyErrors
    df_work = df.copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)
    
    target_date = pd.to_datetime(train_end_date)
        
    # 2. Mantener solo columnas necesarias y crear el Target desplazado
    cols_to_keep = [y, x] + controls
    df_work = df_work[cols_to_keep]
    
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    
    # 3. Aislar los features de "Hoy" (target_date) y el valor REALIZADO
    features_t = df_work.loc[[target_date]].copy()
    realized_value = features_t[target_col].values[0]
    
    # 4. Crear el dataset de entrenamiento (Toda la historia hasta target_date)
    # Eliminamos NaNs para que la regresión no falle
    df_train = df_work.loc[:target_date].dropna().copy()
    
    # 5. Formular la ecuación
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{target_col}') ~ Q('{x}')"
        
    # 6. Entrenar modelo OLS (Media Clásica)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols_mod = smf.ols(formula=equation, data=df_train).fit(cov_type="HC1")
        ols_pred = ols_mod.predict(exog=features_t).values[0]
        
    predictions = []
    
    # 7. Entrenar y Predecir Regresión Cuantílica (Bucle)
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=df_train, formula=equation)
            try:
                reg = mod.fit(q=q, vcov="robust", max_iter=2000)
            except ValueError:
                reg = mod.fit(q=q, vcov="iid", max_iter=2000)
                print(f"Warning: Fallback to iid covariance for q={q}")
                
            pred_value = reg.predict(exog=features_t).values[0]
            
            predictions.append({
                "Quantile": q,
                "Forecast": pred_value,
                "Realized": realized_value,
                "OLS_Mean": ols_pred
            })
            
    # Formatear el DataFrame de resultados
    results_df = pd.DataFrame(predictions)

    from tqdm import tqdm
# from auxi.qreg import q_reg
# from auxi.distfit import mde_distfit



    # ---------------------------------------------------------
    # 8. VISUALIZACIÓN IN-SAMPLE
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    quantile_labels = [f"{int(q * 100)}%" if q <= 1 else str(q) for q in quantiles]
    x_positions = np.arange(len(quantiles))
    
    # A) Bar chart: Quantile forecasts
    plt.bar(x_positions, results_df["Forecast"], color="steelblue", alpha=0.6, label="Quantile Forecast (GaR)")

    # B) Scatterplot: Realized value (red triangle)
    plt.scatter(x_positions, [realized_value]*len(quantiles),
                color="crimson", marker="^", s=100, zorder=5,
                label=f"Realized Value (h={h})")

    # C) Scatterplot: OLS prediction (orange circle)
    plt.scatter(x_positions, [ols_pred]*len(quantiles),
                color="darkorange", marker="o", s=80, zorder=5,
                label="Mean Prediction (OLS)")

    plt.title(f"In-Sample Direct Forecast: {train_end_date} (Horizon $h={h}$)\nTarget: {y}", fontsize=14, pad=15)
    plt.xticks(x_positions, quantile_labels)
    plt.xlabel("Quantiles ($\\tau$)", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")
    
    plt.tight_layout()
    plt.show()

    return results_df


def get_oos_predictions(df: pd.DataFrame,
                        x: str,
                        y: str,
                        tau: float,
                        h: int = 1,
                        controls: list[str] = None,
                        train_fraction: float = 0.8,
                        test_start_date: str = None) -> tuple[pd.Series, pd.Series]:
    """
    Trains the quantile regression model and generates Out-of-Sample predictions.

    The train / test split can be specified in two ways (mutually exclusive):
      - test_start_date (str, 'YYYY-MM-DD'): train on all data strictly before
        that date, test on everything from that date onward.  Use this after
        rolling-origin horizon selection to ensure the test window was never
        seen during h selection.
      - train_fraction (float, default 0.8): fallback fraction-based split when
        test_start_date is not provided.

    Returns a tuple of two pandas Series: (Realized_OOS, Forecasted_OOS)
    """
    if controls is None:
        controls = []

    cols_to_keep = [y, x] + controls
    df_work = df[cols_to_keep].copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)

    # Create the shifted target
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    df_work = df_work.dropna()

    # Split data chronologically
    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        df_train = df_work[df_work.index < test_start_dt].copy()
        df_test  = df_work[df_work.index >= test_start_dt].copy()
    else:
        split_idx = int(len(df_work) * train_fraction)
        df_train  = df_work.iloc[:split_idx].copy()
        df_test   = df_work.iloc[split_idx:].copy()

    # Formulate equation
    if controls:
        control_str = " + ".join([f"Q('{c}')" for c in controls])
        equation = f"Q('{target_col}') ~ Q('{x}') + {control_str}"
    else:
        equation = f"Q('{target_col}') ~ Q('{x}')"

    # Fit model on training data only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.quantreg(data=df_train, formula=equation)
        try:
            reg = mod.fit(q=tau, vcov="robust", max_iter=2000)
        except ValueError:
            reg = mod.fit(q=tau, vcov="iid", max_iter=2000)

    # Generate OOS predictions
    pred_test     = reg.predict(exog=df_test)
    realized_test = df_test[target_col]

    # Return both series, preserving the DatetimeIndex
    return realized_test, pred_test


# =============================================================================
# SECTION 5 - DIRECT FORECASTING PLOTTERS
# =============================================================================

def _plot_coef_panel(ax, results_df: pd.DataFrame, ols_params: dict, vars_to_plot: list, title: str):
    """
    Helper function to render a single panel in the Koenker-Bassett dashboard.
    Plots the QR coefficient path, adds significance stars, and draws the OLS baseline.
    """
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']
    
    for i, var in enumerate(vars_to_plot):
        color = colors[i % len(colors)]
        var_df = results_df[results_df["Regressor"] == var].sort_values("Tau")
        
        ax.plot(var_df["Tau"], var_df["Coefficient"], marker='o', linestyle='-', 
                color=color, linewidth=2, label=f"QR: {var}")
        
        for _, row in var_df.iterrows():
            stars = row["Significance"]
            if pd.notna(stars) and stars != "":
                ax.annotate(stars, 
                            (row["Tau"], row["Coefficient"]), 
                            xytext=(0, 6), textcoords='offset points',
                            color=color, fontsize=14, fontweight='bold', ha='center')
        
        ols_val = ols_params.get(f"Q('{var}')", 0)
        ax.axhline(y=ols_val, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"OLS: {var}")
        
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(title, pad=10, fontsize=12)
    ax.set_xlabel(r"Quantile ($\tau$)")
    ax.set_ylabel(r"Coefficient ($\beta_{\tau}$)")
    ax.grid(True, linestyle="--", alpha=0.5)


def _add_regression_lines(ax, df_panel: pd.DataFrame, x_col: str, y_col: str, 
                          quantiles: list[float], controls: list[str], scatter_color: str):
    """
    Helper function for the scatter plot dashboard.
    Draws the raw scatter points, an OLS line, and multiple Quantile Regression lines.
    Evaluates controls at their sample mean to plot correctly in 2D space.
    """
    # 1. Plot raw scatter points
    ax.scatter(df_panel[x_col], df_panel[y_col], color=scatter_color, alpha=0.4, s=20)
    
    # 2. X-axis range for drawing continuous lines
    x_min, x_max = df_panel[x_col].min(), df_panel[x_col].max()
    x_vals = np.linspace(x_min, x_max, 100)
    
    # 3. Create Prediction DataFrame (Holding controls strictly at their mean)
    pred_df = pd.DataFrame({x_col: x_vals})
    if controls:
        for c in controls:
            pred_df[c] = df_panel[c].mean()
            
    # 4. Build Formula
    control_str = (" + " + " + ".join([f"Q('{c}')" for c in controls])) if controls else ""
    formula = f"Q('{y_col}') ~ Q('{x_col}')" + control_str
    
    # 5. OLS Baseline Line
    try:
        mod_ols = smf.ols(formula=formula, data=df_panel).fit()
        y_ols = mod_ols.predict(pred_df)
        ax.plot(x_vals, y_ols, color="black", linestyle="--", linewidth=2.5, label="OLS Mean")
    except Exception as e:
        print(f"OLS plot failed for {y_col} ~ {x_col}: {e}")
        
    # 6. Quantile Lines (using a coolwarm colormap to differentiate tails)
    cmap = plt.get_cmap("coolwarm")
    line_colors = [cmap(i) for i in np.linspace(0, 1, len(quantiles))]
    
    for q, color in zip(quantiles, line_colors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod_q = smf.quantreg(formula=formula, data=df_panel)
            try:
                res_q = mod_q.fit(q=q, max_iter=2000)
            except ValueError:
                res_q = mod_q.fit(q=q, max_iter=2000, vcov="iid")
            
            y_q = res_q.predict(pred_df)
            ax.plot(x_vals, y_q, color=color, linewidth=2, label=f"$\\tau = {q}$")


def plot_forecasted_scatters(df: pd.DataFrame, 
                             x: str, 
                             y: str, 
                             quantiles: list[float], 
                             h_short: int = 3, 
                             h_long: int = 12, 
                             controls: list[str] = None) -> None:
    """
    Renders a 1x2 scatter plot dashboard with OLS and Quantile regression lines.

    Panel A: Short-term horizon (h_short) — Main Regressor vs Future Y
    Panel B: Long-term horizon (h_long)  — Main Regressor vs Future Y
    """
    if controls is None:
        controls = []

    # --- 1. Data Preparation ---
    cols_to_keep = [x, y] + controls
    df_work = df[cols_to_keep].copy()

    y_target_short = f"{y}_target_h{h_short}"
    y_target_long = f"{y}_target_h{h_long}"

    df_work[y_target_short] = df_work[y].shift(-h_short)
    df_work[y_target_long] = df_work[y].shift(-h_long)

    df_short = df_work[[x, y, y_target_short] + controls].dropna()
    df_long = df_work[[x, y, y_target_long] + controls].dropna()

    # --- 2. Figure Setup ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(f"Predictive Power of Full Specification over Future {y}",
                 fontsize=16, fontweight='bold', y=0.98)

    # --- PANEL A: Short-Term Risk ---
    _add_regression_lines(axes[0], df_short, x, y_target_short, quantiles, controls, scatter_color="steelblue")
    axes[0].set_title(f"Panel A: Short-Term Risk\nFuture {y} (t+{h_short}) vs. Current {x} (t)")
    axes[0].set_xlabel(f"{x} (Current)")
    axes[0].set_ylabel(f"{y} (t+{h_short})")

    # --- PANEL B: Long-Term Risk ---
    _add_regression_lines(axes[1], df_long, x, y_target_long, quantiles, controls, scatter_color="darkred")
    axes[1].set_title(f"Panel B: Long-Term Risk\nFuture {y} (t+{h_long}) vs. Current {x} (t)")
    axes[1].set_xlabel(f"{x} (Current)")
    axes[1].set_ylabel(f"{y} (t+{h_long})")

    # --- 3. Aesthetics & Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    num_cols = min(6, len(quantiles) + 1)
    fig.legend(handles, labels, loc="upper center", ncol=num_cols, bbox_to_anchor=(0.5, 0.04))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.12)
    plt.show()


def plot_contemporaneous_vs_predictive_coefs(df: pd.DataFrame, 
                                             x: str, 
                                             y: str, 
                                             h: int = 1, 
                                             controls: list[str] = None,
                                             errors: str = "robust") -> None:
    """
    Renders a 2x2 dashboard comparing Koenker-Bassett Quantile Regression coefficients.
    
    Row 1: Contemporaneous Impact (Time t)
    Row 2: Predictive Impact (Time t+h)
    Column 1: Main Regressor
    Column 2: Control Variables
    """
    if controls is None:
        controls = []
        
    cols_to_keep = [x, y] + controls
    df_work = df[cols_to_keep].copy()
    
    y_target = f"{y}_target_h{h}"
    df_work[y_target] = df_work[y].shift(-h)
    
    df_c = df_work[[x, y] + controls].dropna()
    df_p = df_work[[x, y_target] + controls].dropna()
    
    res_c = multiple_q_regs(data=df_c, vars_x=x, vars_y=y, controls=controls, errors=errors)
    res_p = multiple_q_regs(data=df_p, vars_x=x, vars_y=y_target, controls=controls, errors=errors)
    
    control_str = (" + " + " + ".join([f"Q('{c}')" for c in controls])) if controls else ""
    eq_c = f"Q('{y}') ~ Q('{x}')" + control_str
    eq_p = f"Q('{y_target}') ~ Q('{x}')" + control_str
    
    ols_c = smf.ols(formula=eq_c, data=df_c).fit(cov_type="HC3")
    ols_p = smf.ols(formula=eq_p, data=df_p).fit(cov_type="HC3")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    display_controls = " + ".join(controls) if controls else "None"
    fig.suptitle(f"Quantile Regression Coefficients: {x} & Controls over Future {y}\n"
                 f"Contemporaneous:  $Y(t) = \\alpha + \\beta X(t) + {display_controls}$\n"
                 f"Predictive:       $Y(t+{h}) = \\alpha + \\beta X(t) + {display_controls}$", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    _plot_coef_panel(axes[0, 0], res_c, ols_c.params, [x], 
                     title=f"Panel A: Contemporaneous Risk\nCurrent {y} vs. Current {x}")
    
    if controls:
        _plot_coef_panel(axes[0, 1], res_c, ols_c.params, controls, 
                         title=f"Panel B: Contemporaneous Controls\nCurrent {y} vs. Controls")
    else:
        axes[0, 1].axis('off')

    _plot_coef_panel(axes[1, 0], res_p, ols_p.params, [x], 
                     title=f"Panel C: Predictive Risk (h={h})\nFuture {y} (t+{h}) vs. Current {x}")
    
    if controls:
        _plot_coef_panel(axes[1, 1], res_p, ols_p.params, controls, 
                         title=f"Panel D: Predictive Controls (h={h})\nFuture {y} (t+{h}) vs. Controls")
    else:
        axes[1, 1].axis('off')
        
    handles_main, labels_main = axes[0, 0].get_legend_handles_labels()
    handles_ctrl, labels_ctrl = axes[0, 1].get_legend_handles_labels() if controls else ([], [])
    
    fig.legend(handles_main + handles_ctrl, labels_main + labels_ctrl, 
               loc="upper center", ncol=min(4, len(labels_main + labels_ctrl)), bbox_to_anchor=(0.5, 0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.3)
    plt.show()
