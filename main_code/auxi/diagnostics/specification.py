"""Quantile-regression model specification diagnostics.

Moved from auxi/diagnostics.py during the 2026-06-26 backend reorg.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

from auxi.qreg import q_reg


def dq_test(df: pd.DataFrame, x: str, y: str, tau: float, controls: list[str] = None, 
            extra_instruments: list[str] = None, n_hit_lags: int = 4, 
            include_constant: bool = True, include_fitted_vals: bool = True, **kwargs) -> dict:
    """Performs the Dynamic Quantile (DQ) Test to verify hit sequence unpredictability."""
    cols_to_check = [y, x]
    if controls: cols_to_check.extend(controls)
    if extra_instruments: cols_to_check.extend(extra_instruments)
        
    df_clean = df.dropna(subset=list(set(cols_to_check))).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = q_reg(df=df_clean, x=x, y=y, tau=tau, controls=controls, **kwargs)
    
    y_real = reg.model.endog
    y_pred = reg.fittedvalues
    hits = np.where(y_real < y_pred, 1 - tau, -tau)
    
    instruments_dict = {}
    if include_constant: instruments_dict['Constant'] = np.ones(len(hits))
    if include_fitted_vals: instruments_dict['Y_pred'] = y_pred
    
    for i in range(1, n_hit_lags + 1):
        instruments_dict[f'Hit_lag_{i}'] = np.roll(hits, i)
        
    if extra_instruments:
        for inst in extra_instruments:
            if inst in df_clean.columns:
                instruments_dict[inst] = df_clean[inst].values
            else:
                print(f"Warning: Instrument '{inst}' not found.")
                
    if not instruments_dict: raise ValueError("Instrument matrix is empty.")
                
    X_instruments = pd.DataFrame(instruments_dict)
    hits_clean = hits[n_hit_lags:]
    X_clean = X_instruments.iloc[n_hit_lags:]
    X_matrix = X_clean.values
    
    hits_vector = hits_clean.reshape(-1, 1)
    x_t_hit = X_matrix.T @ hits_vector
    
    try:
        x_t_x_inv = np.linalg.inv(X_matrix.T @ X_matrix)
    except np.linalg.LinAlgError:
        x_t_x_inv = np.linalg.pinv(X_matrix.T @ X_matrix)
        
    dq_stat = float((x_t_hit.T @ x_t_x_inv @ x_t_hit) / (tau * (1 - tau)))
    p_val = 1 - stats.chi2.cdf(dq_stat, X_matrix.shape[1])
    
    ols_model = sm.OLS(hits_clean, X_clean).fit()
    corr_matrix = X_clean.corr()
    
    try:
        adf_res = adfuller(hits_clean)
        adf_pval = adf_res[1]
    except Exception:
        adf_pval = np.nan
        
    return {
        "Tau": tau, "DQ_Statistic": round(dq_stat, 4), "P_Value": round(p_val, 4),
        "Conclusion": "Pass (White Noise)" if p_val > 0.05 else "Fail (Autocorr)",
        "Hits_Series": pd.Series(hits_clean, index=df_clean.index[n_hit_lags:]),
        "OLS_Summary": ols_model.summary(), "Corr_Matrix": corr_matrix, "ADF_Pval": adf_pval
    }

def plot_advanced_dq_diagnostics(res_dict: dict):
    """Renders an aesthetically perfected dashboard for the DQ Test."""
    hits_series = res_dict["Hits_Series"]
    tau = res_dict["Tau"]
    p_val = res_dict["P_Value"]
    
    print("=" * 80)
    print("OLS AUXILIARY REGRESSION (Instrument Significance Checking):")
    print(res_dict["OLS_Summary"].tables[1]) 
    print("=" * 80)
    
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.8, 1.2], hspace=0.35, wspace=0.2)
    
    status_color = '#2ca02c' if p_val > 0.01 else '#d62728' 
    conclusion_text = "PASS (White Noise)" if p_val > 0.01 else "FAIL (Autocorrelation)"
    
    fig.suptitle(f"Dynamic Quantile (DQ) Test Dashboard | $\\tau = {tau}$\n"
                 f"DQ Stat: {res_dict['DQ_Statistic']:.3f} | p-value: {p_val:.4f}  $\\rightarrow$  {conclusion_text}", 
                 fontsize=18, fontweight='bold', color=status_color, y=0.96)
    
    ax_hits = fig.add_subplot(gs[0, :])
    ax_hits.plot(hits_series.index, hits_series, color=status_color, linewidth=1.2, alpha=0.8)
    ax_hits.scatter(hits_series.index, hits_series, color=status_color, s=10, alpha=0.5) 
    ax_hits.set_title(f"Hits Sequence Over Time (ADF Stationarity p-val: {res_dict['ADF_Pval']:.3f})", fontsize=14, pad=10)
    ax_hits.axhline(0, color='black', linestyle='-', linewidth=1.2)
    ax_hits.grid(True, linestyle='--', alpha=0.5)
    
    ax_acf = fig.add_subplot(gs[1, 0])
    plot_acf(hits_series, ax=ax_acf, title="ACF of Hits", color=status_color, vlines_kwargs={"colors": status_color})
    ax_acf.grid(True, linestyle='--', alpha=0.5)
    
    ax_pacf = fig.add_subplot(gs[1, 1])
    plot_pacf(hits_series, ax=ax_pacf, title="PACF of Hits", method='ywm', color=status_color, vlines_kwargs={"colors": status_color})
    ax_pacf.grid(True, linestyle='--', alpha=0.5)
    
    ax_heat = fig.add_subplot(gs[2, :])
    mask = np.triu(np.ones_like(res_dict["Corr_Matrix"], dtype=bool), k=1) 
    cmap = sns.diverging_palette(230, 20, as_cmap=True) 
    sns.heatmap(res_dict["Corr_Matrix"], mask=mask, annot=True, cmap=cmap, fmt=".2f", 
                ax=ax_heat, vmin=-1, vmax=1, center=0, linewidths=1, linecolor='white')
    ax_heat.set_title("Instrument Multicollinearity Check", fontsize=14, pad=15)
    
    fig.patch.set_facecolor('#f8f9fa')
    for ax in [ax_hits, ax_acf, ax_pacf, ax_heat]: ax.set_facecolor('#ffffff')
    plt.show()

def wald_test(df: pd.DataFrame, x: str, y: str, tau1: float, tau2: float, 
              controls: list = None, n_bootstraps: int = 200, block_size: int = 10, **kwargs) -> dict:
    """Performs the Quantile Wald Test (Koenker-Bassett) using Moving Block Bootstrap."""
    print(f"Starting Wald Test (tau1={tau1} vs tau2={tau2})...")
    
    cols_to_check = [y, x]
    if controls: cols_to_check.extend(controls)
    df_clean = df.dropna(subset=list(set(cols_to_check))).reset_index(drop=True).copy()
    
    param_name = f"Q('{x}')"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg1_orig = q_reg(df=df_clean, x=x, y=y, tau=tau1, controls=controls, **kwargs)
        reg2_orig = q_reg(df=df_clean, x=x, y=y, tau=tau2, controls=controls, **kwargs)
        
    actual_diff = reg1_orig.params[param_name] - reg2_orig.params[param_name]
    
    if controls:
        equation = f"Q('{y}') ~ Q('{x}') + " + " + ".join([f"Q('{c}')" for c in controls])
    else:
        equation = f"Q('{y}') ~ Q('{x}')"
        
    n_obs = len(df_clean)
    boot_diffs = []
    
    for _ in range(n_bootstraps):
        indices = []
        while len(indices) < n_obs:
            start_idx = np.random.randint(0, n_obs - block_size + 1)
            indices.extend(range(start_idx, start_idx + block_size))
        
        df_boot = df_clean.iloc[indices[:n_obs]].copy()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod_boot = smf.quantreg(data=df_boot, formula=equation)
                b1 = mod_boot.fit(q=tau1, vcov="iid", max_iter=2000).params[param_name]
                b2 = mod_boot.fit(q=tau2, vcov="iid", max_iter=2000).params[param_name]
                boot_diffs.append(b1 - b2)
        except Exception: continue 
            
    var_diff = np.var(boot_diffs, ddof=1)
    wald_stat = (actual_diff ** 2) / var_diff
    p_val = 1 - stats.chi2.cdf(wald_stat, df=1)
    
    return {
        "Tau_1": tau1, "Tau_2": tau2, "Beta_1": reg1_orig.params[param_name], "Beta_2": reg2_orig.params[param_name],
        "Actual_Difference": actual_diff, "Bootstrapped_Variance": var_diff, "Wald_Statistic": wald_stat,
        "P_Value": p_val, "Conclusion": "Reject H0" if p_val < 0.05 else "Fail to Reject",
        "Boot_Distribution": boot_diffs
    }

def plot_wald_diagnostics(wald_res: dict):
    """Renders a visual report of the Wald Test Bootstrap Distribution."""
    plt.figure(figsize=(10, 6))
    status_color = '#d62728' if wald_res["P_Value"] < 0.05 else '#2ca02c' 
    
    sns.histplot(wald_res["Boot_Distribution"], kde=True, color='tab:blue', alpha=0.4, stat="density", linewidth=0)
    plt.axvline(0, color='black', linestyle='-', linewidth=2, label='Null Hypothesis (Diff = 0)')
    plt.axvline(wald_res["Actual_Difference"], color=status_color, linestyle='--', linewidth=2.5, 
                label=f'Observed Diff ({wald_res["Actual_Difference"]:.3f})')
    
    plt.title(f"Bootstrap Distribution of the Difference in Slopes\np-value: {wald_res['P_Value']:.4f}", fontsize=14)
    plt.xlabel(f"Difference: $\\beta_{{{wald_res['Tau_1']}}} - \\beta_{{{wald_res['Tau_2']}}}$")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def q_arch_test(df: pd.DataFrame, x: str, y: str, tau: float, controls: list = None, p_lags: int = 5, **kwargs) -> dict:
    """Performs the Quantile ARCH (Q-ARCH) LM Test on the residuals."""
    cols_to_check = [y, x]
    if controls: cols_to_check.extend(controls)
    df_clean = df.dropna(subset=list(set(cols_to_check))).copy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = q_reg(df=df_clean, x=x, y=y, tau=tau, controls=controls, **kwargs)
        
    sq_resid = reg.resid.values ** 2
    arch_dict = {'Sq_Resid_t': sq_resid}
    for i in range(1, p_lags + 1): arch_dict[f'Sq_Resid_t_minus_{i}'] = np.roll(sq_resid, i)
        
    df_arch = pd.DataFrame(arch_dict).iloc[p_lags:].copy()
    
    Y_arch = df_arch['Sq_Resid_t']
    X_arch = sm.add_constant(df_arch.drop(columns=['Sq_Resid_t']))
    ols_arch = sm.OLS(Y_arch, X_arch).fit()
    
    lm_stat = len(df_arch) * ols_arch.rsquared
    p_val = 1 - stats.chi2.cdf(lm_stat, df=p_lags)
    
    return {
        "Tau": tau, "Lags": p_lags, "LM_Statistic": lm_stat, "P_Value": p_val,
        "Conclusion": "Pass (No ARCH)" if p_val >= 0.05 else "Fail (Volatility Clustering)",
        "OLS_Summary": ols_arch.summary(),
        "Sq_Resid_Series": pd.Series(df_arch['Sq_Resid_t'].values, index=df_clean.index[p_lags:])
    }

def plot_q_arch_diagnostics(arch_res: dict):
    """Visual Dashboard for the Quantile ARCH test."""
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)
    status_color = '#2ca02c' if arch_res["P_Value"] >= 0.05 else '#d62728' 
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(arch_res["Sq_Resid_Series"].index, arch_res["Sq_Resid_Series"], color=status_color, linewidth=1, alpha=0.8)
    ax1.set_title(f"Squared Quantile Residuals Over Time ($\hat{{\epsilon}}^2_{{{arch_res['Tau']}}}$)", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_acf(arch_res["Sq_Resid_Series"], ax=ax2, lags=20, color=status_color, vlines_kwargs={"colors": status_color})
    ax2.set_title(f"ACF of Squared Residuals", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(f"Q-ARCH Diagnostic Dashboard | $\\tau = {arch_res['Tau']}$ | p-val: {arch_res['P_Value']:.4f}", 
                 fontsize=15, fontweight='bold', color=status_color)
    plt.show()

def qarx_stability_test(df: pd.DataFrame, y_col: str, x_cols: list, tau: float = 0.50,
                        lags: int = 1, n_boot: int = 199, block_size: int = None,
                        random_state: int = None) -> dict:
    """
    Koenker-Xiao (2004) QAR unit-root (local stability) test with a BOOTSTRAP
    critical value.

    Error-correction QAR(p) model estimated at quantile ``tau``:

        d y_t = mu(tau) + rho(tau) y_{t-1} + sum_j gamma_j(tau) d y_{t-j}
                + beta(tau)' X_t + eps_t

    H0: rho(tau) = 0  (unit root at tau)   vs   H1: rho(tau) < 0 (mean-reverting).

    Why not a Dickey-Fuller critical value
    --------------------------------------
    The t-ratio t(tau) = rho_hat / se(rho_hat) does NOT follow the standard
    Dickey-Fuller distribution; Koenker-Xiao show its limit is a tau- and
    nuisance-dependent convex combination of the DF and the standard-normal laws.
    A fixed DF cutoff (e.g. -2.86) is therefore invalid. Here the null
    distribution of t(tau) is obtained by a recursive residual bootstrap that
    imposes the unit-root null (rho = 0): fit the restricted model
    d y ~ const + d y-lags + X, regenerate d y* recursively from those
    coefficients with moving-block resampled residuals, rebuild y* by cumulating
    d y*, hold X fixed, refit the unrestricted model, and collect t*(tau).
    Returns a left-tail bootstrap p-value and the 5% bootstrap critical value.
    """
    rng = np.random.default_rng(random_state)
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    x_cols = list(x_cols)

    d = df.copy()
    d['dy'] = d[y_col].diff()
    d['y_lag1'] = d[y_col].shift(1)
    dy_lag_cols = []
    for i in range(1, lags + 1):
        c = f'dy_lag{i}'
        d[c] = d['dy'].shift(i)
        dy_lag_cols.append(c)

    use_cols = ['dy', 'y_lag1'] + dy_lag_cols + x_cols
    d = d.dropna(subset=use_cols).copy()
    n = len(d)
    if n < 30:
        raise ValueError(f"qarx_stability_test: only {n} usable rows after differencing/lagging.")

    yv = d['dy'].to_numpy(dtype=float)
    indep = ['y_lag1'] + dy_lag_cols + x_cols
    X_unr = sm.add_constant(d[indep].to_numpy(dtype=float), has_constant='add')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = QuantReg(yv, X_unr).fit(q=tau)
    rho_idx = 1  # col 0 = const, col 1 = y_lag1
    rho_hat = float(res.params[rho_idx])
    se_rho = float(res.bse[rho_idx])
    t_obs = rho_hat / se_rho if se_rho > 0 else np.nan

    # Restricted (null, rho=0) fit:  dy ~ const + dy-lags + X
    rest_cols = dy_lag_cols + x_cols
    if rest_cols:
        X_res = sm.add_constant(d[rest_cols].to_numpy(dtype=float), has_constant='add')
    else:
        X_res = np.ones((n, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_r = QuantReg(yv, X_res).fit(q=tau)
    params_r = np.asarray(res_r.params, dtype=float)
    resid_r = yv - res_r.predict(X_res)

    mu = params_r[0]
    gammas = params_r[1:1 + lags]
    betas = params_r[1 + lags:]
    Xexog = d[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((n, 0))
    y0 = float(d['y_lag1'].iloc[0])

    if block_size is None:
        block_size = max(1, int(round(n ** (1 / 3))))

    t_boot = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, n - block_size + 1))
            idx.extend(range(s, s + block_size))
        e_star = resid_r[np.array(idx[:n])]

        dy_star = np.empty(n)
        if lags > 0:
            dy_star[:lags] = yv[:lags]
        for t in range(lags, n):
            v = mu + e_star[t]
            for j in range(lags):
                v += gammas[j] * dy_star[t - 1 - j]
            if betas.size:
                v += Xexog[t].dot(betas)
            dy_star[t] = v

        y_star = y0 + np.cumsum(dy_star)
        y_lag1_star = np.concatenate(([y0], y_star[:-1]))

        cols = [y_lag1_star]
        for i in range(1, lags + 1):
            cols.append(np.concatenate((np.full(i, np.nan), dy_star[:-i])))
        for k in range(Xexog.shape[1]):
            cols.append(Xexog[:, k])
        Xb = sm.add_constant(np.column_stack(cols), has_constant='add')
        m = ~np.isnan(Xb).any(axis=1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rb = QuantReg(dy_star[m], Xb[m]).fit(q=tau)
            se_b = float(rb.bse[1])
            if se_b > 0:
                t_boot.append(float(rb.params[1] / se_b))
        except Exception:
            continue

    t_boot = np.asarray(t_boot, dtype=float)
    cv05 = float(np.nanpercentile(t_boot, 5)) if t_boot.size else np.nan
    pval = float(np.mean(t_boot <= t_obs)) if (t_boot.size and np.isfinite(t_obs)) else np.nan
    is_stable = bool(pval < 0.05) if not np.isnan(pval) else None  # reject unit root -> stable

    return {"Tau": tau, "Rho": rho_hat, "t_stat": t_obs,
            "Boot_pvalue": pval, "Boot_CV_5pct": cv05, "N_boot": int(t_boot.size),
            "is_stable": is_stable, "model_summary": res.summary()}
