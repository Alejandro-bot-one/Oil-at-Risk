"""Goodness-of-fit diagnostics for distribution fits (JSU, Skew-t).

These diagnostics belong with the other diagnostics (one place per type),
but they delegate to the fitters that still live in auxi.distribution_analysis.

Moved from auxi/distribution_analysis.py during the 2026-06-26 backend reorg.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu, kstest

# Fitters and PDFs stay in auxi.distribution_analysis - we import them here.
from auxi.distribution_analysis import (
    fit_jsu,
    fit_skewt,
    jsu_pdf,
    jsu_cdf,
    jsu_sample,
    jsu_summary,
    jsu_plot,
    skewt_pdf,
    skewt_cdf,
    skewt_summary,
    scipy_skewt,
)


def jsu_ks_test(returns, gamma, delta, loc, scale, M=1000, alpha=0.10,
                random_state=None):
    """
    Bias-corrected Kolmogorov-Smirnov goodness-of-fit test for the JSU fit.

    The standard KS p-value formula assumes parameters were chosen *before*
    seeing the data.  Because we estimated them by MLE on the same sample the
    p-value is anti-conservative (too small).  We correct this via a
    parametric bootstrap / Monte Carlo procedure:

        1. Compute the real KS statistic D_real on the observed data.
        2. For each of M iterations:
              a. Generate fake data of the same length from JSU(γ,δ,μ,σ).
              b. Re-fit JSU to the fake data via MLE → new params.
              c. Compute D_fake between fake ECDF and re-fitted CDF.
        3. True p-value = #{D_fake ≥ D_real} / M

    Parameters
    ----------
    returns              : sorted 1-D numpy array – observed returns
    gamma, delta,
    loc, scale           : float – MLE parameters already fitted to `returns`
    M                    : int   – number of Monte Carlo replications
                                   (1 000 adequate, 10 000 for publication)
    alpha                : float – significance level (default 0.10)
    random_state         : int or None – seed for reproducibility

    Returns
    -------
    dict with keys:
        ks_statistic  : float – D_real
        p_value       : float – Monte Carlo p-value
        reject_null   : bool  – True when p_value < alpha
        simulated_D   : array – the M simulated D_fake values
    """
    if random_state is not None:
        np.random.seed(random_state)

    returns = np.sort(np.asarray(returns, dtype=float))
    n   = len(returns)
    edf = np.arange(1, n + 1) / n

    # ── Step 1: real KS statistic ─────────────────────────────────────────
    cdf_real  = jsu_cdf(returns, gamma, delta, loc, scale)
    d_real    = np.max(np.abs(edf - cdf_real))

    # ── Steps 2–3: parametric bootstrap ───────────────────────────────────
    simulated_D = np.empty(M)

    for i in range(M):
        # a. Generate fake data under H0
        fake_data = jsu_sample(gamma, delta, loc, scale, n)  # already sorted

        # b. Re-estimate all four parameters on the fake data
        g_f, d_f, l_f, s_f = fit_jsu(fake_data)

        # c. KS distance for this fake world
        fake_cdf    = jsu_cdf(fake_data, g_f, d_f, l_f, s_f)
        simulated_D[i] = np.max(np.abs(edf - fake_cdf))

    # ── Step 4: Monte Carlo p-value ───────────────────────────────────────
    p_value = np.sum(simulated_D >= d_real) / M

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("KOLMOGOROV-SMIRNOV TEST  (Monte Carlo corrected)")
    print("=" * 50)
    print(f"  KS Statistic  D_real    : {d_real:.6f}")
    print(f"  Replications  M         : {M:,}")
    print(f"  Significance  α         : {alpha:.2f}")
    print(f"  Monte Carlo p-value     : {p_value:.4f}")
    print("-" * 50)
    if p_value < alpha:
        print("  CONCLUSION: REJECT H₀ — JSU does NOT fit the data.")
    else:
        print("  CONCLUSION: FAIL TO REJECT H₀ — JSU fits the data.")
    print("=" * 50)

    return {
        "ks_statistic": d_real,
        "p_value":      p_value,
        "reject_null":  p_value < alpha,
        "simulated_D":  simulated_D,
    }

def fit_and_diagnose_jsu(returns, ticker="Asset", M=1000,
                          alpha=0.10, random_state=None):
    """
    End-to-end convenience function that runs the full JSU pipeline:

        1. Fit JSU via MLE
        2. Print parameter summary
        3. Produce diagnostic plots
        4. Run the Monte Carlo KS test

    Parameters
    ----------
    returns      : 1-D array-like – raw (unsorted) asset returns
    ticker       : str            – name used in titles / labels
    M            : int            – Monte Carlo replications for KS test
    alpha        : float          – significance level
    random_state : int or None    – RNG seed

    Returns
    -------
    params : dict  – {'gamma', 'delta', 'loc', 'scale'}
    ks     : dict  – output of jsu_ks_test (see its docstring)
    """
    returns = np.sort(np.asarray(returns, dtype=float))

    # 1. Fit
    gamma, delta, loc, scale = fit_jsu(returns)

    # 2. Summary
    jsu_summary(gamma, delta, loc, scale)

    # 3. Plots
    jsu_plot(returns, gamma, delta, loc, scale, ticker=ticker)

    # 4. KS test
    ks = jsu_ks_test(returns, gamma, delta, loc, scale,
                     M=M, alpha=alpha, random_state=random_state)

    params = {"gamma": gamma, "delta": delta, "loc": loc, "scale": scale}
    return params, ks

def evaluate_oos_pit(realized_returns, forecasted_params, bins=10):
    """
    Realiza el test de la Transformada Integral de Probabilidad (PIT) para 
    evaluar la calibración de las predicciones de densidad Out-of-Sample.
    
    Parameters:
    -----------
    realized_returns : array-like
        La serie de rendimientos reales que ocurrieron en t+h.
    forecasted_params : list of lists o np.array
        Los parámetros [a, b, loc, scale] de la distribución Johnson SU 
        pronosticados para cada día (alineados con realized_returns).
    bins : int
        Número de barras para el histograma (se recomienda entre 10 y 20).
        
    Returns:
    --------
    u_t : np.array
        La serie temporal de los valores PIT calculados.
    ks_stat : float
        El estadístico D del test de Kolmogorov-Smirnov.
    ks_pval : float
        El p-value del test KS.
    """
    
    # 1. Asegurar dimensiones
    if len(realized_returns) != len(forecasted_params):
        raise ValueError("La longitud de los rendimientos reales y los parámetros pronosticados debe ser idéntica.")
        
    # 2. Calcular u_t = F(y_{t+h} | Theta_t)
    u_t = []
    for y, params in zip(realized_returns, forecasted_params):
        a, b, loc, scale = params
        
        # Si por algún motivo el optimizador falló en ese día, saltamos la iteración
        if np.isnan(scale) or scale <= 0:
            u_t.append(np.nan)
            continue
            
        # Evaluar en qué percentil de nuestra distribución cayó el rendimiento real
        u = johnsonsu.cdf(y, a, b, loc=loc, scale=scale)
        u_t.append(u)
        
    u_t = np.array(u_t)
    # Limpiar NaNs (si los hubiera) para el test estadístico
    u_t_clean = u_t[~np.isnan(u_t)]
    
    # 3. Test de Kolmogorov-Smirnov (KS)
    # H0: u_t sigue una distribución Uniforme(0, 1)
    ks_stat, ks_pval = kstest(u_t_clean, 'uniform')
    
    # 4. Visualización del Histograma PIT
    plt.figure(figsize=(9, 6))
    
    # density=True asegura que el área total del histograma sume 1
    # Esto permite compararlo con la línea horizontal y=1 de la Uniforme Teórica
    counts, edges, patches = plt.hist(u_t_clean, bins=bins, density=True, 
                                      alpha=0.6, color='steelblue', edgecolor='black')
    
    # Línea ideal: Distribución Uniforme (PDF de U(0,1) es una línea plana en y=1)
    plt.axhline(1.0, color='crimson', linestyle='--', linewidth=2.5,
                label='Perfect Calibration $U(0,1)$')

    title = (f"Out-of-Sample PIT Diagnostic\n"
             f"Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Quantile / Probability ($u_t$ Values)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, max(1.5, max(counts) * 1.2)) # Dar espacio visual por arriba
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    return u_t_clean, ks_stat, ks_pval

def oos_pit_calibration(u_t, grid=None):
    """
    Out-of-Sample calibration curve (PP-plot of the PIT).

    Given the Probability Integral Transform series ``u_t`` produced by
    ``evaluate_oos_pit`` / ``evaluate_oos_pit_skewt``, compute the empirical
    CDF of the PIT on a fine probability grid. This is the cumulative
    counterpart of the PIT histogram: under perfect calibration the PIT is
    ``U(0, 1)`` and its empirical CDF lies on the 45-degree diagonal.

    For each nominal probability ``tau`` (the "predicted CDF", X-axis), the
    empirical coverage (the "empirical CDF", Y-axis) is the fraction of
    realized returns that fell at or below the model's ``tau``-quantile::

        F_emp(tau) = mean( u_t <= tau ) .

    Family-agnostic: ``u_t`` carries the same meaning for the Johnson SU and
    the Skewed-t fits, so a single function serves both.

    Parameters
    ----------
    u_t : array-like
        PIT values in [0, 1]. NaNs are dropped automatically.
    grid : array-like, optional
        Probability levels for the X-axis. Defaults to
        ``np.arange(0.01, 1.00, 0.01)`` (0.01 -> 0.99 in 0.01 steps).

    Returns
    -------
    grid : np.ndarray
        The nominal probability grid (X-axis).
    emp_cdf : np.ndarray
        Empirical CDF of the PIT evaluated on ``grid`` (Y-axis).
    cal_mae : float
        Mean absolute calibration error, ``mean(|emp_cdf - grid|)``.
    cal_sup : float
        Maximum absolute deviation from the diagonal, ``max(|emp_cdf - grid|)``
        (a Kolmogorov-style discrepancy).
    """
    if grid is None:
        grid = np.arange(0.01, 1.00, 0.01)
    grid = np.asarray(grid, dtype=float)

    u = np.asarray(u_t, dtype=float)
    u = u[~np.isnan(u)]
    if u.size == 0:
        raise ValueError("u_t contains no valid (non-NaN) PIT values.")

    # Empirical CDF of the PIT at each grid point: F_emp(tau) = mean(u <= tau).
    emp_cdf = np.array([np.mean(u <= tau) for tau in grid])

    deviation = np.abs(emp_cdf - grid)
    cal_mae = float(deviation.mean())
    cal_sup = float(deviation.max())

    return grid, emp_cdf, cal_mae, cal_sup

def plot_oos_pit_calibration(u_t, grid=None, title="OOS PIT Calibration (PP-plot)",
                             curve_label="Predicted distribution"):
    """
    Plot the Out-of-Sample PIT calibration curve (PP-plot).

    Draws the blue 45-degree diagonal (perfect calibration) and the red
    empirical-CDF-of-PIT curve. Deviations of the red line *above* the
    diagonal mean the model's quantiles are too high (realizations land below
    them too often, i.e. the predicted distribution is shifted right / tails
    too wide on that side); deviations *below* mean the opposite.

    Parameters
    ----------
    u_t : array-like
        PIT values from ``evaluate_oos_pit`` (NaNs dropped automatically).
    grid : array-like, optional
        Probability grid; defaults to ``np.arange(0.01, 1.00, 0.01)``.
    title : str
        Plot title.
    curve_label : str
        Legend label for the empirical (red) curve.

    Returns
    -------
    grid, emp_cdf, cal_mae, cal_sup : see ``oos_pit_calibration``.
    """
    grid, emp_cdf, cal_mae, cal_sup = oos_pit_calibration(u_t, grid=grid)

    plt.figure(figsize=(7.5, 7.5))

    # Perfect-calibration reference (blue diagonal).
    plt.plot([0, 1], [0, 1], color='royalblue', linestyle='--', linewidth=2.5,
             label='Perfect calibration (45°)')

    # Actual predicted distribution (red curve).
    plt.plot(grid, emp_cdf, color='crimson', linewidth=2.0, marker='o',
             markersize=3, label=curve_label)

    full_title = (f"{title}\n"
                  f"Calibration MAE = {cal_mae:.4f}   |   "
                  f"Max deviation = {cal_sup:.4f}")
    plt.title(full_title, fontsize=13, pad=15)
    plt.xlabel(r"Predicted / nominal probability  $\tau$", fontsize=12)
    plt.ylabel("Empirical CDF of PIT", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left')
    plt.grid(linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

    return grid, emp_cdf, cal_mae, cal_sup

def fit_and_diagnose_skewt(data, ticker="Asset", M=100, alpha=0.10, random_state=None):
    """
    Fits the Skewed-t distribution via MLE and runs diagnostics:
      - KS test against the fitted distribution
      - Bootstrap KS critical value
      - Two-panel plot: PDF overlay + QQ-plot
    
    Returns
    -------
    params : tuple  (df, a, loc, scale)
    ks_stat : float
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    df_hat, a_hat, loc_hat, scale_hat = fit_skewt(data)
    skewt_summary(df_hat, a_hat, loc_hat, scale_hat)

    # KS test
    ks_stat, ks_pval = kstest(data, lambda x: scipy_skewt.cdf(x, df_hat, a_hat, loc=loc_hat, scale=scale_hat))
    print(f"\nKS Test  — stat: {ks_stat:.4f}  p-value: {ks_pval:.4f}")

    # Bootstrap KS critical value
    if random_state is not None:
        np.random.seed(random_state)
        
    boot_ks = []
    
    # Disable warnings for the simulation loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(M):
            # FIXED: Inverse Transform Sampling to bypass .rvs() size error
            u = np.random.uniform(0, 1, len(data))
            sample = scipy_skewt.ppf(u, df_hat, a_hat, loc=loc_hat, scale=scale_hat)
            
            s, _ = kstest(sample, lambda x: scipy_skewt.cdf(x, df_hat, a_hat, loc=loc_hat, scale=scale_hat))
            boot_ks.append(s)
            
    cv = np.quantile(boot_ks, 1 - alpha)
    reject = ks_stat > cv
    print(f"Bootstrap CV ({int((1-alpha)*100)}%) : {cv:.4f} (KS statistic)  =>  {'REJECT H0' if reject else 'Fail to reject H0'}")  

    # Plot
    x_grid = np.linspace(data.min(), data.max(), 500)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Skewed-t MLE Fit — {ticker}", fontsize=13, fontweight="bold")

    # PDF overlay
    axes[0].hist(data, bins=50, density=True, alpha=0.4, color="steelblue", edgecolor="white", label="Empirical")
    axes[0].plot(x_grid, scipy_skewt.pdf(x_grid, df_hat, a_hat, loc=loc_hat, scale=scale_hat),
                 color="crimson", lw=2, label="Skewed-t MLE")
    axes[0].set_title("PDF Overlay")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # QQ-plot
    n = len(data)
    emp_q = np.sort(data)
    theo_q = scipy_skewt.ppf(np.linspace(1/(n+1), n/(n+1), n), df_hat, a_hat, loc=loc_hat, scale=scale_hat)
    axes[1].scatter(theo_q, emp_q, s=6, alpha=0.5, color="steelblue")
    lims = [min(theo_q.min(), emp_q.min()), max(theo_q.max(), emp_q.max())]
    axes[1].plot(lims, lims, "r--", lw=1.5)
    axes[1].set_title("QQ-Plot (Theoretical vs Empirical)")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Empirical Quantiles")

    plt.tight_layout()
    plt.show()

    return (df_hat, a_hat, loc_hat, scale_hat), ks_stat

def evaluate_oos_pit_skewt(realized_returns, forecasted_params, bins=15):
    """
    Computes the Probability Integral Transform (PIT) for the Azzalini Skewed-t
    distribution and produces a calibration diagnostic plot.

    Parameters
    ----------
    realized_returns  : array-like of realized returns.
    forecasted_params : list of (df, a, loc, scale) tuples — one per OOS date.
    bins              : number of histogram bins (default 15).

    Returns
    -------
    u_t_clean : np.ndarray of PIT values (filtered NaNs).
    ks_stat   : Kolmogorov-Smirnov statistic vs Uniform[0,1].
    ks_pval   : KS p-value.
    """
    from scipy.stats import kstest

    u_t = []
    for r, params in zip(realized_returns, forecasted_params):
        try:
            df_param, a, loc, scale = params
            u = float(scipy_skewt.cdf(r, df_param, a, loc=loc, scale=scale))
            u_t.append(u)
        except Exception:
            u_t.append(np.nan)

    u_t = np.array(u_t)
    u_t_clean = u_t[~np.isnan(u_t)]

    ks_stat, ks_pval = kstest(u_t_clean, 'uniform')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(u_t_clean, bins=bins, density=True, color='steelblue',
             edgecolor='white', alpha=0.8)
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2,
                label='Ideal Uniform(0,1)')
    ax1.set_title(f'PIT Histogram — Azzalini Skewed-t\n'
                  f'KS stat = {ks_stat:.4f},  p-value = {ks_pval:.4f}',
                  fontsize=13)
    ax1.set_xlabel('PIT value u(t)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    sorted_u = np.sort(u_t_clean)
    n = len(sorted_u)
    theoretical = np.linspace(0, 1, n)
    ax2.plot(theoretical, sorted_u, color='steelblue', linewidth=1.5,
             label='Empirical CDF')
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2,
             label='Ideal Uniform')
    ax2.set_title('PIT Q-Q Plot vs Uniform(0,1)', fontsize=13)
    ax2.set_xlabel('Theoretical Quantile', fontsize=12)
    ax2.set_ylabel('Empirical Quantile', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return u_t_clean, ks_stat, ks_pval
