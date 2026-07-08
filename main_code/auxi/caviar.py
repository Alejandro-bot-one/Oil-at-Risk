"""
CAViaR con variables indicador / severidad
===========================================
Regresión cuantílica tipo CAViaR (Engle & Manganelli, 2004) con dos variantes
de variables adicionales que capturan las rupturas de las fronteras cuantílicas:

  Variante _i (indicador binario):
    upside_breach   = 1{y_t > Q_high(y_t | x_{t-h})}
    downside_breach = 1{y_t < Q_low(y_t  | x_{t-h})}

  Variante _s (severidad — distancia absoluta):
    upside_severity   = max(0, y_t - Q_high(y_t | x_{t-h}))   >= 0
    downside_severity = max(0, Q_low(y_t  | x_{t-h}) - y_t)   >= 0

Los indicadores se computan con conciencia del horizonte h: comparan el valor
realizado en t con la frontera predicha h pasos antes, evitando lookahead.
El panel de entrada NUNCA se muta.

Arquitectura por capas (SoC):
  Capa 1 (helpers puros):    _compute_quantile_bounds, _compute_breaches,
                             _compute_breach_severity,
                             compute_breach_indicators (h-aware, sin lookahead),
                             compute_breach_severity_indicators (h-aware, sin lookahead)
  Capa 2 (estimación in-s.): caviar_i, multiple_caviar_i,
                             caviar_s, multiple_caviar_s
  Capa 3 (visualización):    plot_breach_diagnostics,
                             plot_caviar_i_results, plot_caviar_s_results

Para direct forecasting con indicadores CAViaR, usar compute_breach_indicators
(binario) o compute_breach_severity_indicators (severidad) para generar las
columnas y pasarlas como controls a las funciones estándar de auxi/qreg.py.

Referencia:
  Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional Autoregressive
  Value at Risk by Regression Quantiles. JBES, 22(4), 367-381.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from auxi.qreg import q_reg, plot_quantile_coefs, plot_pseudo_r2


# =============================================================================
# CAPA 1 - HELPERS PRIVADOS (lógica pura, sin estado)
# =============================================================================

def _compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high, **kwargs):
    """
    Ajusta dos regresiones cuantílicas in-sample (cola baja y cola alta) sobre
    la especificación dada y devuelve sus predicciones in-sample Q(y_t | x_t).

    Parameters
    ----------
    df       : DataFrame con los features y el target.
    vars_x   : list[str] regresores de la especificación.
    vars_y   : str, nombre del target.
    tau_low  : float, cuantil de la frontera baja.
    tau_high : float, cuantil de la frontera alta.
    **kwargs : reenviados a q_reg (p.ej. vcov="robust").

    Returns
    -------
    DataFrame indexado igual que df con columnas ["Bound_Low", "Bound_High"].
    No muta df.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg_low = q_reg(df=df, x=vars_x[0], y=vars_y, tau=tau_low,
                        controls=vars_x[1:] or None, **kwargs)
        reg_high = q_reg(df=df, x=vars_x[0], y=vars_y, tau=tau_high,
                         controls=vars_x[1:] or None, **kwargs)

    bounds = pd.DataFrame(
        {"Bound_Low": reg_low.predict(df),
         "Bound_High": reg_high.predict(df)},
        index=df.index,
    )
    return bounds


def _compute_breaches(realized, bounds):
    """
    Lógica pura de comparación: marca dónde el realizado cae fuera de las
    fronteras cuantílicas.

      upside   = 1{realized > Bound_High}
      downside = 1{realized < Bound_Low}

    Donde el bound es NaN (fila sin predicción computable), el indicador es NaN
    -- para no confundir "no breach" con "no computable".

    Parameters
    ----------
    realized : Series del valor realizado (target).
    bounds   : DataFrame con columnas ["Bound_Low", "Bound_High"].

    Returns
    -------
    (upside, downside) : tupla de dos Series con valores {0.0, 1.0} o NaN.
    """
    realized = pd.Series(realized).astype(float)
    low = bounds["Bound_Low"]
    high = bounds["Bound_High"]

    upside = (realized > high).astype(float)
    downside = (realized < low).astype(float)

    # Propagar NaN donde el bound no es computable.
    upside[high.isna()] = np.nan
    downside[low.isna()] = np.nan

    upside.name = "upside_breach"
    downside.name = "downside_breach"
    return upside, downside


def _compute_breach_severity(realized, bounds):
    """
    Severity of quantile-boundary breaches: absolute distance from the
    realized value to the violated boundary.

      upside_severity   = max(0, realized - Bound_High)   >= 0
      downside_severity = max(0, Bound_Low - realized)    >= 0

    Where the bound is NaN the severity is NaN (not 0).

    Parameters
    ----------
    realized : Series of the realized target.
    bounds   : DataFrame with columns ["Bound_Low", "Bound_High"].

    Returns
    -------
    (upside_severity, downside_severity) : tuple of two Series.
    """
    realized = pd.Series(realized).astype(float)
    low = bounds["Bound_Low"]
    high = bounds["Bound_High"]

    diff_high = realized - high
    diff_low = low - realized

    upside_severity = diff_high.clip(lower=0.0)
    downside_severity = diff_low.clip(lower=0.0)

    upside_severity[high.isna()] = np.nan
    downside_severity[low.isna()] = np.nan

    upside_severity.name = "upside_severity"
    downside_severity.name = "downside_severity"
    return upside_severity, downside_severity


def compute_breach_indicators(df, vars_x, vars_y, h,
                              breach_quantiles=None,
                              train_fraction=0.8,
                              test_start_date=None):
    """
    Indicadores de breach h-aware sin lookahead.

    Ajusta las fronteras cuantílicas (tau_low, tau_high) a horizonte h sobre
    los datos de entrenamiento, predice las fronteras para todo el panel, las
    lagea h posiciones, y compara con el valor realizado:

      breach_t = 1{y_t viola la frontera predicha en t-h}

    Las primeras h posiciones son NaN (no existe predicción anterior).

    Parameters
    ----------
    df               : DataFrame con features y target. No se muta.
    vars_x           : list[str] | str, regresores de la especificación.
    vars_y           : str, target.
    h                : int, horizonte de pronóstico (pasos adelante).
    breach_quantiles : list[float], default [0.05, 0.95].
    train_fraction   : float, default 0.8. Fracción del panel usada para
                       ajustar las fronteras (ignorado si test_start_date).
    test_start_date  : str 'YYYY-MM-DD', opcional. Si se da, entrena con
                       datos estrictamente anteriores a esa fecha.

    Returns
    -------
    DataFrame indexado como df con columnas
    ['upside_breach', 'downside_breach'] — valores {0.0, 1.0, NaN}.
    """
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]
    if isinstance(vars_x, str):
        vars_x = [vars_x]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    cols = [vars_y] + list(vars_x)
    work = df[cols].copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    target_col = f"{vars_y}_target_h{h}"
    work[target_col] = work[vars_y].shift(-h)

    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        train = work[work.index < test_start_dt]
    else:
        complete = work.dropna()
        split_idx = int(len(complete) * train_fraction)
        split_date = complete.index[split_idx]
        train = work[work.index < split_date]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="iid")
        try:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="iid")

    boundary_low = pd.Series(reg_low.predict(work), index=work.index)
    boundary_high = pd.Series(reg_high.predict(work), index=work.index)

    boundary_low_lagged = boundary_low.shift(h)
    boundary_high_lagged = boundary_high.shift(h)

    realized = df[vars_y]
    upside = (realized > boundary_high_lagged).astype(float)
    downside = (realized < boundary_low_lagged).astype(float)

    upside[boundary_high_lagged.isna()] = np.nan
    downside[boundary_low_lagged.isna()] = np.nan

    return pd.DataFrame({
        "upside_breach": upside,
        "downside_breach": downside,
    }, index=df.index)


def compute_breach_severity_indicators(df, vars_x, vars_y, h,
                                       breach_quantiles=None,
                                       train_fraction=0.8,
                                       test_start_date=None):
    """
    Severity-based breach indicators, h-aware and lookahead-free.

    Same as compute_breach_indicators but returns the absolute distance from
    the realized value to the violated boundary instead of a binary flag:

      upside_severity_t   = max(0, y_t - Q_high(y_t | x_{t-h}))   >= 0
      downside_severity_t = max(0, Q_low(y_t  | x_{t-h}) - y_t)   >= 0

    The first h rows are NaN (no prior prediction available).

    Parameters
    ----------
    df               : DataFrame with features and target. Never mutated.
    vars_x           : list[str] | str, specification regressors.
    vars_y           : str, target.
    h                : int, forecast horizon (steps ahead).
    breach_quantiles : list[float], default [0.05, 0.95].
    train_fraction   : float, default 0.8 (ignored if test_start_date given).
    test_start_date  : str 'YYYY-MM-DD', optional.

    Returns
    -------
    DataFrame indexed like df with columns
    ['upside_severity', 'downside_severity'].
    """
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]
    if isinstance(vars_x, str):
        vars_x = [vars_x]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    cols = [vars_y] + list(vars_x)
    work = df[cols].copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    target_col = f"{vars_y}_target_h{h}"
    work[target_col] = work[vars_y].shift(-h)

    if test_start_date is not None:
        test_start_dt = pd.to_datetime(test_start_date)
        train = work[work.index < test_start_dt]
    else:
        complete = work.dropna()
        split_idx = int(len(complete) * train_fraction)
        split_date = complete.index[split_idx]
        train = work[work.index < split_date]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_low = q_reg(train, x=vars_x[0], y=target_col, tau=tau_low,
                            controls=vars_x[1:] or None, vcov="iid")
        try:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="robust")
        except ValueError:
            reg_high = q_reg(train, x=vars_x[0], y=target_col, tau=tau_high,
                             controls=vars_x[1:] or None, vcov="iid")

    boundary_low = pd.Series(reg_low.predict(work), index=work.index)
    boundary_high = pd.Series(reg_high.predict(work), index=work.index)

    boundary_low_lagged = boundary_low.shift(h)
    boundary_high_lagged = boundary_high.shift(h)

    realized = df[vars_y]
    bounds_lagged = pd.DataFrame({
        "Bound_Low": boundary_low_lagged,
        "Bound_High": boundary_high_lagged,
    }, index=df.index)

    upside_sev, downside_sev = _compute_breach_severity(realized, bounds_lagged)

    return pd.DataFrame({
        "upside_severity": upside_sev,
        "downside_severity": downside_sev,
    }, index=df.index)


# =============================================================================
# CAPA 2 - ESTIMACIÓN IN-SAMPLE (hermanas de q_reg / multiple_q_regs)
# =============================================================================

def caviar_i(df, vars_x, vars_y, tau, breach_quantiles=None,
             errors="robust", **kwargs):
    """
    Regresión cuantílica tipo CAViaR con dos variables indicador binarias.

    Paso a paso:
      1. tau_low = min(breach_quantiles), tau_high = max(breach_quantiles).
      2. Bounds in-sample con la MISMA especificación (vars_x).
      3. Indicadores upside/downside (efímeros).
      4. Sobre una copia interna, añade los indicadores y corre la q-reg
         y ~ vars_x + upside_breach + downside_breach en el cuantil tau.

    Parameters
    ----------
    df              : DataFrame con features y target. NUNCA se muta.
    vars_x          : list[str] | str, regresores de la especificación.
    vars_y          : str, target.
    tau             : float, cuantil de la regresión CAViaR.
    breach_quantiles: list[float], default [0.05, 0.95]. Sus extremos definen
                      las fronteras del breach.
    errors          : str, vcov para la regresión (default "robust").
    **kwargs        : reenviados a q_reg.

    Returns
    -------
    (reg, indicators) : tupla.
        reg        -> resultado de regresión ajustado (statsmodels).
        indicators -> DataFrame con upside_breach, downside_breach
                      (+ Bound_Low, Bound_High para inspección).
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    bounds = _compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high)
    upside, downside = _compute_breaches(df[vars_y], bounds)

    indicators = pd.DataFrame(
        {"upside_breach": upside,
         "downside_breach": downside,
         "Bound_Low": bounds["Bound_Low"],
         "Bound_High": bounds["Bound_High"]},
        index=df.index,
    )

    # Copia de trabajo: el panel del usuario nunca se toca.
    work = df.copy()
    work["upside_breach"] = upside
    work["downside_breach"] = downside

    all_x = list(vars_x) + ["upside_breach", "downside_breach"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov=errors, **kwargs)
        except ValueError:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov="iid", **kwargs)

    return reg, indicators


def multiple_caviar_i(data, vars_x, vars_y, quantiles=None,
                      breach_quantiles=None, errors="robust"):
    """
    Corre caviar_i a través de varios cuantiles y devuelve una tabla con el
    MISMO esquema que multiple_q_regs (auxi/qreg.py), incluyendo las filas de
    upside_breach y downside_breach.

    Eficiencia clave
    ----------------
    Los indicadores dependen sólo de breach_quantiles, NO del tau de
    estimación. Se computan UNA sola vez antes del bucle y se reutilizan en
    todos los cuantiles.

    Parameters
    ----------
    data             : DataFrame con features y target. No se muta.
    vars_x           : list[str] | str, especificación.
    vars_y           : str, target.
    quantiles        : list[float], default [0.05, 0.25, 0.50, 0.75, 0.95].
    breach_quantiles : list[float], default [0.05, 0.95].
    errors           : str, vcov (default "robust").

    Returns
    -------
    master_df : DataFrame ordenado por (Regressor, Tau) con columnas
        Dependent Variable, Regressor, Tau, Coefficient, Significance,
        Pseudo R-Squared.
    """
    import statsmodels.formula.api as smf

    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    def get_stars(p_value):
        if p_value < 0.01: return '***'
        elif p_value < 0.05: return '**'
        elif p_value < 0.10: return '*'
        else: return ''

    # Indicadores: computados UNA vez, reutilizados en todos los taus.
    bounds = _compute_quantile_bounds(data, vars_x, vars_y, tau_low, tau_high)
    upside, downside = _compute_breaches(data[vars_y], bounds)

    work = data.copy()
    work["upside_breach"] = upside
    work["downside_breach"] = downside

    all_indep_vars = list(vars_x) + ["upside_breach", "downside_breach"]
    rhs = " + ".join([f"Q('{v}')" for v in all_indep_vars])
    equation = f"Q('{vars_y}') ~ {rhs}"

    res_dict = {
        "Dependent Variable": [], "Regressor": [], "Tau": [],
        "Coefficient": [], "Significance": [], "Pseudo R-Squared": [],
    }

    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=work, formula=equation)
            try:
                reg = mod.fit(q=q, vcov=errors)
            except ValueError:
                reg = mod.fit(q=q, vcov="iid")

            pseudo_r2 = getattr(reg, 'prsquared', np.nan)
            for var in all_indep_vars:
                param_name = f"Q('{var}')"
                res_dict["Dependent Variable"].append(vars_y)
                res_dict["Regressor"].append(var)
                res_dict["Tau"].append(q)
                res_dict["Coefficient"].append(reg.params[param_name])
                res_dict["Significance"].append(get_stars(reg.pvalues[param_name]))
                res_dict["Pseudo R-Squared"].append(pseudo_r2)

    master_df = pd.DataFrame(res_dict)
    master_df = master_df.sort_values(by=["Regressor", "Tau"]).reset_index(drop=True)
    return master_df


# =============================================================================
# CAPA 2 (cont.) — ESTIMACIÓN IN-SAMPLE: SEVERIDAD
# =============================================================================

def caviar_s(df, vars_x, vars_y, tau, breach_quantiles=None,
             errors="robust", **kwargs):
    """
    CAViaR with severity (absolute distance) regressors instead of binary
    indicators.

    Same as caviar_i but uses:
      upside_severity   = max(0, y_t - Q_high)   >= 0
      downside_severity = max(0, Q_low - y_t)    >= 0

    Parameters
    ----------
    df              : DataFrame. Never mutated.
    vars_x          : list[str] | str, specification regressors.
    vars_y          : str, target.
    tau             : float, quantile for the CAViaR regression.
    breach_quantiles: list[float], default [0.05, 0.95].
    errors          : str, vcov (default "robust").
    **kwargs        : forwarded to q_reg.

    Returns
    -------
    (reg, indicators) : tuple.
        reg        -> fitted regression result (statsmodels).
        indicators -> DataFrame with upside_severity, downside_severity,
                      Bound_Low, Bound_High.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    bounds = _compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high)
    upside_sev, downside_sev = _compute_breach_severity(df[vars_y], bounds)

    indicators = pd.DataFrame(
        {"upside_severity": upside_sev,
         "downside_severity": downside_sev,
         "Bound_Low": bounds["Bound_Low"],
         "Bound_High": bounds["Bound_High"]},
        index=df.index,
    )

    work = df.copy()
    work["upside_severity"] = upside_sev
    work["downside_severity"] = downside_sev

    all_x = list(vars_x) + ["upside_severity", "downside_severity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov=errors, **kwargs)
        except ValueError:
            reg = q_reg(df=work, x=all_x[0], y=vars_y, tau=tau,
                        controls=all_x[1:], vcov="iid", **kwargs)

    return reg, indicators


def multiple_caviar_s(data, vars_x, vars_y, quantiles=None,
                      breach_quantiles=None, errors="robust"):
    """
    Run caviar_s across a quantile grid and return a master_df with the
    same schema as multiple_q_regs / multiple_caviar_i.

    Severity indicators are computed once and reused across all taus.

    Parameters
    ----------
    data             : DataFrame. Never mutated.
    vars_x           : list[str] | str, specification.
    vars_y           : str, target.
    quantiles        : list[float], default 21-point grid.
    breach_quantiles : list[float], default [0.05, 0.95].
    errors           : str, vcov (default "robust").

    Returns
    -------
    master_df : DataFrame sorted by (Regressor, Tau) with columns
        Dependent Variable, Regressor, Tau, Coefficient, Significance,
        Pseudo R-Squared.
    """
    import statsmodels.formula.api as smf

    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)
    if tau_low == tau_high:
        raise ValueError(
            f"breach_quantiles degenerados: min == max == {tau_low}. "
            "Se necesitan al menos dos cuantiles distintos."
        )

    def get_stars(p_value):
        if p_value < 0.01: return '***'
        elif p_value < 0.05: return '**'
        elif p_value < 0.10: return '*'
        else: return ''

    bounds = _compute_quantile_bounds(data, vars_x, vars_y, tau_low, tau_high)
    upside_sev, downside_sev = _compute_breach_severity(data[vars_y], bounds)

    work = data.copy()
    work["upside_severity"] = upside_sev
    work["downside_severity"] = downside_sev

    all_indep_vars = list(vars_x) + ["upside_severity", "downside_severity"]
    rhs = " + ".join([f"Q('{v}')" for v in all_indep_vars])
    equation = f"Q('{vars_y}') ~ {rhs}"

    res_dict = {
        "Dependent Variable": [], "Regressor": [], "Tau": [],
        "Coefficient": [], "Significance": [], "Pseudo R-Squared": [],
    }

    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.quantreg(data=work, formula=equation)
            try:
                reg = mod.fit(q=q, vcov=errors)
            except ValueError:
                reg = mod.fit(q=q, vcov="iid")

            pseudo_r2 = getattr(reg, 'prsquared', np.nan)
            for var in all_indep_vars:
                param_name = f"Q('{var}')"
                res_dict["Dependent Variable"].append(vars_y)
                res_dict["Regressor"].append(var)
                res_dict["Tau"].append(q)
                res_dict["Coefficient"].append(reg.params[param_name])
                res_dict["Significance"].append(get_stars(reg.pvalues[param_name]))
                res_dict["Pseudo R-Squared"].append(pseudo_r2)

    master_df = pd.DataFrame(res_dict)
    master_df = master_df.sort_values(by=["Regressor", "Tau"]).reset_index(drop=True)
    return master_df


# =============================================================================
# CAPA 3 - VISUALIZACIÓN (mirror de qreg.py, máximo reuso)
# =============================================================================

def plot_breach_diagnostics(ax, df, vars_x, vars_y, breach_quantiles=None,
                            **kwargs):
    """
    Sub-función atómica: serie temporal con las fronteras cuantílicas
    (Bound_Low / Bound_High) y los puntos de breach marcados, al estilo del
    panel de bounds & breaches de auxi/forecasting.py.

    Recalcula bounds y breaches internamente (helpers de capa 1). No muta df.
    """
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]
    tau_low = min(breach_quantiles)
    tau_high = max(breach_quantiles)

    bounds = _compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high)
    upside, downside = _compute_breaches(df[vars_y], bounds)

    realized = df[vars_y]
    ax.plot(realized.index, realized, color="black", linewidth=1.0,
            alpha=0.8, label=f"Realizado {vars_y}")
    ax.plot(bounds.index, bounds["Bound_Low"], color="crimson",
            linestyle="--", linewidth=1.5,
            label=f"Frontera baja ($\\tau$={tau_low})")
    ax.plot(bounds.index, bounds["Bound_High"], color="steelblue",
            linestyle="--", linewidth=1.5,
            label=f"Frontera alta ($\\tau$={tau_high})")
    ax.fill_between(bounds.index, bounds["Bound_Low"], bounds["Bound_High"],
                    color="gray", alpha=0.1)

    down_pts = realized[downside == 1.0]
    up_pts = realized[upside == 1.0]
    ax.scatter(down_pts.index, down_pts, color="red", s=35, zorder=5,
               label="Downside breach")
    ax.scatter(up_pts.index, up_pts, color="blue", s=35, zorder=5,
               label="Upside breach")

    n_breaches = int(np.nansum(upside)) + int(np.nansum(downside))
    ax.set_title(f"Quantile Bounds & Breaches (total={n_breaches})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Value of {vars_y}")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_caviar_i_results(data, vars_x, vars_y, breach_quantiles=None,
                          quantiles=None, errors="robust"):
    """
    Orquestador (mirror de plot_quantile_results de auxi/qreg.py). Construye un
    dashboard 2x2 para el modelo CAViaR con indicadores y devuelve el master_df.

    Paneles:
      [0,0] Coefs de los regresores de la especificación (plot_quantile_coefs).
      [0,1] Coefs de upside_breach / downside_breach por tau -- el resultado
            estrella CAViaR (plot_quantile_coefs).
      [1,0] Pseudo R^2 (plot_pseudo_r2).
      [1,1] plot_breach_diagnostics (bounds + breaches en el tiempo).

    Returns
    -------
    master_df : el output numérico de multiple_caviar_i.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    results_df = multiple_caviar_i(
        data=data, vars_x=vars_x, vars_y=vars_y, quantiles=quantiles,
        breach_quantiles=breach_quantiles, errors=errors,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"CAViaR (indicadores) Dashboard: {vars_y} ~ {vars_x}",
                 fontsize=14, fontweight="bold")

    plot_quantile_coefs(axes[0, 0], results_df, list(vars_x),
                        title=f"Regresores de la especificación: {vars_x}")
    plot_quantile_coefs(axes[0, 1], results_df,
                        ["upside_breach", "downside_breach"],
                        title="Variables indicador (breach)")
    plot_pseudo_r2(axes[1, 0], results_df)
    plot_breach_diagnostics(axes[1, 1], data, vars_x=vars_x, vars_y=vars_y,
                            breach_quantiles=breach_quantiles)

    plt.tight_layout()
    plt.show()

    return results_df


def plot_caviar_s_results(data, vars_x, vars_y, breach_quantiles=None,
                          quantiles=None, errors="robust"):
    """
    2x2 dashboard for the CAViaR severity model — mirrors plot_caviar_i_results.

    Panels:
      [0,0] Coefficients of the specification regressors (plot_quantile_coefs).
      [0,1] Coefficients of upside_severity / downside_severity by tau.
      [1,0] Pseudo R^2 (plot_pseudo_r2).
      [1,1] Breach diagnostics (bounds + breaches timeline).

    Returns
    -------
    master_df : the numeric output of multiple_caviar_s.
    """
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    if quantiles is None:
        quantiles = [0.01] + list(np.round(np.arange(0.05, 0.951, 0.05), 2)) + [0.99]
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]

    results_df = multiple_caviar_s(
        data=data, vars_x=vars_x, vars_y=vars_y, quantiles=quantiles,
        breach_quantiles=breach_quantiles, errors=errors,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"CAViaR (severity) Dashboard: {vars_y} ~ {vars_x}",
                 fontsize=14, fontweight="bold")

    plot_quantile_coefs(axes[0, 0], results_df, list(vars_x),
                        title=f"Specification regressors: {vars_x}")
    plot_quantile_coefs(axes[0, 1], results_df,
                        ["upside_severity", "downside_severity"],
                        title="Severity regressors (breach distance)")
    plot_pseudo_r2(axes[1, 0], results_df)
    plot_breach_diagnostics(axes[1, 1], data, vars_x=vars_x, vars_y=vars_y,
                            breach_quantiles=breach_quantiles)

    plt.tight_layout()
    plt.show()

    return results_df
