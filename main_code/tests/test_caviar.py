import matplotlib
matplotlib.use("Agg")  # backend sin ventana para tests
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from auxi.caviar import (
    _compute_quantile_bounds, _compute_breaches, _compute_breach_severity,
    compute_breach_indicators, compute_breach_severity_indicators,
    caviar_i, multiple_caviar_i,
    caviar_s, multiple_caviar_s,
    plot_breach_diagnostics, plot_caviar_i_results, plot_caviar_s_results,
)


def test_compute_quantile_bounds_shape_and_columns(synthetic_panel):
    bounds = _compute_quantile_bounds(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau_low=0.05, tau_high=0.95,
    )
    # Mismo índice que el panel de entrada.
    assert list(bounds.index) == list(synthetic_panel.index)
    # Exactamente las dos columnas de fronteras.
    assert list(bounds.columns) == ["Bound_Low", "Bound_High"]


def test_compute_quantile_bounds_low_below_high(synthetic_panel):
    bounds = _compute_quantile_bounds(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau_low=0.05, tau_high=0.95,
    )
    # La frontera baja debe quedar por debajo de la alta en la gran mayoría
    # de las filas (quantile crossing puntual es admisible, masivo no).
    frac_ok = (bounds["Bound_Low"] < bounds["Bound_High"]).mean()
    assert frac_ok > 0.95


def test_compute_quantile_bounds_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    _compute_quantile_bounds(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau_low=0.05, tau_high=0.95,
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_compute_breaches_binary_values():
    realized = pd.Series([1.0, 5.0, -5.0, 0.5], index=range(4))
    bounds = pd.DataFrame(
        {"Bound_Low": [-2.0, -2.0, -2.0, -2.0],
         "Bound_High": [2.0, 2.0, 2.0, 2.0]},
        index=range(4),
    )
    upside, downside = _compute_breaches(realized, bounds)
    # Sólo {0,1} (NaN aparte) y la dirección correcta.
    assert list(upside.values) == [0, 1, 0, 0]
    assert list(downside.values) == [0, 0, 1, 0]


def test_compute_breaches_nan_bound_gives_nan():
    realized = pd.Series([1.0, 5.0], index=range(2))
    bounds = pd.DataFrame(
        {"Bound_Low": [np.nan, -2.0],
         "Bound_High": [np.nan, 2.0]},
        index=range(2),
    )
    upside, downside = _compute_breaches(realized, bounds)
    # Fila con bound NaN -> indicador NaN (no 0).
    assert np.isnan(upside.iloc[0])
    assert np.isnan(downside.iloc[0])
    # Fila válida -> upside breach.
    assert upside.iloc[1] == 1
    assert downside.iloc[1] == 0


# =====================================================================
# _compute_breach_severity (Layer 1 severity helper)
# =====================================================================

def test_compute_breach_severity_values():
    realized = pd.Series([1.0, 5.0, -5.0, 0.5], index=range(4))
    bounds = pd.DataFrame(
        {"Bound_Low": [-2.0, -2.0, -2.0, -2.0],
         "Bound_High": [2.0, 2.0, 2.0, 2.0]},
        index=range(4),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    assert list(upside.values) == [0.0, 3.0, 0.0, 0.0]
    assert list(downside.values) == [0.0, 0.0, 3.0, 0.0]


def test_compute_breach_severity_nan_bound_gives_nan():
    realized = pd.Series([1.0, 5.0], index=range(2))
    bounds = pd.DataFrame(
        {"Bound_Low": [np.nan, -2.0],
         "Bound_High": [np.nan, 2.0]},
        index=range(2),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    assert np.isnan(upside.iloc[0])
    assert np.isnan(downside.iloc[0])
    assert upside.iloc[1] == 3.0
    assert downside.iloc[1] == 0.0


def test_compute_breach_severity_sign_convention():
    """Both upside_severity >= 0 and downside_severity >= 0 (absolute value)."""
    realized = pd.Series([10.0, -10.0, 0.0], index=range(3))
    bounds = pd.DataFrame(
        {"Bound_Low": [-2.0, -2.0, -2.0],
         "Bound_High": [2.0, 2.0, 2.0]},
        index=range(3),
    )
    upside, downside = _compute_breach_severity(realized, bounds)
    assert (upside >= 0).all()
    assert (downside >= 0).all()


def test_caviar_i_returns_reg_and_indicators(synthetic_panel):
    reg, indicators = caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    # El reg ajustado expone params (statsmodels).
    assert hasattr(reg, "params")
    # Los indicadores aparecen como regresores en la regresión.
    param_names = list(reg.params.index)
    assert any("upside_breach" in p for p in param_names)
    assert any("downside_breach" in p for p in param_names)
    # indicators es un DataFrame con las dos columnas binarias.
    assert "upside_breach" in indicators.columns
    assert "downside_breach" in indicators.columns


def test_caviar_i_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_caviar_i_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        caviar_i(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            tau=0.5, breach_quantiles=[0.5, 0.5],
        )


def test_caviar_i_default_breach_quantiles(synthetic_panel):
    # Sin breach_quantiles -> usa [0.05, 0.95] sin lanzar.
    reg, indicators = caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5,
    )
    assert hasattr(reg, "params")


def test_multiple_caviar_i_schema(synthetic_panel):
    master = multiple_caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    expected_cols = {
        "Dependent Variable", "Regressor", "Tau",
        "Coefficient", "Significance", "Pseudo R-Squared",
    }
    assert expected_cols.issubset(set(master.columns))


def test_multiple_caviar_i_includes_indicators(synthetic_panel):
    master = multiple_caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    regressors = set(master["Regressor"].unique())
    # Especificación + ambos indicadores.
    assert {"gpr", "vix", "upside_breach", "downside_breach"}.issubset(regressors)


def test_multiple_caviar_i_one_row_per_regressor_per_tau(synthetic_panel):
    quantiles = [0.25, 0.5, 0.75]
    master = multiple_caviar_i(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=quantiles, breach_quantiles=[0.05, 0.95],
    )
    # 4 regresores (gpr, vix, upside, downside) x 3 taus = 12 filas.
    assert len(master) == 4 * len(quantiles)


# =====================================================================
# compute_breach_indicators (h-aware, sin lookahead)
# =====================================================================

def test_compute_breach_indicators_shape(synthetic_panel):
    indicators = compute_breach_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert list(indicators.columns) == ["upside_breach", "downside_breach"]
    assert len(indicators) == len(synthetic_panel)


def test_compute_breach_indicators_first_h_are_nan(synthetic_panel):
    for h in [1, 3, 5]:
        indicators = compute_breach_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=h, breach_quantiles=[0.05, 0.95],
        )
        assert indicators["upside_breach"].iloc[:h].isna().all()
        assert indicators["downside_breach"].iloc[:h].isna().all()


def test_compute_breach_indicators_binary_values(synthetic_panel):
    indicators = compute_breach_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    valid = indicators.dropna()
    assert set(valid["upside_breach"].unique()).issubset({0.0, 1.0})
    assert set(valid["downside_breach"].unique()).issubset({0.0, 1.0})


def test_compute_breach_indicators_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    compute_breach_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_compute_breach_indicators_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        compute_breach_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=1, breach_quantiles=[0.5, 0.5],
        )


def test_compute_breach_indicators_test_start_date(synthetic_panel):
    mid = synthetic_panel.index[200]
    indicators = compute_breach_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
        test_start_date=str(mid.date()),
    )
    assert len(indicators) == len(synthetic_panel)
    valid = indicators.dropna()
    assert set(valid["upside_breach"].unique()).issubset({0.0, 1.0})


def test_compute_breach_indicators_detects_outliers(synthetic_panel):
    indicators = compute_breach_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert indicators["upside_breach"].sum() > 0
    assert indicators["downside_breach"].sum() > 0


# =====================================================================
# compute_breach_severity_indicators (h-aware, sin lookahead)
# =====================================================================

def test_compute_breach_severity_indicators_shape(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert list(indicators.columns) == ["upside_severity", "downside_severity"]
    assert len(indicators) == len(synthetic_panel)


def test_compute_breach_severity_indicators_first_h_are_nan(synthetic_panel):
    for h_val in [1, 3, 5]:
        indicators = compute_breach_severity_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=h_val, breach_quantiles=[0.05, 0.95],
        )
        assert indicators["upside_severity"].iloc[:h_val].isna().all()
        assert indicators["downside_severity"].iloc[:h_val].isna().all()


def test_compute_breach_severity_indicators_sign_convention(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    valid = indicators.dropna()
    assert (valid["upside_severity"] >= 0).all()
    assert (valid["downside_severity"] >= 0).all()


def test_compute_breach_severity_indicators_continuous_values(synthetic_panel):
    """Severity is continuous, not binary — some values should differ from 0/1."""
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    valid = indicators.dropna()
    up_nonzero = valid["upside_severity"][valid["upside_severity"] > 0]
    if len(up_nonzero) > 0:
        assert not set(up_nonzero.values).issubset({0.0, 1.0})


def test_compute_breach_severity_indicators_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_compute_breach_severity_indicators_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        compute_breach_severity_indicators(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            h=1, breach_quantiles=[0.5, 0.5],
        )


def test_compute_breach_severity_indicators_test_start_date(synthetic_panel):
    mid = synthetic_panel.index[200]
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
        test_start_date=str(mid.date()),
    )
    assert len(indicators) == len(synthetic_panel)
    valid = indicators.dropna()
    assert (valid["upside_severity"] >= 0).all()
    assert (valid["downside_severity"] >= 0).all()


def test_compute_breach_severity_indicators_detects_outliers(synthetic_panel):
    indicators = compute_breach_severity_indicators(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        h=1, breach_quantiles=[0.05, 0.95],
    )
    assert (indicators["upside_severity"] > 0).any()
    assert (indicators["downside_severity"] > 0).any()


# =====================================================================
# Capa 3 — visualización
# =====================================================================

def test_plot_breach_diagnostics_runs(synthetic_panel):
    fig, ax = plt.subplots()
    # No debe lanzar; dibuja sobre el ax dado.
    plot_breach_diagnostics(
        ax, synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        breach_quantiles=[0.05, 0.95],
    )
    # Algo se ha dibujado (líneas de bounds + scatter de breaches).
    assert len(ax.get_lines()) >= 1
    plt.close(fig)


def test_plot_caviar_i_results_returns_master_df(synthetic_panel):
    master = plot_caviar_i_results(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        breach_quantiles=[0.05, 0.95], quantiles=[0.25, 0.5, 0.75],
    )
    # Devuelve el master_df (como plot_quantile_results).
    assert "Coefficient" in master.columns
    regressors = set(master["Regressor"].unique())
    assert {"gpr", "vix", "upside_breach", "downside_breach"}.issubset(regressors)
    plt.close("all")


# =====================================================================
# caviar_s / multiple_caviar_s (Layer 2 — severity estimators)
# =====================================================================

def test_caviar_s_returns_reg_and_indicators(synthetic_panel):
    reg, indicators = caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    assert hasattr(reg, "params")
    param_names = list(reg.params.index)
    assert any("upside_severity" in p for p in param_names)
    assert any("downside_severity" in p for p in param_names)
    assert "upside_severity" in indicators.columns
    assert "downside_severity" in indicators.columns


def test_caviar_s_does_not_mutate_input(synthetic_panel):
    before = synthetic_panel.copy()
    caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5, breach_quantiles=[0.05, 0.95],
    )
    pd.testing.assert_frame_equal(synthetic_panel, before)


def test_caviar_s_degenerate_taus_raise(synthetic_panel):
    with pytest.raises(ValueError):
        caviar_s(
            synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
            tau=0.5, breach_quantiles=[0.5, 0.5],
        )


def test_caviar_s_default_breach_quantiles(synthetic_panel):
    reg, indicators = caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        tau=0.5,
    )
    assert hasattr(reg, "params")


def test_multiple_caviar_s_schema(synthetic_panel):
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    expected_cols = {
        "Dependent Variable", "Regressor", "Tau",
        "Coefficient", "Significance", "Pseudo R-Squared",
    }
    assert expected_cols.issubset(set(master.columns))


def test_multiple_caviar_s_includes_severity(synthetic_panel):
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=[0.25, 0.5, 0.75], breach_quantiles=[0.05, 0.95],
    )
    regressors = set(master["Regressor"].unique())
    assert {"gpr", "vix", "upside_severity", "downside_severity"}.issubset(regressors)


def test_multiple_caviar_s_one_row_per_regressor_per_tau(synthetic_panel):
    quantiles = [0.25, 0.5, 0.75]
    master = multiple_caviar_s(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        quantiles=quantiles, breach_quantiles=[0.05, 0.95],
    )
    assert len(master) == 4 * len(quantiles)


# =====================================================================
# plot_caviar_s_results (Layer 3 — severity dashboard)
# =====================================================================

def test_plot_caviar_s_results_returns_master_df(synthetic_panel):
    master = plot_caviar_s_results(
        synthetic_panel, vars_x=["gpr", "vix"], vars_y="Brent_Return",
        breach_quantiles=[0.05, 0.95], quantiles=[0.25, 0.5, 0.75],
    )
    assert "Coefficient" in master.columns
    regressors = set(master["Regressor"].unique())
    assert {"gpr", "vix", "upside_severity", "downside_severity"}.issubset(regressors)
    plt.close("all")
