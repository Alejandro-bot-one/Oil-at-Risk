# CAViaR con variables indicador (`caviar_i`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el módulo `auxi/caviar.py` que estima una regresión cuantílica tipo CAViaR añadiendo dos variables indicador binarias (upside/downside breach) derivadas in-sample de la propia especificación, sin contaminar el panel de entrada.

**Architecture:** Tres capas siguiendo SoC, en paralelo a `auxi/qreg.py`. Capa 1: helpers privados puros (`_compute_quantile_bounds`, `_compute_breaches`). Capa 2: estimación (`caviar_i` devuelve `(reg, indicators)`; `multiple_caviar_i` devuelve un `master_df`). Capa 3: visualización (`plot_breach_diagnostics` nueva + reuso de `plot_quantile_coefs`/`plot_pseudo_r2` de `qreg.py`; orquestador `plot_caviar_i_results`). Los indicadores son efímeros: viven dentro de la llamada y nunca se escriben en el panel del usuario.

**Tech Stack:** Python, pandas, numpy, statsmodels (`smf.quantreg`), matplotlib, pytest 7.4.4.

---

## File Structure

- **Create:** `auxi/caviar.py` — todo el módulo CAViaR (3 capas).
- **Create:** `tests/__init__.py` — marca `tests` como paquete (vacío).
- **Create:** `tests/conftest.py` — fixture de panel sintético reutilizable.
- **Create:** `tests/test_caviar.py` — tests de las capas 1 y 2 (lógica numérica).
- **Reuse (no modificar):** `auxi/qreg.py` — `q_reg`, `plot_quantile_coefs`, `plot_pseudo_r2`.

**Nota sobre testing de plots:** las funciones de la Capa 3 (matplotlib) no se testean con asserts numéricos; se validan con un test de "humo" que las ejecuta con backend `Agg` y comprueba que no lanzan excepción y devuelven el `master_df` esperado.

**Nota sobre imports:** `auxi/` no tiene `__init__.py` (es un dir de módulos sueltos importados como `from auxi.qreg import ...` desde la raíz). Dentro de `caviar.py` el import es `from auxi.qreg import q_reg, plot_quantile_coefs, plot_pseudo_r2`. Los tests se ejecutan desde la raíz del proyecto (`main code/`) con `python -m pytest` para que `auxi` y `tests` sean importables.

---

## Task 1: Andamiaje de tests y fixture de panel sintético

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Crear `tests/__init__.py` vacío**

Crear el archivo `tests/__init__.py` con contenido vacío (un único salto de línea).

- [ ] **Step 2: Crear la fixture de panel sintético**

Crear `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel():
    """
    Panel diario sintético reproducible para tests de CAViaR.

    - Índice DatetimeIndex de 400 días hábiles.
    - `Brent_Return`: target, ruido gaussiano con outliers inyectados en
      posiciones conocidas para forzar breaches deterministas.
    - `gpr`, `vix`: dos features de especificación.
    """
    rng = np.random.default_rng(42)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)

    gpr = rng.normal(0.0, 1.0, n)
    vix = rng.normal(0.0, 1.0, n)
    # Target con relación débil a los features + ruido.
    brent = 0.3 * gpr - 0.2 * vix + rng.normal(0.0, 1.0, n)

    # Inyectar outliers extremos en posiciones conocidas (garantizan breaches).
    up_pos = [50, 150, 250]      # outliers positivos -> upside breach
    down_pos = [80, 180, 280]    # outliers negativos -> downside breach
    brent[up_pos] = 8.0
    brent[down_pos] = -8.0

    df = pd.DataFrame(
        {"Brent_Return": brent, "gpr": gpr, "vix": vix},
        index=idx,
    )
    return df
```

- [ ] **Step 3: Verificar que pytest descubre el directorio**

Run: `python -m pytest tests/ -q`
Expected: `no tests ran` (0 tests recolectados, sin errores de colección).

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add synthetic panel fixture for caviar tests"
```

---

## Task 2: `_compute_quantile_bounds` (Capa 1)

**Files:**
- Create: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_caviar.py`:

```python
import numpy as np
import pandas as pd
import pytest

from auxi.caviar import _compute_quantile_bounds


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
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'auxi.caviar'` (o `ImportError`).

- [ ] **Step 3: Crear `auxi/caviar.py` con la cabecera y `_compute_quantile_bounds`**

Crear `auxi/caviar.py`:

```python
"""
CAViaR con variables indicador (caviar_i)
=========================================
Regresión cuantílica tipo CAViaR (Engle & Manganelli, 2004) en su
especificación más sencilla: dos variables indicador binarias que señalan si
el valor realizado cayó fuera de las fronteras cuantílicas de la propia
especificación.

  upside_breach   = 1{y_t > Q(y_t | x_t; tau_high)}
  downside_breach = 1{y_t < Q(y_t | x_t; tau_low)}

Los indicadores se computan in-sample y de forma contemporánea (sin shift): el
desfase respecto al objetivo de pronóstico lo aporta luego el shift del target
en el direct forecasting. Son efímeros — viven dentro de la llamada, se usan
como regresores y se descartan. El panel de entrada NUNCA se muta.

Arquitectura por capas (SoC), en paralelo a auxi/qreg.py:
  Capa 1 (helpers puros): _compute_quantile_bounds, _compute_breaches
  Capa 2 (estimación):    caviar_i, multiple_caviar_i
  Capa 3 (visualización): plot_breach_diagnostics, plot_caviar_i_results
                          (+ reuso de plot_quantile_coefs, plot_pseudo_r2)

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
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add _compute_quantile_bounds helper for caviar"
```

---

## Task 3: `_compute_breaches` (Capa 1)

**Files:**
- Modify: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_caviar.py`:

```python
from auxi.caviar import _compute_breaches


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
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -k breaches -q`
Expected: FAIL con `ImportError: cannot import name '_compute_breaches'`.

- [ ] **Step 3: Añadir `_compute_breaches` a `auxi/caviar.py`**

Insertar después de `_compute_quantile_bounds`:

```python
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
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -k breaches -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add _compute_breaches helper for caviar"
```

---

## Task 4: `caviar_i` (Capa 2 — estimación de un solo tau)

**Files:**
- Modify: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_caviar.py`:

```python
from auxi.caviar import caviar_i


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
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -k caviar_i -q`
Expected: FAIL con `ImportError: cannot import name 'caviar_i'`.

- [ ] **Step 3: Añadir `caviar_i` a `auxi/caviar.py`**

Insertar una nueva sección de capa 2 y la función. Añadir el separador y la función después de `_compute_breaches`:

```python
# =============================================================================
# CAPA 2 - ESTIMACIÓN (hermanas de q_reg / multiple_q_regs)
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
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -k caviar_i -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add caviar_i estimator returning (reg, indicators)"
```

---

## Task 5: `multiple_caviar_i` (Capa 2 — tabla multi-cuantil)

**Files:**
- Modify: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_caviar.py`:

```python
from auxi.caviar import multiple_caviar_i


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
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -k multiple -q`
Expected: FAIL con `ImportError: cannot import name 'multiple_caviar_i'`.

- [ ] **Step 3: Añadir `multiple_caviar_i` a `auxi/caviar.py`**

Insertar después de `caviar_i`:

```python
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
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
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
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -k multiple -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Ejecutar la suite completa de capas 1-2**

Run: `python -m pytest tests/test_caviar.py -q`
Expected: PASS (todos, ~12 passed).

- [ ] **Step 6: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add multiple_caviar_i producing master_df table"
```

---

## Task 6: `plot_breach_diagnostics` (Capa 3 — sub-función atómica)

**Files:**
- Modify: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test de humo que falla**

Añadir a `tests/test_caviar.py` (al inicio del archivo, fijar backend no interactivo):

```python
import matplotlib
matplotlib.use("Agg")  # backend sin ventana para tests
import matplotlib.pyplot as plt

from auxi.caviar import plot_breach_diagnostics


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
```

Nota: el `import matplotlib; matplotlib.use("Agg")` va al principio del archivo de test, antes de cualquier `import matplotlib.pyplot`. Si ya hubiera un import de pyplot arriba por tareas previas, mover estas dos líneas por encima de él.

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -k breach_diagnostics -q`
Expected: FAIL con `ImportError: cannot import name 'plot_breach_diagnostics'`.

- [ ] **Step 3: Añadir `plot_breach_diagnostics` a `auxi/caviar.py`**

Insertar la sección de capa 3 y la función después de `multiple_caviar_i`:

```python
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
    ax.set_title(f"Fronteras cuantílicas & breaches (total={n_breaches})")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"Valor de {vars_y}")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -k breach_diagnostics -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add plot_breach_diagnostics panel for caviar"
```

---

## Task 7: `plot_caviar_i_results` (Capa 3 — orquestador dashboard)

**Files:**
- Modify: `auxi/caviar.py`
- Test: `tests/test_caviar.py`

- [ ] **Step 1: Escribir el test de humo que falla**

Añadir a `tests/test_caviar.py`:

```python
from auxi.caviar import plot_caviar_i_results


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
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `python -m pytest tests/test_caviar.py -k caviar_i_results -q`
Expected: FAIL con `ImportError: cannot import name 'plot_caviar_i_results'`.

- [ ] **Step 3: Añadir `plot_caviar_i_results` a `auxi/caviar.py`**

Insertar después de `plot_breach_diagnostics`:

```python
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
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
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
```

- [ ] **Step 4: Ejecutar el test para ver que pasa**

Run: `python -m pytest tests/test_caviar.py -k caviar_i_results -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Ejecutar la suite completa**

Run: `python -m pytest tests/test_caviar.py -q`
Expected: PASS (todos, ~14 passed).

- [ ] **Step 6: Commit**

```bash
git add auxi/caviar.py tests/test_caviar.py
git commit -m "feat: add plot_caviar_i_results dashboard orchestrator"
```

---

## Verification Final

- [ ] **Suite completa verde**

Run: `python -m pytest tests/ -v`
Expected: todos los tests PASS, sin warnings de import.

- [ ] **El panel del usuario nunca se muta**

Confirmado por `test_compute_quantile_bounds_does_not_mutate_input` y
`test_caviar_i_does_not_mutate_input`. No se añade ninguna columna persistente
al `df` de entrada en ninguna función pública.

- [ ] **Reuso real de qreg.py**

`caviar.py` importa `q_reg`, `plot_quantile_coefs`, `plot_pseudo_r2` de
`auxi/qreg.py` — no duplica esa lógica.
