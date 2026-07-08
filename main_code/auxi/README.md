# auxi/ — Mapa de módulos

Backend de utilidades del TFM. Organizado por propósito.

## Motores y estimadores
- **qreg.py** — Quantile-regression engine (`q_reg`, `multiple_q_regs`, plotters), direct-forecasting estimators (`direct_forecasting`, `insample_direct_forecasting`, `get_oos_predictions`), y sus plotters.
- **caviar.py** — CAViaR con indicadores binarios. `compute_breach_indicators` (h-aware, sin lookahead) para direct forecasting; `caviar_i`, `multiple_caviar_i` para análisis de especificación in-sample.
- **distribution_analysis.py** — Fitters JSU y Skew-t (`fit_jsu`, `fit_skewt`), MDE (`mde_jsu_weighted`, `mde_distfit_skewt`), PDFs, generadores de parámetros OOS, comparators.
- **predictive_density.py** — Densidades predictivas sobre las distribuciones ajustadas.
- **risk_metrics.py** y **risk_metrics_boosted.py** — VaR y CVaR condicional/histórico.
- **vulnerability_metrics.py** — Entropía de cola y skewness en el tiempo.
- **data.py** — Carga y actualización de datos (`import_data`, `update_brent`).
- **descriptive.py** — Estadística descriptiva y selección de ventana.

## Diagnósticos (subpackage)
- **diagnostics/specification.py** — Tests del modelo cuantílico: DQ, Wald, Q-ARCH, QAR stability.
- **diagnostics/direct_forecasting.py** — Evaluación de pronóstico: Kupiec, Christoffersen, fallout, pinball cross-horizon, residual ACF.
- **diagnostics/distribution_fitting.py** — Goodness-of-fit para JSU/Skew-t: KS, PIT calibration, fit-and-diagnose bundles.
- **diagnostics/series.py** — Estacionariedad (ADF), filtros de tendencia (Hamilton).

Acceso: `import auxi.diagnostics as diags`. Todas las funciones públicas se reexportan en `__init__.py`.

## Documentación de diseño
- `docs/superpowers/specs/2026-06-25-caviar-indicator-design.md` — Diseño de CAViaR con indicadores.
- `docs/superpowers/specs/2026-06-26-backend-reorg-design.md` — Diseño del reorg actual.
