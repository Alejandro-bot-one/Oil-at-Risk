# Oil-at-Risk

**A Framework for Studying Oil Price Vulnerabilities**

Master Thesis (TFM) — MSc in Quantitative Economic Analysis (MQuEA), Universidad Autónoma de Madrid
Author: **J. Alejandro Rodríguez Garrido**
Supervisors: Diego Eduardo Fresoli and Pilar Poncela Blanco
Academic Year 2025–2026 · Submission: June 2026

---

## Abstract

Crude oil prices act as a key macroeconomic variable, often serving as a primary input in global inflation dynamics, industrial production costs, and international trade balances. The propagation of geopolitical tensions through energy markets can generate heightened volatility, which carries macroeconomic consequences and potential systemic risks for policymakers, firms, and institutional investors. Many traditional econometric approaches assume that the average historical effect of a geopolitical shock adequately represents the underlying stochastic process. However, in the context of crude oil prices and geopolitical shocks, where data tend to exhibit substantial heteroskedasticity, asymmetric responses, and influential tail observations that can distort linear estimates, this assumption may prove overly restrictive.

To address the limitations of conditional mean forecasting, this research adapts the **Growth-at-Risk** framework (Adrian, Boyarchenko & Giannone, 2019) to model the distribution of future crude oil price growth. Rather than focusing solely on average baseline effects, the analysis estimates how fluctuations in geopolitical risk shift specific quantiles of the return distribution, capturing heterogeneity across outcomes and describing both downside and upside market risks. The empirical strategy relies on the European Brent crude oil spot price and a dictionary-based Geopolitical Risk Index. Drawing on daily returns and a set of control variables — including a realised-volatility proxy, the U.S. dollar index, and a shipping-cost measure — the study specifies multiple quantile regressions to isolate the tail-risk imprint of geopolitical uncertainty at a one-trading-week horizon. A backtesting framework combining the Kupiec unconditional-coverage test and the dynamic quantile test evaluates the conditional risk measures against realised outcomes, and an entropy-based indicator is examined for early-warning content. The full analytical pipeline is implemented in a self-contained, open-source Python module to promote replicability.

**Keywords:** Geopolitical Risk, Brent Crude Oil, Quantile Regression, Growth-at-Risk, Tail Risk, Volatility, Johnson SU, CAViaR, Expected Shortfall, Kullback–Leibler Divergence.
**JEL Codes:** C21, C22, C53, C58, F51, G17, Q43.

---

## What this project is

**Oil-at-Risk (OaR)** adapts the Growth-at-Risk / Vulnerable Growth methodology to the crude oil market. The empirical pipeline is:

1. **Conditioning variables** — geopolitical risk (GPR/GPRD), economic policy uncertainty (EPU), realised volatility, and other financial/macro controls are selected and preprocessed.
2. **Quantile regression** of Brent returns on these risk factors, estimated at multiple quantile levels (5th, 10th, 25th percentiles, among others), including a CAViaR variant where the quantile process is autoregressive on past breaches.
3. **Full conditional distribution** recovered by fitting a smooth density — Johnson SU or Azzalini & Capitanio skew-t — to the estimated quantiles via Minimum Distance Estimation, following Adrian et al. (2019).
4. **Oil-at-Risk** read off the fitted lower tail at the target confidence level and forecast horizon (Value-at-Risk and Expected Shortfall / CVaR, both tails: OaR downside and GaR upside).
5. **Entropy-based early warning system** — Kullback–Leibler tail divergence against a Normal baseline, tested as a leading indicator via cross-correlation and Granger causality.
6. **Backtesting** — Kupiec unconditional coverage, Christoffersen conditional coverage/independence, the Engle–Manganelli dynamic quantile test, rolling-window pinball loss, and Diebold–Mariano tests for equal predictive accuracy across model variants.

The full pipeline is implemented as a self-contained, tested, open-source Python package (`main_code/auxi/`) consumed by a series of Jupyter notebooks — the analysis surface, one notebook per research question.

## Repository structure

```
TFM/
├── main_code/              ← the Python package + notebooks (see below)
├── data/                    ← raw inputs (data/raw/*.xlsx) and generated panels
│   ├── raw/                 ← Brent, VOIL, GPR, EPU, REIA, BADI, controls, inflation, US oil stats
│   ├── daily_panel.csv      ← merged daily panel (generated, cached)
│   └── monthly_panel.csv    ← merged monthly panel (generated, cached)
├── references/              ← the papers the methodology rests on, organized by topic
│   ├── methodology/          Adrian et al. (2019), Azzalini & Capitanio (2003), ...
│   ├── qreg/                 Koenker & Bassett (1978), Koenker inference, ...
│   ├── qar models/           CAViaR, quantile autoregression
│   ├── gpr/                  Caldara & Iacoviello geopolitical risk index
│   ├── oil/                  oil price / macroeconomy literature
│   ├── related literature/   geopolitical oil-price-risk applications
│   └── risk managment/       Diebold, Gunther & Tay (1998)
├── results/                  ← saved figures (results/plots/) and OOS output (results/oos/)
└── _others/                   exploratory side analysis (tourism indicators, not part of OaR)
```

### `main_code/` — the empirical engine

```
main_code/
├── auxi/                    ← BACKEND: importable Python package
│   ├── qreg.py                quantile-regression engine + direct-forecasting estimators
│   ├── caviar.py              CAViaR breach indicators (binary `_i` and severity `_s`, h-aware)
│   ├── distribution_analysis.py  Johnson SU & skew-t fitters (MLE + MDE)
│   ├── predictive_density.py  in-sample predictive density surface (Adrian et al. Fig. 1 replica)
│   ├── risk_metrics.py / risk_metrics_boosted.py  VaR & CVaR, readable vs. vectorized
│   ├── vulnerability_metrics.py  tail KL-divergence (entropy) & time-varying skewness
│   ├── data.py                 FRED download, panel construction, `import_data`
│   ├── descriptive.py          descriptive statistics, rolling-window selection
│   └── diagnostics/            SUBPACKAGE: specification tests, forecast evaluation,
│                                distribution goodness-of-fit, stationarity/trend tests
├── *.ipynb                  ← FRONTEND: descriptive, specification, direct forecasting,
│                                distribution analysis, risk metrics, common factors
├── tests/                   ← pytest suite (synthetic-panel fixture + module tests)
├── docs/superpowers/        ← design specs and implementation plans (the "why")
└── context/                 ← persistent project memory (architecture, conventions,
                                 decisions, known errors, glossary, workflow)
```

For a module-by-module map see `main_code/auxi/README.md`; for the full architectural write-up (data flow, dependency graph, design decisions) see `main_code/context/`.

## Data

- **Brent** crude spot price (FRED `DCOILBRENTEU`); returns are `pct_change() * 100`.
- **GPR / GPRD** — Geopolitical Risk Index, daily variant (Caldara & Iacoviello).
- **EPU** — Economic Policy Uncertainty index (Baker, Bloom & Davis).
- **VOIL / OVX** and **VIX** — CBOE crude oil and equity volatility indices.
- Additional controls in `data/raw/`: REIA, BADI (shipping-cost proxy), inflation rate, US oil production, US weekly stocks, weekly product supplied.

`main_code/auxi/data.py` merges these into daily and monthly panels (`data/daily_panel.csv`, `data/monthly_panel.csv`), the single input every estimator consumes.

## Running the code

The backend has no packaged dependency file yet; the modules import `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `fredapi`, and `tqdm`. Install these in your environment, then from `main_code/`:

```bash
# run the test suite
python -m pytest tests/ -v

# open a notebook (Jupyter or JupyterLab)
jupyter lab direct_forecasting.ipynb
```

Notebooks import the backend rather than redefining estimators inline (`import auxi.qreg as fc`, `import auxi.diagnostics as diags`); a "Restart Kernel → Run All" pass on a notebook is the project's integration/smoke test.

## Key references

| Reference | Role |
|---|---|
| Adrian, Boyarchenko & Giannone (2019), "Vulnerable Growth," *American Economic Review* 109(4) | Core Growth-at-Risk framework |
| Engle & Manganelli (2004), "CAViaR," *Journal of Business & Economic Statistics* 22(4) | CAViaR specification and dynamic quantile test |
| Koenker & Bassett (1978), "Regression Quantiles," *Econometrica* 46(1) | Quantile regression foundations |
| Johnson (1949), "Systems of Frequency Curves," *Biometrika* 36(1/2) | Johnson SU distribution |
| Azzalini & Capitanio (2003), skew-t distribution | Second candidate return density |
| Kupiec (1995), "Verifying Risk Measurement Models," *Journal of Derivatives* 3(2) | Unconditional coverage test |
| Christoffersen (1998), "Evaluating Interval Forecasts," *International Economic Review* 39(4) | Conditional coverage / independence test |
| Diebold & Mariano (1995) | Equal predictive accuracy test |

The full literature base is organized by topic under `references/`.

## Project conventions

This is an academic codebase: every modelling choice is traceable to a reference and named in the relevant module docstring, and reproducibility outranks cleverness. Backtests are lookahead-free by construction (breach bounds and OOS forecasts are always fit on the training slice only). See `main_code/context/conventions.md` and `main_code/context/decisions.md` for the full set of standing design decisions.
