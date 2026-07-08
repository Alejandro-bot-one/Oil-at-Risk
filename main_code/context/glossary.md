# Glossary — Oil-at-Risk TFM

> Context file for Claude. Two parts: (1) how Alejandro communicates, so I read his intent
> correctly, and (2) the domain, data, and code vocabulary of this project, so terms mean the
> same thing to both of us. Add a term the first time it causes ambiguity.

## How Alejandro communicates

- **Who he is.** Economic researcher and consultant writing a master's thesis (TFM = *Trabajo
  Fin de Máster*) for the MQuEA programme (MSc in Quantitative Economic Analysis, UAM). Strong
  in statistics, econometrics, maths, and programming; wants the *full picture, top to bottom*,
  and values technical depth plus creativity.
- **Address him as "Alejandro" in every reply.** (Project rule.)
- **Language.** He writes to me mostly in **English**, but the project's planning docs, specs,
  and some docstrings are in **Spanish** (e.g. `CAPA` = layer, `frontera` = bound/boundary,
  `efímero` = ephemeral, `borrador` = draft, `correccion` = correction). Code-surface docstrings
  are English. Deliverables and replies: English unless he switches.
- **Style.** Fast, informal, with frequent typos ("porpouse", "frontend"/"fronend", "erros",
  "existint", "Glosary", "yoursel"). Read for **intent**, don't get snagged on spelling. He
  prefers **concise, direct** answers — minimal preamble, no filler, cut words that don't add
  meaning.
- **Standards he holds me to.** Don't answer unless highly confident; think twice and look for
  flaws in my own reasoning; corroborate with more than one source; say so when I don't know.
- **Names that may come up.** "Pilar" is a supervisor/reviewer (see
  `deliverable/drafts/borrador_0_correcion pilar.pdf`).

## Methodology & domain terms

- **Oil-at-Risk (OaR)** — this project's adaptation of Growth-at-Risk to crude oil; the
  downside (left-tail) risk of Brent returns. In `risk_metrics.py`, left tail = OaR.
- **Growth-at-Risk (GaR)** — the Adrian, Boyarchenko & Giannone (2019) "Vulnerable Growth"
  framework: model the *conditional quantiles* of a target, fit a density, read tail risk over
  time. Here the upside (right tail) is labelled GaR.
- **Quantile regression (QR)** — estimating conditional quantiles `Q(y | x; tau)` instead of
  the mean; the engine of the whole project (`q_reg`, Koenker & Bassett 1978).
- **tau (τ)** — the quantile level being estimated, e.g. 0.05, 0.5, 0.95.
- **quantile grid / 21-point grid** — the set of taus swept across a regression. Two defaults
  coexist: `multiple_q_regs` still uses the sparse `[0.05, 0.25, 0.50, 0.75, 0.95]`, while the
  orchestrators and caviar table functions use the dense 21-point grid `[0.01, 0.05, 0.10, …,
  0.95, 0.99]` (`[0.01] + np.round(np.arange(0.05, 0.951, 0.05), 2) + [0.99]`).
- **CAViaR** — Conditional Autoregressive Value at Risk (Engle & Manganelli, 2004); here
  realized as quantile regression augmented with binary **breach indicators**.
- **Breach** — a row where the realized value falls outside a quantile bound.
  `upside_breach = 1{y > Bound_High}`, `downside_breach = 1{y < Bound_Low}`; `NaN` where the
  bound is not computable (never silently 0).
- **Severity (breach severity)** — the absolute distance from the realized value to the
  violated quantile boundary. `upside_severity = max(0, y_t - Q_high)` (>= 0),
  `downside_severity = max(0, Q_low - y_t)` (>= 0). Both non-negative; zero when no breach.
  Used in the `caviar_s` variant. Contrast with the binary breach in `caviar_i`.
- **Bounds / boundaries (`Bound_Low`, `Bound_High`)** — the low- and high-tail conditional
  quantile predictions that define the breach region (`frontera` in the Spanish docs).
- **Direct forecasting (DF)** — forecasting `y_{t+h}` by quantile-regressing an **h-shifted**
  target; it *is* quantile regression, hence it lives in `qreg.py`/`caviar.py`.
- **h (horizon)** — forecast horizon in periods ahead. The first `h` rows of an h-lagged series
  are `NaN`.
- **In-sample vs OOS** — in-sample fits on the whole sample (specification work, the predictive
  density surface); **out-of-sample (OOS)** fits on a training slice and predicts forward
  (forecast evaluation). OOS must be **lookahead-free**.
- **Lookahead (bias)** — letting future data inform a past prediction; the cardinal sin of any
  backtest here. See `known_errors.md` #1.
- **train_fraction / test_start_date** — the two ways to set the OOS train/test split.
- **MDE (Minimum Distance Estimation)** — fitting a parametric density to the *forecasted
  quantiles* (the quantile fan) by minimizing distance, turning QR output into a smooth density.
- **JSU (Johnson SU)** — Johnson (1949) flexible four-parameter distribution; one of the two
  sanctioned return densities. Params often `(a, b, loc, scale)` → `cond_a, cond_b, cond_loc,
  cond_scale` in outputs.
- **Skew-t (Azzalini)** — Azzalini & Capitanio (2003) skewed Student-t; the second sanctioned
  density.
- **Predictive density surface** — the 3-D waterfall of stacked in-sample conditional densities
  over time, replicating Adrian et al. (2019) Figure 1 (`predictive_density.py`).
- **VaR / CVaR** — Value at Risk (a tail quantile) and Conditional VaR / Expected Shortfall
  (mean beyond the quantile). Computed for both tails, **conditional** (from the fitted JSU) and
  **unconditional** (historical simulation).
- **Tail entropy / relative entropy / KL divergence** — Kullback–Leibler divergence of the
  fitted density from a Normal baseline, decomposed **Full / Left / Right**, used as a
  vulnerability measure (`vulnerability_metrics.py`).
- **Expanding vs rolling window** — OOS backtests either grow the training window from the start
  (expanding) or use a fixed-size most-recent window (rolling). The direct-forecasting evaluation
  uses a **rolling** window of **1000** observations (`window_size=1000`): it slides forward one
  origin at a time, dropping the oldest row as it adds a new one, so the model forgets old regimes.
- **Forecast origin** — the date `t` from which a forecast is made; the model is trained on the
  window ending just before `t` and predicts `y_{t+h}`. The rolling backtest steps `t` across the
  whole test set.
- **Rolling-window pinball loss** — the average pinball (tick) loss over all rolling forecast
  origins for a given horizon and quantile; the fair OOS error metric produced by
  `compute_rolling_pinball` (one line per tau in `plot_rolling_pinball`). Replaced the old
  single-split `OOS_Loss`/`IS_Loss` metric — see `known_errors.md` #9.
- **`eval_taus`** — the notebook's list of quantiles evaluated together by the rolling backtest,
  `[0.05, 0.50, 0.95]` (downside tail / median / upside tail).
- **PIT (Probability Integral Transform)** — maps realized values through the predicted CDF;
  uniform PIT ⇒ well-calibrated density. Basis of PIT calibration diagnostics.
- **Pinball / tick loss** — the asymmetric quantile loss; the scoring rule for quantile
  forecasts (`pinball_loss`).
- **Fallout** — breach/exception rate of a quantile forecast over time.
- **Coverage tests** — Kupiec (unconditional coverage), Christoffersen (conditional coverage /
  independence): is the breach frequency right, and are breaches independent over time?
- **Diebold-Mariano (DM) test** — test for equal predictive accuracy between two models
  (Diebold & Mariano, 1995). Regresses the loss differential $d_t = L_{1,t} - L_{2,t}$ on a
  constant with HAC standard errors (bandwidth $h-1$, rectangular kernel). Alpha < 0 means
  Model 1 has lower average loss. `compute_dm_comparison` is the orchestrator.
- **Tick loss series** — element-wise (non-averaged) asymmetric quantile loss; `tick_loss_series`
  returns a vector whose mean equals `pinball_loss`. Used as input to the DM test.
- **Long-run variance (LRV)** — HAC-consistent variance estimator that accounts for the
  MA(h-1) serial correlation in h-step direct forecast errors. The DM test uses the
  rectangular kernel: $\hat\gamma_0 + 2\sum_{k=1}^{h-1}\hat\gamma_k$.
- **DQ test** — Engle–Manganelli Dynamic Quantile test for quantile forecast adequacy.
- **Wald test / Q-ARCH test / QAR(X) stability** — specification diagnostics for the quantile
  model (`diagnostics/specification.py`).
- **ADF** — Augmented Dickey–Fuller stationarity test (`diagnostics/series.py`).
- **Hamilton filter** — Hamilton's (2018) regression-based alternative to the HP filter for
  trend/cycle decomposition.
- **CCF (Cross-Correlation Function)** — `r(h) = cor(X[0..N-h-1], Y[h..N-1])` for h >= 0.
  Convention: h > 0 means X leads Y. `compute_ccf` in `diagnostics/ews.py`.
- **Granger causality** — F-test comparing restricted (Y ~ own lags) vs unrestricted (Y ~ own
  lags + X lags) model. Lag selected by AIC/BIC. `granger_causality_test` in `diagnostics/ews.py`.
- **Anticipation test** — combined CCF + Granger test for one (X, Y) pair; `compute_anticipation_test`.
- **EWS battery** — runs the anticipation test across multiple indicators against a single target;
  `compute_ews_battery` returns a summary DataFrame.
- **Coherence test** — pairwise CCF among indicator series to check internal consistency;
  `compute_coherence_test`.

## Data series & variables

- **Brent** — Brent crude price (FRED `DCOILBRENTEU`); `Brent_Return` = `pct_change()*100`.
- **VOIL / OVX** — CBOE crude oil volatility index (FRED `OVXCLS`).
- **VIX** — CBOE equity volatility index (FRED `VIXCLS`).
- **GPR / GPRD** — Geopolitical Risk index (Caldara & Iacoviello); GPRD is the daily variant.
  A central regressor (see `gpr_descriptive.ipynb`).
- **EPU** — Economic Policy Uncertainty index (Baker, Bloom & Davis).
- **REIA, BADI, controls** — additional control series in `data/raw/` (inflation, US oil
  production, US weekly stocks, weekly product supplied, etc.) merged into the panel.
- **Panel** — the merged, feature-engineered dataset (`daily_panel.csv`, `monthly_panel.csv`);
  the single input every estimator consumes.
- **Realized_Volatility** — squared Brent return; lags written `Brent_Return (t-1)`, moving
  averages `Realized_Volatility_MA7`.

## Code-specific terms

- **`master_df`** — the tidy long-format results table (`Dependent Variable`, `Regressor`,
  `Tau`, `Coefficient`, `Significance`, `Pseudo R-Squared`) produced by `multiple_q_regs` /
  `multiple_caviar_i`; the numeric source feeding both tables and plots.
- **specification** — a chosen `(vars_x, vars_y, controls, quantiles, h)` combination under test.
- **`vars_x` vs `x`** — `vars_x` is the full regressor list; `x` is a single regressor (helpers
  coerce a string to a one-element list).
- **`_i` suffix** — indicator (binary) variant. **`_s` suffix** — severity (absolute distance)
  variant. Both follow the three-layer architecture.
- **ephemeral indicators** — breach columns created inside a call on a `.copy()` and discarded;
  the user's panel is never mutated.
- **diags** — the conventional alias: `import auxi.diagnostics as diags`.
- **fc** — alias for the quantile/forecasting engine in notebooks: `import auxi.qreg as fc`
  (historically `auxi.forecasting`, now deleted).
- **boosted** — the vectorized `risk_metrics_boosted.py`, same API/numbers as `risk_metrics.py`.
- **superpowers** — the `docs/superpowers/` spec+plan workflow folder (design records).
