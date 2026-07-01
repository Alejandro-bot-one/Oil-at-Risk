# Tribunal Review — "Oil-at-Risk: A Framework for Studying Oil Price Vulnerabilities"

Reviewer stance: adversarial PhD committee. The aim below is not to praise but to break the thesis wherever it can be broken, so that Alejandro can fix it before a real tribunal does. Points are graded **[FATAL]** (threatens a central claim), **[MAJOR]** (a chapter needs rework), **[MODERATE]** (a section needs work), **[MINOR]** (presentation/proofing), and **[Q]** (a question you must be able to answer on your feet).

---

## 1. Internal inconsistencies that will be found in the first ten minutes

### 1.1 [FATAL] The out-of-sample period does not agree with itself
The forecast-evaluation chapter says the test span is **2 Jan 2023 to 23 Apr 2026**, with **N = 864** origins for the standard model and **N = 517** for the CAViaR variants (Section 6.1, Table 1, and Appendix 11.1). The early-warning chapter says the entropy series is evaluated over **June 2012 to April 2026, 2,167 daily observations** (Section 7.6). These cannot both be true if the entropy is built from the *same* out-of-sample conditional densities as the risk metrics. Either:
- the entropy uses **in-sample** fitted densities (in which case the "early warning" claim is contaminated by look-ahead and collapses), or
- the OOS density series really runs from 2012 and the coverage table is computed on a much shorter sub-window (in which case the two chapters are describing different experiments and you must say so explicitly).

Additionally, 2,167 trading days is roughly 8.6 years, which matches neither "June 2012 – April 2026" (~13.8 years, ~3,480 trading days) nor the 2023–2026 backtest window. **You must reconcile these numbers or the whole empirical section reads as internally incoherent.**

### 1.2 [MAJOR] Two headline confidence levels float through the thesis unlabelled
The forecast backtests are reported at **τ = 0.95** (Table 1: π̂ ≈ 0.938). The risk metrics (VaR/ES) are at **97.5%** (Section 5.9). A reader cannot tell which is "the" model calibration. State once, prominently, that the *distributional* object is evaluated at 0.95 for pinball/coverage and that VaR/ES are read at 0.975 from the fitted density, and keep the notation consistent.

### 1.3 [MAJOR] The diagnosed model is not the deployed model
The in-sample specification uses **RV_{t−1}**; the out-of-sample model uses **RV_t** (Section 5.2/5.4). Every in-sample diagnostic (ARCH-LM, QAR unit root, Koenker–Machado R¹, Wald, quantile crossing) therefore validates a specification that is *not* the one generating the risk metrics and the early-warning signal. The diagnostics chapter certifies the wrong equation. At minimum, re-run the core diagnostics on the RV_t specification, or justify why the swap is innocuous.

---

## 2. Econometric validity of the backtests

### 2.1 [FATAL] Overlapping h = 5 forecasts invalidate the iid-based coverage tests
Kupiec, Christoffersen, the KS test on the PIT, and the traffic-light test all assume an **iid** hit/PIT sequence. Direct 5-step forecasts produce **overlapping** windows, so the hit sequence is MA(4)-dependent by construction. Consequences:
- Kupiec unconditional coverage: variance of the breach count is understated, so the LR statistic is size-distorted.
- Christoffersen: it only models **first-order** Markov dependence, but your induced dependence runs to lag 4. "Passing" it certifies nothing about lags 2–4.
- KS on the PIT: critical values assume iid uniform; with serially dependent PITs the nominal 5% is wrong.
You already apply a HAC correction (bandwidth h−1) to Diebold–Mariano precisely because of this MA(h−1) structure. The same logic condemns the coverage tests, yet they are run naively. Either use block-bootstrap/HAC-robust versions or restrict to non-overlapping origins (h-spaced), and re-report.

### 2.2 [MAJOR] RMSE and MAPE are improper scores for a quantile
Table 3 reports RMSE = 4.23 and **MAPE ≈ 892%–902%** for a 0.95-quantile forecast. MAPE explodes because daily returns are near zero (division by ~0); an ~900% figure is meaningless and should be deleted. RMSE is not a proper scoring rule for a quantile either. Report the **pinball loss** (which you correctly identify as the unique proper score) and drop the rest, or you invite the charge that you do not know what you are scoring.

### 2.3 [MAJOR] The three models' losses are not compared on a common sample
Standard model: N = 864 (avg tick loss 0.2380). CAViaR variants: N = 517 (0.2290, 0.3317). The averages are over **different windows**, so "CAViaR-indicator has the lowest tick loss" is not a like-for-like statement. Diebold–Mariano requires a **paired** loss differential on the **same** origins. Restrict all three to the common overlapping window before comparing, then re-state the ranking.

### 2.4 [MODERATE] Selecting CAViaR-indicator contradicts your own DM result
Every DM comparison is insignificant even at 10% (Table 2), yet CAViaR-indicator is "carried forward as a tail-robust complement" on the basis of a **statistically insignificant** loss edge computed on a different sample (see 2.3). This is post-hoc selection against your own evidence. Either report it as "no model dominates" or provide a significance-backed criterion.

### 2.5 [Q] Traffic-light zones at 97.5%
The classic Basel traffic light (green 0–4, yellow 5–9, red ≥10) is defined for **99% VaR over 250 days**. At α = 2.5% the expected count over 250 days is ~6.25, so the classic thresholds do not apply. You say you recompute zones from the binomial — good, but state the recalibrated boundaries explicitly, otherwise readers will assume the wrong ones.

---

## 3. The early-warning chapter is the weakest and most attackable

### 3.1 [FATAL] Over-differencing of an already-stationary return
Section 7.3 states that **both** the entropy series **and the Brent return** are first-differenced before the CCF and Granger tests, "to ensure stationarity." The return series is *already* stationary (it is itself a log-difference of price). Differencing it again is **over-differencing**: it injects a spurious MA(1) unit root (a −1 root), fabricating negative autocorrelation and distorting every cross-correlation and F-statistic that follows. This alone can generate the large, "highly significant" Granger F-statistics you report. Do not difference the return. Difference the entropy only if a unit-root test on the entropy demands it, and report that test.

### 3.2 [FATAL] The claimed "lead" is partly mechanical
The entropy at origin t is the KL divergence of the **h = 5-step-ahead** conditional density, i.e. an object built from information at t to describe t+5. It is *designed* to look forward five days. Finding that it "leads" returns by 7–16 days is therefore partially built into the construction, not a discovery. You must show the lead survives after removing the mechanical horizon offset (e.g. align the entropy to the date its density describes, t+h, before computing the CCF), otherwise the early-warning result is an artefact of the pipeline.

### 3.3 [MAJOR] Effect sizes are weak and oversold
Peak cross-correlations are r = 0.311 (full), 0.285 (right), and **−0.149** (left). A |r| ≈ 0.15 signal, even if "significant" under a Bartlett band of ±0.042, is a very thin basis for an "operationally meaningful" trading/hedging tool. The prose ("sufficiently long to permit meaningful risk-management responses") overstates what r ≈ 0.15–0.31 can support. Temper the language and report the implied R² (≈ 1–10%).

### 3.4 [MAJOR] Granger with up to 100 lags on daily data
p_max = 100, selected p = 24–33, giving 67 parameters. On over-differenced daily series (3.1) with an entropy regressor that is itself smoothed/autocorrelated (built from rolling densities), high-order Granger F-tests are exactly where spurious predictability appears. You concede overfitting "cannot be excluded" but then still headline the result. Add a genuine **out-of-sample** forecasting horse-race (does adding lagged entropy lower OOS pinball/MSE?) — Granger in-sample F-tests are not evidence of forecastability.

### 3.5 [MODERATE] Baseline construction may leak
Section 5.8/7.2 mention an unconditional JSU baseline "fitted by MLE to the historical returns." If "historical" means the full 1990–2026 sample, then an OOS early-warning series is being compared to a **full-sample** (future-inclusive) reference — look-ahead. Confirm the baseline is estimated only on data available at each origin (or on a pre-sample hold-out), and say so.

### 3.6 [MODERATE] Misattributed normality claim
Section 7.2 says Brent returns are "approximately Gaussian in the central mass, a regularity the Kolmogorov–Smirnov tests in Section 5.6 are consistent with." The KS test in 5.6 is a **PIT-uniformity** test on the conditional JSU fit; it says nothing about the marginal normality of raw returns. Two different tests are being conflated. Remove or substantiate with an actual normality test on returns.

---

## 4. Distribution-fitting step (JSU via MDE)

### 4.1 [MAJOR] Contradiction on "extreme" quantiles
Section 5.9 argues the framework avoids estimating extreme quantiles directly ("estimating the 99th percentile with precision requires very large samples") and instead reads 2.5% from the fitted density. But the quantile set fed to the MDE is **{0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99}** — you *do* estimate the 1% and 99% quantiles directly and rely on them. Either drop 0.01/0.99 from the QR step (and genuinely interpolate the tails), or drop the claim that you avoid extreme-quantile estimation. As written it is self-contradictory.

### 4.2 [MAJOR] MDE objective is unweighted and tail-dominated
Eq. (5.x) minimises the **unweighted** sum of squared quantile distances over K = 7 points. In return units the 1%/99% points are an order of magnitude larger than the 25%/50%/75% points, so the sum of squares is dominated by the tails and the central fit can be poor — the opposite of what you want if you also read a 2.5% VaR and integrate an ES. Justify the equal-weight L2 choice, or use a weighted MDE (e.g. weight by 1/quantile-spacing or by density) and show robustness.

### 4.3 [MODERATE] Four parameters, seven points, three residual d.f.
With K = 7 and four JSU parameters, the "estimator" is close to interpolation, and the 2.5% VaR is anchored by only two nearby fitted quantiles (1% and 5%). Report the sensitivity of VaR/ES to (i) the quantile grid, (ii) dropping the extreme points, (iii) an alternative family (skewed-t, which you cite but never actually run head-to-head despite Contribution 3 claiming a comparison). **Contribution 3 promises a skewed-t vs JSU comparison "within the same pipeline"; the results section never delivers the comparison.** That is a promised deliverable that is missing.

### 4.4 [MODERATE] KL against a Normal baseline is mostly a kurtosis meter
KL(p‖q) with q Gaussian and p heavy-tailed is dominated by the region where q → 0, i.e. it mechanically tracks tail-fatness/kurtosis rather than "regime shift" per se. On a finite 5,000-point grid the tail integrals can be numerically unstable. Report the grid-truncation range and a convergence check; otherwise the entropy series may be measuring numerics as much as economics.

---

## 5. Formula and notation errors

### 5.1 [MAJOR] The severity variables are written as booleans, not magnitudes
Eqs. (5.x) define `upside_severity_t = max(0, y_t − Q_high(...)) ≥ 0` and the downside analogue. The trailing "**≥ 0**" turns a magnitude into a truth value (max(0,·) is trivially ≥ 0), which directly contradicts the surrounding text ("the absolute size of the exceedance"). Delete the "≥ 0". As printed, Model 3 is mathematically ill-defined.

### 5.2 [MAJOR] Breach regressors: prove there is no look-ahead
`upside_breach_t = 1(y_t > Q_high(y_t | x_{t−h}))`. The conditioning notation `Q(y_t | x_{t−h})` is opaque, and it is not visually obvious that the flag entering the origin-t forecast of y_{t+h} uses **only** information dated ≤ t. Rewrite with explicit time stamps showing the breach is lagged relative to the forecast target, and state the exact availability lag. This is the single most likely place a committee will suspect leakage.

### 5.3 [MODERATE] "The quantile of an expectation is not the expectation of a quantile"
Section 5.4. This phrasing is muddled. The correct point is that **conditional quantiles do not obey a law of iterated expectations**, so a multi-step quantile cannot be obtained by chaining one-step quantile forecasts. Rephrase precisely.

### 5.4 [MODERATE] Koenker–Machado degrees of freedom
Section 10.1 states T·V(τ)·R¹(τ) is asymptotically χ² with **k** d.f. Check whether it should be **k−1** (regressors excluding the intercept), and define V(τ). State the block length used in the moving-block bootstrap for the Wald test (Section 10.2) — it is currently unspecified, and the result depends on it.

### 5.5 [MINOR] "Realised volatility" is a misnomer
If RV is a rolling standard deviation of daily returns (as "constructed from the historical variation of crude returns" implies), it is **not** realised volatility in the Andersen–Bollerslev sense (which requires intraday data). Call it rolling/historical volatility, or state the intraday source.

---

## 6. Economics and identification

### 6.1 [MAJOR] Endogeneity / reverse causality of GPR and oil
The GPR index is news-based; oil-price spikes are themselves reported as geopolitical events, so GPR and oil returns are plausibly **simultaneously** determined. The strict-exogeneity assumption Q_τ(u|X)=0 (Appendix 9) is asserted but never defended against this. Address it (lagging GPR helps but does not fully solve it; discuss, or instrument).

### 6.2 [MAJOR] The central economic result is never quantified in the text
The thesis's reason to exist is: does geopolitical risk move the tails of oil returns, and in which direction? Yet there is **no coefficient table** with β₁(τ), signs, magnitudes, and standard errors. The finding is relegated to a figure and a Wald "reject equality." State the estimated sign and size of the GPR effect at, say, τ ∈ {0.05, 0.50, 0.95}, with inference.

### 6.3 [Q] Sign of the effect vs the mechanism
The conclusion says geopolitical risk "compress[es] the lower tail more aggressively than the upper." But your own literature framing (supply-disruption premium, right-tail fattening, precautionary demand) predicts the **right** tail should react. Reconcile: is GPR widening downside, upside, or both? The narrative and the reported direction must agree.

### 6.4 [MAJOR] No external benchmark model
Every model compared is inside your own family (QR, QR+indicator, QR+severity). A committee will ask: does this beat a **standard** tail-risk model — GARCH with skewed-t, EWMA-VaR, EVT-POT, or a direct CAViaR VaR read without the JSU-MDE detour? Without an outside benchmark you cannot claim the pipeline is worth its complexity. In particular, justify the extra MDE-density estimation error versus reading VaR directly off the QR quantiles.

### 6.5 [MODERATE] Horizon mismatch in the VaR comparison
The conditional VaR is a **5-day-return** quantile (correctly, no √t scaling). The unconditional benchmark is "historical simulation over a rolling 1,008 window" — if that is a **1-day** VaR, Figure varcvar compares 5-day vs 1-day objects. Use overlapping 5-day returns for the historical benchmark, or state the scaling.

### 6.6 [MODERATE] GPRD-MA7: calendar vs trading days
Weekends/holidays are removed from the return series, so t indexes **trading** days, yet GPRD-MA7 = (1/7)Σ_{j=0}^{6} GPRD_{t−j} is defended as "a calendar week." Seven trading-day lags span ~9 calendar days, and dropping weekend GPR values discards information (geopolitical news does not stop on weekends). Clarify whether the MA is over calendar or trading days and defend the alignment.

---

## 7. Missing standard content a tribunal will demand

### 7.1 [MAJOR] No descriptive statistics or stationarity tests for the regressors
There is no table of mean/sd/skew/kurtosis, no ADF/KPSS for GPRD-MA7, ΔUSDI, RV, ΔBDI, no correlation matrix. The QAR unit-root test is run only on Brent returns. Add the standard data-diagnostics table.

### 7.2 [MAJOR] The multicollinearity exclusion claim is unsupported
Section 5.2 says candidate regressors were dropped "on grounds of multicollinearity" but no **VIFs** or correlation evidence is shown. Either show the numbers or soften the claim.

### 7.3 [MODERATE] The daily-retraining robustness check has no numbers
Section 5.10 asserts daily vs 30-day retraining is "nearly indistinguishable" but reports no metric. Give the max/mean absolute difference in VaR/ES, or a small table.

### 7.4 [MINOR] Contribution 6 (the code) is only "intended to be publicly available"
A contribution must be delivered. Provide the repository link/DOI, or reclassify it as future work.

---

## 8. Factual / citation issues (verified against primary sources)

### 8.1 [MAJOR] The "1,008-day / 4-year Basel window" is not a Basel prescription
You repeatedly justify W = 1,008 (four trading years) as "the standard calibration window prescribed by Basel III for internal-model VaR" (Sections 5.4, 5.9, 5.10, 7.2). Basel FRTB actually calibrates the 97.5% ES to a **250-day (12-month)** current period and a **250-day stressed** period; the internal-models approach uses 250-day windows, not 1,008 days. Four years is a reasonable **modelling** choice, but attributing it to Basel is incorrect. Fix the citation: keep 1,008 days as your own design choice and stop calling it the Basel standard.
Sources: BIS d457 explanatory note; BIS d436 (Revisions to minimum capital requirements for market risk); FRTB (Wikipedia summary).

### 8.2 [MINOR] Traffic-light provenance
The classic traffic light (green 0–4 / yellow 5–9 / red ≥10) is the 1996 Basel amendment for **99% VaR / 250 days**. You apply it at 97.5%; cite the recalibration (Costanzino–Curran you already cite for ES) and give the 2.5% zone boundaries.

### 8.3 [Q] 2025–2026 working papers
Kilian–Plante–Richter (2026), Verduzco-Bustos–Zanetti (2026), Brignone–Gambetti–Ricci (2025), Pinchetti (2024) are cited as working papers. Confirm the versions, that titles/years match the latest drafts, and that you are not over-relying on unpublished results for load-bearing claims.

---

## 9. Presentation and proofing

- **[MINOR]** Numerous figure captions read "Enter Caption" (Graphical Annex: Brent evolution, the JSU 3D surface figure, the Iran figure, the OOS-horizons figure). Incomplete.
- **[MINOR]** Filenames/typos leak into the document: `insaple_specification.png`, "Value -at-Risk", "CAViAR", "christofen", `fig:irance`, "Iran Wars Eve." Proofread; use professional captions.
- **[MINOR]** Introduction overstates OLS behaviour: outliers do not generally make "slope coefficients collapse close to zero" — they inflate variance and can bias in either direction. Rephrase.
- **[MINOR]** The 3D density surface (Figure jsu3d) is described as **in-sample**; an out-of-sample example would be more persuasive given the thesis's OOS emphasis.

---

## 10. The ten questions to rehearse for the defence

1. Why does the entropy sample (2,167 obs from 2012) not match the backtest sample (864 obs from 2023)? Which densities feed the early-warning test?
2. Your h = 5 forecasts overlap. Why are Kupiec/Christoffersen/KS run as if the hit sequence were iid?
3. You differenced an already-stationary return before Granger/CCF. Does the result survive without over-differencing?
4. How much of the entropy "lead" is mechanical, given the entropy is a 5-day-ahead object by construction?
5. Show me the sign and magnitude of the GPR coefficient in the lower vs upper tail, with standard errors.
6. Where is the promised skewed-t vs JSU comparison (Contribution 3)?
7. Why is the MDE unweighted, and how sensitive is the 2.5% VaR to the quantile grid and to the 1%/99% points?
8. Prove the CAViaR breach flags use no future information.
9. What does the JSU-MDE pipeline add over a standard GARCH-skewed-t or a direct CAViaR VaR?
10. On what basis is 1,008 days "the Basel standard" when FRTB uses 250-day windows?

---

## Overall assessment

The framework is coherent in ambition and the writing is fluent, but as it stands the empirical spine has three load-bearing cracks: (a) an internal sample-size contradiction between the risk chapter and the early-warning chapter, (b) iid-based backtests applied to overlapping multi-step forecasts, and (c) an early-warning result resting on over-differenced series and a partly mechanical lead. Any one of these, unaddressed, is enough to dominate a viva. They are all fixable, and none requires abandoning the thesis: reconcile the samples, make the backtests HAC/block-robust or non-overlapping, redo the CCF/Granger without over-differencing and with an OOS horse-race, add a coefficient table and an external benchmark, and correct the Basel attribution. Do that and the contribution stands on much firmer ground.
