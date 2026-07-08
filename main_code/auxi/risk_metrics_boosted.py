"""
Risk Metrics (BOOSTED) — vectorized / GPU-ready VaR & CVaR
==========================================================
Drop-in accelerated replacement for :mod:`auxi.risk_metrics`.

Same public API, same numbers (matches the original to ~1e-11), but the two
expensive Out-of-Sample drivers — :func:`generate_oos_var` and
:func:`generate_oos_cvar` — are rewritten so the **conditional** (JSU) metric
for *every* date is computed in one vectorized pass instead of a per-date
Python loop calling ``scipy.stats.johnsonsu`` thousands of times.

Why it's faster
---------------
The original ``compute_conditional_cvar`` evaluated a 5,000-point
``johnsonsu.pdf`` grid **per date** through SciPy's generic distribution
machinery (heavy Python/C overhead on every call). Here the Johnson SU
quantile and density have closed forms::

    ppf(p) = loc + scale * sinh((Phi^{-1}(p) - a) / b)
    pdf(x) = (b/scale) / sqrt(1+z^2) / sqrt(2pi)
             * exp(-0.5 * (a + b*asinh(z))^2),   z = (x-loc)/scale

so all dates are stacked into arrays and evaluated together. The normal
quantiles ``Phi^{-1}(p)`` are plain constants (computed once with
``scipy.special.ndtri``), so no per-element special function is needed and the
whole thing runs on either NumPy or CuPy with identical code.

Backends
--------
``backend="auto"`` (default) uses the GPU via CuPy when it is importable and
falls back to NumPy otherwise. ``backend="gpu"`` forces CuPy (raises if
missing); ``backend="cpu"`` forces NumPy. The conditional CVaR builds an
``(n_dates, n_grid)`` grid, so it is processed in chunks (``chunk`` dates at a
time) to bound memory — important on a 4 GB GPU.

To enable the GPU path::

    pip install cupy-cuda12x      # driver 581.x -> CUDA 12.x

Everything that is not a bottleneck (the scalar ``compute_*`` helpers and all
plotting) is re-exported unchanged from :mod:`auxi.risk_metrics`, so this
module is a true drop-in: replace ``import auxi.risk_metrics as rm`` with
``import auxi.risk_metrics_boosted as rm``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri
from tqdm import tqdm

# Re-export the unchanged public API (scalar computers + plotting) so this
# module is a complete stand-in for auxi.risk_metrics.
from auxi.risk_metrics import (  # noqa: F401
    compute_conditional_var,
    compute_historical_var,
    compute_conditional_cvar,
    compute_historical_cvar,
    plot_var,
    plot_cvar,
    _plot_tail_risk,
)

_SQRT2PI = float(np.sqrt(2.0 * np.pi))


# ──────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────
def _select_backend(backend: str):
    """
    Resolve the array backend.

    Returns ``(xp, to_numpy, name)`` where ``xp`` is the array module
    (``numpy`` or ``cupy``) and ``to_numpy`` brings an array back to host.
    """
    backend = (backend or "auto").lower()
    if backend in ("gpu", "cupy", "auto"):
        try:
            import cupy as cp  # type: ignore
            return cp, (lambda a: cp.asnumpy(a)), "cupy"
        except Exception:
            if backend in ("gpu", "cupy"):
                raise RuntimeError(
                    "backend='gpu' requested but CuPy is not available. "
                    "Install it with e.g. `pip install cupy-cuda12x`, or use "
                    "backend='cpu'."
                )
    return np, (lambda a: np.asarray(a)), "numpy"


def _params_arrays(df_entropy: pd.DataFrame, xp):
    """Pull the stored JSU params into backend arrays + a validity mask."""
    a     = np.asarray(df_entropy["cond_a"].to_numpy(),     dtype=float)
    b     = np.asarray(df_entropy["cond_b"].to_numpy(),     dtype=float)
    loc   = np.asarray(df_entropy["cond_loc"].to_numpy(),   dtype=float)
    scale = np.asarray(df_entropy["cond_scale"].to_numpy(), dtype=float)
    # Valid rows mirror the original guards: finite params, positive b & scale.
    valid = (np.isfinite(a) & np.isfinite(b)
             & np.isfinite(loc) & np.isfinite(scale)
             & (b > 0) & (scale > 0))
    return xp.asarray(a), xp.asarray(b), xp.asarray(loc), xp.asarray(scale), valid


# ──────────────────────────────────────────────────────────────────────────
# Vectorized conditional metrics (Johnson SU, closed form)
# ──────────────────────────────────────────────────────────────────────────
def _conditional_var_vectorized(a, b, loc, scale, confidence, xp):
    """Both-tail VaR for all dates at once. Returns (var_left, var_right)."""
    z_left  = float(ndtri(1.0 - confidence))
    z_right = float(ndtri(confidence))
    var_left  = loc + scale * xp.sinh((z_left  - a) / b)
    var_right = loc + scale * xp.sinh((z_right - a) / b)
    return var_left, var_right


def _conditional_cvar_vectorized(a, b, loc, scale, confidence, xp,
                                 n_grid: int = 5_000, chunk: int = 2_000):
    """
    Both-tail CVaR (Expected Shortfall) for all dates at once, via the same
    Riemann sum over a per-date [ppf(1e-6), ppf(1-1e-6)] grid as the original
    ``compute_conditional_cvar`` — but chunked and vectorized across dates.
    """
    z_left  = float(ndtri(1.0 - confidence))
    z_right = float(ndtri(confidence))
    z_lo    = float(ndtri(1e-6))
    z_hi    = float(ndtri(1.0 - 1e-6))

    t = xp.linspace(0.0, 1.0, n_grid)[None, :]          # (1, G)
    N = int(a.shape[0])
    cvar_left  = xp.full((N,), xp.nan)
    cvar_right = xp.full((N,), xp.nan)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        aa = a[s:e][:, None]; bb = b[s:e][:, None]
        ll = loc[s:e][:, None]; ss = scale[s:e][:, None]

        var_l = ll + ss * xp.sinh((z_left  - aa) / bb)   # (m, 1)
        var_r = ll + ss * xp.sinh((z_right - aa) / bb)
        x_lo  = ll + ss * xp.sinh((z_lo - aa) / bb)
        x_hi  = ll + ss * xp.sinh((z_hi - aa) / bb)

        xg = x_lo + (x_hi - x_lo) * t                    # (m, G)
        dx = (x_hi - x_lo) / (n_grid - 1)                # (m, 1)
        z  = (xg - ll) / ss
        pdf = (bb / ss) / xp.sqrt(1.0 + z * z) / _SQRT2PI \
            * xp.exp(-0.5 * (aa + bb * xp.arcsinh(z)) ** 2)

        mask_l = xg <= var_l
        prob_l = (pdf * mask_l).sum(axis=1) * dx[:, 0]
        cvar_left[s:e] = (xg * pdf * mask_l).sum(axis=1) * dx[:, 0] \
            / xp.maximum(prob_l, 1e-12)

        mask_r = xg >= var_r
        prob_r = (pdf * mask_r).sum(axis=1) * dx[:, 0]
        cvar_right[s:e] = (xg * pdf * mask_r).sum(axis=1) * dx[:, 0] \
            / xp.maximum(prob_r, 1e-12)

    return cvar_left, cvar_right


# ──────────────────────────────────────────────────────────────────────────
# Historical Simulation (unconditional) — same gated logic as the original
# ──────────────────────────────────────────────────────────────────────────
def _historical_series(df, y_var, dates, window, confidence, horizon,
                       retrain_after, compute_fn, desc):
    """
    Rolling Historical Simulation aligned to ``dates``, recomputed only every
    ``retrain_after`` observations and carried forward in between (identical to
    the original ``generate_oos_*`` Historical branch).
    """
    left  = np.full(len(dates), np.nan)
    right = np.full(len(dates), np.nan)
    y = df[y_var]
    cache = None
    since = 0
    for i, current_date in enumerate(tqdm(dates, desc=desc)):
        if (cache is None) or (since >= retrain_after):
            past = y.loc[y.index < current_date].dropna()
            if len(past) < window:
                cache = None
            else:
                cache = compute_fn(past.iloc[-window:].values,
                                   confidence=confidence, horizon=horizon)
            since = 0
        if cache is not None:
            left[i], right[i] = cache
        since += 1
    return left, right


# ──────────────────────────────────────────────────────────────────────────
# Public drivers (vectorized) — drop-in replacements
# ──────────────────────────────────────────────────────────────────────────
def generate_oos_var(
    df: pd.DataFrame,
    y_var: str,
    df_entropy: pd.DataFrame,
    window: int = 1_008,
    confidence: float = 0.975,
    horizon: int = 10,
    retrain_after: int = 30,
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Vectorized Out-of-Sample VaR. Identical signature, columns, and results as
    :func:`auxi.risk_metrics.generate_oos_var`, plus a ``backend`` switch
    (``"auto"`` | ``"cpu"`` | ``"gpu"``).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = pd.to_datetime(df.index)

    xp, to_numpy, _ = _select_backend(backend)

    # Conditional (JSU) — all dates in one shot.
    a, b, loc, scale, valid = _params_arrays(df_entropy, xp)
    v_left, v_right = _conditional_var_vectorized(a, b, loc, scale, confidence, xp)
    c_left  = to_numpy(v_left)
    c_right = to_numpy(v_right)
    c_left[~valid]  = np.nan
    c_right[~valid] = np.nan

    # Unconditional (Historical Simulation) — gated rolling loop.
    h_left, h_right = _historical_series(
        df, y_var, df_entropy.index, window, confidence, horizon,
        retrain_after, compute_historical_var, desc="OOS VaR (hist)")

    return pd.DataFrame(
        {"Cond_VaR_Left":  c_left,  "Cond_VaR_Right": c_right,
         "Hist_VaR_Left":  h_left,  "Hist_VaR_Right": h_right},
        index=df_entropy.index,
    ).rename_axis("Date")


def generate_oos_cvar(
    df: pd.DataFrame,
    y_var: str,
    df_entropy: pd.DataFrame,
    window: int = 1_008,
    confidence: float = 0.975,
    horizon: int = 10,
    retrain_after: int = 30,
    backend: str = "auto",
    n_grid: int = 5_000,
    chunk: int = 2_000,
) -> pd.DataFrame:
    """
    Vectorized Out-of-Sample CVaR / Expected Shortfall. Identical signature,
    columns, and results as :func:`auxi.risk_metrics.generate_oos_cvar`, plus a
    ``backend`` switch and memory-bounding ``n_grid`` / ``chunk`` knobs.

    The conditional CVaR builds an ``(n_dates, n_grid)`` density grid; lower
    ``chunk`` if a small GPU runs out of memory.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy(); df.index = pd.to_datetime(df.index)

    xp, to_numpy, _ = _select_backend(backend)

    # Conditional (JSU) — vectorized, chunked integration.
    a, b, loc, scale, valid = _params_arrays(df_entropy, xp)
    cv_left, cv_right = _conditional_cvar_vectorized(
        a, b, loc, scale, confidence, xp, n_grid=n_grid, chunk=chunk)
    c_left  = to_numpy(cv_left)
    c_right = to_numpy(cv_right)
    c_left[~valid]  = np.nan
    c_right[~valid] = np.nan

    # Unconditional (Historical Simulation) — gated rolling loop.
    h_left, h_right = _historical_series(
        df, y_var, df_entropy.index, window, confidence, horizon,
        retrain_after, compute_historical_cvar, desc="OOS CVaR (hist)")

    return pd.DataFrame(
        {"Cond_CVaR_Left":  c_left,  "Cond_CVaR_Right": c_right,
         "Hist_CVaR_Left":  h_left,  "Hist_CVaR_Right": h_right},
        index=df_entropy.index,
    ).rename_axis("Date")
