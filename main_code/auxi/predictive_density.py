"""
Predictive Density Surface (Growth-at-Risk over time)
=====================================================
Replicates Figure 1 of Adrian, Boyarchenko & Giannone (2019, *Vulnerable
Growth*): the one-period (h-step) ahead **in-sample predictive density** of a
target variable, estimated via quantile regression, fitted to a smooth
parametric density at every date, and stacked over time as a 3-D waterfall
surface.

Pipeline (mirrors the rest of ``auxi`` — a ``compute_*`` core separated from a
``plot_*`` renderer, plus a one-call convenience wrapper):

    1. Shift the target h periods ahead and fit ONE quantile regression per
       quantile on the FULL sample (in-sample, as in the paper).
    2. At each (sub-sampled) date t, predict the conditional quantiles from the
       contemporaneous X(t) and controls, then fit a Johnson SU density to
       those quantiles via Minimum Distance Estimation (``mde_jsu_weighted``).
    3. Evaluate the fitted density on a fixed value grid -> one density row per
       date. The stack of rows is the surface.

The distribution fitter is injected via ``mde_fn`` and defaults to Johnson SU,
so swapping to the Azzalini skew-t (``auxi.distribution_analysis.mde_distfit_skewt``) is
a one-line change without touching this module.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from tqdm import tqdm

from auxi.qreg import q_reg
from auxi.distribution_analysis import mde_jsu_weighted, jsu_pdf


# ──────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class DensitySurface:
    """
    Output of :func:`compute_density_surface`.

    Keeping the computed surface as a self-contained object means it can be
    re-plotted, exported, or fed to downstream diagnostics without recomputing
    the (expensive) quantile regressions and MDE fits.

    Attributes
    ----------
    dates       : pd.DatetimeIndex          – one entry per surface row (time axis).
    value_grid  : np.ndarray, shape (G,)    – evaluation points of the target (value axis).
    density     : np.ndarray, shape (T, G)  – fitted PDF; row t = density at dates[t].
    params      : pd.DataFrame               – per-date JSU params [gamma, delta, loc, scale].
    meta        : dict                       – call metadata (x, y, h, quantiles, ...).
    """
    dates: pd.DatetimeIndex
    value_grid: np.ndarray
    density: np.ndarray
    params: pd.DataFrame
    meta: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# 1. Compute core (no plotting)
# ──────────────────────────────────────────────────────────────────────────
def compute_density_surface(df: pd.DataFrame,
                            x: str,
                            y: str,
                            quantiles: list[float],
                            h: int = 1,
                            controls: list[str] = None,
                            freq: str | None = "QE",
                            n_grid: int = 200,
                            value_range: tuple[float, float] = None,
                            mde_fn=mde_jsu_weighted,
                            verbose: bool = True,
                            _cp=None):
    """
    Compute the in-sample h-step-ahead predictive density of ``y`` at each date
    and stack the densities into a time x value surface.

    Parameters
    ----------
    df           : DataFrame with a DatetimeIndex (coerced if necessary) holding
                   ``y``, ``x`` and every name in ``controls``.
    x            : main conditioning variable, e.g. ``"GPRD_MA7"``.
    y            : target variable to forecast, e.g. ``"Brent_Return"``.
    quantiles    : probability levels for the quantile regressions, e.g.
                   ``[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]``.
    h            : forecast horizon (number of periods ahead).
    controls     : list of additional regressors (``None`` -> no controls).
    freq         : pandas offset alias used to sub-sample the time axis so the
                   3-D surface stays legible (``"QE"`` quarter-end, ``"ME"``
                   month-end, ``"YE"`` year-end). ``None`` keeps every row.
    n_grid       : resolution of the value axis.
    value_range  : (lo, hi) bounds of the value axis. Defaults to the
                   [0.5, 99.5] percentiles of the realized ``y``.
    mde_fn       : distribution fitter mapping (fcq, tau_levels) -> 4 params.
                   Defaults to Johnson SU (``mde_jsu_weighted``).
    verbose      : show a progress bar over dates.

    Returns
    -------
    DensitySurface
    """
    # ── Cache helpers (DensitySurface = 3 files: npz + parquet + json) ──────
    def _cache_stem(cp):
        """Strip .parquet suffix if passed so stems are consistent."""
        return Path(str(cp).replace(".parquet", ""))

    def _cache_exists(cp):
        s = _cache_stem(cp)
        return (Path(f"{s}_arrays.npz").exists()
                and Path(f"{s}_params.parquet").exists()
                and Path(f"{s}_meta.json").exists())

    def _cache_load(cp):
        s = _cache_stem(cp)
        arrays = np.load(f"{s}_arrays.npz", allow_pickle=False)
        dates  = pd.DatetimeIndex(arrays["dates"])
        params = pd.read_parquet(f"{s}_params.parquet")
        with open(f"{s}_meta.json") as _f:
            meta = json.load(_f)
        return DensitySurface(dates=dates, value_grid=arrays["value_grid"],
                              density=arrays["density"], params=params, meta=meta)

    def _cache_save(surface, cp):
        s = _cache_stem(cp)
        Path(s).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{s}_arrays.npz",
                            value_grid=surface.value_grid,
                            density=surface.density,
                            dates=np.array(surface.dates.astype(str)))
        surface.params.to_parquet(f"{s}_params.parquet")
        with open(f"{s}_meta.json", "w") as _f:
            json.dump(surface.meta, _f, indent=2)

    if _cp is not None and _cache_exists(_cp):
        return _cache_load(_cp)
    # ─────────────────────────────────────────────────────────────────────────

    if controls is None:
        controls = []

    # --- 0. Tidy frame & datetime index --------------------------------------
    cols = [y, x] + controls
    df_work = df[cols].copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)
    df_work = df_work.sort_index()

    # --- 1. Shifted target + full-sample training set ------------------------
    target_col = f"{y}_target_h{h}"
    df_work[target_col] = df_work[y].shift(-h)
    df_train = df_work.dropna(subset=[target_col, x] + controls).copy()

    # --- 2. Fit ONE quantile regression per quantile on the full sample ------
    #     (in-sample == the paper's Figure 1)
    fitted = {}
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted[q] = q_reg(df=df_train, x=x, y=target_col, tau=q,
                                  controls=controls, vcov="robust", max_iter=2000)
            except ValueError:
                fitted[q] = q_reg(df=df_train, x=x, y=target_col, tau=q,
                                  controls=controls, vcov="iid", max_iter=2000)

    # --- 3. Pick the evaluation dates ----------------------------------------
    #     Any date with complete features is a candidate; sub-sample to `freq`.
    feats = df_work[[x] + controls].dropna()
    if freq is not None:
        idx_ser = pd.Series(feats.index, index=feats.index)
        eval_dates = pd.DatetimeIndex(idx_ser.resample(freq).last().dropna().values)
    else:
        eval_dates = feats.index

    # --- 4. Value grid --------------------------------------------------------
    if value_range is None:
        lo = np.nanpercentile(df_work[y].values, 0.5)
        hi = np.nanpercentile(df_work[y].values, 99.5)
    else:
        lo, hi = value_range
    value_grid = np.linspace(lo, hi, n_grid)

    # --- 5. Per-date: predict quantiles -> MDE fit -> density row ------------
    rows, kept_dates, param_records = [], [], []
    iterator = tqdm(eval_dates, desc=f"Density surface (h={h})") if verbose else eval_dates

    for date in iterator:
        features_t = df_work.loc[[date]]
        # Forecast the conditional quantiles at this date
        fcq = []
        for q in quantiles:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fcq.append(fitted[q].predict(exog=features_t).values[0])

        # Fit the parametric density to the forecasted quantiles
        try:
            params = mde_fn(fcq=fcq, tau_levels=quantiles)
        except Exception:
            continue
        params = np.asarray(params, dtype=float)
        if np.any(~np.isfinite(params)) or params[1] <= 0 or params[3] <= 0:
            continue  # skip degenerate fits to keep a clean mesh

        gamma, delta, loc, scale = params
        rows.append(jsu_pdf(value_grid, gamma, delta, loc, scale))
        kept_dates.append(date)
        param_records.append({"gamma": gamma, "delta": delta,
                              "loc": loc, "scale": scale})

    if not rows:
        raise RuntimeError("No valid densities were produced — check inputs / horizon.")

    kept_index = pd.DatetimeIndex(kept_dates)
    surface = DensitySurface(
        dates=kept_index,
        value_grid=value_grid,
        density=np.vstack(rows),
        params=pd.DataFrame(param_records, index=kept_index),
        meta=dict(x=x, y=y, h=h, quantiles=list(quantiles),
                  controls=list(controls), freq=freq),
    )
    return surface


# ──────────────────────────────────────────────────────────────────────────
# 2. 3-D waterfall renderer
# ──────────────────────────────────────────────────────────────────────────
def plot_density_surface(surface: DensitySurface,
                        cmap="magma",
                        elev: float = 28,
                        azim: float = -60,
                        n_date_ticks: int = 8,
                        figsize: tuple = (12, 8)):
    """
    Render a :class:`DensitySurface` as the Adrian-Boyarchenko-Giannone style
    3-D waterfall: target value on the front horizontal axis, time receding
    into depth, density on the vertical axis, coloured mesh.

    Parameters
    ----------
    cmap : str | Colormap
        Colormap for the surface (default ``"magma"``).

    Returns
    -------
    (fig, ax) so the figure stays composable.
    """
    s = surface
    t_pos = np.arange(len(s.dates))                  # numeric time positions
    Xg, Yg = np.meshgrid(s.value_grid, t_pos)        # (T, G) grids
    Z = s.density

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    ax.plot_surface(Xg, Yg, Z, cmap=cmap,
                    rstride=1, cstride=2, linewidth=0.15,
                    edgecolor="0.4", antialiased=True, alpha=0.95)

    # Time tick labels along the depth axis
    tick_idx = np.linspace(0, len(s.dates) - 1, min(n_date_ticks, len(s.dates)))
    tick_idx = np.unique(tick_idx.astype(int))
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([s.dates[i].strftime("%Y") for i in tick_idx])

    h, y = s.meta.get("h"), s.meta.get("y")
    ax.set_xlabel(f"{y}  (t+{h})", labelpad=10)
    ax.set_ylabel("Date", labelpad=12)
    ax.set_zlabel("Density", labelpad=6)
    ax.view_init(elev=elev, azim=azim)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_edgecolor("white")
        pane.pane.set_alpha(0.0)
    ax.grid(True, alpha=0.3)

    return fig, ax


# ──────────────────────────────────────────────────────────────────────────
# 3. One-call convenience wrapper
# ──────────────────────────────────────────────────────────────────────────
def density_surface(df: pd.DataFrame,
                    x: str,
                    y: str,
                    quantiles: list[float],
                    h: int = 1,
                    controls: list[str] = None,
                    freq: str | None = "QE",
                    n_grid: int = 200,
                    value_range: tuple[float, float] = None,
                    mde_fn=mde_jsu_weighted,
                    show: bool = True,
                    **plot_kwargs) -> DensitySurface:
    """
    Compute and plot the predictive density surface in a single call.
    """
    surface = compute_density_surface(
        df=df, x=x, y=y, quantiles=quantiles, h=h, controls=controls,
        freq=freq, n_grid=n_grid, value_range=value_range, mde_fn=mde_fn,
    )
    plot_density_surface(surface, **plot_kwargs)
    if show:
        plt.show()
    return surface
