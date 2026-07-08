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


@pytest.fixture
def synthetic_ews_pair():
    """Leader/follower pair with known lag for EWS tests.

    - leader: sine wave (period 50) + small noise.
    - follower: same sine shifted forward by 5 periods + noise.
    - Ground truth: h* = 5 (leader leads follower by 5).
    """
    rng = np.random.default_rng(123)
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n, dtype=float)
    signal = np.sin(2 * np.pi * t / 50)
    leader = pd.Series(signal + rng.normal(0, 0.1, n), index=idx, name="leader")
    follower = pd.Series(
        np.roll(signal, 5) + rng.normal(0, 0.1, n), index=idx, name="follower"
    )
    return leader, follower


@pytest.fixture
def independent_pair():
    """Two independent Gaussian noise series (no causal relationship)."""
    rng = np.random.default_rng(456)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = pd.Series(rng.normal(0, 1, n), index=idx, name="x")
    y = pd.Series(rng.normal(0, 1, n), index=idx, name="y")
    return x, y
