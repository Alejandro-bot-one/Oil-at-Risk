# Backend Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate `auxi/` modules by purpose — direct-forecasting estimators live with their model engine (`qreg.py`, `caviar.py`); all evaluation diagnostics live in a new `auxi/diagnostics/` subpackage organized by diagnostic type. Delete `auxi/forecasting.py`.

**Architecture:** Single-file reshuffle for `qreg.py` and `caviar.py` (one concern each). `auxi/diagnostics.py` becomes `auxi/diagnostics/` subpackage with four thematic submodules (`specification`, `direct_forecasting`, `distribution_fitting`, `series`). Caviar's DF functions are thin wrappers (Approach C): they augment the panel with breach indicators and delegate to the qreg/diagnostics counterparts.

**Tech Stack:** Python 3.10+, statsmodels, pandas, numpy, matplotlib, scipy, pytest. Jupyter notebooks for the consumer-side smoke tests.

**Related spec:** [`docs/superpowers/specs/2026-06-26-backend-reorg-design.md`](../specs/2026-06-26-backend-reorg-design.md)

---

## File Structure

**Created in this plan:**
- `auxi/diagnostics/__init__.py`
- `auxi/diagnostics/specification.py`
- `auxi/diagnostics/direct_forecasting.py`
- `auxi/diagnostics/distribution_fitting.py`
- `auxi/diagnostics/series.py`
- `auxi/README.md`
- `.gitignore` (Task 0)

**Modified in this plan:**
- `auxi/qreg.py` (append DF helpers, estimators, plotters)
- `auxi/caviar.py` (append DF wrappers + lookahead docstring note)
- `auxi/distribution_analysis.py` (remove 7 diagnostic functions)
- `direct_forecasting.ipynb` (1 import changed, 1 added, 7 calls redirected, 1 cell handled)
- `distribution_analysis.ipynb` (1 import changed, 1 added, 5 calls redirected)

**Deleted in this plan:**
- `auxi/forecasting.py`
- `auxi/diagnostics.py`

---

## Task 0: Initialize git for rollback safety

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Initialize the git repo**

Run:
```
git init
```

Expected output: `Initialized empty Git repository in ...`

- [ ] **Step 2: Create `.gitignore` at the project root**

Create `.gitignore` with this content:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Pytest
.pytest_cache/

# Editor/OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Output / temp
*.parquet
```

- [ ] **Step 3: Initial commit**

```
git add .gitignore
git add auxi tests docs
git status
```

Verify the staged file list looks reasonable (no `__pycache__`, no `.parquet`). Then:

```
git commit -m "chore: initial commit before auxi reorg"
```

- [ ] **Step 4: Tag the pre-reorg state**

```
git tag pre-reorg-2026-06-26
```

This gives you a fixed rollback target: `git reset --hard pre-reorg-2026-06-26` if anything goes wrong later.

---

## Task 1: Baseline verification

**Files:** none (read-only)

- [ ] **Step 1: Confirm the existing tests pass**

Run:
```
python -m pytest tests/test_caviar.py -v
```

Expected: all tests pass.

If they don't pass: stop, fix the existing tests before doing any reorg. The plan assumes a green baseline.

- [ ] **Step 2: Snapshot what the current public surface looks like**

Run:
```
python -c "import auxi.forecasting as fc; print(sorted(n for n in dir(fc) if not n.startswith('_')))"
python -c "import auxi.diagnostics as diags; print(sorted(n for n in dir(diags) if not n.startswith('_')))"
python -c "import auxi.qreg as qr; print(sorted(n for n in dir(qr) if not n.startswith('_')))"
python -c "import auxi.caviar as cv; print(sorted(n for n in dir(cv) if not n.startswith('_')))"
python -c "import auxi.distribution_analysis as da; print(sorted(n for n in dir(da) if not n.startswith('_')))"
```

Save the output somewhere (a scratch file). After the reorg, re-running these should resolve every function name that existed before (minus `select_horizon_rolling_origin` which we're deleting on purpose).

---

## Task 2: Move DF helpers, estimators, and plotters from `forecasting.py` → `qreg.py`

**Files:**
- Modify: `auxi/qreg.py` (append three new sections)
- Read: `auxi/forecasting.py` (don't modify yet — kept until Task 7 for safety)

- [ ] **Step 1: Write the smoke test (expected to fail before the move)**

Create temporary test file `tests/test_reorg_qreg.py`:

```python
"""Smoke tests for the qreg.py post-reorg public surface."""
import auxi.qreg as qr


def test_pinball_loss_resolvable():
    assert callable(qr.pinball_loss)


def test_direct_forecasting_resolvable():
    assert callable(qr.direct_forecasting)


def test_insample_direct_forecasting_resolvable():
    assert callable(qr.insample_direct_forecasting)


def test_get_oos_predictions_resolvable():
    assert callable(qr.get_oos_predictions)


def test_plot_forecasted_scatters_resolvable():
    assert callable(qr.plot_forecasted_scatters)


def test_plot_contemporaneous_vs_predictive_coefs_resolvable():
    assert callable(qr.plot_contemporaneous_vs_predictive_coefs)
```

- [ ] **Step 2: Run the smoke test — expect failure**

```
python -m pytest tests/test_reorg_qreg.py -v
```

Expected: all 6 tests FAIL with `AttributeError: module 'auxi.qreg' has no attribute 'pinball_loss'` (etc.).

- [ ] **Step 3: Append `tqdm` import to `auxi/qreg.py`**

`auxi/qreg.py` currently imports `warnings`, `numpy`, `pandas`, `matplotlib.pyplot`, `statsmodels.api`, `statsmodels.formula.api`. The DF functions need additional imports. After line 7 (`import statsmodels.formula.api as smf`), add:

```python
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf
```

- [ ] **Step 4: Append Section 3 (DF helpers) to `auxi/qreg.py`**

Append at the end of the file:

```python
# =============================================================================
# SECTION 3 — DIRECT FORECASTING HELPERS
# =============================================================================

def pinball_loss(tau, y_true, y_pred):
    """Asymmetric (Tick) Loss for a given quantile tau."""
    import numpy as np
    error = y_true - y_pred
    loss = np.where(error < 0, (1 - tau) * np.abs(error), tau * np.abs(error))
    return np.mean(loss)
```

- [ ] **Step 5: Append Section 4 (DF estimators) to `auxi/qreg.py`**

Copy the bodies of `direct_forecasting` (lines 37–85 of `auxi/forecasting.py`), `insample_direct_forecasting` (lines 87–214), and `get_oos_predictions` (lines 1187–1252) of the current `forecasting.py`, in that order, under a fresh section header:

```python
# =============================================================================
# SECTION 4 — DIRECT FORECASTING ESTIMATORS
# =============================================================================

# (paste direct_forecasting here)
# (paste insample_direct_forecasting here)
# (paste get_oos_predictions here)
```

Use `auxi/forecasting.py` as your source — copy the function bodies verbatim, including their docstrings. Do not change behavior.

- [ ] **Step 6: Append Section 5 (DF plotters + private helpers) to `auxi/qreg.py`**

Copy `_plot_coef_panel` (line 310), `_add_regression_lines` (line 342), `plot_forecasted_scatters` (line 394), `plot_contemporaneous_vs_predictive_coefs` (line 474) from `forecasting.py`, in that order:

```python
# =============================================================================
# SECTION 5 — DIRECT FORECASTING PLOTTERS
# =============================================================================

# (paste _plot_coef_panel here)
# (paste _add_regression_lines here)
# (paste plot_forecasted_scatters here)
# (paste plot_contemporaneous_vs_predictive_coefs here)
```

- [ ] **Step 7: Run the smoke test — expect pass**

```
python -m pytest tests/test_reorg_qreg.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 8: Run the existing caviar tests as a regression check**

```
python -m pytest tests/test_caviar.py -v
```

Expected: all pass. `caviar.py` imports `q_reg, plot_quantile_coefs, plot_pseudo_r2` from `qreg` — adding new functions to `qreg.py` shouldn't break that.

- [ ] **Step 9: Commit**

```
git add auxi/qreg.py tests/test_reorg_qreg.py
git commit -m "refactor: move DF helpers, estimators, plotters from forecasting to qreg"
```

---

## Task 3: Convert `auxi/diagnostics.py` → `auxi/diagnostics/` subpackage (specification + series + __init__)

**Files:**
- Delete: `auxi/diagnostics.py`
- Create: `auxi/diagnostics/__init__.py`
- Create: `auxi/diagnostics/specification.py`
- Create: `auxi/diagnostics/series.py`

- [ ] **Step 1: Write the smoke test (expected to fail because the new subpackage doesn't exist yet)**

Create `tests/test_reorg_diagnostics_subpkg.py`:

```python
"""Smoke tests for the diagnostics/ subpackage public surface (Stage 1)."""
import auxi.diagnostics as diags


def test_dq_test_resolvable():
    assert callable(diags.dq_test)


def test_wald_test_resolvable():
    assert callable(diags.wald_test)


def test_q_arch_test_resolvable():
    assert callable(diags.q_arch_test)


def test_qarx_stability_test_resolvable():
    assert callable(diags.qarx_stability_test)


def test_adf_test_all_resolvable():
    assert callable(diags.adf_test_all)


def test_hamilton_filter_resolvable():
    assert callable(diags.hamilton_filter)


def test_submodule_specification_importable():
    from auxi.diagnostics import specification
    assert callable(specification.dq_test)


def test_submodule_series_importable():
    from auxi.diagnostics import series
    assert callable(series.adf_test_all)
```

- [ ] **Step 2: Run smoke test — expect PASS partially (existing names) and FAIL on the submodule imports**

```
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: the 6 top-level resolvable tests PASS (because `auxi/diagnostics.py` still exists), the 2 submodule tests FAIL with `ImportError`.

- [ ] **Step 3: Create the subpackage directory and `specification.py`**

```
mkdir auxi/diagnostics
```

Create `auxi/diagnostics/specification.py` with this preamble:

```python
"""Quantile-regression model specification diagnostics.

Moved from auxi/diagnostics.py during the 2026-06-26 backend reorg.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

from auxi.qreg import q_reg
```

Then append the bodies of the following functions verbatim from `auxi/diagnostics.py`:
- `dq_test` (line 31)
- `plot_advanced_dq_diagnostics` (line 97)
- `wald_test` (line 144)
- `plot_wald_diagnostics` (line 197)
- `q_arch_test` (line 215)
- `plot_q_arch_diagnostics` (line 245)
- `qarx_stability_test` (line 266)

- [ ] **Step 4: Create `auxi/diagnostics/series.py`**

```python
"""Series/data utilities (stationarity tests, trend filters).

Moved from auxi/diagnostics.py during the 2026-06-26 backend reorg.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
```

Then append the bodies of `adf_test_all` (line 394 of `auxi/diagnostics.py`) and `hamilton_filter` (line 404) verbatim.

- [ ] **Step 5: Create `auxi/diagnostics/__init__.py` with explicit reexports**

```python
"""auxi.diagnostics — subpackage of diagnostics organized by type.

Re-exports the public functions of each submodule so existing usage
(`import auxi.diagnostics as diags; diags.dq_test(...)`) keeps working.
"""
from .specification import (
    dq_test,
    plot_advanced_dq_diagnostics,
    wald_test,
    plot_wald_diagnostics,
    q_arch_test,
    plot_q_arch_diagnostics,
    qarx_stability_test,
)
from .series import (
    adf_test_all,
    hamilton_filter,
)
```

- [ ] **Step 6: Delete the old single-file `auxi/diagnostics.py`**

```
del "auxi/diagnostics.py"
```

(Or `rm auxi/diagnostics.py` in Git Bash.)

- [ ] **Step 7: Clear the Python cache so the old module isn't picked up**

```
del /s /q auxi\__pycache__
```

(Or `rm -rf auxi/__pycache__` in Git Bash.)

- [ ] **Step 8: Run the smoke test — expect all PASS**

```
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: all 8 tests PASS. If any fail with `ImportError` from `specification.py`, check that you also moved the imports it needs (`os`, `gridspec`, `seaborn`, `QuantReg`, `plot_acf`, `scipy.stats`) into the preamble in Step 3.

- [ ] **Step 9: Run the existing caviar tests as regression check**

```
python -m pytest tests/test_caviar.py -v
```

Expected: pass.

- [ ] **Step 10: Commit**

```
git add auxi/diagnostics auxi/diagnostics.py tests/test_reorg_diagnostics_subpkg.py
git status
git commit -m "refactor: convert auxi/diagnostics.py to auxi/diagnostics/ subpackage (specification + series)"
```

`git status` should show `auxi/diagnostics.py` as deleted. If it's still tracked, run `git rm auxi/diagnostics.py` first.

---

## Task 4: Create `auxi/diagnostics/direct_forecasting.py` and update `__init__.py`

**Files:**
- Create: `auxi/diagnostics/direct_forecasting.py`
- Modify: `auxi/diagnostics/__init__.py`

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_reorg_diagnostics_subpkg.py`:

```python
def test_evaluate_direct_forecasting_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.evaluate_direct_forecasting)


def test_compute_fallout_errors_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.compute_fallout_errors)


def test_compute_unconditional_coverage_unified_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.compute_unconditional_coverage_unified)


def test_compute_conditional_coverage_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.compute_conditional_coverage)


def test_diagnose_residual_acf_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.diagnose_residual_acf)


def test_submodule_direct_forecasting_importable():
    from auxi.diagnostics import direct_forecasting
    assert callable(direct_forecasting.evaluate_direct_forecasting)
```

- [ ] **Step 2: Run — expect FAIL on the new tests**

```
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: 8 old tests PASS, 6 new tests FAIL.

- [ ] **Step 3: Create `auxi/diagnostics/direct_forecasting.py`**

```python
"""Direct-forecasting evaluation diagnostics (shared by qreg-DF and caviar-DF).

These functions take a pre-fitted forecast as (realized, forecasted) series
OR refit a quantile-regression model internally on the dataframe.
Either way, the model used is opaque to the caller — the same diagnostic
applies to a qreg forecast and to a caviar forecast.

Moved from auxi/forecasting.py during the 2026-06-26 backend reorg.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf

from auxi.qreg import pinball_loss
```

Then append the bodies of these functions verbatim from `auxi/forecasting.py`, in this order:
- `evaluate_direct_forecasting` (line 218)
- `diagnose_residual_acf` (line 550)
- `compute_fallout_errors` (line 621)
- `plot_fallout_errors` (line 695)
- `evaluate_cumulative_fallout` (line 758)
- `compute_unconditional_coverage_unified` (line 799)
- `plot_unconditional_coverage_unified` (line 848)
- `compute_conditional_coverage` (line 900)
- `plot_conditional_coverage` (line 966)
- `plot_unconditional_coverage` (line 1255)

Do NOT copy `select_horizon_rolling_origin` (line 1033) — it is being deleted on purpose per the spec.

- [ ] **Step 4: Update `auxi/diagnostics/__init__.py` to reexport the new names**

Append after the existing imports:

```python
from .direct_forecasting import (
    evaluate_direct_forecasting,
    diagnose_residual_acf,
    compute_fallout_errors,
    plot_fallout_errors,
    evaluate_cumulative_fallout,
    compute_unconditional_coverage_unified,
    plot_unconditional_coverage_unified,
    compute_conditional_coverage,
    plot_conditional_coverage,
    plot_unconditional_coverage,
)
```

- [ ] **Step 5: Clear cache and run the smoke test — expect all pass**

```
del /s /q auxi\__pycache__
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 6: Commit**

```
git add auxi/diagnostics/direct_forecasting.py auxi/diagnostics/__init__.py tests/test_reorg_diagnostics_subpkg.py
git commit -m "refactor: add diagnostics/direct_forecasting.py (DF evaluation diagnostics)"
```

---

## Task 5: Pull distribution-fitting diagnostics into `auxi/diagnostics/distribution_fitting.py`

**Files:**
- Create: `auxi/diagnostics/distribution_fitting.py`
- Modify: `auxi/diagnostics/__init__.py`
- Modify: `auxi/distribution_analysis.py` (remove 7 functions)

⚠ This task creates a temporary breakage of `distribution_analysis.ipynb` — fixed in Task 9.

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_reorg_diagnostics_subpkg.py`:

```python
def test_jsu_ks_test_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.jsu_ks_test)


def test_evaluate_oos_pit_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.evaluate_oos_pit)


def test_evaluate_oos_pit_skewt_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.evaluate_oos_pit_skewt)


def test_fit_and_diagnose_jsu_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.fit_and_diagnose_jsu)


def test_fit_and_diagnose_skewt_resolvable():
    import auxi.diagnostics as diags
    assert callable(diags.fit_and_diagnose_skewt)


def test_distribution_analysis_loses_diagnostics():
    """After the move, these names should NOT resolve on da anymore."""
    import auxi.distribution_analysis as da
    assert not hasattr(da, "jsu_ks_test"), "jsu_ks_test should be moved to diagnostics"
    assert not hasattr(da, "evaluate_oos_pit"), "evaluate_oos_pit should be moved to diagnostics"
    assert not hasattr(da, "fit_and_diagnose_jsu"), "fit_and_diagnose_jsu should be moved to diagnostics"


def test_distribution_analysis_keeps_fitters():
    """Fitters and engines stay in distribution_analysis."""
    import auxi.distribution_analysis as da
    assert callable(da.fit_jsu)
    assert callable(da.fit_skewt)
    assert callable(da.mde_jsu_weighted)
```

- [ ] **Step 2: Run — expect FAIL on the new tests**

```
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: previous tests PASS, new tests for `diags.jsu_ks_test` etc. FAIL.

- [ ] **Step 3: Create `auxi/diagnostics/distribution_fitting.py`**

```python
"""Goodness-of-fit diagnostics for distribution fits (JSU, Skew-t).

These diagnostics belong with the other diagnostics (one place per type),
but they delegate to the fitters that still live in auxi.distribution_analysis.

Moved from auxi/distribution_analysis.py during the 2026-06-26 backend reorg.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Fitters and PDFs stay in auxi.distribution_analysis — we import them here.
from auxi.distribution_analysis import (
    fit_jsu,
    fit_skewt,
    jsu_pdf,
    jsu_cdf,
    jsu_sample,
    skewt_pdf,
    skewt_cdf,
)
```

Then append the bodies of these functions verbatim from `auxi/distribution_analysis.py`:
- `jsu_ks_test` (line 331)
- `fit_and_diagnose_jsu` (line 421)
- `evaluate_oos_pit` (line 887)
- `oos_pit_calibration` (line 967)
- `plot_oos_pit_calibration` (line 1025)
- `fit_and_diagnose_skewt` (line 1190)
- `evaluate_oos_pit_skewt` (line 1402)

Each function should work standalone after copy — its only dependency on `distribution_analysis` is the fit/pdf helpers we imported above.

- [ ] **Step 4: Update `auxi/diagnostics/__init__.py` with the new reexports**

Append:

```python
from .distribution_fitting import (
    jsu_ks_test,
    evaluate_oos_pit,
    evaluate_oos_pit_skewt,
    oos_pit_calibration,
    plot_oos_pit_calibration,
    fit_and_diagnose_jsu,
    fit_and_diagnose_skewt,
)
```

- [ ] **Step 5: Delete the 7 functions from `auxi/distribution_analysis.py`**

In `auxi/distribution_analysis.py`, delete the *function bodies* of these 7 functions only (the rest of the file stays unchanged):
- `jsu_ks_test` (starts at line 331)
- `fit_and_diagnose_jsu` (starts at line 421)
- `evaluate_oos_pit` (starts at line 887)
- `oos_pit_calibration` (starts at line 967)
- `plot_oos_pit_calibration` (starts at line 1025)
- `fit_and_diagnose_skewt` (starts at line 1190)
- `evaluate_oos_pit_skewt` (starts at line 1402)

For each, delete from the `def ...` line through its `return` / final statement (just before the next `def` or `# =====` boundary).

- [ ] **Step 6: Clear cache and run the smoke test — expect all pass**

```
del /s /q auxi\__pycache__
python -m pytest tests/test_reorg_diagnostics_subpkg.py -v
```

Expected: all PASS, including the `test_distribution_analysis_loses_diagnostics` and `test_distribution_analysis_keeps_fitters` tests.

- [ ] **Step 7: Commit**

```
git add auxi/diagnostics/distribution_fitting.py auxi/diagnostics/__init__.py auxi/distribution_analysis.py tests/test_reorg_diagnostics_subpkg.py
git commit -m "refactor: move distribution-fitting diagnostics to diagnostics/distribution_fitting.py"
```

---

## Task 6: Add caviar DF wrappers (Approach C) to `auxi/caviar.py`

**Files:**
- Modify: `auxi/caviar.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_reorg_caviar_wrappers.py`:

```python
"""Smoke tests for the caviar DF wrappers added by the reorg."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # no display
import auxi.caviar as cv


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"X": x, "Y": y}, index=idx)


def test_caviar_direct_forecasting_resolvable():
    assert callable(cv.caviar_direct_forecasting)


def test_caviar_insample_direct_forecasting_resolvable():
    assert callable(cv.caviar_insample_direct_forecasting)


def test_caviar_get_oos_predictions_resolvable():
    assert callable(cv.caviar_get_oos_predictions)


def test_caviar_plot_forecasted_scatters_resolvable():
    assert callable(cv.caviar_plot_forecasted_scatters)


def test_caviar_plot_contemporaneous_vs_predictive_coefs_resolvable():
    assert callable(cv.caviar_plot_contemporaneous_vs_predictive_coefs)


def test_caviar_evaluate_direct_forecasting_resolvable():
    assert callable(cv.caviar_evaluate_direct_forecasting)


def test_caviar_direct_forecasting_runs_end_to_end(sample_df):
    """One full execution to verify the wrapper plumbs through to qreg.direct_forecasting."""
    result = cv.caviar_direct_forecasting(
        df=sample_df,
        vars_x="X",
        vars_y="Y",
        quantiles=[0.05, 0.5, 0.95],
        h=1,
    )
    assert isinstance(result, pd.DataFrame)
    assert {"Quantile", "Forecast"}.issubset(result.columns)


def test_caviar_get_oos_predictions_runs_end_to_end(sample_df):
    realized, predicted = cv.caviar_get_oos_predictions(
        df=sample_df,
        vars_x="X",
        vars_y="Y",
        tau=0.05,
        h=1,
        train_fraction=0.7,
    )
    assert isinstance(realized, pd.Series)
    assert isinstance(predicted, pd.Series)
    assert len(realized) == len(predicted)


def test_input_panel_is_not_mutated(sample_df):
    """Wrappers must never add upside_breach / downside_breach columns to the caller's df."""
    cols_before = list(sample_df.columns)
    _ = cv.caviar_direct_forecasting(
        df=sample_df, vars_x="X", vars_y="Y",
        quantiles=[0.5], h=1,
    )
    assert list(sample_df.columns) == cols_before
    assert "upside_breach" not in sample_df.columns
    assert "downside_breach" not in sample_df.columns
```

- [ ] **Step 2: Run smoke test — expect FAIL**

```
python -m pytest tests/test_reorg_caviar_wrappers.py -v
```

Expected: all FAIL with `AttributeError`.

- [ ] **Step 3: Add the lookahead caveat to the module docstring of `auxi/caviar.py`**

Find the existing module docstring (the very first triple-quoted string in `auxi/caviar.py`). Append after the last existing line of the docstring, just before the closing `"""`:

```
Limitación conocida (lookahead B):
  Los bounds se ajustan sobre el panel completo. El shift del direct forecasting
  cura sólo el leak trivial (predecir y_t desde y_t); el leak del ajuste in-sample
  sobrevive. Diferido a sesión futura — ver
  docs/superpowers/specs/2026-06-26-backend-reorg-design.md.
```

- [ ] **Step 4: Add the breach-augmentation helper to Capa 1 of `auxi/caviar.py`**

Locate the end of Capa 1 (after `_compute_breaches`, around line 112). Append:

```python
def _augment_with_breaches(df, vars_x, vars_y, breach_quantiles=None):
    """Compute bounds + breaches; return augmented panel copy.

    Helper used by every caviar DF wrapper. df is never mutated.
    See module docstring for the lookahead (B) caveat.
    """
    if breach_quantiles is None:
        breach_quantiles = [0.05, 0.95]
    if isinstance(vars_x, str):
        vars_x = [vars_x]

    bounds = _compute_quantile_bounds(
        df, vars_x, vars_y, min(breach_quantiles), max(breach_quantiles)
    )
    upside, downside = _compute_breaches(df[vars_y], bounds)

    work = df.copy()
    work["upside_breach"] = upside
    work["downside_breach"] = downside
    return work
```

- [ ] **Step 5: Add Capa 4 (DF estimator wrappers) to `auxi/caviar.py`**

Append at the end of the file:

```python
# =============================================================================
# CAPA 4 — DIRECT FORECASTING ESTIMATOR WRAPPERS (Approach C)
# =============================================================================
# Each wrapper: augment panel with breach indicators, delegate to qreg counterpart.
# See module docstring for the lookahead (B) caveat.

from auxi.qreg import (
    direct_forecasting,
    insample_direct_forecasting,
    get_oos_predictions,
    plot_forecasted_scatters,
    plot_contemporaneous_vs_predictive_coefs,
)
from auxi.diagnostics import evaluate_direct_forecasting


def caviar_direct_forecasting(df, vars_x, vars_y, quantiles, h=1, controls=None,
                              breach_quantiles=None, **kwargs):
    """Caviar DF wrapper — augment + delegate to qreg.direct_forecasting."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return direct_forecasting(
        df=work, x=vars_x[0], y=vars_y, quantiles=quantiles, h=h,
        controls=augmented_controls, **kwargs,
    )


def caviar_insample_direct_forecasting(df, vars_x, vars_y, quantiles, train_end_date,
                                       h=1, controls=None, breach_quantiles=None, **kwargs):
    """Caviar in-sample DF wrapper — augment + delegate to qreg."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return insample_direct_forecasting(
        df=work, x=vars_x[0], y=vars_y, quantiles=quantiles,
        train_end_date=train_end_date, h=h, controls=augmented_controls, **kwargs,
    )


def caviar_get_oos_predictions(df, vars_x, vars_y, tau, h=1, controls=None,
                               train_fraction=0.8, test_start_date=None,
                               breach_quantiles=None):
    """Caviar OOS predictions wrapper — augment + delegate to qreg."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return get_oos_predictions(
        df=work, x=vars_x[0], y=vars_y, tau=tau, h=h,
        controls=augmented_controls,
        train_fraction=train_fraction, test_start_date=test_start_date,
    )
```

- [ ] **Step 6: Add Capa 5 (DF plotter wrappers) to `auxi/caviar.py`**

Append:

```python
# =============================================================================
# CAPA 5 — DIRECT FORECASTING PLOTTER WRAPPERS (Approach C)
# =============================================================================

def caviar_plot_forecasted_scatters(df, vars_x, vars_y, quantiles, h_short=3, h_long=12,
                                    controls=None, breach_quantiles=None):
    """Caviar wrapper — augment + delegate to qreg.plot_forecasted_scatters."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return plot_forecasted_scatters(
        df=work, x=vars_x[0], y=vars_y, quantiles=quantiles,
        h_short=h_short, h_long=h_long, controls=augmented_controls,
    )


def caviar_plot_contemporaneous_vs_predictive_coefs(df, vars_x, vars_y, h=1,
                                                    controls=None, errors="robust",
                                                    breach_quantiles=None):
    """Caviar wrapper — augment + delegate to qreg.plot_contemporaneous_vs_predictive_coefs."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return plot_contemporaneous_vs_predictive_coefs(
        df=work, x=vars_x[0], y=vars_y, h=h,
        controls=augmented_controls, errors=errors,
    )
```

- [ ] **Step 7: Add Capa 6 (DF eval wrapper) to `auxi/caviar.py`**

Append:

```python
# =============================================================================
# CAPA 6 — DIRECT FORECASTING EVALUATION WRAPPERS (Approach C)
# =============================================================================

def caviar_evaluate_direct_forecasting(df, vars_x, vars_y, controls=None,
                                       tau=0.05, max_h=90, train_fraction=0.8,
                                       breach_quantiles=None):
    """Caviar wrapper — augment + delegate to diagnostics.evaluate_direct_forecasting."""
    if isinstance(vars_x, str):
        vars_x = [vars_x]
    work = _augment_with_breaches(df, vars_x, vars_y, breach_quantiles)
    augmented_controls = (
        list(vars_x[1:]) + (controls or []) + ["upside_breach", "downside_breach"]
    )
    return evaluate_direct_forecasting(
        df=work, x=vars_x[0], y=vars_y, controls=augmented_controls,
        tau=tau, max_h=max_h, train_fraction=train_fraction,
    )
```

- [ ] **Step 8: Clear cache and run all the smoke tests — expect all pass**

```
del /s /q auxi\__pycache__
python -m pytest tests/test_reorg_caviar_wrappers.py -v
```

Expected: all 9 tests PASS. The end-to-end runs verify the wrapper plumbing through qreg and that `df` is not mutated.

- [ ] **Step 9: Run the original caviar tests — regression check**

```
python -m pytest tests/test_caviar.py -v
```

Expected: pass.

- [ ] **Step 10: Commit**

```
git add auxi/caviar.py tests/test_reorg_caviar_wrappers.py
git commit -m "feat: add caviar DF wrappers (Approach C) — augment panel + delegate"
```

---

## Task 7: Delete `auxi/forecasting.py`

**Files:**
- Delete: `auxi/forecasting.py`

- [ ] **Step 1: Verify nothing in `auxi/` still imports from it**

Run:
```
python -c "import pathlib; [print(p) for p in pathlib.Path('auxi').glob('**/*.py') if 'from auxi.forecasting' in p.read_text(encoding='utf-8') or 'import auxi.forecasting' in p.read_text(encoding='utf-8')]"
```

Expected: empty output. If anything prints, fix that file first before deleting.

- [ ] **Step 2: Delete the file**

```
del "auxi/forecasting.py"
```

- [ ] **Step 3: Clear cache and verify imports still work**

```
del /s /q auxi\__pycache__
python -c "import auxi.qreg, auxi.caviar, auxi.diagnostics; print('OK')"
```

Expected: prints `OK`. No `ImportError`.

- [ ] **Step 4: Run all tests as final regression**

```
python -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add auxi/forecasting.py
git status
git commit -m "refactor: delete auxi/forecasting.py (contents redistributed)"
```

---

## Task 8: Update `direct_forecasting.ipynb`

**Files:**
- Modify: `direct_forecasting.ipynb`

Recommended: open the notebook in Jupyter Lab/Notebook to edit interactively. Use the Find/Replace feature within Jupyter.

- [ ] **Step 1: Change the existing import line**

Find: `import auxi.forecasting as fc`
Replace with: `import auxi.qreg as fc`

(Single occurrence.)

- [ ] **Step 2: Add a new import for diagnostics**

In the same import cell (just below the line you changed), add:

```python
import auxi.diagnostics as diags
```

- [ ] **Step 3: Redirect calls that now live in `auxi.diagnostics`**

Use Jupyter's Find/Replace, "Match Case" enabled. Do each substitution in turn:

| Find | Replace with | Occurrences expected |
|---|---|---|
| `fc.evaluate_direct_forecasting` | `diags.evaluate_direct_forecasting` | 2 |
| `fc.plot_fallout_errors` | `diags.plot_fallout_errors` | 1 |
| `fc.compute_unconditional_coverage_unified` | `diags.compute_unconditional_coverage_unified` | 1 |
| `fc.plot_unconditional_coverage` | `diags.plot_unconditional_coverage` | 1 |
| `fc.compute_conditional_coverage` | `diags.compute_conditional_coverage` | 1 |
| `fc.plot_conditional_coverage` | `diags.plot_conditional_coverage` | 1 |

After all substitutions: 7 redirects done.

- [ ] **Step 4: Handle the cell that calls `fc.select_horizon_rolling_origin`**

The function `select_horizon_rolling_origin` has been deleted in this reorg (spec decision). Find the cell containing `fc.select_horizon_rolling_origin` and comment out its entire body. Add this header comment at the top of the cell:

```python
# DEPRECADO 2026-06-26: select_horizon_rolling_origin fue eliminada en el reorg
# de auxi/. Si se necesita la selección de h, hacer un sweep manual de horizontes
# usando diags.evaluate_direct_forecasting en bucle.
#
# (Cell preserved as documentation, not executed.)

# (resto del código de la celda comentado a continuación)
```

Then prepend `# ` to every code line below in the cell. The cell becomes non-executing.

- [ ] **Step 5: Verify the 5 calls that stayed**

The following calls should NOT have changed — they still use `fc.` and resolve via `auxi.qreg` now:
- `fc.direct_forecasting`
- `fc.insample_direct_forecasting`
- `fc.get_oos_predictions`
- `fc.plot_forecasted_scatters`
- `fc.plot_contemporaneous_vs_predictive_coefs`

Use Jupyter's Find (without Replace) to confirm each still appears, with the same number of occurrences as before the edits.

- [ ] **Step 6: Restart kernel and run all cells**

In Jupyter: Kernel → Restart Kernel and Run All Cells.

Expected: all cells run without errors except the deprecated cell from Step 4 (which is fully commented out and just contains comments — should still "succeed" as a no-op).

If anything errors:
- `AttributeError: module 'auxi.qreg' has no attribute X` → the function is in `diagnostics`, switch to `diags.X`
- `NameError: name 'diags' is not defined` → the `import auxi.diagnostics as diags` cell didn't run yet; check that it's above the cells that use it

- [ ] **Step 7: Commit**

```
git add direct_forecasting.ipynb
git commit -m "refactor: update direct_forecasting.ipynb imports/calls for auxi reorg"
```

---

## Task 9: Update `distribution_analysis.ipynb`

**Files:**
- Modify: `distribution_analysis.ipynb`

- [ ] **Step 1: Change the existing import line**

Find: `from auxi.forecasting import insample_direct_forecasting`
Replace with: `from auxi.qreg import insample_direct_forecasting`

(Single occurrence.)

- [ ] **Step 2: Add the diagnostics import**

In the import cell, just below the line you changed, add:

```python
import auxi.diagnostics as diags
```

- [ ] **Step 3: Redirect distribution-fitting diagnostic calls**

Use Jupyter's Find/Replace, "Match Case" enabled:

| Find | Replace with | Occurrences expected |
|---|---|---|
| `da.fit_and_diagnose_jsu` | `diags.fit_and_diagnose_jsu` | 1 |
| `da.evaluate_oos_pit` | `diags.evaluate_oos_pit` | 1 |
| `da.fit_and_diagnose_skewt` | `diags.fit_and_diagnose_skewt` | 1 |
| `da.evaluate_oos_pit_skewt` | `diags.evaluate_oos_pit_skewt` | 2 |

After all substitutions: 5 redirects done.

- [ ] **Step 4: Verify the calls that stayed**

All other `da.<funcname>` calls (fitters, MDE, comparators, OOS-parameter generators, etc.) should still appear unchanged. Use Find to spot-check at least one (e.g., `da.fit_jsu`, `da.mde_jsu_weighted`).

- [ ] **Step 5: Restart kernel and run all cells**

Kernel → Restart Kernel and Run All Cells.

Expected: all cells run without errors.

If `AttributeError: module 'auxi.distribution_analysis' has no attribute 'fit_and_diagnose_jsu'` → you missed swapping a `da.` to `diags.` somewhere. Use Find on `da.fit_and_diagnose` and `da.evaluate_oos_pit` to locate any missed occurrences.

- [ ] **Step 6: Commit**

```
git add distribution_analysis.ipynb
git commit -m "refactor: update distribution_analysis.ipynb imports/calls for auxi reorg"
```

---

## Task 10: Write `auxi/README.md`

**Files:**
- Create: `auxi/README.md`

- [ ] **Step 1: Create `auxi/README.md`**

```markdown
# auxi/ — Mapa de módulos

Backend de utilidades del TFM. Organizado por propósito.

## Motores y estimadores
- **qreg.py** — Quantile-regression engine (`q_reg`, `multiple_q_regs`, plotters), direct-forecasting estimators (`direct_forecasting`, `insample_direct_forecasting`, `get_oos_predictions`), y sus plotters.
- **caviar.py** — CAViaR con indicadores binarios (`caviar_i`, `multiple_caviar_i`) y wrappers de direct forecasting que delegan a qreg (Approach C). ⚠ Lookahead (B) sin curar — ver módulo docstring.
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
```

- [ ] **Step 2: Commit**

```
git add auxi/README.md
git commit -m "docs: add auxi/README.md as module map"
```

---

## Task 11: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Clear all caches**

```
del /s /q auxi\__pycache__
del /s /q tests\__pycache__
```

- [ ] **Step 2: Smoke-test every module/subpackage import**

```
python -c "import auxi.qreg, auxi.caviar, auxi.diagnostics, auxi.distribution_analysis, auxi.predictive_density, auxi.risk_metrics, auxi.risk_metrics_boosted, auxi.vulnerability_metrics, auxi.data, auxi.descriptive; print('All imports OK')"
```

Expected: prints `All imports OK`.

- [ ] **Step 3: Verify the public surface against the pre-reorg snapshot**

Recompute the dir() lists from Task 1:

```
python -c "import auxi.qreg as qr; print(sorted(n for n in dir(qr) if not n.startswith('_')))"
python -c "import auxi.diagnostics as diags; print(sorted(n for n in dir(diags) if not n.startswith('_')))"
python -c "import auxi.caviar as cv; print(sorted(n for n in dir(cv) if not n.startswith('_')))"
python -c "import auxi.distribution_analysis as da; print(sorted(n for n in dir(da) if not n.startswith('_')))"
```

Compare against the snapshot from Task 1:
- `qr` should now include all the DF function names that used to be in `auxi.forecasting`.
- `diags` should include the new DF/distribution_fitting names plus the original specification + series names.
- `cv` should include the 6 new `caviar_*` wrappers in addition to the originals.
- `da` should NOT include the 7 moved diagnostic functions; it should still include the fitters.

`select_horizon_rolling_origin` should no longer appear anywhere.

- [ ] **Step 4: Run all tests**

```
python -m pytest tests -v
```

Expected: all tests PASS.

- [ ] **Step 5: Final commit and tag**

```
git status
```

Should be clean.

```
git tag post-reorg-2026-06-26
git log --oneline -15
```

Expected log shows the 10 commit messages from Tasks 0–10 in order.

- [ ] **Step 6: (Optional) Clean up the temporary reorg test files**

The three `tests/test_reorg_*.py` files served their purpose during the migration. You can keep them as ongoing regression guards or delete them:

To delete:
```
del tests\test_reorg_qreg.py
del tests\test_reorg_diagnostics_subpkg.py
del tests\test_reorg_caviar_wrappers.py
git add tests
git commit -m "chore: remove temporary reorg smoke tests"
```

Recommendation: **keep them**. They're cheap and they document the post-reorg public API contract.
