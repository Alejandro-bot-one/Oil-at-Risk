"""auxi.diagnostics - subpackage of diagnostics organized by type.

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
from .direct_forecasting import (
    evaluate_direct_forecasting,
    evaluate_direct_forecasting_single,
    compute_rolling_pinball,
    plot_rolling_pinball,
    get_oos_predictions_rolling,
    diagnose_residual_acf,
    compute_fallout_errors,
    plot_fallout_errors,
    evaluate_cumulative_fallout,
    compute_unconditional_coverage_unified,
    plot_unconditional_coverage_unified,
    compute_conditional_coverage,
    plot_conditional_coverage,
    plot_coverage_dashboard,
    plot_unconditional_coverage,
    tick_loss_series,
    diebold_mariano_test,
    compute_dm_comparison,
)
from .distribution_fitting import (
    jsu_ks_test,
    evaluate_oos_pit,
    evaluate_oos_pit_skewt,
    oos_pit_calibration,
    plot_oos_pit_calibration,
    fit_and_diagnose_jsu,
    fit_and_diagnose_skewt,
)
from .ews import (
    compute_ccf,
    granger_causality_test,
    compute_anticipation_test,
    compute_ews_battery,
    compute_coherence_test,
    plot_ccf,
    plot_ews_battery,
    plot_coherence_dashboard,
)
