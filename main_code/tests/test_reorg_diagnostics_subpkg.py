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


def test_evaluate_direct_forecasting_resolvable():
    assert callable(diags.evaluate_direct_forecasting)


def test_compute_fallout_errors_resolvable():
    assert callable(diags.compute_fallout_errors)


def test_compute_unconditional_coverage_unified_resolvable():
    assert callable(diags.compute_unconditional_coverage_unified)


def test_compute_conditional_coverage_resolvable():
    assert callable(diags.compute_conditional_coverage)


def test_diagnose_residual_acf_resolvable():
    assert callable(diags.diagnose_residual_acf)


def test_submodule_direct_forecasting_importable():
    from auxi.diagnostics import direct_forecasting
    assert callable(direct_forecasting.evaluate_direct_forecasting)


def test_jsu_ks_test_resolvable():
    assert callable(diags.jsu_ks_test)


def test_evaluate_oos_pit_resolvable():
    assert callable(diags.evaluate_oos_pit)


def test_evaluate_oos_pit_skewt_resolvable():
    assert callable(diags.evaluate_oos_pit_skewt)


def test_fit_and_diagnose_jsu_resolvable():
    assert callable(diags.fit_and_diagnose_jsu)


def test_fit_and_diagnose_skewt_resolvable():
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
