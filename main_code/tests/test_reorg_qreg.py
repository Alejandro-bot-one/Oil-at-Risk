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
