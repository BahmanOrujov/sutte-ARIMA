import numpy as np
import pytest
from statsmodels.tsa.arima.model import ARIMA

from sutte_arima import (AlphaSutte, SutteARIMA, calculate_metrics, diebold_mariano, find_order,
                         pesaran_timmermann, theil_u2)

R_EXAMPLE = [94.77, 96.23, 98.12, 99.09, 100.04, 100.12, 99.93, 100.09, 101.44, 102.38, 103.68,
             104.12, 104.81, 105.35, 106.36, 106.89, 107.35, 107.21, 107.72, 108.54, 109.57, 112.03]


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))


def sutte_forecast_r(x):
    """Literal translation of sutteForecastR::alpha.sutte (Ahmar, CRAN 0.1), 1-based indexing."""
    n = len(x)
    al = [None] + list(x)
    sutte4 = [None] * (n + 3)
    for i in range(2, n + 1):
        d = al[i] - al[i - 1]
        m = (al[i] + al[i - 1]) / 2
        sutte4[i] = d / m * al[i]
    sutte5 = [None] * (n + 3)
    for i in range(2, n - 1):
        sutte5[i + 2] = (sutte4[i] + sutte4[i + 1] + sutte4[i + 2]) / 3
    sutteoke = {}
    for i in range(2, n - 2):
        sutteoke[i - 1] = sutte5[i + 2] + al[i + 2]   # forecast of al[i + 3]
    # sutteoke[k] (k = 1..n-4) predicts al[k + 4], i.e. 0-based index k + 3
    return {k + 3: v for k, v in sutteoke.items()}


def test_alpha_sutte_matches_reference_r_implementation():
    fitted = AlphaSutte.compute_fitted(np.array(R_EXAMPLE))
    reference = sutte_forecast_r(R_EXAMPLE)
    for idx, value in reference.items():
        assert fitted[idx] == pytest.approx(value, rel=1e-12)


def test_default_hybrid_is_backward_compatible_with_v1():
    y = random_walk()
    res = SutteARIMA(y, order=(0, 1, 1)).fit()
    arima = ARIMA(y, order=(0, 1, 1)).fit().fittedvalues
    sutte = AlphaSutte.compute_fitted(y)
    np.testing.assert_allclose(res.fittedvalues[4:], 0.5 * (arima[4:] + sutte[4:]), rtol=1e-8)
    assert np.isnan(res.fittedvalues[0])          # diffuse start is no longer reported as a fit


def test_find_order_accepts_lowercase_criterion(capsys):
    order = find_order(random_walk(150), max_p=1, max_q=1, ic="aicc", verbose=True)
    assert order[1] == 1
    assert "Recommended order" in capsys.readouterr().out


def test_rolling_forecast_has_no_look_ahead():
    y = random_walk(400, seed=1)
    train, test = y[:300], y[300:]
    for refit in (None, 10):
        res = SutteARIMA(train, order=(1, 1, 0), transform="log", trend="drift").fit()
        base = res.rolling_forecast(test, refit_every=refit)
        altered = test.copy()
        altered[21:] *= 1.5
        alt = res.rolling_forecast(altered, refit_every=refit)
        cols = ["SutteARIMA", "ARIMA", "AlphaSutte", "Naive"]
        np.testing.assert_allclose(base.loc[:21, cols].values, alt.loc[:21, cols].values, rtol=1e-10)
        assert not np.allclose(base.loc[22:, "ARIMA"].values, alt.loc[22:, "ARIMA"].values)


def test_first_rolling_forecast_equals_one_step_forecast():
    y = random_walk(300, seed=2)
    res = SutteARIMA(y[:250], order=(1, 1, 0), transform="log", trend="drift").fit()
    roll = res.rolling_forecast(y[250:])
    fc = res.forecast(steps=1)
    assert roll.loc[0, "ARIMA"] == pytest.approx(fc.loc[1, "ARIMA"], rel=1e-10)
    assert roll.loc[0, "Naive"] == y[249]


def test_log_transform_interval_contains_median():
    y = random_walk(300, seed=3)
    fc = SutteARIMA(y, order=(1, 1, 0), transform="log", trend="drift").fit().forecast(steps=20)
    assert (fc["Lower_95"] > 0).all()
    assert ((fc["Lower_95"] <= fc["ARIMA"]) & (fc["ARIMA"] <= fc["Upper_95"])).all()


def test_optimal_weight_is_bounded_and_evaluate_runs():
    y = random_walk(350, seed=4)
    res = SutteARIMA(y[:300], order=(1, 1, 0), transform="log", trend="drift", weight="optimal").fit()
    assert 0.0 <= res.weight <= 1.0
    table = res.evaluate(y[300:], refit_every=25)
    assert table.loc["Naive", "U2"] == pytest.approx(1.0)
    assert {"DM_MSE_p", "PT_p", "U2"}.issubset(table.columns)


def test_exog_requires_future_values():
    y = random_walk(200, seed=5)
    x = np.zeros(200)
    x[100:105] = 1.0
    res = SutteARIMA(y, order=(1, 1, 0), exog=x).fit()
    with pytest.raises(ValueError):
        res.forecast(steps=3)
    assert len(res.forecast(steps=3, exog=np.zeros(3))) == 3


def test_forecast_comparison_tests():
    rng = np.random.default_rng(6)
    e1, e2 = rng.standard_normal(500), 1.5 * rng.standard_normal(500)
    dm12, dm21 = diebold_mariano(e1, e2), diebold_mariano(e2, e1)
    assert dm12["statistic"] == pytest.approx(-dm21["statistic"])
    assert dm12["statistic"] < 0 and dm12["p_value"] < 0.01

    change = rng.standard_normal(500)
    pt = pesaran_timmermann(change, change)
    assert pt["hit_rate"] == 1.0 and pt["p_value"] < 1e-6

    a = np.cumsum(rng.standard_normal(100))
    naive = np.r_[a[0], a[:-1]]
    assert theil_u2(a, naive, naive) == pytest.approx(1.0)
    assert calculate_metrics(a, naive, benchmark=naive)["U2"] == pytest.approx(1.0)
