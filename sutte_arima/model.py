"""
SutteARIMA: statsmodels-style hybrid of the alpha-Sutte indicator and Box-Jenkins ARIMA.

Version 2.0 keeps the original estimator (equal-weight average of an ARIMA forecast and the
alpha-Sutte forecast on the original scale) as the default, and adds the tools needed to use it
in a statistically sound way:

* ``transform="log"``  - model multiplicative / heteroskedastic series (e.g. prices) in logs;
* ``trend="drift"``    - explicit drift; statsmodels silently drops the constant when d > 0;
* ``exog``             - regression with ARIMA errors (documented semantics, see ``SutteARIMA``);
* ``weight``           - combination weight on the ARIMA component, fixed or ``"optimal"``
                         (Bates & Granger, 1969);
* ``rolling_forecast`` / ``evaluate`` - leakage-free expanding-window one-step-ahead evaluation
                         against the naive (random-walk) benchmark;
* ``diebold_mariano``, ``pesaran_timmermann``, ``theil_u2`` - forecast comparison tests;
* prediction intervals taken from the predictive distribution of the fitted ARIMA model;
* no global warning filters are installed on import.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller


# ==============================================================================
# 0. INTERNAL HELPERS
# ==============================================================================

def _check_transform(transform: Optional[str]) -> Optional[str]:
    if transform is None or (isinstance(transform, str) and transform.lower() == "none"):
        return None
    if isinstance(transform, str) and transform.lower() == "log":
        return "log"
    raise ValueError(f"Unknown transform={transform!r}. Use None or 'log'.")


def _forward(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.log(y) if transform == "log" else y


def _inverse(z: np.ndarray, transform: Optional[str]) -> np.ndarray:
    return np.exp(z) if transform == "log" else z


def _resolve_trend(trend: Any, d: int, D: int = 0) -> Any:
    """Translate the ``"drift"`` alias into a statsmodels trend specification.

    statsmodels' ARIMA writes the trend in levels: ``y_t = trend_t + u_t`` with ``u_t`` integrated.
    A constant is therefore not identified when ``d + D > 0``; a drift (a constant in the
    ``(d + D)``-th difference) is the trend term of power ``d + D``.
    """
    if trend is None:
        return None
    if isinstance(trend, str) and trend.lower() == "drift":
        k = int(d) + int(D)
        return "c" if k == 0 else [0] * k + [1]
    return trend


def _adf_pvalue(x: np.ndarray) -> float:
    """ADF p-value; works whether statsmodels returns a tuple or a result object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        res = adfuller(x, autolag="AIC")
    return float(getattr(res, "pvalue", res[1]))


def _arch_lm(x: np.ndarray, nlags: int) -> Tuple[float, float]:
    """Engle's ARCH-LM statistic and p-value; works with tuple or result-object returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        res = het_arch(x, nlags=nlags)
    stat = getattr(res, "lm", None)
    pval = getattr(res, "lmpval", None)
    if stat is None or pval is None:
        stat, pval = res[0], res[1]
    return float(stat), float(pval)


def _as_2d(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _fit_statsmodels_arima(z: np.ndarray, exog: Optional[np.ndarray], order: Tuple[int, int, int],
                           seasonal_order: Optional[Tuple[int, int, int, int]], trend: Any,
                           **fit_kwargs) -> ARIMAResults:
    """Fit ARIMA without leaking warning filters into the caller's session."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(z, exog=exog, order=order,
                      seasonal_order=seasonal_order if seasonal_order is not None else (0, 0, 0, 0),
                      trend=trend)
        return model.fit(**fit_kwargs)


def _combination_weight(y: np.ndarray, a: np.ndarray, s: np.ndarray, relative: bool) -> float:
    """Least-squares weight w on ``a`` in ``w*a + (1-w)*s`` (Bates & Granger, 1969), clipped to [0, 1].

    Minimises sum(e_s + w (e_a - e_s))^2 with e_a = y - a, e_s = y - s, giving
    w* = sum(e_s (e_s - e_a)) / sum((e_a - e_s)^2).
    """
    mask = ~(np.isnan(y) | np.isnan(a) | np.isnan(s))
    if mask.sum() < 3:
        return 0.5
    scale = y[mask] if relative else 1.0
    e_a = (y[mask] - a[mask]) / scale
    e_s = (y[mask] - s[mask]) / scale
    denom = np.sum((e_a - e_s) ** 2)
    if denom <= 0:
        return 0.5
    return float(np.clip(np.sum(e_s * (e_s - e_a)) / denom, 0.0, 1.0))


# ==============================================================================
# 1. EVALUATION METRICS AND FORECAST-COMPARISON TESTS
# ==============================================================================

def calculate_metrics(actual: Any, predicted: Any, train_data: Optional[Any] = None,
                      benchmark: Optional[Any] = None) -> Dict[str, float]:
    """
    Forecast accuracy metrics computed from actual and predicted values.

      - MAPE (%)  : Mean Absolute Percentage Error
      - MSE, RMSE, MAE
      - sMAPE (%) : Symmetric Mean Absolute Percentage Error
      - TIC       : Theil's inequality coefficient U1 = RMSE / (RMS(actual) + RMS(predicted)).
                    For persistent level series U1 is close to 0 for almost any forecast and is
                    therefore not informative about forecasting skill; use U2.
      - U2        : Theil's U2 = RMSE(model) / RMSE(benchmark). Only returned when ``benchmark``
                    is given (normally the naive forecast y_{t-1}). U2 < 1 means the model beats
                    the benchmark.
      - MASE      : MAE scaled by the in-sample MAE of the one-step naive forecast (Hyndman &
                    Koehler, 2006). MASE < 1 means better than the *in-sample* naive MAE, which is
                    not the same as beating the naive forecast out of sample.
      - MaxAE, MBE (mean bias error, actual - predicted)
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    b = None if benchmark is None else np.asarray(benchmark, dtype=float).ravel()

    mask = ~(np.isnan(a) | np.isnan(p) | np.isinf(a) | np.isinf(p))
    if b is not None:
        mask &= ~(np.isnan(b) | np.isinf(b))
    a, p = a[mask], p[mask]
    if b is not None:
        b = b[mask]
    if len(a) == 0:
        return {}

    errors = a - p
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(errors / a)
        clean_ape = ape[~np.isnan(ape) & ~np.isinf(ape)]
    mape = float(np.mean(clean_ape) * 100.0) if len(clean_ape) > 0 else np.nan

    denom = np.abs(a) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_arr = np.where(denom == 0, 0.0, 200.0 * abs_errors / denom)
    smape = float(np.mean(smape_arr))

    norm_denom = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(p ** 2))
    tic = float(rmse / norm_denom) if norm_denom > 0 else np.nan

    if train_data is not None:
        tr = np.asarray(train_data, dtype=float).ravel()
        tr = tr[~np.isnan(tr)]
        scale = np.mean(np.abs(np.diff(tr)))
        mase = float(mae / scale) if scale > 0 else np.nan
    else:
        mase = np.nan

    out = {
        "MAPE (%)": round(mape, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "sMAPE (%)": round(smape, 4),
        "TIC": round(tic, 4),
        "MASE": round(mase, 4) if not np.isnan(mase) else None,
        "MaxAE": round(float(np.max(abs_errors)), 4),
        "MBE": round(float(np.mean(errors)), 4),
    }
    if b is not None:
        rmse_b = float(np.sqrt(np.mean((a - b) ** 2)))
        out["U2"] = round(rmse / rmse_b, 4) if rmse_b > 0 else np.nan
    return out


def theil_u2(actual: Any, predicted: Any, benchmark: Any) -> float:
    """Theil's U2 = RMSE(predicted) / RMSE(benchmark)."""
    a, p, b = (np.asarray(v, dtype=float).ravel() for v in (actual, predicted, benchmark))
    mask = ~(np.isnan(a) | np.isnan(p) | np.isnan(b))
    rmse_p = np.sqrt(np.mean((a[mask] - p[mask]) ** 2))
    rmse_b = np.sqrt(np.mean((a[mask] - b[mask]) ** 2))
    return float(rmse_p / rmse_b) if rmse_b > 0 else np.nan


def diebold_mariano(errors_1: Any, errors_2: Any, h: int = 1, power: int = 2,
                    harvey_correction: bool = True) -> Dict[str, float]:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy.

    Loss differential d_t = |e1_t|^power - |e2_t|^power. The long-run variance uses the
    autocovariances of d_t up to lag h-1. With ``harvey_correction`` the statistic is multiplied by
    sqrt((T + 1 - 2h + h(h-1)/T) / T) and compared with Student-t(T-1) (Harvey, Leybourne &
    Newbold, 1997).

    H0: equal accuracy. A negative statistic means model 1 has the smaller loss.
    """
    e1 = np.asarray(errors_1, dtype=float).ravel()
    e2 = np.asarray(errors_2, dtype=float).ravel()
    mask = ~(np.isnan(e1) | np.isnan(e2))
    d = np.abs(e1[mask]) ** power - np.abs(e2[mask]) ** power
    T = len(d)
    if T < 3:
        return {"statistic": np.nan, "p_value": np.nan, "mean_loss_differential": np.nan, "n": T}
    dbar = float(np.mean(d))
    dc = d - dbar
    lrv = np.sum(dc ** 2) / T
    for k in range(1, h):
        lrv += 2.0 * np.sum(dc[k:] * dc[:-k]) / T
    if lrv <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "mean_loss_differential": dbar, "n": T}
    dm = dbar / np.sqrt(lrv / T)
    if harvey_correction:
        dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        p = 2.0 * stats.t.sf(abs(dm), df=T - 1)
    else:
        p = 2.0 * stats.norm.sf(abs(dm))
    return {"statistic": float(dm), "p_value": float(p), "mean_loss_differential": dbar, "n": T}


def pesaran_timmermann(actual_change: Any, predicted_change: Any) -> Dict[str, float]:
    """
    Pesaran-Timmermann (1992) test of directional predictive ability.

    Inputs are realised and predicted *changes* (or returns). Observations with a zero realised
    change are dropped. H0: the predicted direction is independent of the realised direction;
    the p-value is one-sided (H1: predictive ability). The statistic is undefined (NaN) when every
    prediction has the same sign.
    """
    a = np.asarray(actual_change, dtype=float).ravel()
    p = np.asarray(predicted_change, dtype=float).ravel()
    mask = ~(np.isnan(a) | np.isnan(p)) & (a != 0)
    a_up, p_up = a[mask] > 0, p[mask] > 0
    T = len(a_up)
    if T < 3:
        return {"hit_rate": np.nan, "statistic": np.nan, "p_value": np.nan, "n": T}
    hit = float(np.mean(a_up == p_up))
    py, px = a_up.mean(), p_up.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hit = p_star * (1 - p_star) / T
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / T \
        + 4 * py * px * (1 - py) * (1 - px) / T ** 2
    if v_hit - v_star <= 0:
        return {"hit_rate": hit, "statistic": np.nan, "p_value": np.nan, "n": T}
    s = (hit - p_star) / np.sqrt(v_hit - v_star)
    return {"hit_rate": hit, "statistic": float(s), "p_value": float(stats.norm.sf(s)), "n": T}


# ==============================================================================
# 2. ALPHA-SUTTE INDICATOR (Ahmar, 2017)
# ==============================================================================

class AlphaSutte:
    """
    alpha-Sutte indicator (Ahmar, 2017; reference implementation: CRAN package
    ``sutteForecastR`` 0.1, function ``alpha.sutte``).

        delta_t = (1/3) * sum_{j=1..3} y_{t-j} * (y_{t-j} - y_{t-j-1}) / ((y_{t-j} + y_{t-j-1}) / 2)
        yhat_t  = y_{t-1} + delta_t

    Because y_{t-j} / mean(y_{t-j}, y_{t-j-1}) = 1 + O(relative change), delta_t is approximately the
    average of the last three changes: the indicator is a short-run momentum extrapolation. If the
    series is a random walk, the one-step error variance is about (4/3) sigma^2, i.e. its RMSE is
    about sqrt(4/3) = 1.155 times that of the naive forecast.
    """

    @staticmethod
    def step_increment(y_t4: float, y_t3: float, y_t2: float, y_t1: float) -> float:
        dx, dy, dz = y_t3 - y_t4, y_t2 - y_t3, y_t1 - y_t2
        mx, my, mz = (y_t3 + y_t4) / 2.0, (y_t2 + y_t3) / 2.0, (y_t1 + y_t2) / 2.0

        tx = (y_t3 * (dx / mx)) if abs(mx) > 1e-12 else dx
        ty = (y_t2 * (dy / my)) if abs(my) > 1e-12 else dy
        tz = (y_t1 * (dz / mz)) if abs(mz) > 1e-12 else dz

        return float((tx + ty + tz) / 3.0)

    @classmethod
    def compute_fitted(cls, series: np.ndarray) -> np.ndarray:
        """One-step-ahead values yhat_t for t >= 4, each using only y_{t-4}, ..., y_{t-1}."""
        series = np.asarray(series, dtype=float)
        n = len(series)
        fitted = np.full(n, np.nan)
        for t in range(4, n):
            delta = cls.step_increment(series[t - 4], series[t - 3], series[t - 2], series[t - 1])
            fitted[t] = series[t - 1] + delta
        return fitted

    @classmethod
    def compute_forecast(cls, series: np.ndarray, steps: int) -> np.ndarray:
        """Recursive multi-step forecast (forecasts are fed back as if they were observations)."""
        hist = np.asarray(series, dtype=float).tolist()
        fc = []
        for _ in range(steps):
            delta = cls.step_increment(hist[-4], hist[-3], hist[-2], hist[-1])
            y_next = hist[-1] + delta
            fc.append(y_next)
            hist.append(y_next)
        return np.asarray(fc, dtype=float)


# ==============================================================================
# 3. ORDER SELECTION
# ==============================================================================

def find_order(data: Any,
               max_p: int = 3,
               max_q: int = 3,
               ic: str = "AICc",
               verbose: bool = False,
               *,
               d: Optional[int] = None,
               transform: Optional[str] = None,
               trend: Any = None,
               exog: Any = None) -> Tuple[int, int, int]:
    """
    Recommended (p, d, q) order for the ARIMA component.

      1. ``d`` from ADF tests on the (optionally log-transformed) series, unless ``d`` is given:
         d = 0 if the level is stationary at 5%, else 1 if the first difference is, else 2.
      2. Grid search over p in [0, max_p] and q in [0, max_q] with the same ``trend`` and ``exog``
         as the final model.
      3. Minimum of the chosen criterion ('AICc', 'AIC', 'BIC' or 'HQIC', case-insensitive). The
         criteria are taken from statsmodels and count every estimated parameter (incl. sigma2).
    """
    transform = _check_transform(transform)
    y = np.asarray(data, dtype=float).ravel()
    n = len(y)
    if n < 8:
        raise ValueError(f"Need at least 8 observations, got {n}.")
    if transform == "log":
        if np.any(y <= 0):
            raise ValueError("transform='log' requires strictly positive data.")
        y = np.log(y)
    X = _as_2d(exog)

    columns = {"AICC": "AICc", "AIC": "AIC", "BIC": "BIC", "HQIC": "HQIC"}
    ic_key = ic.upper()
    if ic_key not in columns:
        raise ValueError(f"Unknown criterion ic='{ic}'. Choose from {sorted(columns.values())}.")

    adf_lvl = _adf_pvalue(y)
    if d is None:
        if adf_lvl < 0.05:
            d = 0
        else:
            d = 1 if _adf_pvalue(np.diff(y)) < 0.05 else 2
    trend_sm = _resolve_trend(trend, d)

    best_score = float("inf")
    best_order = (1, d, 0)
    records = []
    for p, q in product(range(max_p + 1), range(max_q + 1)):
        try:
            fit = _fit_statsmodels_arima(y, X, (p, d, q), None, trend_sm)
        except Exception:
            continue
        scores = {"AICC": float(fit.aicc), "AIC": float(fit.aic), "BIC": float(fit.bic),
                  "HQIC": float(fit.hqic)}
        records.append({"order": (p, d, q), **{columns[k]: round(v, 3) for k, v in scores.items()}})
        if scores[ic_key] < best_score:
            best_score = scores[ic_key]
            best_order = (p, d, q)

    if verbose and records:
        df_rank = pd.DataFrame(records).sort_values(columns[ic_key]).reset_index(drop=True)
        print(f"ADF stationarity: level p-value={adf_lvl:.4f} -> differencing d={d}")
        print(f"Recommended order: {best_order} (min {columns[ic_key]}: {best_score:.3f})")
        print("\nTop candidate orders:")
        print(df_rank.head(5).to_string(index=False))
        print("-" * 60)

    return best_order


# ==============================================================================
# 4. RESULTS
# ==============================================================================

class SutteARIMAResults:
    """
    Results returned by ``SutteARIMA.fit()``: estimates, information criteria, residual
    diagnostics, in-sample accuracy, forecasts and leakage-free out-of-sample evaluation.
    """

    def __init__(self,
                 model: "SutteARIMA",
                 arima_res: ARIMAResults,
                 fittedvalues: np.ndarray,
                 order: Tuple[int, int, int],
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 arima_fitted: Optional[np.ndarray] = None,
                 sutte_fitted: Optional[np.ndarray] = None,
                 weight: float = 0.5):
        self.model = model
        self.arima_results = arima_res
        self.fittedvalues = fittedvalues
        self.arima_fittedvalues = arima_fitted
        self.sutte_fittedvalues = sutte_fitted
        self._order = order
        self._seasonal_order = seasonal_order
        self._weight = float(weight)
        self.endog = model.endog
        self.transform = model.transform

        # Residuals on the original scale (as in v1) and on the modelling scale (used for tests).
        self.resid = self.endog - self.fittedvalues
        with np.errstate(divide="ignore", invalid="ignore"):
            self.resid_model = (_forward(self.endog, self.transform)
                                - _forward(self.fittedvalues, self.transform))

        naive_in = np.r_[np.nan, self.endog[:-1]]
        self.in_sample_metrics = calculate_metrics(
            actual=self.endog[4:], predicted=self.fittedvalues[4:],
            train_data=self.endog, benchmark=naive_in[4:])

        clean_resid = self.resid_model[4:][~np.isnan(self.resid_model[4:])]
        self._dw = float(durbin_watson(clean_resid)) if len(clean_resid) > 0 else np.nan

        if len(clean_resid) > 3:
            jb = jarque_bera(clean_resid)
            self._jb_stat, self._jb_pval = float(jb[0]), float(jb[1])
            self._skew, self._kurtosis = float(jb[2]), float(jb[3])
        else:
            self._jb_stat = self._jb_pval = self._skew = self._kurtosis = np.nan

        self._lb_stat = self._lb_pval = np.nan
        self._lb_lag = 0
        if len(clean_resid) > 6:
            try:
                lb_lag = min(10, len(clean_resid) // 3)
                lb_res = acorr_ljungbox(clean_resid, lags=[lb_lag], return_df=True)
                self._lb_stat = float(lb_res["lb_stat"].iloc[0])
                self._lb_pval = float(lb_res["lb_pvalue"].iloc[0])
                self._lb_lag = lb_lag
            except Exception:
                pass

        self._arch_stat = self._arch_pval = np.nan
        if len(clean_resid) > 30:
            try:
                arch_lags = min(10, len(clean_resid) // 10)
                self._arch_stat, self._arch_pval = _arch_lm(clean_resid - clean_resid.mean(), arch_lags)
            except Exception:
                pass

        try:
            h_test = self.arima_results.test_heteroskedasticity(method="breakvar")
            self._h_stat, self._h_pval = float(h_test[0, 0]), float(h_test[0, 1])
        except Exception:
            self._h_stat = self._h_pval = np.nan

    # ------------------------------------------------------------------ properties
    @property
    def order(self) -> Tuple[int, int, int]:
        """ARIMA order (p, d, q)."""
        return self._order

    @property
    def seasonal_order(self) -> Optional[Tuple[int, int, int, int]]:
        """Seasonal order (P, D, Q, s) if specified."""
        return self._seasonal_order

    @property
    def weight(self) -> float:
        """Combination weight on the ARIMA component (1 - weight goes to alpha-Sutte)."""
        return self._weight

    @property
    def params(self) -> pd.Series:
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.params, index=names)

    @property
    def pvalues(self) -> pd.Series:
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.pvalues, index=names)

    @property
    def bse(self) -> pd.Series:
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.bse, index=names)

    @property
    def aic(self) -> float:
        return float(self.arima_results.aic)

    @property
    def aicc(self) -> float:
        return float(self.arima_results.aicc)

    @property
    def bic(self) -> float:
        return float(self.arima_results.bic)

    @property
    def hqic(self) -> float:
        return float(self.arima_results.hqic)

    @property
    def llf(self) -> float:
        return float(self.arima_results.llf)

    @property
    def durbin_watson(self) -> float:
        return self._dw

    @property
    def ljung_box(self) -> Dict[str, float]:
        """Ljung-Box test on hybrid residuals (modelling scale). H0: no autocorrelation."""
        return {"statistic": round(self._lb_stat, 4), "p_value": round(self._lb_pval, 4), "lags": self._lb_lag}

    @property
    def jarque_bera(self) -> Dict[str, float]:
        return {"statistic": round(self._jb_stat, 4), "p_value": round(self._jb_pval, 4),
                "skew": round(self._skew, 4), "kurtosis": round(self._kurtosis, 4)}

    @property
    def heteroskedasticity(self) -> Dict[str, float]:
        """Break-variance test on the ARIMA innovations. H0: homoskedastic."""
        return {"statistic": round(self._h_stat, 4), "p_value": round(self._h_pval, 4)}

    @property
    def arch_lm(self) -> Dict[str, float]:
        """Engle's ARCH-LM test on hybrid residuals (modelling scale). H0: no ARCH effects."""
        return {"statistic": round(self._arch_stat, 4), "p_value": round(self._arch_pval, 4)}

    @property
    def mape(self) -> float:
        return self.in_sample_metrics.get("MAPE (%)", np.nan)

    @property
    def smape(self) -> float:
        return self.in_sample_metrics.get("sMAPE (%)", np.nan)

    @property
    def rmse(self) -> float:
        return self.in_sample_metrics.get("RMSE", np.nan)

    @property
    def mse(self) -> float:
        return self.in_sample_metrics.get("MSE", np.nan)

    @property
    def mae(self) -> float:
        return self.in_sample_metrics.get("MAE", np.nan)

    @property
    def tic(self) -> float:
        return self.in_sample_metrics.get("TIC", np.nan)

    @property
    def u2(self) -> float:
        """In-sample Theil U2 against the naive forecast y_{t-1}."""
        return self.in_sample_metrics.get("U2", np.nan)

    @property
    def mase(self) -> Optional[float]:
        return self.in_sample_metrics.get("MASE", None)

    @property
    def max_ae(self) -> float:
        return self.in_sample_metrics.get("MaxAE", np.nan)

    @property
    def mbe(self) -> float:
        return self.in_sample_metrics.get("MBE", np.nan)

    # ------------------------------------------------------------------ forecasting
    def forecast(self, steps: int = 1, alpha: float = 0.05, exog: Any = None,
                 interval: str = "arima") -> pd.DataFrame:
        """
        Multi-step forecasts from the end of the sample.

        Columns: ``SutteARIMA`` (hybrid), ``ARIMA``, ``AlphaSutte`` (recursive), ``Naive`` (last
        observation) and the prediction bounds.

        ``interval="arima"`` (default): bounds are quantiles of the predictive distribution of the
        fitted ARIMA model (back-transformed when ``transform="log"``; the ARIMA point forecast is
        then the predictive median). A prediction interval describes the distribution of the future
        observation, not of a particular point forecast, so it does not depend on the weight.
        ``interval="legacy"``: the version 1 heuristic (hybrid-centred, in-sample residual SD scaled
        by the ARIMA standard-error growth). It has no probabilistic justification and is kept only
        to reproduce results obtained with version 1.
        """
        if self.model.exog is not None and exog is None:
            raise ValueError("The model was fitted with exog; future values of exog are required.")
        X = _as_2d(exog)
        fc = self.arima_results.get_forecast(steps=steps, exog=X)
        mean_z = np.asarray(fc.predicted_mean, dtype=float)
        ci_z = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
        a_mean = _inverse(mean_z, self.transform)
        s_mean = AlphaSutte.compute_forecast(self.endog, steps=steps)
        hybrid_mean = self._weight * a_mean + (1.0 - self._weight) * s_mean
        pct = int(round((1.0 - alpha) * 100))

        if interval == "arima":
            lower = _inverse(ci_z[:, 0], self.transform)
            upper = _inverse(ci_z[:, 1], self.transform)
        elif interval == "legacy":
            a_se = np.asarray(fc.se_mean, dtype=float)
            clean = self.resid[4:][~np.isnan(self.resid[4:])]
            sigma = np.std(clean, ddof=1) if len(clean) > 1 else 1.0
            se = sigma * a_se / (a_se[0] if len(a_se) > 0 and a_se[0] > 0 else 1.0)
            z = float(stats.norm.ppf(1.0 - alpha / 2.0))
            lower, upper = hybrid_mean - z * se, hybrid_mean + z * se
        else:
            raise ValueError("interval must be 'arima' or 'legacy'.")

        df = pd.DataFrame({
            "SutteARIMA": hybrid_mean,
            "ARIMA": a_mean,
            "AlphaSutte": s_mean,
            "Naive": np.repeat(self.endog[-1], steps),
            f"Lower_{pct}": lower,
            f"Upper_{pct}": upper,
        }, index=list(range(1, steps + 1)))
        df.index.name = "Horizon"
        return df

    def rolling_forecast(self, test_data: Any, exog: Any = None,
                         refit_every: Optional[int] = None) -> pd.DataFrame:
        """
        Expanding-window one-step-ahead forecasts over ``test_data`` without look-ahead.

        The forecast of observation t uses only observations (and exogenous values) up to t - 1,
        plus the exogenous value of period t itself, which must therefore be known in advance.
        With ``refit_every=None`` the parameters estimated on the training sample are kept and the
        Kalman filter is run through the new data; with ``refit_every=k`` the ARIMA (and an
        ``"optimal"`` weight) are re-estimated every k observations on all data available at that
        point.

        Returns a DataFrame with columns Actual, SutteARIMA, ARIMA, AlphaSutte, Naive.
        """
        test = np.asarray(test_data, dtype=float).ravel()
        H = len(test)
        n0 = len(self.endog)
        y_all = np.r_[self.endog, test]
        if self.transform == "log" and np.any(test <= 0):
            raise ValueError("transform='log' requires strictly positive test data.")
        z_all = _forward(y_all, self.transform)
        X_all = None
        if self.model.exog is not None:
            if exog is None:
                raise ValueError("The model was fitted with exog; exog for the test period is required.")
            X_test = _as_2d(exog)
            if len(X_test) != H:
                raise ValueError("exog for the test period must have one row per test observation.")
            X_all = np.vstack([self.model.exog, X_test])

        sutte_all = AlphaSutte.compute_fitted(y_all)
        arima_pred = np.full(H, np.nan)
        weights = np.full(H, self._weight)
        step = H if not refit_every else int(refit_every)

        for b in range(n0, n0 + H, step):
            e = min(b + step, n0 + H)
            if b == n0:                      # the training fit already uses exactly the data up to n0
                res_b, w_b = self.arima_results, self._weight
            else:
                res_b = _fit_statsmodels_arima(z_all[:b], None if X_all is None else X_all[:b],
                                               self._order, self._seasonal_order, self.model.trend)
                w_b = self._weight
                if self.model.weight_mode == "optimal":
                    a_in = _inverse(np.asarray(res_b.fittedvalues, dtype=float), self.transform)
                    a_in[:self.model._burn] = np.nan
                    w_b = _combination_weight(y_all[:b], a_in, sutte_all[:b], self.transform == "log")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                applied = res_b.apply(z_all[:e], exog=None if X_all is None else X_all[:e])
            arima_pred[b - n0:e - n0] = np.asarray(applied.fittedvalues, dtype=float)[b:e]
            weights[b - n0:e - n0] = w_b

        arima_level = _inverse(arima_pred, self.transform)
        sutte = sutte_all[n0:]
        hybrid = np.where(np.isnan(sutte), arima_level, weights * arima_level + (1.0 - weights) * sutte)
        return pd.DataFrame({
            "Actual": test,
            "SutteARIMA": hybrid,
            "ARIMA": arima_level,
            "AlphaSutte": sutte,
            "Naive": y_all[n0 - 1:n0 + H - 1],
        })

    def evaluate(self, test_data: Any, exog: Any = None, refit_every: Optional[int] = None) -> pd.DataFrame:
        """
        Out-of-sample evaluation of the one-step-ahead forecasts from ``rolling_forecast``:
        accuracy metrics, Theil's U2 and Diebold-Mariano tests against the naive forecast
        (squared and absolute loss), and the Pesaran-Timmermann directional test.
        """
        fc = self.rolling_forecast(test_data, exog=exog, refit_every=refit_every)
        rows = []
        e_naive = fc["Actual"] - fc["Naive"]
        actual_change = fc["Actual"] - fc["Naive"]
        for col in ["SutteARIMA", "ARIMA", "AlphaSutte", "Naive"]:
            m = calculate_metrics(fc["Actual"], fc[col], train_data=self.endog, benchmark=fc["Naive"])
            row = {"Model": col, **m}
            if col != "Naive":
                e = fc["Actual"] - fc[col]
                dm2 = diebold_mariano(e, e_naive, power=2)
                dm1 = diebold_mariano(e, e_naive, power=1)
                pt = pesaran_timmermann(actual_change, fc[col] - fc["Naive"])
                row.update({"DM_MSE": dm2["statistic"], "DM_MSE_p": dm2["p_value"],
                            "DM_MAE": dm1["statistic"], "DM_MAE_p": dm1["p_value"],
                            "Direction_hit": pt["hit_rate"], "PT_p": pt["p_value"]})
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")

    def metrics(self, test_data: Optional[Any] = None, exog: Any = None) -> Dict[str, Any]:
        """
        In-sample metrics (``test_data=None``) or metrics of the static multi-step forecast over
        ``test_data`` (the forecast origin is the end of the training sample). U2 is computed
        against the naive multi-step forecast (the last training observation).
        For one-step-ahead evaluation use ``evaluate``.
        """
        if test_data is None:
            return self.in_sample_metrics
        test = np.asarray(test_data, dtype=float).ravel()
        fc = self.forecast(steps=len(test), exog=exog)
        return calculate_metrics(actual=test, predicted=fc["SutteARIMA"].values,
                                 train_data=self.endog, benchmark=fc["Naive"].values)

    def metrics_table(self, test_data: Optional[Any] = None, exog: Any = None) -> pd.DataFrame:
        m = self.metrics(test_data=test_data, exog=exog)
        col_name = "In-Sample" if test_data is None else "Out-of-Sample (static multi-step)"
        return pd.DataFrame(list(m.items()), columns=["Metric", col_name]).set_index("Metric")

    def params_table(self) -> pd.DataFrame:
        """Parameter estimates: coef, std err, z, P>|z| and 95% confidence interval."""
        ci = np.asarray(self.arima_results.conf_int(), dtype=float)
        names = getattr(self.arima_results, "param_names", None)
        df = pd.DataFrame({
            "coef": np.asarray(self.params),
            "std err": np.asarray(self.bse),
            "z": np.asarray(self.arima_results.tvalues),
            "P>|z|": np.asarray(self.pvalues),
            "[0.025": ci[:, 0],
            "0.975]": ci[:, 1],
        }, index=names if names is not None else list(range(len(self.params))))
        return df.round(4)

    def plot(self, steps: Optional[int] = None, alpha: float = 0.05, figsize: Tuple[int, int] = (10, 5),
             exog: Any = None):
        """Actual vs. fitted values, optionally with a multi-step forecast and its interval."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib: pip install matplotlib")

        fig, ax = plt.subplots(figsize=figsize)
        t_hist = np.arange(len(self.endog))
        ax.plot(t_hist, self.endog, label="Actual", color="#1f77b4", linewidth=1.8)
        ax.plot(t_hist, self.fittedvalues, label="SutteARIMA fitted", color="#ff7f0e", linestyle="--", linewidth=1.5)
        if steps:
            fc_df = self.forecast(steps=steps, alpha=alpha, exog=exog)
            t_fc = np.arange(len(self.endog), len(self.endog) + steps)
            pct = int(round((1.0 - alpha) * 100))
            ax.plot(t_fc, fc_df["SutteARIMA"], label=f"Forecast ({steps} steps)", color="#2ca02c", linewidth=2.0)
            ax.fill_between(t_fc, fc_df[f"Lower_{pct}"], fc_df[f"Upper_{pct}"], color="#2ca02c", alpha=0.18,
                            label=f"{pct}% prediction interval (ARIMA)")
        name = f"SutteARIMA{self.order}" + (f"x{self.seasonal_order}" if self.seasonal_order else "")
        ax.set_title(f"{name} - actual vs. fitted and forecast", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Value")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        return fig, ax

    def summary(self) -> str:
        """statsmodels-style text summary."""
        m = self.in_sample_metrics
        model_str = f"SutteARIMA{self.order}" + (f"x{self.seasonal_order}" if self.seasonal_order else "")
        trend = self.model.trend_input if self.model.trend_input is not None else "statsmodels default"
        lines = [
            "=" * 72,
            f"{'SutteARIMA Model Estimation Results':^72}",
            "=" * 72,
            f" Dep. Variable       : y                      No. Observations : {len(self.endog)}",
            f" Model               : {model_str:<23} Log-Likelihood   : {self.llf:<10.3f}",
            f" Transform           : {str(self.transform):<23} Trend            : {str(trend):<10}",
            f" Weight on ARIMA     : {self._weight:<23.4f} Exog regressors  : {0 if self.model.exog is None else self.model.exog.shape[1]}",
            f" AIC                 : {self.aic:<10.3f}       AICc             : {self.aicc:<10.3f}",
            f" BIC                 : {self.bic:<10.3f}       HQIC             : {self.hqic:<10.3f}",
            "-" * 72,
            f"{'In-sample one-step accuracy (hybrid)':^72}",
            "-" * 72,
            f" MAPE (%)            : {m.get('MAPE (%)', np.nan):<10.4f}       sMAPE (%)        : {m.get('sMAPE (%)', np.nan):<10.4f}",
            f" RMSE                : {m.get('RMSE', np.nan):<10.4f}       MAE              : {m.get('MAE', np.nan):<10.4f}",
            f" Theil U2 vs naive   : {m.get('U2', np.nan):<10.4f}       Theil U1 (TIC)   : {m.get('TIC', np.nan):<10.4f}",
            f" Max Error (MaxAE)   : {m.get('MaxAE', np.nan):<10.4f}       Mean Bias (MBE)  : {m.get('MBE', np.nan):<10.4f}",
            "-" * 72,
            f"{'ARIMA Parameter Estimates':^72}",
            "-" * 72,
            self.params_table().to_string(),
            "-" * 72,
            f"{'Residual Diagnostic Tests (modelling scale)':^72}",
            "-" * 72,
            f" Durbin-Watson       : {self._dw:<10.4f}",
            f" Ljung-Box (Q)       : {self._lb_stat:<10.4f}       Prob(Q)          : {self._lb_pval:<10.4f}",
            f" ARCH-LM             : {self._arch_stat:<10.4f}       Prob(ARCH)       : {self._arch_pval:<10.4f}",
            f" Jarque-Bera (JB)    : {self._jb_stat:<10.4f}       Prob(JB)         : {self._jb_pval:<10.4f}",
            f" Heteroskedasticity  : {self._h_stat:<10.4f}       Prob(H)          : {self._h_pval:<10.4f}",
            f" Skew                : {self._skew:<10.4f}       Kurtosis         : {self._kurtosis:<10.4f}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        return f"<pre style='font-family: monospace;'>{self.summary()}</pre>"


# ==============================================================================
# 5. MODEL
# ==============================================================================

class SutteARIMA:
    """
    Hybrid alpha-Sutte + ARIMA model.

        yhat_t = w * yhat_t^ARIMA + (1 - w) * yhat_t^Sutte

    Parameters
    ----------
    endog : array-like
        Univariate series (original scale).
    order : (p, d, q) or 'auto'
        ARIMA order; 'auto' runs ``find_order`` with the same transform, trend and exog.
    seasonal_order : (P, D, Q, s), optional
    exog : array-like, optional
        Exogenous regressors, entering as a *regression with ARIMA errors*:
        z_t = x_t' beta + u_t with u_t ~ ARIMA(p, d, q), where z_t is the (transformed) series.
        With d = 1 this implies Delta z_t = Delta x_t' beta + ARMA error, so a step dummy in x
        produces a permanent level shift, a pulse dummy a one-period level shift, and a ramp
        (cumulative sum of a window dummy) a change in the growth rate during the window.
        Every regressor must be known when the forecast is made: never build exog from the
        contemporaneous value of the series (e.g. a dummy for "today's return > 8%").
    trend : str, list or None
        statsmodels trend specification, or ``"drift"`` for a constant in the d-th difference.
        ``None`` keeps the statsmodels default ('c' if d = 0, no trend if d > 0).
    transform : None or 'log'
        'log' models log(y); the ARIMA forecast is back-transformed with exp (predictive median).
        The alpha-Sutte component is always computed on the original scale, as in Ahmar (2017).
    weight : float in [0, 1] or 'optimal', default 0.5
        Weight on the ARIMA component. 0.5 reproduces the original SutteARIMA. 'optimal' estimates
        the least-squares weight from the in-sample one-step errors (Bates & Granger, 1969),
        using relative errors when ``transform='log'``.

    Examples
    --------
    >>> res = SutteARIMA(prices, order=(1, 1, 0), transform="log", trend="drift").fit()
    >>> res.evaluate(test_prices, refit_every=21)
    """

    def __init__(self,
                 endog: Any,
                 order: Union[Tuple[int, int, int], str] = "auto",
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 *,
                 exog: Any = None,
                 trend: Any = None,
                 transform: Optional[str] = None,
                 weight: Union[float, str] = 0.5):
        self.endog = np.asarray(endog, dtype=float).ravel()
        if len(self.endog) < 8:
            raise ValueError(f"SutteARIMA requires at least 8 observations, got {len(self.endog)}.")
        self.transform = _check_transform(transform)
        if self.transform == "log" and np.any(self.endog <= 0):
            raise ValueError("transform='log' requires strictly positive data.")
        self.exog = _as_2d(exog)
        if self.exog is not None and len(self.exog) != len(self.endog):
            raise ValueError("exog must have the same number of rows as endog.")
        self.seasonal_order = seasonal_order
        self.trend_input = trend

        if isinstance(weight, str):
            if weight.lower() != "optimal":
                raise ValueError("weight must be a number in [0, 1] or 'optimal'.")
            self.weight_mode = "optimal"
            self.weight = 0.5
        else:
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError("weight must lie in [0, 1].")
            self.weight_mode = "fixed"
            self.weight = float(weight)

        if order == "auto" or order is None:
            self.order = find_order(self.endog, verbose=False, transform=self.transform,
                                    trend=trend, exog=self.exog)
        elif isinstance(order, (tuple, list)) and len(order) == 3:
            self.order = (int(order[0]), int(order[1]), int(order[2]))
        else:
            raise ValueError(
                f"Invalid order={order}. Use a (p, d, q) tuple such as (0, 1, 1) or order='auto'.")

        D = int(seasonal_order[1]) if seasonal_order else 0
        s = int(seasonal_order[3]) if seasonal_order else 0
        self.trend = _resolve_trend(trend, self.order[1], D)
        self._burn = self.order[1] + D * s

    def fit(self, **fit_kwargs) -> SutteARIMAResults:
        """Estimate the ARIMA component (MLE via the Kalman filter) and build the hybrid."""
        z = _forward(self.endog, self.transform)
        arima_res = _fit_statsmodels_arima(z, self.exog, self.order, self.seasonal_order, self.trend, **fit_kwargs)

        arima_fitted = _inverse(np.asarray(arima_res.fittedvalues, dtype=float).copy(), self.transform)
        arima_fitted[:self._burn] = np.nan          # diffuse start: not a genuine prediction
        sutte_fitted = AlphaSutte.compute_fitted(self.endog)

        if self.weight_mode == "optimal":
            w = _combination_weight(self.endog, arima_fitted, sutte_fitted, self.transform == "log")
        else:
            w = self.weight

        hybrid = np.where(np.isnan(sutte_fitted), arima_fitted, w * arima_fitted + (1.0 - w) * sutte_fitted)
        return SutteARIMAResults(model=self, arima_res=arima_res, fittedvalues=hybrid, order=self.order,
                                 seasonal_order=self.seasonal_order, arima_fitted=arima_fitted,
                                 sutte_fitted=sutte_fitted, weight=w)
