"""
SutteARIMA: Statsmodels-style Python Implementation
Combining Alpha-Sutte Indicator and Box-Jenkins ARIMA.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. COMPREHENSIVE TIME SERIES EVALUATION METRICS ENGINE
# ==============================================================================

def calculate_evaluation_metrics(actual: Any,
                                 predicted: Any,
                                 train_series: Optional[Any] = None) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of time series forecasting evaluation metrics:
      - MAE    : Mean Absolute Error
      - MSE    : Mean Squared Error
      - RMSE   : Root Mean Squared Error
      - MAPE   : Mean Absolute Percentage Error (%)
      - MdAPE  : Median Absolute Percentage Error (%)
      - sMAPE  : Symmetric Mean Absolute Percentage Error (%)
      - MASE   : Mean Absolute Scaled Error (against in-sample naive diff)
      - R2     : Coefficient of Determination (R-squared)
      - MaxAE  : Maximum Absolute Error
      - MBE    : Mean Bias Error
      - TIC    : Theil's Inequality Coefficient (0 = perfect fit)
      - U2     : Theil's U2 Statistic (forecast quality vs naive)
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()

    if len(a) != len(p):
        raise ValueError(f"Actual length ({len(a)}) must match predicted length ({len(p)}).")

    mask = ~(np.isnan(a) | np.isnan(p) | np.isinf(a) | np.isinf(p))
    a, p = a[mask], p[mask]
    n = len(a)
    if n == 0:
        return {}

    errors = a - p
    abs_errors = np.abs(errors)

    # MAE, MSE, RMSE
    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    # Percentage errors (MAPE, MdAPE)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(errors / a)
        ape_clean = ape[~np.isnan(ape) & ~np.isinf(ape)]

    mape = float(np.mean(ape_clean) * 100.0) if len(ape_clean) > 0 else np.nan
    mdape = float(np.median(ape_clean) * 100.0) if len(ape_clean) > 0 else np.nan

    # sMAPE (%)
    denom = np.abs(a) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_arr = np.where(denom == 0, 0.0, 200.0 * abs_errors / denom)
    smape = float(np.mean(smape_arr))

    # MASE
    if train_series is not None:
        t = np.asarray(train_series, dtype=float).ravel()
        scale = np.mean(np.abs(np.diff(t)))
        mase = float(mae / scale) if scale > 0 else np.nan
    else:
        mase = np.nan

    # R2
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    # MaxAE, MBE
    max_ae = float(np.max(abs_errors))
    mbe = float(np.mean(errors))

    # Theil's Inequality Coefficient (TIC)
    norm_denom = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(p ** 2))
    tic = float(rmse / norm_denom) if norm_denom > 0 else np.nan

    return {
        "n": n,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "MdAPE (%)": round(mdape, 4),
        "sMAPE (%)": round(smape, 4),
        "MASE": round(mase, 4) if not np.isnan(mase) else None,
        "R2": round(r2, 4),
        "MaxAE": round(max_ae, 4),
        "MBE": round(mbe, 4),
        "TIC": round(tic, 4),
    }


# ==============================================================================
# 2. ALPHA-SUTTE INDICATOR
# ==============================================================================

class AlphaSutte:
    """Exact Alpha-Sutte Indicator implementation (Ahmar, 2017)."""

    @staticmethod
    def step_increment(y_t4: float, y_t3: float, y_t2: float, y_t1: float) -> float:
        dx, dy, dz = y_t3 - y_t4, y_t2 - y_t3, y_t1 - y_t2
        mx, my, mz = (y_t3 + y_t4) / 2.0, (y_t2 + y_t3) / 2.0, (y_t1 + y_t2) / 2.0

        tx = (y_t3 * (dx / mx)) if abs(mx) > 1e-12 else dx
        ty = (y_t2 * (dy / my)) if abs(my) > 1e-12 else dy
        tz = (y_t1 * (dz / mz)) if abs(mz) > 1e-12 else dz

        return float((tx + ty + tz) / 3.0)

    @classmethod
    def predict_next(cls, y_t4: float, y_t3: float, y_t2: float, y_t1: float) -> float:
        return float(y_t1 + cls.step_increment(y_t4, y_t3, y_t2, y_t1))

    @classmethod
    def compute_fitted(cls, series: np.ndarray) -> np.ndarray:
        n = len(series)
        fitted = np.full(n, np.nan)
        for t in range(4, n):
            fitted[t] = cls.predict_next(series[t - 4], series[t - 3], series[t - 2], series[t - 1])
        return fitted

    @classmethod
    def compute_forecast(cls, series: np.ndarray, steps: int) -> np.ndarray:
        hist = series.copy().tolist()
        fc = []
        for _ in range(steps):
            nxt = cls.predict_next(hist[-4], hist[-3], hist[-2], hist[-1])
            fc.append(nxt)
            hist.append(nxt)
        return np.asarray(fc, dtype=float)


# ==============================================================================
# 3. RESULTS WRAPPER (STATSMODELS COMPLIANT)
# ==============================================================================

class SutteARIMAResults:
    """
    Results class for SutteARIMA model estimation.
    Designed with the familiar statsmodels interface.
    """

    def __init__(self,
                 model: SutteARIMA,
                 arima_res: ARIMAResults,
                 fittedvalues: np.ndarray,
                 order: Tuple[int, int, int]):
        self.model = model
        self.arima_results = arima_res
        self.fittedvalues = fittedvalues
        self.order = order

        self.endog = model.endog
        self.resid = self.endog - self.fittedvalues

        # Compute in-sample metrics directly on historical data (for t >= 4)
        self.in_sample_metrics = calculate_evaluation_metrics(
            self.endog[4:], self.fittedvalues[4:], self.endog
        )

        # Residual diagnostics
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        self.durbin_watson = float(durbin_watson(clean_resid)) if len(clean_resid) > 0 else np.nan
        jb_res = jarque_bera(clean_resid)
        self.jarque_bera_stat = float(jb_res[0])
        self.jarque_bera_pval = float(jb_res[1])

    @property
    def aic(self) -> float:
        """ARIMA component AIC."""
        return float(self.arima_results.aic)

    @property
    def bic(self) -> float:
        """ARIMA component BIC."""
        return float(self.arima_results.bic)

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generates out-of-sample forecasts for `steps` periods ahead with prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Number of periods ahead to forecast.
        alpha : float, default=0.05
            Significance level for prediction intervals (0.05 gives 95% CI).

        Returns
        -------
        pd.DataFrame
            Contains point forecasts (SutteARIMA, ARIMA, AlphaSutte) and Lower/Upper bounds.
        """
        arima_fc = self.arima_results.get_forecast(steps=steps)
        a_mean = np.asarray(arima_fc.predicted_mean, dtype=float)
        a_se = np.asarray(arima_fc.se_mean, dtype=float)

        s_mean = AlphaSutte.compute_forecast(self.endog, steps=steps)
        hybrid_mean = 0.5 * (a_mean + s_mean)

        # Variance scaling for prediction intervals
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        hybrid_sigma = np.std(clean_resid, ddof=1) if len(clean_resid) > 1 else 1.0
        se_scale = a_se / (a_se[0] if a_se[0] > 0 else 1.0)
        hybrid_se = hybrid_sigma * se_scale

        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        pct = int(round((1.0 - alpha) * 100))

        df = pd.DataFrame({
            "SutteARIMA": hybrid_mean,
            "ARIMA": a_mean,
            "AlphaSutte": s_mean,
            f"Lower_{pct}": hybrid_mean - z * hybrid_se,
            f"Upper_{pct}": hybrid_mean + z * hybrid_se,
        }, index=list(range(1, steps + 1)))
        df.index.name = "Horizon"
        return df

    def metrics(self, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns full evaluation metrics dictionary:
          - If `test_data` is None: returns In-Sample metrics on historical data.
          - If `test_data` is provided: forecasts len(test_data) steps and evaluates Out-of-Sample metrics!
        """
        if test_data is None:
            return self.in_sample_metrics

        test = np.asarray(test_data, dtype=float).ravel()
        fc = self.forecast(steps=len(test))["SutteARIMA"].values
        return calculate_evaluation_metrics(test, fc, self.endog)

    def summary(self) -> str:
        """Returns a clean, professional statsmodels-style summary table."""
        m = self.in_sample_metrics
        order_str = str(self.order)
        lines = [
            "=" * 70,
            f"{'SutteARIMA Model Estimation Results':^70}",
            "=" * 70,
            f" Dep. Variable       : y                      No. Observations : {len(self.endog)}",
            f" Model               : SutteARIMA{order_str:<10} Date             : Available",
            f" ARIMA AIC           : {self.aic:<10.3f}       ARIMA BIC        : {self.bic:<10.3f}",
            "-" * 70,
            f"{'In-Sample Forecasting Accuracy Metrics':^70}",
            "-" * 70,
            f" MAE                 : {m.get('MAE', np.nan):<10.4f}       MSE              : {m.get('MSE', np.nan):<10.4f}",
            f" RMSE                : {m.get('RMSE', np.nan):<10.4f}       R-squared (R2)   : {m.get('R2', np.nan):<10.4f}",
            f" MAPE (%)            : {m.get('MAPE (%)', np.nan):<10.4f}%      sMAPE (%)        : {m.get('sMAPE (%)', np.nan):<10.4f}%",
            f" MdAPE (%)           : {m.get('MdAPE (%)', np.nan):<10.4f}%      Theil's IC (TIC) : {m.get('TIC', np.nan):<10.4f}",
            f" Max Error (MaxAE)   : {m.get('MaxAE', np.nan):<10.4f}       Mean Bias (MBE)  : {m.get('MBE', np.nan):<10.4f}",
            "-" * 70,
            f"{'Residual Diagnostic Checks':^70}",
            "-" * 70,
            f" Durbin-Watson       : {self.durbin_watson:<10.4f}       Jarque-Bera (p)  : {self.jarque_bera_pval:<10.4f}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ==============================================================================
# 4. MAIN STATSMODELS-STYLE SUTTE-ARIMA CLASS
# ==============================================================================

class SutteARIMA:
    """
    SutteARIMA Model (Statsmodels-Style Interface).

    Parameters
    ----------
    endog : array-like
        The observed time series data (y).
    order : Tuple[int, int, int], optional
        The (p, d, q) order of the ARIMA model. If None (default), the order
        is automatically selected by minimizing AICc over a grid search.

    Example
    -------
    >>> from sutte_arima import SutteARIMA
    >>> data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8]
    >>> model = SutteARIMA(data)
    >>> res = model.fit()
    >>> print(res.summary())
    >>> forecasts = res.forecast(steps=5)
    """

    def __init__(self,
                 endog: Any,
                 order: Optional[Tuple[int, int, int]] = None):
        self.endog = np.asarray(endog, dtype=float).ravel()
        if len(self.endog) < 8:
            raise ValueError(f"SutteARIMA requires at least 8 observations, got {len(self.endog)}.")
        self.order = order

    def _auto_order(self, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
        """Automatically selects (p, d, q) via ADF test and AICc grid search."""
        y = self.endog
        # Stationarity check for d
        stat, pval, _, _, _, _ = adfuller(y, autolag="AIC")
        d = 0 if pval < 0.05 else (1 if adfuller(np.diff(y), autolag="AIC")[1] < 0.05 else 2)

        best_score = float("inf")
        best_order = (1, d, 0)
        n = len(y)

        for p, q in product(range(max_p + 1), range(max_q + 1)):
            try:
                mod = ARIMA(y, order=(p, d, q))
                fit = mod.fit()
                k = p + q + 1
                aicc = fit.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0) if (n - k - 1) > 0 else fit.aic
                if aicc < best_score:
                    best_score = aicc
                    best_order = (p, d, q)
            except Exception:
                continue

        return best_order

    def fit(self) -> SutteARIMAResults:
        """
        Fits the SutteARIMA model.

        Returns
        -------
        SutteARIMAResults
            Results object with .summary(), .forecast(), .metrics(), .fittedvalues, .resid.
        """
        chosen_order = self.order or self._auto_order()

        # Fit ARIMA
        arima_model = ARIMA(self.endog, order=chosen_order)
        arima_res = arima_model.fit()

        # In-sample Alpha-Sutte
        sutte_fitted = AlphaSutte.compute_fitted(self.endog)
        arima_fitted = np.asarray(arima_res.fittedvalues, dtype=float)

        # Hybrid in-sample fitted values
        hybrid_fitted = np.full(len(self.endog), np.nan)
        for i in range(len(self.endog)):
            if np.isnan(sutte_fitted[i]):
                hybrid_fitted[i] = arima_fitted[i]
            else:
                hybrid_fitted[i] = 0.5 * (arima_fitted[i] + sutte_fitted[i])

        return SutteARIMAResults(
            model=self,
            arima_res=arima_res,
            fittedvalues=hybrid_fitted,
            order=chosen_order
        )
