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
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. EVALUATION METRICS ENGINE
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
      - MBE    : Mean Bias Error (directional bias)
      - TIC    : Theil's Inequality Coefficient (0 = perfect forecast, 1 = worst)
      - U2     : Theil's U2 Statistic (forecast vs naive)
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
# 3. RESULTS CLASS (CALLABLE PROPERTIES, MATRICES & TABLES)
# ==============================================================================

class SutteARIMAResults:
    """
    Results class for SutteARIMA model estimation.
    Provides direct callable access to order, parameters, diagnostics,
    metrics matrices, and multi-step forecasts.
    """

    def __init__(self,
                 model: SutteARIMA,
                 arima_res: ARIMAResults,
                 fittedvalues: np.ndarray,
                 order: Tuple[int, int, int]):
        self.model = model
        self.arima_results = arima_res
        self.fittedvalues = fittedvalues
        self._order = order

        self.endog = model.endog
        self.resid = self.endog - self.fittedvalues

        # In-sample metrics on training series (for t >= 4)
        self.in_sample_metrics = calculate_evaluation_metrics(
            self.endog[4:], self.fittedvalues[4:], self.endog
        )

        # Residual diagnostics
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        self._dw = float(durbin_watson(clean_resid)) if len(clean_resid) > 0 else np.nan
        jb_res = jarque_bera(clean_resid)
        self._jb_stat = float(jb_res[0])
        self._jb_pval = float(jb_res[1])

    # --------------------------------------------------------------------------
    # Direct code-accessible properties
    # --------------------------------------------------------------------------
    @property
    def order(self) -> Tuple[int, int, int]:
        """Returns the ARIMA order tuple (p, d, q). Example: (0, 1, 1)."""
        return self._order

    @property
    def params(self) -> pd.Series:
        """Returns fitted ARIMA coefficients (AR, MA, trend/drift, variance)."""
        return self.arima_results.params

    @property
    def pvalues(self) -> pd.Series:
        """Returns p-values of the fitted ARIMA coefficients."""
        return self.arima_results.pvalues

    @property
    def bse(self) -> pd.Series:
        """Returns standard errors of the fitted ARIMA coefficients."""
        return self.arima_results.bse

    @property
    def aic(self) -> float:
        """Akaike Information Criterion (AIC)."""
        return float(self.arima_results.aic)

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion (BIC)."""
        return float(self.arima_results.bic)

    @property
    def aicc(self) -> float:
        """Small-sample Corrected Akaike Information Criterion (AICc)."""
        k = len(self.params)
        n = len(self.endog)
        if (n - k - 1) > 0:
            return float(self.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0))
        return self.aic

    @property
    def llf(self) -> float:
        """Log-Likelihood value of the model."""
        return float(self.arima_results.llf)

    @property
    def durbin_watson(self) -> float:
        """Durbin-Watson statistic for residual autocorrelation (~2.0 is ideal)."""
        return self._dw

    @property
    def jarque_bera(self) -> Dict[str, float]:
        """Jarque-Bera normality test on residuals."""
        return {"statistic": round(self._jb_stat, 4), "p_value": round(self._jb_pval, 4)}

    # --------------------------------------------------------------------------
    # Matrices & Tables
    # --------------------------------------------------------------------------
    def params_table(self) -> pd.DataFrame:
        """
        Matrix of parameter estimates, standard errors, z-statistics,
        p-values, and 95% confidence intervals.
        """
        conf_int = self.arima_results.conf_int()
        df = pd.DataFrame({
            "coef": self.params,
            "std err": self.bse,
            "z": self.arima_results.tvalues if hasattr(self.arima_results, "tvalues") else self.params / self.bse,
            "P>|z|": self.pvalues,
            "[0.025": conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0],
            "0.975]": conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1],
        })
        return df.round(4)

    def diagnostics_table(self, lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Matrix of Ljung-Box autocorrelation test across lags.
        Tests the null hypothesis that residuals are white noise (p >= 0.05).
        """
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        n = len(clean_resid)
        default_lags = [l for l in [4, 8, 12, 16] if l < n]
        test_lags = lags or (default_lags if default_lags else [min(2, n - 1)])

        lb = acorr_ljungbox(clean_resid, lags=test_lags, return_df=True)
        lb["White_Noise (p >= 0.05)"] = lb["lb_pvalue"] >= 0.05
        return lb.round(4)

    def metrics(self, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns all evaluation metrics as a dictionary:
          - If `test_data` is None: returns in-sample metrics on historical data.
          - If `test_data` is provided: forecasts len(test_data) steps and evaluates on test data!
        """
        if test_data is None:
            return self.in_sample_metrics

        test = np.asarray(test_data, dtype=float).ravel()
        fc = self.forecast(steps=len(test))["SutteARIMA"].values
        return calculate_evaluation_metrics(test, fc, self.endog)

    def metrics_table(self, test_data: Optional[Any] = None) -> pd.DataFrame:
        """
        Returns all evaluation metrics formatted as a pandas DataFrame matrix.
        """
        m = self.metrics(test_data=test_data)
        mode = "In-Sample (Historical)" if test_data is None else "Out-of-Sample (Test Data)"
        df = pd.DataFrame(list(m.items()), columns=["Metric", f"Value ({mode})"])
        return df.set_index("Metric")

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generates out-of-sample forecasts for `steps` periods ahead with prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Number of periods ahead to forecast.
        alpha : float, default=0.05
            Significance level (alpha=0.05 gives 95% Confidence Interval).

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

    def forecast_table(self, steps: int = 1, test_data: Optional[Any] = None) -> pd.DataFrame:
        """
        Detailed forecast matrix showing step-by-step components, prediction intervals,
        and (if test_data is passed) Actual values, Absolute Error, and APE.
        """
        h = len(test_data) if test_data is not None else steps
        df = self.forecast(steps=h)

        if test_data is not None:
            t = np.asarray(test_data, dtype=float).ravel()
            df.insert(0, "Actual", t)
            df["Abs_Error"] = np.abs(df["Actual"] - df["SutteARIMA"])
            df["APE (%)"] = np.abs(df["Actual"] - df["SutteARIMA"]) / np.abs(df["Actual"]) * 100.0

        return df.round(4)

    def summary(self) -> str:
        """Returns a clean, statsmodels-style summary report."""
        m = self.in_sample_metrics
        order_str = str(self.order)
        lines = [
            "=" * 70,
            f"{'SutteARIMA Model Estimation Results':^70}",
            "=" * 70,
            f" Dep. Variable       : y                      No. Observations : {len(self.endog)}",
            f" Model               : SutteARIMA{order_str:<10} Log-Likelihood   : {self.llf:<10.3f}",
            f" ARIMA AIC           : {self.aic:<10.3f}       ARIMA AICc       : {self.aicc:<10.3f}",
            f" ARIMA BIC           : {self.bic:<10.3f}       Durbin-Watson    : {self.durbin_watson:<10.4f}",
            "-" * 70,
            f"{'In-Sample Forecasting Accuracy Matrix':^70}",
            "-" * 70,
            f" MAE                 : {m.get('MAE', np.nan):<10.4f}       MSE              : {m.get('MSE', np.nan):<10.4f}",
            f" RMSE                : {m.get('RMSE', np.nan):<10.4f}       R-squared (R2)   : {m.get('R2', np.nan):<10.4f}",
            f" MAPE (%)            : {m.get('MAPE (%)', np.nan):<10.4f}%      sMAPE (%)        : {m.get('sMAPE (%)', np.nan):<10.4f}%",
            f" MdAPE (%)           : {m.get('MdAPE (%)', np.nan):<10.4f}%      Theil's IC (TIC) : {m.get('TIC', np.nan):<10.4f}",
            f" Max Error (MaxAE)   : {m.get('MaxAE', np.nan):<10.4f}       Mean Bias (MBE)  : {m.get('MBE', np.nan):<10.4f}",
            "-" * 70,
            f"{'ARIMA Parameter Estimates':^70}",
            "-" * 70,
        ]

        # Add parameters table
        pt = self.params_table()
        lines.append(pt.to_string())
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ==============================================================================
# 4. SUTTE-ARIMA MAIN MODEL CLASS
# ==============================================================================

class SutteARIMA:
    """
    SutteARIMA Model (Statsmodels-Style Interface).

    Parameters
    ----------
    endog : array-like
        The observed time series data (y).
    order : Tuple[int, int, int]
        The (p, d, q) order for ARIMA. You must provide `order=(p, d, q)`.
        Example: `order=(0, 1, 1)` or `order=(1, 1, 0)`.
        (To inspect or select orders automatically, call `SutteARIMA.auto_select_order(data)`).

    Example
    -------
    >>> from sutte_arima import SutteARIMA
    >>> data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8]
    >>> model = SutteARIMA(data, order=(0, 1, 1))
    >>> res = model.fit()
    >>> print(res.order)         # (0, 1, 1)
    >>> print(res.params)        # coefficients
    >>> print(res.metrics())     # evaluation metrics matrix
    >>> print(res.forecast(5))   # 5-step forecast
    """

    def __init__(self,
                 endog: Any,
                 order: Optional[Tuple[int, int, int]] = None):
        self.endog = np.asarray(endog, dtype=float).ravel()
        if len(self.endog) < 8:
            raise ValueError(f"SutteARIMA requires at least 8 observations, got {len(self.endog)}.")

        if order is None:
            raise ValueError(
                "You must specify the ARIMA order=(p, d, q), e.g. SutteARIMA(data, order=(0, 1, 1)).\n"
                "If you want to view the automatic order selection table and find the optimal order, "
                "call: order, table = SutteARIMA.auto_select_order(data)"
            )

        if not (isinstance(order, (tuple, list)) and len(order) == 3):
            raise ValueError(f"order must be a tuple of 3 integers (p, d, q), got {order}.")

        self.order = (int(order[0]), int(order[1]), int(order[2]))

    def fit(self) -> SutteARIMAResults:
        """
        Fits the SutteARIMA model with the specified order.

        Returns
        -------
        SutteARIMAResults
            Results object with callable attributes (.order, .params, .pvalues,
            .aic, .bic, .metrics(), .metrics_table(), .forecast(), .summary()).
        """
        arima_model = ARIMA(self.endog, order=self.order)
        arima_res = arima_model.fit()

        sutte_fitted = AlphaSutte.compute_fitted(self.endog)
        arima_fitted = np.asarray(arima_res.fittedvalues, dtype=float)

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
            order=self.order
        )

    # --------------------------------------------------------------------------
    # Standalone Order Exploration & Selection Tool
    # --------------------------------------------------------------------------
    @classmethod
    def auto_select_order(cls,
                          data: Any,
                          max_p: int = 3,
                          max_q: int = 3,
                          max_d: int = 2,
                          criterion: str = "aicc") -> Tuple[Tuple[int, int, int], pd.DataFrame]:
        """
        Explores all ARIMA(p, d, q) candidate specifications and returns
        the best order along with the complete Information Criteria Comparison Matrix.

        Parameters
        ----------
        data : array-like
            The time series data to analyze.
        max_p : int, default=3
            Maximum AR order to test.
        max_q : int, default=3
            Maximum MA order to test.
        max_d : int, default=2
            Maximum differencing order to test.
        criterion : str, default='aicc'
            Sorting criterion: 'aicc', 'aic', or 'bic'.

        Returns
        -------
        best_order : Tuple[int, int, int]
            The optimal (p, d, q) order minimizing the criterion.
        comparison_matrix : pd.DataFrame
            Matrix of all tested candidate orders with p, d, q, AIC, AICc, BIC, LogLik.
        """
        y = np.asarray(data, dtype=float).ravel()
        n = len(y)

        # 1. Determine d via ADF test
        stat, pval, _, _, _, _ = adfuller(y, autolag="AIC")
        d = 0 if pval < 0.05 else (1 if adfuller(np.diff(y), autolag="AIC")[1] < 0.05 else 2)
        d_candidates = [d]
        if d == 0 and max_d >= 1:
            d_candidates.append(1)

        records = []
        for curr_d in d_candidates:
            for p, q in product(range(max_p + 1), range(max_q + 1)):
                try:
                    mod = ARIMA(y, order=(p, curr_d, q))
                    res = mod.fit()
                    k = p + q + 1
                    aic = float(res.aic)
                    bic = float(res.bic)
                    aicc = float(aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)) if (n - k - 1) > 0 else aic

                    records.append({
                        "order": f"({p}, {curr_d}, {q})",
                        "p": p, "d": curr_d, "q": q,
                        "AIC": round(aic, 3),
                        "AICc": round(aicc, 3),
                        "BIC": round(bic, 3),
                        "LogLik": round(float(res.llf), 3),
                    })
                except Exception:
                    continue

        df = pd.DataFrame(records)
        sort_col = "AICc" if criterion == "aicc" else ("AIC" if criterion == "aic" else "BIC")
        df = df.sort_values(sort_col).reset_index(drop=True)

        best_row = df.iloc[0]
        best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
        return best_order, df
