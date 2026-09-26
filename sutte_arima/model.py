"""
SutteARIMA: Minimal, Statsmodels-Style Time Series Forecasting Model
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
# 1. EVALUATION METRICS ENGINE (Calculated strictly from model predictions)
# ==============================================================================

def calculate_metrics(actual: Any, predicted: Any, train_data: Optional[Any] = None) -> Dict[str, float]:
    """
    Computes standard forecasting accuracy metrics from actual and predicted values:
      - MAPE (%) : Mean Absolute Percentage Error (primary metric in Sutte literature)
      - MSE      : Mean Squared Error
      - RMSE     : Root Mean Squared Error
      - MAE      : Mean Absolute Error
      - sMAPE (%) : Symmetric Mean Absolute Percentage Error
      - TIC      : Theil's Inequality Coefficient (0 = perfect forecast, 1 = worst)
      - MASE     : Mean Absolute Scaled Error (vs. naive historical diff)
      - MaxAE    : Maximum Absolute Error
      - MBE      : Mean Bias Error
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()

    mask = ~(np.isnan(a) | np.isnan(p) | np.isinf(a) | np.isinf(p))
    a, p = a[mask], p[mask]
    n = len(a)
    if n == 0:
        return {}

    errors = a - p
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(errors / a)
        clean_ape = ape[~np.isnan(ape) & ~np.isinf(ape)]
    mape = float(np.mean(clean_ape) * 100.0) if len(clean_ape) > 0 else np.nan

    # sMAPE (%)
    denom = np.abs(a) + np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_arr = np.where(denom == 0, 0.0, 200.0 * abs_errors / denom)
    smape = float(np.mean(smape_arr))

    # Theil's TIC
    norm_denom = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(p ** 2))
    tic = float(rmse / norm_denom) if norm_denom > 0 else np.nan

    # MASE
    if train_data is not None:
        tr = np.asarray(train_data, dtype=float).ravel()
        scale = np.mean(np.abs(np.diff(tr)))
        mase = float(mae / scale) if scale > 0 else np.nan
    else:
        mase = np.nan

    return {
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


# ==============================================================================
# 2. ALPHA-SUTTE INDICATOR
# ==============================================================================

class AlphaSutte:
    """Alpha-Sutte Indicator implementation (Ahmar, 2017)."""

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
        n = len(series)
        fitted = np.full(n, np.nan)
        for t in range(4, n):
            delta = cls.step_increment(series[t - 4], series[t - 3], series[t - 2], series[t - 1])
            fitted[t] = series[t - 1] + delta
        return fitted

    @classmethod
    def compute_forecast(cls, series: np.ndarray, steps: int) -> np.ndarray:
        hist = series.copy().tolist()
        fc = []
        for _ in range(steps):
            delta = cls.step_increment(hist[-4], hist[-3], hist[-2], hist[-1])
            y_next = hist[-1] + delta
            fc.append(y_next)
            hist.append(y_next)
        return np.asarray(fc, dtype=float)


# ==============================================================================
# 3. HELPER: FIND RECOMMENDED (p, d, q) ORDER
# ==============================================================================

def find_order(data: Any,
               max_p: int = 3,
               max_q: int = 3,
               verbose: bool = True) -> Tuple[int, int, int]:
    """
    Analyzes the dataset and returns the recommended (p, d, q) order for ARIMA.

    Workflow:
      1. Runs ADF unit-root test to determine differencing order d (0, 1, or 2).
      2. Grid-searches candidate (p, d, q) combinations.
      3. Selects the order that minimizes small-sample corrected AIC (AICc).

    Parameters
    ----------
    data : array-like
        The input time-series data.
    max_p : int, default=3
        Maximum autoregressive order to test.
    max_q : int, default=3
        Maximum moving average order to test.
    verbose : bool, default=True
        Whether to print stationarity results and candidate rankings.

    Returns
    -------
    Tuple[int, int, int]
        The recommended (p, d, q) order.
    """
    y = np.asarray(data, dtype=float).ravel()
    n = len(y)
    if n < 8:
        raise ValueError(f"Need at least 8 observations, got {n}.")

    # 1. Determine d via ADF test
    adf_lvl = adfuller(y, autolag="AIC")[1]
    if adf_lvl < 0.05:
        d = 0
    else:
        adf_d1 = adfuller(np.diff(y), autolag="AIC")[1]
        d = 1 if adf_d1 < 0.05 else 2

    # 2. Grid-search across candidate orders
    best_score = float("inf")
    best_order = (1, d, 0)
    records = []

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        try:
            fit = ARIMA(y, order=(p, d, q)).fit()
            k = p + q + 1
            aicc = float(fit.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)) if (n - k - 1) > 0 else float(fit.aic)
            records.append({"order": (p, d, q), "AICc": round(aicc, 3), "AIC": round(float(fit.aic), 3)})
            if aicc < best_score:
                best_score = aicc
                best_order = (p, d, q)
        except Exception:
            continue

    if verbose:
        df_rank = pd.DataFrame(records).sort_values("AICc").reset_index(drop=True)
        print(f"Stationarity: Level p-val={adf_lvl:.4f} -> Recommended differencing d={d}")
        print(f"Recommended Order: {best_order} (AICc: {best_score:.3f})")
        print("\nTop Candidate Orders:")
        print(df_rank.head(4).to_string(index=False))
        print("-" * 50)

    return best_order


# ==============================================================================
# 4. SUTTE-ARIMA RESULTS (STATSMODELS COMPLIANT)
# ==============================================================================

class SutteARIMAResults:
    """
    Results class returned by model.fit().
    Provides direct access to all model estimates, evaluation metrics, and forecasts.
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

        # In-sample residuals and metrics (computed strictly from model predictions)
        self.resid = self.endog - self.fittedvalues
        self.in_sample_metrics = calculate_metrics(
            actual=self.endog[4:],
            predicted=self.fittedvalues[4:],
            train_data=self.endog
        )

        # Residual diagnostics
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        self._dw = float(durbin_watson(clean_resid)) if len(clean_resid) > 0 else np.nan
        jb = jarque_bera(clean_resid)
        self._jb_stat = float(jb[0])
        self._jb_pval = float(jb[1])

    # Direct properties
    @property
    def order(self) -> Tuple[int, int, int]:
        """Returns the ARIMA order tuple (p, d, q)."""
        return self._order

    @property
    def params(self) -> pd.Series:
        """Returns fitted ARIMA coefficients."""
        return self.arima_results.params

    @property
    def pvalues(self) -> pd.Series:
        """Returns p-values of parameter estimates."""
        return self.arima_results.pvalues

    @property
    def bse(self) -> pd.Series:
        """Returns standard errors of coefficients."""
        return self.arima_results.bse

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return float(self.arima_results.aic)

    @property
    def aicc(self) -> float:
        """Corrected Akaike Information Criterion."""
        k = len(self.params)
        n = len(self.endog)
        return float(self.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)) if (n - k - 1) > 0 else self.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return float(self.arima_results.bic)

    @property
    def llf(self) -> float:
        """Log-Likelihood."""
        return float(self.arima_results.llf)

    @property
    def durbin_watson(self) -> float:
        """Durbin-Watson statistic for residual autocorrelation."""
        return self._dw

    @property
    def jarque_bera(self) -> Dict[str, float]:
        """Jarque-Bera normality test on residuals."""
        return {"statistic": round(self._jb_stat, 4), "p_value": round(self._jb_pval, 4)}

    # Matrices
    def metrics(self, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns all evaluation metrics:
          - If test_data is None: returns in-sample metrics computed on training series.
          - If test_data is provided: forecasts len(test_data) steps and evaluates on test data!
        """
        if test_data is None:
            return self.in_sample_metrics

        test = np.asarray(test_data, dtype=float).ravel()
        fc = self.forecast(steps=len(test))["SutteARIMA"].values
        return calculate_metrics(actual=test, predicted=fc, train_data=self.endog)

    def metrics_table(self, test_data: Optional[Any] = None) -> pd.DataFrame:
        """Returns all evaluation metrics as a clean pandas DataFrame matrix."""
        m = self.metrics(test_data=test_data)
        col_name = "In-Sample" if test_data is None else "Out-of-Sample (Test Data)"
        return pd.DataFrame(list(m.items()), columns=["Metric", col_name]).set_index("Metric")

    def params_table(self) -> pd.DataFrame:
        """Returns parameter estimates matrix: coef, std err, z, P>|z|, and 95% CI."""
        ci = self.arima_results.conf_int()
        return pd.DataFrame({
            "coef": self.params,
            "std err": self.bse,
            "z": self.arima_results.tvalues if hasattr(self.arima_results, "tvalues") else self.params / self.bse,
            "P>|z|": self.pvalues,
            "[0.025": ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0],
            "0.975]": ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1],
        }).round(4)

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generates out-of-sample forecasts for `steps` ahead with prediction intervals.
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

    def summary(self) -> str:
        """Returns clean, statsmodels-style summary table."""
        m = self.in_sample_metrics
        lines = [
            "=" * 70,
            f"{'SutteARIMA Model Estimation Results':^70}",
            "=" * 70,
            f" Dep. Variable       : y                      No. Observations : {len(self.endog)}",
            f" Model               : SutteARIMA{str(self.order):<10} Log-Likelihood   : {self.llf:<10.3f}",
            f" ARIMA AIC           : {self.aic:<10.3f}       ARIMA AICc       : {self.aicc:<10.3f}",
            f" ARIMA BIC           : {self.bic:<10.3f}       Durbin-Watson    : {self.durbin_watson:<10.4f}",
            "-" * 70,
            f"{'Model Accuracy Matrix (MAPE, MSE, RMSE, MAE)':^70}",
            "-" * 70,
            f" MAPE (%)            : {m.get('MAPE (%)', np.nan):<10.4f}%      sMAPE (%)        : {m.get('sMAPE (%)', np.nan):<10.4f}%",
            f" MSE                 : {m.get('MSE', np.nan):<10.4f}       RMSE             : {m.get('RMSE', np.nan):<10.4f}",
            f" MAE                 : {m.get('MAE', np.nan):<10.4f}       Theil's IC (TIC) : {m.get('TIC', np.nan):<10.4f}",
            f" Max Error (MaxAE)   : {m.get('MaxAE', np.nan):<10.4f}       Mean Bias (MBE)  : {m.get('MBE', np.nan):<10.4f}",
            "-" * 70,
            f"{'ARIMA Parameter Estimates':^70}",
            "-" * 70,
            self.params_table().to_string(),
            "=" * 70,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ==============================================================================
# 5. MAIN SUTTE-ARIMA CLASS
# ==============================================================================

class SutteARIMA:
    """
    SutteARIMA: Statsmodels-style Hybrid Forecasting Model.

    Parameters
    ----------
    endog : array-like
        The univariate time series data.
    order : Tuple[int, int, int]
        The (p, d, q) order for ARIMA.
        You can assign it directly (e.g. order=(1, 1, 0)),
        or find the best order via `order = find_order(data)`.

    Example
    -------
    >>> from sutte_arima import SutteARIMA, find_order
    >>> data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8]
    >>> order = find_order(data)             # step 1: get recommended (p, d, q)
    >>> model = SutteARIMA(data, order=order) # step 2: pass to model
    >>> res = model.fit()                    # step 3: fit
    >>> print(res.summary())                 # see all info
    """

    def __init__(self, endog: Any, order: Tuple[int, int, int]):
        self.endog = np.asarray(endog, dtype=float).ravel()
        if len(self.endog) < 8:
            raise ValueError(f"SutteARIMA requires at least 8 observations, got {len(self.endog)}.")

        if not (isinstance(order, (tuple, list)) and len(order) == 3):
            raise ValueError(
                "You must specify order=(p, d, q), e.g. SutteARIMA(data, order=(1, 1, 0)).\n"
                "Tip: To find the recommended order automatically, call: order = find_order(data)"
            )
        self.order = (int(order[0]), int(order[1]), int(order[2]))

    def fit(self) -> SutteARIMAResults:
        """Fits SutteARIMA and returns SutteARIMAResults."""
        arima_res = ARIMA(self.endog, order=self.order).fit()
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
