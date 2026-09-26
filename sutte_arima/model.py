"""
SutteARIMA: Minimal, Statsmodels-Style Time Series Forecasting Model
Combining Alpha-Sutte Indicator and Box-Jenkins ARIMA.
Designed specifically for econometricians and empirical researchers.
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
      - MAPE (%)  : Mean Absolute Percentage Error (primary metric in Sutte literature)
      - MSE       : Mean Squared Error
      - RMSE      : Root Mean Squared Error
      - MAE       : Mean Absolute Error
      - sMAPE (%) : Symmetric Mean Absolute Percentage Error
      - TIC       : Theil's Inequality Coefficient (0 = perfect forecast, 1 = worst)
      - MASE      : Mean Absolute Scaled Error (vs. naive historical diff)
      - MaxAE     : Maximum Absolute Error
      - MBE       : Mean Bias Error
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

    # MAPE (%)
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
               ic: str = "AICc",
               verbose: bool = False) -> Tuple[int, int, int]:
    """
    Analyzes the dataset and returns the recommended (p, d, q) order for ARIMA.

    Econometric Workflow:
      1. Unit-Root Test: Runs Augmented Dickey-Fuller (ADF) test to identify
         the integration/differencing order d (0 for level-stationary, 1, or 2).
      2. Grid-Search: Fits candidate (p, d, q) combinations across p in [0, max_p]
         and q in [0, max_q].
      3. Parsimonious Selection: Selects the specification that minimizes the
         chosen Information Criterion (default AICc for small/finite sample correction).

    Parameters
    ----------
    data : array-like
        The input time-series data.
    max_p : int, default=3
        Maximum autoregressive order to test.
    max_q : int, default=3
        Maximum moving average order to test.
    ic : str, default='AICc'
        Information criterion to minimize ('AICc', 'AIC', 'BIC', or 'HQIC').
    verbose : bool, default=False
        Whether to print stationarity diagnostics and candidate ranking table.

    Returns
    -------
    Tuple[int, int, int]
        The recommended (p, d, q) order.
    """
    y = np.asarray(data, dtype=float).ravel()
    n = len(y)
    if n < 8:
        raise ValueError(f"Need at least 8 observations, got {n}.")

    ic_key = ic.upper()
    valid_ics = {"AICC", "AIC", "BIC", "HQIC"}
    if ic_key not in valid_ics:
        raise ValueError(f"Unknown criterion ic='{ic}'. Choose from {sorted(valid_ics)}.")

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
            aic_val = float(fit.aic)
            bic_val = float(fit.bic)
            hqic_val = float(fit.hqic) if hasattr(fit, "hqic") else aic_val
            aicc_val = float(aic_val + (2.0 * k * (k + 1.0)) / (n - k - 1.0)) if (n - k - 1) > 0 else aic_val

            scores = {
                "AICC": aicc_val,
                "AIC": aic_val,
                "BIC": bic_val,
                "HQIC": hqic_val
            }
            score = scores[ic_key]

            records.append({
                "order": (p, d, q),
                "AICc": round(aicc_val, 3),
                "AIC": round(aic_val, 3),
                "BIC": round(bic_val, 3),
                "HQIC": round(hqic_val, 3)
            })

            if score < best_score:
                best_score = score
                best_order = (p, d, q)
        except Exception:
            continue

    if verbose and len(records) > 0:
        df_rank = pd.DataFrame(records).sort_values(ic if ic != "AICC" else "AICc").reset_index(drop=True)
        print(f"ADF Stationarity: Level p-val={adf_lvl:.4f} -> Recommended differencing d={d}")
        print(f"Recommended Order: {best_order} (Min {ic}: {best_score:.3f})")
        print("\nTop Candidate Orders:")
        print(df_rank.head(5).to_string(index=False))
        print("-" * 60)

    return best_order


# ==============================================================================
# 4. SUTTE-ARIMA RESULTS (STATSMODELS COMPLIANT)
# ==============================================================================

class SutteARIMAResults:
    """
    Results class returned by SutteARIMA.fit().
    Provides direct access to all model estimates, evaluation metrics, diagnostics, and forecasts.
    """

    def __init__(self,
                 model: SutteARIMA,
                 arima_res: ARIMAResults,
                 fittedvalues: np.ndarray,
                 order: Tuple[int, int, int],
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        self.model = model
        self.arima_results = arima_res
        self.fittedvalues = fittedvalues
        self._order = order
        self._seasonal_order = seasonal_order
        self.endog = model.endog

        # In-sample residuals and metrics (computed strictly from hybrid predictions)
        self.resid = self.endog - self.fittedvalues
        self.in_sample_metrics = calculate_metrics(
            actual=self.endog[4:],
            predicted=self.fittedvalues[4:],
            train_data=self.endog
        )

        # Residual diagnostics
        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        self._dw = float(durbin_watson(clean_resid)) if len(clean_resid) > 0 else np.nan

        # Jarque-Bera normality test: (jb_stat, pvalue, skew, kurtosis)
        if len(clean_resid) > 3:
            jb = jarque_bera(clean_resid)
            self._jb_stat = float(jb[0])
            self._jb_pval = float(jb[1])
            self._skew = float(jb[2])
            self._kurtosis = float(jb[3])
        else:
            self._jb_stat = self._jb_pval = self._skew = self._kurtosis = np.nan

        # Ljung-Box test for residual autocorrelation (white noise check)
        if len(clean_resid) > 6:
            try:
                lb_lag = min(10, len(clean_resid) // 3)
                lb_res = acorr_ljungbox(clean_resid, lags=[lb_lag], return_df=True)
                self._lb_stat = float(lb_res["lb_stat"].iloc[0])
                self._lb_pval = float(lb_res["lb_pvalue"].iloc[0])
                self._lb_lag = lb_lag
            except Exception:
                self._lb_stat = self._lb_pval = np.nan
                self._lb_lag = 0
        else:
            self._lb_stat = self._lb_pval = np.nan
            self._lb_lag = 0

    # --------------------------------------------------------------------------
    # Direct Model Properties
    # --------------------------------------------------------------------------
    @property
    def order(self) -> Tuple[int, int, int]:
        """Returns the ARIMA order tuple (p, d, q)."""
        return self._order

    @property
    def seasonal_order(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns the seasonal order tuple (P, D, Q, s) if specified."""
        return self._seasonal_order

    @property
    def params(self) -> pd.Series:
        """Returns fitted ARIMA/SARIMAX parameter estimates."""
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.params, index=names)

    @property
    def pvalues(self) -> pd.Series:
        """Returns p-values of parameter estimates."""
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.pvalues, index=names)

    @property
    def bse(self) -> pd.Series:
        """Returns standard errors of coefficients."""
        names = getattr(self.arima_results, "param_names", None)
        return pd.Series(self.arima_results.bse, index=names)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return float(self.arima_results.aic)

    @property
    def aicc(self) -> float:
        """Corrected Akaike Information Criterion for finite samples."""
        k = len(self.params)
        n = len(self.endog)
        return float(self.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)) if (n - k - 1) > 0 else self.aic

    @property
    def bic(self) -> float:
        """Bayesian / Schwarz Information Criterion."""
        return float(self.arima_results.bic)

    @property
    def hqic(self) -> float:
        """Hannan-Quinn Information Criterion."""
        if hasattr(self.arima_results, "hqic"):
            return float(self.arima_results.hqic)
        k = len(self.params)
        n = len(self.endog)
        return float(-2.0 * self.llf + 2.0 * k * np.log(np.log(n)))

    @property
    def llf(self) -> float:
        """Log-Likelihood."""
        return float(self.arima_results.llf)

    @property
    def durbin_watson(self) -> float:
        """Durbin-Watson statistic for residual autocorrelation (~2.0 = no autocorrelation)."""
        return self._dw

    @property
    def ljung_box(self) -> Dict[str, float]:
        """Ljung-Box Portmanteau test for residual autocorrelation (H0: white noise)."""
        return {
            "statistic": round(self._lb_stat, 4),
            "p_value": round(self._lb_pval, 4),
            "lags": self._lb_lag
        }

    @property
    def jarque_bera(self) -> Dict[str, float]:
        """Jarque-Bera test for normality of residuals (H0: normal distribution)."""
        return {
            "statistic": round(self._jb_stat, 4),
            "p_value": round(self._jb_pval, 4),
            "skew": round(self._skew, 4),
            "kurtosis": round(self._kurtosis, 4)
        }

    # Accuracy shortcuts
    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error (%)."""
        return self.in_sample_metrics.get("MAPE (%)", np.nan)

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return self.in_sample_metrics.get("RMSE", np.nan)

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return self.in_sample_metrics.get("MAE", np.nan)

    # --------------------------------------------------------------------------
    # Tables and Matrices
    # --------------------------------------------------------------------------
    def metrics(self, test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns all forecasting accuracy metrics:
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
        names = getattr(self.arima_results, "param_names", None)
        z_vals = self.arima_results.tvalues if hasattr(self.arima_results, "tvalues") else self.params / self.bse

        ci_low = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0]
        ci_high = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1]

        df = pd.DataFrame({
            "coef": np.asarray(self.params),
            "std err": np.asarray(self.bse),
            "z": np.asarray(z_vals),
            "P>|z|": np.asarray(self.pvalues),
            "[0.025": np.asarray(ci_low),
            "0.975]": np.asarray(ci_high),
        }, index=names if names is not None else list(range(len(self.params))))
        return df.round(4)

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generates out-of-sample forecasts for `steps` ahead with prediction intervals.

        Parameters
        ----------
        steps : int, default=1
            Number of future periods to forecast.
        alpha : float, default=0.05
            Significance level for prediction intervals (default 0.05 for 95% CI).

        Returns
        -------
        pd.DataFrame
            DataFrame containing SutteARIMA, ARIMA, AlphaSutte, and prediction bounds.
        """
        arima_fc = self.arima_results.get_forecast(steps=steps)
        a_mean = np.asarray(arima_fc.predicted_mean, dtype=float)
        a_se = np.asarray(arima_fc.se_mean, dtype=float)

        s_mean = AlphaSutte.compute_forecast(self.endog, steps=steps)
        hybrid_mean = 0.5 * (a_mean + s_mean)

        clean_resid = self.resid[4:][~np.isnan(self.resid[4:])]
        hybrid_sigma = np.std(clean_resid, ddof=1) if len(clean_resid) > 1 else 1.0
        se_scale = a_se / (a_se[0] if len(a_se) > 0 and a_se[0] > 0 else 1.0)
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

    def plot(self, steps: Optional[int] = None, alpha: float = 0.05, figsize: Tuple[int, int] = (10, 5)):
        """
        Single-line visual diagnostic chart: Actual vs. Fitted vs. Forecast.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Please install it via: pip install matplotlib")

        fig, ax = plt.subplots(figsize=figsize)
        t_hist = np.arange(len(self.endog))
        ax.plot(t_hist, self.endog, label="Actual", color="#1f77b4", linewidth=1.8)
        ax.plot(t_hist, self.fittedvalues, label="SutteARIMA Fitted", color="#ff7f0e", linestyle="--", linewidth=1.5)

        if steps is not None and steps > 0:
            fc_df = self.forecast(steps=steps, alpha=alpha)
            t_fc = np.arange(len(self.endog), len(self.endog) + steps)
            pct = int(round((1.0 - alpha) * 100))
            ax.plot(t_fc, fc_df["SutteARIMA"], label=f"Forecast ({steps} steps)", color="#2ca02c", linewidth=2.0)
            ax.fill_between(
                t_fc,
                fc_df[f"Lower_{pct}"],
                fc_df[f"Upper_{pct}"],
                color="#2ca02c",
                alpha=0.18,
                label=f"{pct}% Confidence Interval"
            )

        model_name = f"SutteARIMA{self.order}"
        if self.seasonal_order:
            model_name += f"x{self.seasonal_order}"
        ax.set_title(f"{model_name} — Actual vs. Fitted & Forecast", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time Index", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        return fig, ax

    def summary(self) -> str:
        """Returns clean, statsmodels-style summary table for econometricians."""
        m = self.in_sample_metrics
        model_str = f"SutteARIMA{self.order}"
        if self.seasonal_order:
            model_str += f"x{self.seasonal_order}"

        lines = [
            "=" * 72,
            f"{'SutteARIMA Model Estimation Results':^72}",
            "=" * 72,
            f" Dep. Variable       : y                      No. Observations : {len(self.endog)}",
            f" Model               : {model_str:<23} Log-Likelihood   : {self.llf:<10.3f}",
            f" AIC                 : {self.aic:<10.3f}       AICc             : {self.aicc:<10.3f}",
            f" BIC                 : {self.bic:<10.3f}       HQIC             : {self.hqic:<10.3f}",
            "-" * 72,
            f"{'Model Accuracy Matrix (MAPE, MSE, RMSE, MAE)':^72}",
            "-" * 72,
            f" MAPE (%)            : {m.get('MAPE (%)', np.nan):<10.4f}%      sMAPE (%)        : {m.get('sMAPE (%)', np.nan):<10.4f}%",
            f" MSE                 : {m.get('MSE', np.nan):<10.4f}       RMSE             : {m.get('RMSE', np.nan):<10.4f}",
            f" MAE                 : {m.get('MAE', np.nan):<10.4f}       Theil's IC (TIC) : {m.get('TIC', np.nan):<10.4f}",
            f" Max Error (MaxAE)   : {m.get('MaxAE', np.nan):<10.4f}       Mean Bias (MBE)  : {m.get('MBE', np.nan):<10.4f}",
            "-" * 72,
            f"{'ARIMA Parameter Estimates':^72}",
            "-" * 72,
            self.params_table().to_string(),
            "-" * 72,
            f"{'Residual Diagnostic Tests':^72}",
            "-" * 72,
            f" Durbin-Watson       : {self.durbin_watson:<10.4f}       (residual autocorrelation ~ 2.0)",
            f" Ljung-Box (Q)       : {self._lb_stat:<10.4f}       Prob(Q)          : {self._lb_pval:<10.4f} (white noise)",
            f" Jarque-Bera (JB)    : {self._jb_stat:<10.4f}       Prob(JB)         : {self._jb_pval:<10.4f} (normality)",
            f" Skew                : {self._skew:<10.4f}       Kurtosis         : {self._kurtosis:<10.4f}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter Notebooks."""
        return f"<pre style='font-family: monospace;'>{self.summary()}</pre>"


# ==============================================================================
# 5. MAIN SUTTE-ARIMA CLASS (ONE-LINER FIT FOR ECONOMETRICIANS)
# ==============================================================================

class SutteARIMA:
    """
    SutteARIMA: Statsmodels-style Hybrid Forecasting Model.
    Designed for econometricians: minimal code, rich statistical diagnostics.

    Parameters
    ----------
    endog : array-like
        The univariate time series data.
    order : Tuple[int, int, int] or 'auto', default='auto'
        The (p, d, q) order for the ARIMA component.
        - 'auto' (default): Automatically conducts ADF unit-root test for differencing
          and grid-searches orders minimizing AICc.
        - tuple (p, d, q): Specific order (e.g. order=(0, 1, 1)).
    seasonal_order : Tuple[int, int, int, int], optional
        The (P, D, Q, s) seasonal specification (e.g. (2, 1, 0, 12)).

    Examples
    --------
    >>> from sutte_arima import SutteARIMA
    >>> data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8]
    >>> # 1-liner with automatic order selection:
    >>> res = SutteARIMA(data).fit()
    >>> print(res.summary())
    >>>
    >>> # Or with specific order (e.g. from Box-Jenkins identification):
    >>> res = SutteARIMA(data, order=(0, 1, 1)).fit()
    """

    def __init__(self,
                 endog: Any,
                 order: Union[Tuple[int, int, int], str] = "auto",
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        self.endog = np.asarray(endog, dtype=float).ravel()
        if len(self.endog) < 8:
            raise ValueError(f"SutteARIMA requires at least 8 observations, got {len(self.endog)}.")

        self.seasonal_order = seasonal_order

        # Automatic or explicit order selection
        if order == "auto" or order is None:
            self.order = find_order(self.endog, verbose=False)
        elif isinstance(order, (tuple, list)) and len(order) == 3:
            self.order = (int(order[0]), int(order[1]), int(order[2]))
        else:
            raise ValueError(
                f"Invalid order={order}. Please specify order as a (p, d, q) tuple like (0, 1, 1), "
                f"or order='auto' (default) to determine it automatically."
            )

    def fit(self) -> SutteARIMAResults:
        """Fits SutteARIMA and returns SutteARIMAResults with full econometric diagnostics."""
        if self.seasonal_order is not None:
            arima_res = ARIMA(self.endog, order=self.order, seasonal_order=self.seasonal_order).fit()
        else:
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
            order=self.order,
            seasonal_order=self.seasonal_order
        )
