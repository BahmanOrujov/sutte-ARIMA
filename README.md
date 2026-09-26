# SutteARIMA: Statsmodels-Style Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for the **SutteARIMA** hybrid time series forecasting model, designed with the familiar **statsmodels** interface (`model = SutteARIMA(data, order=(p, d, q))`, `res = model.fit()`, `res.summary()`).

SutteARIMA combines the non-parametric **Alpha-Sutte (α-Sutte) Indicator** with the parametric **Box-Jenkins ARIMA** model by averaging their forecasts:

$$\hat{Z}_t^{\text{SutteARIMA}} = \frac{\hat{Z}_t^{\text{ARIMA}} + \hat{Z}_t^{\alpha\text{-Sutte}}}{2}$$

---

## Installation Directly from GitHub

```bash
pip install --upgrade git+https://github.com/BahmanOrujov/sutte-ARIMA.git
```

---

## Quickstart (Statsmodels Interface)

Specify your chosen `order=(p, d, q)` and fit the model:

```python
from sutte_arima import SutteARIMA

# 1. Your data
data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8, 16.5, 17.2]
train, test = data[:9], data[9:]

# 2. Initialize with your chosen order=(p, d, q) and Fit
model = SutteARIMA(train, order=(0, 1, 1))
res = model.fit()

# 3. View Full Estimation & Evaluation Metrics Summary
print(res.summary())
```

---

## Calling Key Attributes Directly in Code

Access any model parameter or statistic directly via code attributes:

```python
print("ARIMA Order:       ", res.order)           # Returns: (0, 1, 1)
print("AIC / AICc / BIC:  ", res.aic, res.aicc, res.bic)
print("Log-Likelihood:    ", res.llf)
print("Coefficients:      \n", res.params)
print("P-values:          \n", res.pvalues)
print("Standard Errors:   \n", res.bse)
print("Durbin-Watson Stat:", res.durbin_watson)  # ~2.0 indicates no residual autocorrelation
print("Jarque-Bera Test:  ", res.jarque_bera)     # Normality test: {'statistic': ..., 'p_value': ...}
print("Fitted Values:     \n", res.fittedvalues)
print("Residuals:         \n", res.resid)
```

---

## All Model Matrices & Tables

All diagnostics and metrics can be exported or printed as pandas DataFrame matrices:

### 1. Evaluation Metrics Matrix (`res.metrics_table()`)
```python
# In-Sample accuracy on training data:
print(res.metrics_table())

# Out-of-Sample accuracy on held-out test data:
print(res.metrics_table(test))
```
| Metric | Value (Out-of-Sample) | Description |
|---|---|---|
| **MAE** | 0.7878 | Mean Absolute Error |
| **MSE** | 0.7230 | Mean Squared Error |
| **RMSE** | 0.8503 | Root Mean Squared Error |
| **MAPE (%)** | 4.7133% | Mean Absolute Percentage Error |
| **MdAPE (%)** | 4.7493% | Median Absolute Percentage Error |
| **sMAPE (%)** | 4.8440% | Symmetric MAPE |
| **MASE** | 1.3131 | Mean Absolute Scaled Error (vs. naive benchmark) |
| **R²** | 0.9511 | Coefficient of Determination |
| **MaxAE** | 1.1817 | Maximum Absolute Error |
| **MBE** | 0.7878 | Mean Bias Error (directional bias) |
| **TIC** | 0.0264 | Theil's Inequality Coefficient (0 = perfect fit) |

### 2. Parameters Matrix (`res.params_table()`)
```python
print(res.params_table())
```
Returns a matrix with `coef`, `std err`, `z`, `P>|z|`, `[0.025`, and `0.975]`.

### 3. Residual Diagnostic Matrix (`res.diagnostics_table()`)
```python
print(res.diagnostics_table())
```
Returns a matrix of the **Ljung-Box** test across lags checking if residuals are Gaussian White Noise ($p \ge 0.05$).

### 4. Step-by-Step Forecast Matrix (`res.forecast_table()`)
```python
print(res.forecast_table(steps=5, test_data=test))
```
Returns a matrix showing: `Actual`, `SutteARIMA`, `ARIMA`, `AlphaSutte`, `Lower_95`, `Upper_95`, `Abs_Error`, and `APE (%)`.

---

## How Order $(p, d, q)$ is Selected

ARIMA order identification follows the classical **Box-Jenkins methodology**:

1. **Determining $d$ (Differencing Order)**:
   - Runs the **Augmented Dickey-Fuller (ADF)** unit-root test on level data $Y_t$.
   - If non-stationary ($p \ge 0.05$), first differencing $\Delta Y_t = Y_t - Y_{t-1}$ is applied.
   - If stationary ($p < 0.05$), $d = 1$. Otherwise, differencing is repeated ($d = 2$).
2. **Determining $p$ and $q$**:
   - Autoregressive order $p$ is guided by significant spikes in the Partial Autocorrelation Function (PACF).
   - Moving Average order $q$ is guided by significant spikes in the Autocorrelation Function (ACF).
3. **Information Criteria Minimization (AICc)**:
   - Candidate models are estimated via Maximum Likelihood and ranked by **AICc**:
     $$\text{AICc} = -2\ln(L) + 2k + \frac{2k(k+1)}{N - k - 1}$$
   - The specification minimizing AICc provides the optimal balance of fit and parsimony.

### Inspect the Order Selection Matrix On Demand:
```python
best_order, search_matrix = SutteARIMA.auto_select_order(train, max_p=3, max_q=3)

print("Recommended Order:", best_order)
print(search_matrix.head(10))
```

---

## Recommended Orders for Common Datasets

| Dataset | Recommended $(p, d, q)$ | Reason / Source |
|---|---|---|
| **CRAN `sutteForecastR` Reference Series** | **`ARIMA(1, 1, 0)`** | ADF level non-stationary ($p=0.91$), diff stationary ($p=0.0009$), minimum AICc = 45.91. |
| **Turkiye Sigorta Stock Price History** | **`ARIMA(2, 1, 3)`** or **`ARIMA(0, 1, 0)`** | First diff stationary ($p=0.0000$). `(2,1,3)` minimizes AICc; `(0,1,0)` minimizes BIC. |
| **COVID-19 Spain Cases** | **`ARIMA(2, 2, 1)`** | Ahmar & Boj (2020), *Science of Total Environment*. |
| **IBEX 35 Stock Index** | **`ARIMA(0, 1, 0)`** (with drift) | Ahmar & Boj (2020), *Science of Total Environment*. |
| **Food Grain Yield (Rice, Pulses)** | **`ARIMA(0, 1, 1)`** | Ahmar et al. (2023), *Forecasting MDPI*. |
| **Infant Mortality Rate** | **`ARIMA(0, 2, 2)`** | Ahmar et al. (2022), *CMC*. |

---

## References

1. Ahmar, A.S. (2017). "α-Sutte Indicator: Suatu Pendekatan Baru dalam Peramalan Data." OSF Preprints, doi:10.17605/osf.io/rknsv.
2. Ahmar, A.S., & Boj, E. (2020). "SutteARIMA: Short-term forecasting method, a case: Covid-19 and stock market in Spain." Science of The Total Environment, 729, 138883.
3. Ahmar, A.S., Boj del Val, E., et al. (2022). "SutteARIMA: A Novel Method for Forecasting the Infant Mortality Rate in Indonesia." Computers, Materials & Continua, 70(3), 6007-6022.
4. Ahmar, A.S., Singh, P.K., et al. (2023). "Comparison of ARIMA, SutteARIMA, and Holt-Winters, and NNAR Models to Predict Food Grain in India." Forecasting, 5(1), 138-152.
