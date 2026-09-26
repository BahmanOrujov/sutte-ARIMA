# SutteARIMA: Statsmodels-Style Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A streamlined Python library for the **SutteARIMA** hybrid time series forecasting model, designed with the clean **statsmodels** interface.

$$\hat{Z}_t^{\text{SutteARIMA}} = \frac{\hat{Z}_t^{\text{ARIMA}} + \hat{Z}_t^{\alpha\text{-Sutte}}}{2}$$

---

## Installation Directly from GitHub

```bash
pip install --upgrade git+https://github.com/BahmanOrujov/sutte-ARIMA.git
```

---

## Ultra-Simple Usage (Just 3 Lines)

```python
from sutte_arima import SutteARIMA, find_order

data = [94.77, 96.23, 98.12, 99.09, 100.04, 100.12, 99.93, 100.09,
        101.44, 102.38, 103.68, 104.12, 104.81, 105.35, 106.36, 106.89]

# 1. Call find_order to get the recommended (p, d, q)
order = find_order(data)

# 2. Pass order directly into SutteARIMA and fit
model = SutteARIMA(data, order=order)
res = model.fit()

# 3. View Full Estimation Results & Accuracy Matrix
print(res.summary())
```

### Summary Output:
```text
======================================================================
                 SutteARIMA Model Estimation Results                  
======================================================================
 Dep. Variable       : y                      No. Observations : 16
 Model               : SutteARIMA(1, 1, 0)  Log-Likelihood   : -13.446   
 ARIMA AIC           : 30.892           ARIMA AICc       : 31.815    
 ARIMA BIC           : 32.308           Durbin-Watson    : 1.7523    
----------------------------------------------------------------------
             Model Accuracy Matrix (MAPE, MSE, RMSE, MAE)             
----------------------------------------------------------------------
 MAPE (%)            : 0.4394    %      sMAPE (%)        : 0.4395    %
 MSE                 : 0.3333           RMSE             : 0.5773    
 MAE                 : 0.4498           Theil's IC (TIC) : 0.0028    
 Max Error (MaxAE)   : 1.2758           Mean Bias (MBE)  : -0.0151   
----------------------------------------------------------------------
                      ARIMA Parameter Estimates                       
----------------------------------------------------------------------
     coef  std err       z   P>|z|  [0.025  0.975]
0  0.8222   0.1522  5.4035  0.0000  0.5240  1.1205
1  0.3262   0.1518  2.1495  0.0316  0.0288  0.6237
======================================================================
```

---

## Direct Code Attributes

Access any model property with code:

```python
print("ARIMA Order:       ", res.order)           # (1, 1, 0)
print("Coefficients:      \n", res.params)
print("P-values:          \n", res.pvalues)
print("AIC / AICc / BIC:  ", res.aic, res.aicc, res.bic)
print("Durbin-Watson:     ", res.durbin_watson)  # ~2.0 indicates no residual autocorrelation
print("Jarque-Bera:       ", res.jarque_bera)     # Normality test
print("Fitted Values:     \n", res.fittedvalues)
print("Residuals:         \n", res.resid)
```

---

## All Evaluation Matrices

All metrics are calculated **strictly after the model runs** from the model's actual predictions:

```python
# 1. In-sample evaluation matrix:
print(res.metrics_table())

# 2. Out-of-sample evaluation matrix (on held-out test data):
test_data = [107.35, 107.21, 107.72]
print(res.metrics_table(test_data))
```

### Metrics Included:
- **MAPE (%)**: Mean Absolute Percentage Error (primary decision criterion in Sutte literature)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **sMAPE (%)**: Symmetric MAPE
- **TIC**: Theil's Inequality Coefficient ($0 = \text{perfect forecast}$)
- **MASE**: Mean Absolute Scaled Error
- **MaxAE**: Maximum Absolute Error
- **MBE**: Mean Bias Error

> **Note on $R^2$ in Time Series:**
> In classical time series forecasting (especially with non-stationary data and differencing), $R^2$ is not an appropriate metric because trended series artificially inflate $R^2 \approx 0.99$ even on poor fits. Following the peer-reviewed papers by Dr. Ansari Saleh Ahmar, the model relies on **MAPE, MSE, RMSE, MAE, and TIC**.

---

## Multi-Step Forecasting

```python
forecast_df = res.forecast(steps=5, alpha=0.05)
print(forecast_df)
```

| Horizon | SutteARIMA | ARIMA | AlphaSutte | Lower_95 | Upper_95 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 107.4558 | 107.3258 | 107.5858 | 106.2744 | 108.6372 |
| 2 | 108.0090 | 107.6841 | 108.3339 | 105.5534 | 110.4646 |
| 3 | 108.4863 | 107.9787 | 108.9939 | 104.6470 | 112.3257 |
| 4 | 108.9593 | 108.2210 | 109.6975 | 103.6906 | 114.2279 |
| 5 | 109.4119 | 108.4201 | 110.4037 | 102.7063 | 116.1176 |

---

## Manual Order Specification

If you already have your order:

```python
model = SutteARIMA(data, order=(0, 1, 1))
res = model.fit()
```

---

## How `find_order(data)` Selects $(p, d, q)$

1. **Differencing $d$**: Runs Augmented Dickey-Fuller (ADF) test. If level data has $p \ge 0.05$, differences the series until stationary ($d = 1$ or $d = 2$).
2. **Orders $p, q$**: Estimates candidate models and selects the $(p, d, q)$ specification that minimizes **AICc** (small-sample corrected Akaike Information Criterion).
