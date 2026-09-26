# SutteARIMA: Statsmodels-Style Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for the **SutteARIMA** hybrid time series forecasting model, designed with the familiar **statsmodels** interface (`model = SutteARIMA(...)`, `res = model.fit()`, `res.summary()`).

SutteARIMA combines the non-parametric **Alpha-Sutte (α-Sutte) Indicator** with the parametric **Box-Jenkins ARIMA** model by averaging their forecasts:

$$\hat{Z}_t^{\text{SutteARIMA}} = \frac{\hat{Z}_t^{\text{ARIMA}} + \hat{Z}_t^{\alpha\text{-Sutte}}}{2}$$

---

## Installation Directly from GitHub

```bash
pip install git+https://github.com/BahmanOrujov/sutte-ARIMA.git
```

Or clone and import locally:
```bash
git clone https://github.com/BahmanOrujov/sutte-ARIMA.git
```

---

## Quickstart (Statsmodels Interface)

Just 3 lines of code:

```python
from sutte_arima import SutteARIMA

# 1. Your data
data = [10.2, 10.8, 11.5, 11.9, 12.4, 13.1, 13.8, 14.5, 15.0, 15.8, 16.5, 17.2]
train, test = data[:9], data[9:]

# 2. Initialize and Fit (ARIMA order is auto-selected if omitted!)
model = SutteARIMA(train)
res = model.fit()

# 3. View Full Model Estimation & Accuracy Summary
print(res.summary())
```

### Output:
```text
======================================================================
                 SutteARIMA Model Estimation Results                  
======================================================================
 Dep. Variable       : y                      No. Observations : 9
 Model               : SutteARIMA(0, 2, 0)  Date             : Available
 ARIMA AIC           : -3.381           ARIMA BIC        : -3.435    
----------------------------------------------------------------------
                In-Sample Forecasting Accuracy Metrics                
----------------------------------------------------------------------
 MAE                 : 0.0995           MSE              : 0.0163    
 RMSE                : 0.1278           R-squared (R2)   : 0.9813    
 MAPE (%)            : 0.7103    %      sMAPE (%)        : 0.7106    %
 MdAPE (%)           : 0.5600    %      Theil's IC (TIC) : 0.0046    
 Max Error (MaxAE)   : 0.2091           Mean Bias (MBE)  : 0.0159    
----------------------------------------------------------------------
                      Residual Diagnostic Checks                      
----------------------------------------------------------------------
 Durbin-Watson       : 1.1721           Jarque-Bera (p)  : 0.8021    
======================================================================
```

---

## Multi-Step Forecasting

Generate multi-step ahead forecasts with 95% Confidence Intervals:

```python
forecast_df = res.forecast(steps=5, alpha=0.05)
print(forecast_df)
```

| Horizon | SutteARIMA | ARIMA | AlphaSutte | Lower_95 | Upper_95 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 15.5740 | 15.50 | 15.6480 | 15.2961 | 15.8519 |
| 2 | 16.1386 | 16.00 | 16.2772 | 15.5171 | 16.7601 |
| 3 | 16.6905 | 16.50 | 16.8811 | 15.6506 | 17.7305 |
| 4 | 17.2602 | 17.00 | 17.5205 | 15.7379 | 18.7825 |
| 5 | 17.8282 | 17.50 | 18.1564 | 15.7670 | 19.8894 |

---

## Comprehensive Evaluation Metrics

Evaluate your model with one command:

```python
# 1. In-Sample Metrics (on training data)
print(res.metrics())

# 2. Out-of-Sample Metrics (on held-out test data)
print(res.metrics(test))
```

### Supported Metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE (%)**: Mean Absolute Percentage Error
- **MdAPE (%)**: Median Absolute Percentage Error
- **sMAPE (%)**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error (vs. naive benchmark)
- **R²**: Coefficient of Determination
- **MaxAE**: Maximum Absolute Error
- **MBE**: Mean Bias Error (directional bias)
- **TIC**: Theil's Inequality Coefficient (0 = perfect fit)
- **DW**: Durbin-Watson Residual Autocorrelation Test
- **JB**: Jarque-Bera Residual Normality Test

---

## Manual Order Specification

If you know the order from research (e.g. `ARIMA(0, 1, 1)` or `ARIMA(2, 2, 1)`):

```python
model = SutteARIMA(train, order=(0, 1, 1))
res = model.fit()
```

---

## References

1. Ahmar, A.S. (2017). "α-Sutte Indicator: Suatu Pendekatan Baru dalam Peramalan Data." OSF Preprints, doi:10.17605/osf.io/rknsv.
2. Ahmar, A.S., & Boj, E. (2020). "SutteARIMA: Short-term forecasting method, a case: Covid-19 and stock market in Spain." Science of The Total Environment, 729, 138883.
3. Ahmar, A.S., Boj del Val, E., et al. (2022). "SutteARIMA: A Novel Method for Forecasting the Infant Mortality Rate in Indonesia." Computers, Materials & Continua, 70(3), 6007-6022.
4. Ahmar, A.S., Singh, P.K., et al. (2023). "Comparison of ARIMA, SutteARIMA, and Holt-Winters, and NNAR Models to Predict Food Grain in India." Forecasting, 5(1), 138-152.
