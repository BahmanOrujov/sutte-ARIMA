# SutteARIMA: Econometric Time Series Forecasting in Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SutteARIMA** is a specialized time series forecasting library that combines the non-parametric **$\alpha$-Sutte Indicator** (Ahmar, 2017) with the classical **Box-Jenkins ARIMA** model (Ahmar et al., 2019):

$$\hat{Z}_t^{\text{SutteARIMA}} = \frac{\hat{Z}_t^{\text{ARIMA}} + \hat{Z}_t^{\alpha\text{-Sutte}}}{2}$$

Built specifically for **econometricians, financial analysts, and empirical researchers**—delivering maximum statistical value with minimal code.

---

## Installation

```bash
pip install --upgrade git+https://github.com/BahmanOrujov/sutte-ARIMA.git
```

---

## Quickstart: Less Code, Maximum Value (Just 2 Lines)

Econometricians are researchers, not software engineers. You do not need to write boilerplate setup loops:

```python
from sutte_arima import SutteARIMA

# 1. Fit model (automatic unit-root test & AICc order selection by default!)
res = SutteARIMA(data).fit()

# 2. Print full econometric estimation results & diagnostic tests
print(res.summary())
```

---

## Econometric Summary Table

The estimation summary provides the exact statistical diagnostics econometricians rely on (matching Statsmodels, Stata, and EViews standards):

```text
========================================================================
                  SutteARIMA Model Estimation Results                   
========================================================================
 Dep. Variable       : y                      No. Observations : 132
 Model               : SutteARIMA(0, 1, 1)    Log-Likelihood   : -629.598  
 AIC                 : 1263.195               AICc             : 1263.288  
 BIC                 : 1268.945               HQIC             : 1265.532  
------------------------------------------------------------------------
              Model Accuracy Matrix (MAPE, MSE, RMSE, MAE)              
------------------------------------------------------------------------
 MAPE (%)            : 9.3010 %               sMAPE (%)        : 9.3832 %
 MSE                 : 1099.6808              RMSE             : 33.1614   
 MAE                 : 25.1785                Theil's IC (TIC) : 0.0577    
 Max Error (MaxAE)   : 126.6663               Mean Bias (MBE)  : 0.0318    
------------------------------------------------------------------------
                       ARIMA Parameter Estimates                        
------------------------------------------------------------------------
             coef   std err         z     P>|z|     [0.025     0.975]
ma.L1      0.3730    0.0886    4.2082    0.0000     0.1993     0.5468
sigma2   873.9665   94.2428    9.2736    0.0000   689.2540  1058.6791
------------------------------------------------------------------------
                       Residual Diagnostic Tests                        
------------------------------------------------------------------------
 Durbin-Watson       : 1.6834        (residual autocorrelation ~ 2.0)
 Ljung-Box (Q)       : 56.4081       Prob(Q)          : 0.0000 (white noise)
 Jarque-Bera (JB)    : 21.7344       Prob(JB)         : 0.0000 (normality)
 Skew                : -0.4515       Kurtosis         : 4.8055    
========================================================================
```

---

## Econometric Methodology: How the Recommended Order $(p, d, q)$ is Selected

A core principle in econometrics is the **Box-Jenkins methodology**: Identification $\to$ Estimation $\to$ Diagnostic Checking $\to$ Forecasting. `SutteARIMA` automates this sequence using formal econometric tests:

### Step 1: Unit-Root & Integration Order ($d$) via ADF Test
Before estimating ARMA terms, the time series must be tested for weak stationarity:
- The **Augmented Dickey-Fuller (ADF)** test checks the null hypothesis:
  $$H_0: \text{The series has a unit root (non-stationary)}$$
- If the level series $y_t$ yields $p_{\text{ADF}} < 0.05$, the series is $I(0)$ $\implies d = 0$.
- If $p_{\text{ADF}} \ge 0.05$, the first difference $\Delta y_t = y_t - y_{t-1}$ is tested. If $p_{\text{ADF}} < 0.05$, the series is $I(1)$ $\implies d = 1$.
- If still non-stationary, second differencing $\Delta^2 y_t$ is applied $\implies d = 2$.

### Step 2: Information Criteria Minimization across Candidate $(p, q)$
`SutteARIMA` grid-searches candidate AR ($p \in [0, p_{\max}]$) and MA ($q \in [0, q_{\max}]$) orders and computes small-sample corrected criteria:

$$\text{AICc} = -2\ln(\hat{L}) + 2k + \frac{2k(k+1)}{n - k - 1}$$

- **Why AICc over standard AIC?** For finite sample sizes ($n < 200$) common in macroeconomic and financial series, standard AIC has a negative bias and tends to overfit (selecting excess lags). AICc adds a finite-sample penalty that protects against over-parameterization.
- **Alternative Information Criteria available**:
  - **AIC (Akaike)**: Minimizes one-step-ahead forecast error variance.
  - **BIC / SBC (Schwarz-Bayesian)**: Penalizes parameters with $k \ln(n)$, strongly favoring parsimony and consistent order identification.
  - **HQIC (Hannan-Quinn)**: Moderate penalty $2k \ln(\ln(n))$.

You can specify the selection criterion directly:
```python
from sutte_arima import find_order

# Select order minimizing BIC or AICc
order_bic = find_order(data, ic="BIC", verbose=True)
order_aicc = find_order(data, ic="AICc", verbose=True)
```

### Step 3: Coefficient Significance & Principle of Parsimony ($P > |z| < 0.05$)
Econometric best practice requires that every estimated coefficient be statistically significant:
$$\text{Reject } H_0: \beta_j = 0 \quad \text{if } P > |z| < 0.05$$
Parameters with $p \ge 0.05$ introduce noise and increase forecast variance, and should be eliminated.

### Step 4: Residual Diagnostic Validation
A valid econometric model must have residuals $\hat{\varepsilon}_t$ that satisfy:
1. **No Serial Correlation (White Noise)**: Verified by **Ljung-Box $Q$-test** ($p > 0.05$) and **Durbin-Watson statistic** ($DW \approx 2.0$).
2. **Normality**: Tested via **Jarque-Bera ($JB$) test** ($p > 0.05$, Skewness $\approx 0$, Kurtosis $\approx 3$).

---

## Case Study: Photo Analysis & Order Decision

Below is an econometric audit of the empirical estimation table from the classic **Airline Passenger numbers (`#Passengers`, $n=132$)**:

```text
Statespace Model Results: SARIMAX(0, 1, 1)x(2, 1, 1, 12)
Dep. Variable: #Passengers | Observations: 132 | Log-Likelihood: -443.013
AIC: 896.026 | BIC: 909.922 | HQIC: 901.669
------------------------------------------------------------------------------
                 coef    std err          z      P>|z|      [0.025      0.975]
ma.L1         -0.2983      0.077     -3.870      0.000      -0.449      -0.147
ar.S.L12       0.7097      0.236      3.005      0.003       0.247       1.173
ar.S.L24       0.2895      0.100      2.884      0.004       0.093       0.486
ma.S.L12      -0.9803      2.266     -0.433      0.665      -5.422       3.461
sigma2        88.2925    180.492      0.489      0.625    -265.465     442.050
------------------------------------------------------------------------------
Ljung-Box (Q): 38.27 (p=0.55) | Jarque-Bera (JB): 0.00 (p=1.00)
Heteroskedasticity (H): 1.62 (p=0.13) | Skew: -0.00 | Kurtosis: 2.99
```

### Econometric Analysis: Which Order to Use?

1. **Non-Seasonal Component $\to$ Order `(0, 1, 1)`**:
   - The parameter `ma.L1` has coefficient $-0.2983$, $z = -3.870$, and **$P > |z| = 0.0000$**. It is **statistically significant** at $p < 0.001$.
   - The 95% confidence interval $[-0.449, -0.147]$ excludes zero.
   - Non-seasonal AR order is 0 ($p = 0$), differencing is 1 ($d = 1$), and MA order is 1 ($q = 1$).
   - **Conclusion**: **`order = (0, 1, 1)` is the exact order to use!**

2. **Seasonal Insignificance & Parsimony (`ma.S.L12`)**:
   - `ar.S.L12` ($p = 0.003$) and `ar.S.L24` ($p = 0.004$) are significant.
   - However, **`ma.S.L12` has $P > |z| = 0.665$** ($p \gg 0.05$) with a standard error ($2.266$) larger than the coefficient itself, and a 95% CI $[-5.42, 3.46]$ that straddles zero.
   - In econometrics, keeping statistically insignificant parameters violates parsimony and causes numerical instability (evidenced by the inflated standard error of $\sigma^2 = 180.49$).
   - If using seasonal terms, the parsimonious model is `seasonal_order=(2, 1, 0, 12)`.
   - In SutteARIMA, the **$\alpha$-Sutte indicator** dynamically captures trend and momentum without requiring fragile seasonal moving-average estimations.

### Using This in Code:

```python
from sutte_arima import SutteARIMA

# Option A: Core SutteARIMA with the identified significant order (0, 1, 1)
model = SutteARIMA(data, order=(0, 1, 1))
res = model.fit()
print(res.summary())

# Option B: SutteARIMA with parsimonious seasonal order
model_seasonal = SutteARIMA(data, order=(0, 1, 1), seasonal_order=(2, 1, 0, 12))
res_seasonal = model_seasonal.fit()
```

---

## Multi-Step Forecasting with Prediction Intervals

Generate out-of-sample forecasts with calibrated $95\%$ confidence bounds:

```python
forecast_df = res.forecast(steps=6, alpha=0.05)
print(forecast_df)
```

| Horizon | SutteARIMA | ARIMA | AlphaSutte | Lower_95 | Upper_95 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 407.36 | 426.17 | 388.55 | 342.11 | 472.61 |
| 2 | 405.18 | 426.17 | 384.19 | 294.35 | 516.02 |
| 3 | 409.34 | 426.17 | 392.52 | 266.83 | 551.86 |
| 4 | 407.34 | 426.17 | 388.51 | 239.00 | 575.68 |
| 5 | 407.35 | 426.17 | 388.54 | 216.66 | 598.05 |
| 6 | 407.36 | 426.17 | 388.55 | 196.44 | 618.27 |

---

## 1-Line Visual Diagnostics

Econometricians verify fits visually. If `matplotlib` is installed:

```python
# Plot Actual vs. Fitted and 12-step Forecast with 95% Confidence Interval
res.plot(steps=12)
```

---

## Direct Code Attributes

Quickly access key metrics for reports, tables, or Monte Carlo loops:

```python
res.aic             # Akaike Information Criterion
res.aicc            # Small-sample corrected AIC
res.bic             # Bayesian Information Criterion
res.hqic            # Hannan-Quinn Information Criterion
res.llf             # Log-Likelihood
res.params          # Estimated parameter series (with names: ma.L1, sigma2, etc.)
res.pvalues         # Coefficient p-values
res.durbin_watson   # Durbin-Watson statistic
res.ljung_box       # Ljung-Box test dict: {'statistic', 'p_value', 'lags'}
res.jarque_bera     # Jarque-Bera test dict: {'statistic', 'p_value', 'skew', 'kurtosis'}
res.mape            # In-sample MAPE (%)
res.rmse            # In-sample RMSE
res.fittedvalues    # Model fitted values
res.resid           # In-sample residuals
```

---

## Out-of-Sample Evaluation Matrix

Evaluate out-of-sample forecast accuracy against a test holdout:

```python
train_data = data[:-12]
test_data  = data[-12:]

model = SutteARIMA(train_data, order=(0, 1, 1))
res = model.fit()

# Evaluates 12-step forecast against actual test holdout
print(res.metrics_table(test_data))
```

### Metrics Included:
- **MAPE (%)**: Mean Absolute Percentage Error (benchmark metric in Sutte literature)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **sMAPE (%)**: Symmetric Mean Absolute Percentage Error
- **TIC**: Theil's Inequality Coefficient ($0 = \text{perfect forecast}$)
- **MASE**: Mean Absolute Scaled Error
- **MaxAE**: Maximum Absolute Error
- **MBE**: Mean Bias Error

---

## References

1. Ahmar, A. S. (2017). "A Comparison of the α-Sutte Indicator and Moving Average (MA) Method in Forecasting Stock Prices." *International Journal of Advanced Science and Technology*.
2. Ahmar, A. S., et al. (2019). "SutteARIMA: Short-Term Forecasting Method." *Journal of Physics: Conference Series*.
3. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.
4. Hurvich, C. M., & Tsai, C.-L. (1989). "Regression and time series model selection in small samples." *Biometrika*, 76(2), 297–307.
