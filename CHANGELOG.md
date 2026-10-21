# Changelog

## 2.0.0

### Added
- `transform="log"` in `SutteARIMA` and `find_order`: the ARIMA component is estimated on `ln(y)` and back-transformed (predictive median); α-Sutte stays on the original scale.
- `trend=` (statsmodels specification) with the `"drift"` alias, which adds the trend term of power `d + D` so that the differenced series has a constant.
- `exog=`: exogenous regressors as a regression with ARIMA errors; future values are required by `forecast`, `rolling_forecast` and `evaluate`. Semantics (pulse / step / window / ramp) and the look-ahead rule are documented.
- `weight=`: combination weight on the ARIMA component, fixed in [0, 1] (default 0.5) or `"optimal"` (Bates-Granger least-squares weight from in-sample one-step errors, relative errors under the log transform).
- `SutteARIMAResults.rolling_forecast()`: expanding-window one-step-ahead forecasts without look-ahead, optional re-estimation every `refit_every` observations.
- `SutteARIMAResults.evaluate()`: accuracy of hybrid, ARIMA, α-Sutte and naive forecasts with Theil's U2, Diebold-Mariano (squared and absolute loss) and Pesaran-Timmermann tests.
- `diebold_mariano()`, `pesaran_timmermann()`, `theil_u2()`; `calculate_metrics(..., benchmark=)` returns `U2`.
- `Naive` column in `forecast()`; `arima_fittedvalues`, `sutte_fittedvalues`, `resid_model`, `weight`, `u2`, `arch_lm` on the results object; ARCH-LM line in `summary()`.
- `tests/` (pytest), including a literal translation of the reference R function `sutteForecastR::alpha.sutte`.

### Changed
- `forecast()` prediction intervals now come from the ARIMA predictive distribution (`interval="arima"`). The 1.0 heuristic is available as `interval="legacy"`.
- `find_order()` and `SutteARIMAResults.aicc` use statsmodels' information criteria, which count every estimated parameter (including `sigma2`); 1.0 used `k = p + q + 1`. Selected orders can differ in borderline cases.
- The first `d + D*s` ARIMA fitted values (diffuse initialisation) are reported as `NaN` instead of 0.
- Residual diagnostics are computed on the modelling scale.
- `metrics(test)` also reports `U2` against the naive multi-step forecast.

### Fixed
- Importing the package no longer installs global warning filters (`warnings.filterwarnings("ignore")` affected the whole user session).
- `find_order(..., ic="aicc", verbose=True)` (lower-case criterion) raised a `KeyError`.

### Migration from 1.0
- Point forecasts and fitted values with default arguments are unchanged (`weight=0.5`, no transform, no trend).
- To reproduce 1.0 prediction intervals pass `interval="legacy"`.
- New arguments after `seasonal_order` are keyword-only.
