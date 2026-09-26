"""
SutteARIMA: Statsmodels-style Hybrid Time Series Forecasting Library
"""

from .model import SutteARIMA, SutteARIMAResults, AlphaSutte, calculate_evaluation_metrics

__version__ = "1.0.0"
__all__ = ["SutteARIMA", "SutteARIMAResults", "AlphaSutte", "calculate_evaluation_metrics"]
