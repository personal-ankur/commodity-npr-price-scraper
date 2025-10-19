# flake8: noqa
"""
Analyst package housing the machine learning pipeline for price trend forecasting.

This package provides utilities to ingest the historical commodity pricing dataset,
engineer predictive features, train and evaluate forecasting models, and generate
30-day ahead projections for the supported commodities.
"""

from .pipeline import PriceForecastPipeline, ForecastResult  # noqa: F401

