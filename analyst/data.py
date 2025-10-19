"""
Data ingestion and cleansing utilities for the pricing dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .calendar import NepaliCalendarIndexer, NepaliDate, compute_date_index

PRICE_COLUMNS = ("fine_gold", "standard_gold", "silver")


@dataclass
class PriceDataBundle:
    """Container for the cleaned dataframe and auxiliary calendar indexer."""

    frame: pd.DataFrame
    calendar: NepaliCalendarIndexer


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV path does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def clean_price_data(df: pd.DataFrame) -> PriceDataBundle:
    _validate_columns(df, ["date", "year", "month", "day", *PRICE_COLUMNS])

    working = df.copy()
    # Ensure correct dtypes.
    working["year"] = working["year"].astype(int)
    working["day"] = working["day"].astype(int)
    for column in PRICE_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    calendar = NepaliCalendarIndexer(working)
    working = compute_date_index(working)
    working["month_ordinal"] = working["month"].map(calendar.month_to_ordinal)

    # Flag missing reports before imputing.
    working["standard_gold_reported"] = (working["standard_gold"] > 0).astype(int)
    working.loc[working["standard_gold"] == 0, "standard_gold"] = np.nan

    # Interpolate missing values conservatively and back/forward fill residuals.
    working["standard_gold"] = working["standard_gold"].interpolate(method="linear", limit_direction="both")

    # Replace any improbable artifacts (e.g., negative prices) with NaN then interpolate.
    for col in PRICE_COLUMNS:
        working.loc[working[col] <= 0, col] = np.nan
        working[col] = working[col].interpolate(method="linear", limit_direction="both")
        working[col] = working[col].ffill().bfill()

    # Add aggregated statistics helpful for downstream modelling.
    working["price_spread_gold"] = working["fine_gold"] - working["standard_gold"]
    working["gold_silver_ratio"] = working["fine_gold"] / working["silver"]
    working["fine_gold_log"] = np.log1p(working["fine_gold"])
    working["silver_log"] = np.log1p(working["silver"])
    working["standard_gold_log"] = np.log1p(working["standard_gold"])

    return PriceDataBundle(frame=working, calendar=calendar)


def load_price_data(csv_path: Path) -> PriceDataBundle:
    raw = load_raw_data(csv_path)
    return clean_price_data(raw)

