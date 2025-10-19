"""
Utilities for working with Nepali calendar dates in a time-series friendly manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

NEPALI_MONTHS: Sequence[str] = (
    "Baisakh",
    "Jestha",
    "Ashad",
    "Shrawan",
    "Bhadra",
    "Ashoj",
    "Kartik",
    "Mansir",
    "Poush",
    "Magh",
    "Falgun",
    "Chaitra",
)


@dataclass(frozen=True)
class NepaliDate:
    """Simple representation of a Nepali calendar date."""

    year: int
    month: str
    day: int

    def as_key(self) -> Tuple[int, str]:
        return self.year, self.month

    def as_string(self) -> str:
        return f"{self.year}-{self.month}-{self.day:02d}"


class NepaliCalendarIndexer:
    """
    Helper that mimics calendar arithmetic for the Nepali calendar leveraging the historical data.

    The pipeline relies on this helper to increment dates when producing forward-looking forecasts.
    We derive month lengths from the observed dataset. When we encounter unseen (year, month) pairs,
    we fall back to the maximum observed length for that month across all years. If no prior data is
    available, a conservative default of 32 days is used.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot initialise NepaliCalendarIndexer with an empty dataframe.")

        self.months: List[str] = list(NEPALI_MONTHS)
        self.month_to_ordinal: Dict[str, int] = {month: idx + 1 for idx, month in enumerate(self.months)}

        # Maximum day per (year, month) pair as observed.
        grouped = df.groupby(["year", "month"])["day"].max()
        self._year_month_max: Dict[Tuple[int, str], int] = grouped.to_dict()

        # Global maximum day per month across all years.
        self._month_max: Dict[str, int] = df.groupby("month")["day"].max().to_dict()

        # Precompute safe default for unseen combinations.
        self._default_max_day: int = int(max(self._month_max.values(), default=32))

    def month_name_to_ordinal(self, month_name: str) -> int:
        try:
            return self.month_to_ordinal[month_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Nepali month name: {month_name}") from exc

    def ordinal_to_month_name(self, month_ordinal: int) -> str:
        index = (month_ordinal - 1) % len(self.months)
        return self.months[index]

    def get_month_length(self, year: int, month: str) -> int:
        year_specific = self._year_month_max.get((year, month))
        month_wide = self._month_max.get(month, self._default_max_day)
        if year_specific is None:
            return month_wide
        return max(year_specific, month_wide)

    def next_day(self, date: NepaliDate) -> NepaliDate:
        month_length = self.get_month_length(date.year, date.month)
        if date.day < month_length:
            return NepaliDate(year=date.year, month=date.month, day=date.day + 1)

        # Advance to the first day of the next month.
        ordinal = self.month_to_ordinal.get(date.month)
        if ordinal is None:
            raise ValueError(f"Unknown month name encountered: {date.month}")
        next_index = ordinal % len(self.months)
        next_month = self.months[next_index]
        next_year = date.year + 1 if next_index == 0 else date.year
        return NepaliDate(year=next_year, month=next_month, day=1)

    def advance(self, start: NepaliDate, steps: int) -> List[NepaliDate]:
        """Return a list of NepaliDate stepping forward from ``start`` (exclusive)."""
        cursor = start
        result: List[NepaliDate] = []
        for _ in range(steps):
            cursor = self.next_day(cursor)
            result.append(cursor)
        return result

    def build_base_features(self, dates: Iterable[NepaliDate]) -> pd.DataFrame:
        """
        Construct deterministic calendar features from the provided iterable of dates.
        The resulting frame includes both polynomial and harmonic representations of seasonality.
        """
        records = []
        for idx, date in enumerate(dates):
            month_ord = self.month_name_to_ordinal(date.month)
            record = {
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "month_ordinal": month_ord,
                "relative_index": idx,
            }
            records.append(record)

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return frame

        # Derive polynomial trend terms.
        frame["trend_linear"] = frame["relative_index"]
        frame["trend_quadratic"] = frame["relative_index"] ** 2

        # Encode cyclical components for month and day within the month.
        month_cycle = 12.0
        day_cycle = frame["day"].max() if frame["day"].max() else 32.0
        frame["month_sin"] = np.sin(2 * np.pi * frame["month_ordinal"] / month_cycle)
        frame["month_cos"] = np.cos(2 * np.pi * frame["month_ordinal"] / month_cycle)
        frame["day_sin"] = np.sin(2 * np.pi * frame["day"] / day_cycle)
        frame["day_cos"] = np.cos(2 * np.pi * frame["day"] / day_cycle)

        return frame


def compute_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a monotonically increasing ``date_id`` column preserving chronological order.
    """
    month_order = {name: idx for idx, name in enumerate(NEPALI_MONTHS)}

    def sort_key(series: pd.Series) -> pd.Series:
        if series.name == "month":
            return series.map(month_order)
        return series

    ordered = df.sort_values(["year", "month", "day"], key=sort_key)
    ordered = ordered.reset_index(drop=True)
    ordered["date_id"] = ordered.index.astype(int)
    return ordered
