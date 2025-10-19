"""
Feature engineering components for the price forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .calendar import NepaliCalendarIndexer, NepaliDate


def create_base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct deterministic calendar and trend features from the supplied dataframe.
    """
    if df.empty:
        raise ValueError("Cannot build base feature frame from an empty dataframe.")

    origin = df["date_id"].min()
    features = pd.DataFrame(index=df.index)

    features["date_id"] = df["date_id"]
    features["relative_index"] = df["date_id"] - origin
    features["trend_linear"] = features["relative_index"]
    features["trend_quadratic"] = features["relative_index"] ** 2
    features["trend_cubic"] = features["relative_index"] ** 3

    features["year"] = df["year"]
    features["year_relative"] = df["year"] - df["year"].min()
    features["month_ordinal"] = df["month_ordinal"]
    features["day"] = df["day"]

    month_cycle = 12.0
    day_cycle = 32.0
    features["month_sin"] = np.sin(2 * np.pi * df["month_ordinal"] / month_cycle)
    features["month_cos"] = np.cos(2 * np.pi * df["month_ordinal"] / month_cycle)
    features["day_sin"] = np.sin(2 * np.pi * df["day"] / day_cycle)
    features["day_cos"] = np.cos(2 * np.pi * df["day"] / day_cycle)

    features["week_index"] = (features["relative_index"] // 7).astype(int)
    features["quarter_index"] = (features["month_ordinal"] - 1) // 3

    # Fraction within the year to capture intra-annual seasonality.
    features["year_fraction"] = (df["month_ordinal"] - 1 + (df["day"] - 1) / day_cycle) / month_cycle
    features["year_fraction_sin"] = np.sin(2 * np.pi * features["year_fraction"])
    features["year_fraction_cos"] = np.cos(2 * np.pi * features["year_fraction"])

    return features


def create_future_base_features(
    calendar: NepaliCalendarIndexer,
    last_date: NepaliDate,
    horizon: int,
    start_date_id: int,
    origin_date_id: int = 0,
    origin_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate base features for the next ``horizon`` days following ``last_date``.
    """
    if origin_year is None:
        origin_year = last_date.year

    future_dates = calendar.advance(last_date, horizon)
    records: List[Dict[str, float]] = []
    for offset, date in enumerate(future_dates, start=1):
        date_id = start_date_id + offset
        relative_index = date_id - origin_date_id
        month_ord = calendar.month_name_to_ordinal(date.month)
        year_fraction = (month_ord - 1 + (date.day - 1) / 32.0) / 12.0

        record = {
            "date_id": date_id,
            "relative_index": relative_index,
            "trend_linear": relative_index,
            "trend_quadratic": relative_index ** 2,
            "trend_cubic": relative_index ** 3,
            "year": date.year,
            "year_relative": date.year - origin_year,
            "month_ordinal": month_ord,
            "day": date.day,
            "month_sin": np.sin(2 * np.pi * month_ord / 12.0),
            "month_cos": np.cos(2 * np.pi * month_ord / 12.0),
            "day_sin": np.sin(2 * np.pi * date.day / 32.0),
            "day_cos": np.cos(2 * np.pi * date.day / 32.0),
            "week_index": (relative_index // 7),
            "quarter_index": (month_ord - 1) // 3,
            "year_fraction": year_fraction,
            "year_fraction_sin": np.sin(2 * np.pi * year_fraction),
            "year_fraction_cos": np.cos(2 * np.pi * year_fraction),
            "year_future_indicator": int(date.year > last_date.year),
        }
        records.append(record)

    return pd.DataFrame.from_records(records, index=None)


@dataclass
class FeatureBuilderConfig:
    target_column: str
    lags: Sequence[int]
    rolling_windows: Sequence[int]
    horizon: int
    fourier_order: int = 3
    seasonal_period: int = 365


class FeatureBuilder:
    """
    Constructs lagged, rolling, and harmonic features for a particular target column.
    """

    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config
        self.feature_columns_: Optional[List[str]] = None
        self.required_history_: int = max(
            max(config.lags, default=1), max(config.rolling_windows, default=1)
        )

    @property
    def target(self) -> str:
        return self.config.target_column

    def build_training_matrix(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = self._assemble_feature_frame(df, base_features)
        features = features.dropna()
        X = features.drop(columns=[self.target])
        y = features[self.target]
        self.feature_columns_ = X.columns.tolist()
        return X, y

    def forecast(
        self,
        df: pd.DataFrame,
        base_features: pd.DataFrame,
        model,
        calendar: NepaliCalendarIndexer,
        horizon: int,
    ) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise RuntimeError("FeatureBuilder must be fitted before forecasting.")

        start_date_id = int(df["date_id"].iloc[-1])
        origin_date_id = int(df["date_id"].iloc[0])
        origin_year = int(df["year"].min())
        last_row = df.iloc[-1]
        last_date = NepaliDate(
            year=int(last_row["year"]),
            month=str(last_row["month"]),
            day=int(last_row["day"]),
        )

        future_base = create_future_base_features(
            calendar=calendar,
            last_date=last_date,
            horizon=horizon,
            start_date_id=start_date_id,
            origin_date_id=origin_date_id,
            origin_year=origin_year,
        )

        # Maintain a working history series for recursive forecasting.
        history_series = df[self.target].copy()
        required_length = self.required_history_ + 1
        if len(history_series) < required_length:
            raise ValueError(
                f"Insufficient history ({len(history_series)}) for target {self.target}; "
                f"need at least {required_length} observations."
            )

        forecast_records: List[Dict[str, float]] = []
        future_dates = calendar.advance(last_date, horizon)
        for step in range(horizon):
            base_row = future_base.iloc[step]
            feature_row = self._assemble_single_row(
                base_row=base_row,
                history_series=history_series,
            )
            ordered_row = [feature_row[col] for col in self.feature_columns_]
            X_row = np.asarray(ordered_row, dtype=float).reshape(1, -1)
            y_hat = float(model.predict(X_row)[0])

            predicted_date = future_dates[step]
            record = {
                "forecast_date": predicted_date.as_string(),
                "year": predicted_date.year,
                "month": predicted_date.month,
                "day": predicted_date.day,
                "date_id": int(base_row["date_id"]),
                f"{self.target}_prediction": y_hat,
            }
            forecast_records.append(record)

            # Append prediction for recursive lags and rolling statistics.
            history_series = pd.concat(
                [history_series, pd.Series([y_hat], index=[history_series.index[-1] + 1])]
            )

        return pd.DataFrame(forecast_records)

    # Internal helpers -----------------------------------------------------------------

    def _assemble_feature_frame(self, df: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        target_series = df[self.target]
        composed = base_features.copy()
        composed = composed.join(self._lag_features(target_series))
        composed = composed.join(self._rolling_features(target_series))
        composed = composed.join(self._volatility_features(target_series))
        composed = composed.join(self._fourier_features(base_features))
        composed[self.target] = target_series
        return composed

    def _lag_features(self, series: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(index=series.index)
        for lag in self.config.lags:
            frame[f"{self.target}_lag_{lag}"] = series.shift(lag)
        return frame

    def _rolling_features(self, series: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(index=series.index)
        shifted = series.shift(1)
        for window in self.config.rolling_windows:
            rolling = shifted.rolling(window=window, min_periods=window)
            frame[f"{self.target}_roll_mean_{window}"] = rolling.mean()
            frame[f"{self.target}_roll_std_{window}"] = rolling.std()
            frame[f"{self.target}_roll_min_{window}"] = rolling.min()
            frame[f"{self.target}_roll_max_{window}"] = rolling.max()
        return frame

    def _volatility_features(self, series: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(index=series.index)
        diff_1 = series.diff().shift(1)
        diff_7 = series.diff(7).shift(1)
        pct_1 = series.pct_change().shift(1)
        pct_7 = series.pct_change(periods=7).shift(1)

        frame[f"{self.target}_diff_1"] = diff_1
        frame[f"{self.target}_diff_7"] = diff_7
        frame[f"{self.target}_pct_change_1"] = pct_1
        frame[f"{self.target}_pct_change_7"] = pct_7

        rolling_volatility = series.pct_change().rolling(window=14, min_periods=10).std().shift(1)
        frame[f"{self.target}_volatility_14"] = rolling_volatility
        frame[f"{self.target}_log_lag_1"] = np.log1p(series.shift(1))
        frame[f"{self.target}_log_lag_7"] = np.log1p(series.shift(7))
        return frame

    def _fourier_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=base_features.index)
        date_ids = base_features["date_id"]
        for order in range(1, self.config.fourier_order + 1):
            angle = 2 * np.pi * order * date_ids / self.config.seasonal_period
            frame[f"fourier_sin_{order}"] = np.sin(angle)
            frame[f"fourier_cos_{order}"] = np.cos(angle)
        return frame

    def _assemble_single_row(self, base_row: pd.Series, history_series: pd.Series) -> Dict[str, float]:
        row = base_row.to_dict()
        row.update(self._single_lag_features(history_series))
        row.update(self._single_rolling_features(history_series))
        row.update(self._single_volatility_features(history_series))
        row.update(self._single_fourier_features(base_row["date_id"]))
        return row

    def _single_lag_features(self, history_series: pd.Series) -> Dict[str, float]:
        data: Dict[str, float] = {}
        series_length = len(history_series)
        for lag in self.config.lags:
            if series_length < lag:
                raise ValueError(
                    f"Insufficient history length ({series_length}) for lag {lag} while forecasting {self.target}."
                )
            data[f"{self.target}_lag_{lag}"] = float(history_series.iloc[-lag])
        return data

    def _single_rolling_features(self, history_series: pd.Series) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for window in self.config.rolling_windows:
            if len(history_series) < window:
                window_slice = history_series.values
            else:
                window_slice = history_series.iloc[-window:].values
            data[f"{self.target}_roll_mean_{window}"] = float(np.mean(window_slice))
            if window > 1 and len(window_slice) > 1:
                data[f"{self.target}_roll_std_{window}"] = float(np.std(window_slice, ddof=1))
            else:
                data[f"{self.target}_roll_std_{window}"] = 0.0
            data[f"{self.target}_roll_min_{window}"] = float(np.min(window_slice))
            data[f"{self.target}_roll_max_{window}"] = float(np.max(window_slice))
        return data

    def _single_volatility_features(self, history_series: pd.Series) -> Dict[str, float]:
        data: Dict[str, float] = {}
        diff_series = history_series.diff()
        data[f"{self.target}_diff_1"] = float(diff_series.iloc[-1])
        if len(history_series) > 7:
            data[f"{self.target}_diff_7"] = float(history_series.iloc[-1] - history_series.iloc[-8])
        else:
            data[f"{self.target}_diff_7"] = float("nan")

        pct_change = history_series.pct_change()
        data[f"{self.target}_pct_change_1"] = float(pct_change.iloc[-1])
        if len(history_series) > 7:
            data[f"{self.target}_pct_change_7"] = float(history_series.iloc[-1] / history_series.iloc[-8] - 1)
        else:
            data[f"{self.target}_pct_change_7"] = float("nan")

        volatility_window = history_series.pct_change().iloc[-14:]
        data[f"{self.target}_volatility_14"] = float(volatility_window.std())
        data[f"{self.target}_log_lag_1"] = float(np.log1p(history_series.iloc[-1]))
        if len(history_series) > 7:
            data[f"{self.target}_log_lag_7"] = float(np.log1p(history_series.iloc[-7]))
        else:
            data[f"{self.target}_log_lag_7"] = float("nan")
        return data

    def _single_fourier_features(self, date_id: float) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for order in range(1, self.config.fourier_order + 1):
            angle = 2 * np.pi * order * date_id / self.config.seasonal_period
            data[f"fourier_sin_{order}"] = float(np.sin(angle))
            data[f"fourier_cos_{order}"] = float(np.cos(angle))
        return data
