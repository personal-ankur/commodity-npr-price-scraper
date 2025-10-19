"""
High-level orchestration for the price trend forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import PRICE_COLUMNS, load_price_data
from .features import FeatureBuilder, FeatureBuilderConfig, create_base_feature_frame
from .models import ModelArtifact, ModelSelector, default_candidates


@dataclass
class ForecastResult:
    forecast_frame: pd.DataFrame
    trend_summary: pd.DataFrame
    diagnostics: Dict[str, ModelArtifact]

    def save(self, forecast_path: Path, summary_path: Path) -> None:
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.forecast_frame.to_csv(forecast_path, index=False)
        self.trend_summary.to_csv(summary_path, index=False)


class PriceForecastPipeline:
    """
    End-to-end pipeline that reads historical price data, engineers features,
    selects and trains models, and produces 30-day forecasts for each commodity.
    """

    def __init__(
        self,
        data_path: Path = Path("data/prices.csv"),
        horizon: int = 30,
        output_dir: Optional[Path] = Path("out"),
        random_state: int = 42,
        mode: str = "robust",
        training_window: Optional[int] = None,
        prediction_margin: float = 0.02,
        verbose: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.horizon = horizon
        self.output_dir = Path(output_dir) if output_dir else None
        self.random_state = random_state
        self.mode = mode
        self.verbose = verbose
        self.training_window = training_window
        self.prediction_margin = prediction_margin

        if self.mode not in {"robust", "fast"}:
            raise ValueError("mode must be either 'robust' or 'fast'")
        if self.training_window is not None and self.training_window <= 0:
            raise ValueError("training_window must be a positive integer when provided.")
        if self.prediction_margin < 0:
            raise ValueError("prediction_margin must be non-negative.")

    def run(self) -> ForecastResult:
        if self.verbose:
            print(f"[pipeline] Loading data from {self.data_path} ...", flush=True)

        data_bundle = load_price_data(self.data_path)
        calendar = data_bundle.calendar
        frame = data_bundle.frame.sort_values("date_id")
        if self.training_window:
            if self.verbose:
                print(f"[pipeline] Restricting training data to the last {self.training_window} days.", flush=True)
            frame = frame.tail(self.training_window).reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)

        if self.verbose:
            print("[pipeline] Engineering base feature matrix ...", flush=True)
        base_features = create_base_feature_frame(frame)

        diagnostics: Dict[str, ModelArtifact] = {}
        forecast_frames: List[pd.DataFrame] = []

        history_length = len(frame)
        for target in PRICE_COLUMNS:
            if self.verbose:
                print(f"[pipeline] Training models for '{target}' ...", flush=True)
            builder = self._make_feature_builder(target, history_length=history_length)
            X, y = builder.build_training_matrix(frame, base_features)

            n_samples = len(X)
            if n_samples < 3:
                raise ValueError(
                    f"Not enough samples ({n_samples}) to train models for '{target}'. "
                    "Consider increasing the training window or reducing feature lags."
                )

            selector = ModelSelector(
                candidates=default_candidates(random_state=self.random_state, mode=self.mode),
                scoring="neg_mean_absolute_error",
                n_splits=self._determine_cv_splits(n_samples),
                random_state=self.random_state,
            )
            artifact = selector.fit(X.to_numpy(), y.to_numpy(), feature_names=X.columns)
            diagnostics[target] = artifact
            if self.verbose:
                print(f"[pipeline] Selected model '{artifact.name}' for '{target}'.", flush=True)

            forecast_df = builder.forecast(
                df=frame,
                base_features=base_features,
                model=artifact.estimator,
                calendar=calendar,
                horizon=self.horizon,
            )
            forecast_df = self._constrain_forecast(forecast_df, frame, target)
            forecast_frames.append(forecast_df)

        if self.verbose:
            print("[pipeline] Combining forecasts ...", flush=True)
        combined_forecast = self._combine_forecasts(forecast_frames)
        trend_summary = self._summarise_trends(combined_forecast, diagnostics)

        result = ForecastResult(
            forecast_frame=combined_forecast,
            trend_summary=trend_summary,
            diagnostics=diagnostics,
        )

        if self.output_dir:
            forecast_path = self.output_dir / "forecast_next_30_days.csv"
            summary_path = self.output_dir / "forecast_summary.csv"
            result.save(forecast_path=forecast_path, summary_path=summary_path)
            if self.verbose:
                print(f"[pipeline] Results written to {self.output_dir.resolve()}", flush=True)

        return result

    # Internal ------------------------------------------------------------------------

    def _make_feature_builder(self, target: str, history_length: int) -> FeatureBuilder:
        base_lags = [1, 2, 3, 5, 7, 14, 21, 28]
        base_windows = [3, 7, 14, 28]

        min_samples_desired = max(8, self.horizon // 2)
        max_lag_allowed = max(1, history_length - min_samples_desired)
        lags = [lag for lag in base_lags if lag <= max_lag_allowed and lag < history_length]
        if len(lags) < 2:
            lags = base_lags[:2]
            lags = [lag for lag in lags if lag < history_length] or [1]

        max_window_allowed = max(2, history_length - min_samples_desired)
        windows = [window for window in base_windows if window <= max_window_allowed]
        windows = [window for window in windows if window < history_length]
        if len(windows) < 2:
            fallback = [w for w in base_windows if w < history_length]
            windows = fallback[:2] if fallback else [max(2, history_length - 1)]

        config = FeatureBuilderConfig(
            target_column=target,
            lags=tuple(lags),
            rolling_windows=tuple(windows),
            horizon=self.horizon,
            fourier_order=4,
            seasonal_period=365,
        )
        return FeatureBuilder(config)

    def _determine_cv_splits(self, n_samples: int) -> int:
        max_splits = 2 if self.mode == "fast" else 4
        splits = min(max_splits, n_samples - 1)
        return max(splits, 2)

    def _constrain_forecast(self, forecast_df: pd.DataFrame, history_df: pd.DataFrame, target: str) -> pd.DataFrame:
        if self.prediction_margin == 0:
            return forecast_df

        recent = history_df[target].dropna()
        if recent.empty:
            return forecast_df

        margin = self.prediction_margin
        lower = float(recent.min() * (1 - margin))
        upper = float(recent.max() * (1 + margin))
        column = f"{target}_prediction"
        forecast_df[column] = forecast_df[column].clip(lower=lower, upper=upper)

        if self.verbose:
            print(
                f"[pipeline] Applied bounds for '{target}' within [{lower:,.2f}, {upper:,.2f}].",
                flush=True,
            )
        return forecast_df

    def _combine_forecasts(self, forecast_frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not forecast_frames:
            raise ValueError("No forecast frames to combine.")

        combined = forecast_frames[0].copy()
        combined = combined.set_index(["date_id", "forecast_date", "year", "month", "day"])

        for frame in forecast_frames[1:]:
            indexed = frame.set_index(["date_id", "forecast_date", "year", "month", "day"])
            combined = combined.join(indexed, how="inner")

        combined = combined.reset_index()
        combined = combined.sort_values("date_id").reset_index(drop=True)
        return combined

    def _summarise_trends(self, forecast_df: pd.DataFrame, diagnostics: Dict[str, ModelArtifact]) -> pd.DataFrame:
        summaries: List[Dict[str, float]] = []
        horizon_index = np.arange(1, len(forecast_df) + 1)

        for target in PRICE_COLUMNS:
            column = f"{target}_prediction"
            if column not in forecast_df.columns:
                continue

            series = forecast_df[column].astype(float)

            start_value = float(series.iloc[0])
            end_value = float(series.iloc[-1])
            absolute_change = end_value - start_value
            percent_change = (absolute_change / start_value) if start_value != 0 else np.nan

            slope, intercept = np.polyfit(horizon_index, series, deg=1)
            volatility = float(np.std(series))

            trend_label = self._classify_trend(percent_change)
            mae = diagnostics[target].mean_absolute_error if target in diagnostics else np.nan
            confidence = float(np.exp(-mae / (start_value + 1e-9)))

            summaries.append(
                {
                    "target": target,
                    "starting_price": start_value,
                    "ending_price": end_value,
                    "absolute_change": absolute_change,
                    "percent_change": percent_change,
                    "estimated_slope_per_day": float(slope),
                    "forecast_volatility": volatility,
                    "trend_direction": trend_label,
                    "cv_mean_absolute_error": mae,
                    "confidence_indicator": confidence,
                }
            )

        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df.sort_values("absolute_change", key=np.abs, ascending=False)
        return summary_df

    @staticmethod
    def _classify_trend(percent_change: float) -> str:
        if not np.isfinite(percent_change):
            return "undetermined"
        threshold = 0.02  # 2% change over the horizon counts as material.
        if percent_change > threshold:
            return "upward"
        if percent_change < -threshold:
            return "downward"
        return "stable"
