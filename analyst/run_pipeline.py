"""
Command-line entry point for running the price forecasting pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PriceForecastPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 30-day price trend forecast using historical commodity prices."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/prices.csv"),
        help="Path to the historical prices CSV.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Forecast horizon in days.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="Directory for saving forecast artefacts.",
    )
    parser.add_argument(
        "--mode",
        choices=("robust", "fast"),
        default="robust",
        help="Model search strategy. Use 'fast' for quicker, lighter-weight runs.",
    )
    parser.add_argument(
        "--training-window",
        type=int,
        default=None,
        help="Number of most recent days to use for training/analysis. Use all data when omitted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose pipeline logging.",
    )
    return parser.parse_args()


def format_currency(value: float) -> str:
    return f"NPR {value:,.2f}"


def main() -> None:
    args = parse_args()
    pipeline = PriceForecastPipeline(
        data_path=args.data,
        horizon=args.horizon,
        output_dir=args.output_dir,
        mode=args.mode,
        training_window=args.training_window,
        verbose=not args.quiet,
    )
    result = pipeline.run()

    trend_df = result.trend_summary.copy()
    trend_df["starting_price"] = trend_df["starting_price"].apply(format_currency)
    trend_df["ending_price"] = trend_df["ending_price"].apply(format_currency)
    trend_df["absolute_change"] = trend_df["absolute_change"].apply(format_currency)
    trend_df["percent_change"] = trend_df["percent_change"].apply(lambda x: f"{x * 100:.2f}%")
    trend_df["confidence_indicator"] = trend_df["confidence_indicator"].apply(lambda x: f"{x:.3f}")
    trend_df["cv_mean_absolute_error"] = trend_df["cv_mean_absolute_error"].apply(format_currency)

    print("\n=== 30-Day Forecast Trend Summary ===")
    print(trend_df.to_string(index=False))
    print("\nForecast data saved to:", (args.output_dir / "forecast_next_30_days.csv").resolve())
    print("Summary saved to:", (args.output_dir / "forecast_summary.csv").resolve())


if __name__ == "__main__":
    main()
