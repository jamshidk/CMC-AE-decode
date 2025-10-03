#!/usr/bin/env python3
"""Command line utility to train and evaluate a SPY directional model."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from spy_predictor import (
    build_training_dataset,
    compute_features,
    download_spy_data,
    evaluate_model,
    predict_next_direction,
    train_directional_model,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01", help="Start date for data download")
    parser.add_argument("--end", default=None, help="End date for data download")
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of walk-forward splits to use during evaluation",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    raw = download_spy_data(start=args.start, end=args.end)
    features = compute_features(raw)
    if not features:
        raise SystemExit("Not enough data returned to compute features.")

    X, y = build_training_dataset(features)

    evaluation = evaluate_model(X, y, splits=args.splits)
    print("Walk-forward evaluation:")
    for metric, value in evaluation.to_dict().items():
        print(f"  {metric}: {value:.3f}")

    pipeline = train_directional_model(X, y)
    latest_features = features[-1].as_vector()
    proba = predict_next_direction(pipeline, latest_features)
    print(
        "\nProbability SPY closes higher on the next session: "
        f"{proba:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
