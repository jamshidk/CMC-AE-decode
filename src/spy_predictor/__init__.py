"""Utilities for downloading data and predicting SPY movements."""

from .data import PriceBar, download_spy_data
from .features import FeatureRow, compute_features
from .model import (
    build_training_dataset,
    evaluate_model,
    train_directional_model,
    predict_next_direction,
)

__all__ = [
    "PriceBar",
    "FeatureRow",
    "download_spy_data",
    "compute_features",
    "build_training_dataset",
    "evaluate_model",
    "train_directional_model",
    "predict_next_direction",
]
