# CMC-AE-decode

This repository provides a minimal example of building a machine-learning pipeline to
predict the next-day directional movement of the SPDR S&P 500 ETF Trust (ticker: SPY).

## Getting Started

1. (Optional) Create a virtual environment.

2. Run the training and prediction script:

   ```bash
   python scripts/predict_spy.py --start 2015-01-01
   ```

   The script will download historical SPY data from Stooq, engineer a set of
   technical features, evaluate a lightweight logistic regression model using walk-forward
   validation, and finally report the probability that SPY closes higher on the next trading
   session.

## Library Usage

The core functionality is exposed through the `spy_predictor` package and only relies on
Python's standard library:

- `download_spy_data` downloads OHLCV data from Stooq.
- `compute_features` engineers technical indicators and creates a direction label.
- `build_training_dataset` prepares the features and target arrays.
- `evaluate_model` performs walk-forward validation.
- `train_directional_model` trains a logistic regression classifier.
- `predict_next_direction` outputs the probability of an upward move for the next day.

You can compose these utilities to experiment with alternative models or feature sets.

## Testing

Unit tests can be run with `pytest`:

```bash
pytest
```
