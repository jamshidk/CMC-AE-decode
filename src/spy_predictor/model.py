"""Simple logistic regression implementation for SPY direction prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .features import FeatureRow


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass
class StandardScaler:
    means: List[float]
    stds: List[float]

    @classmethod
    def fit(cls, X: Sequence[Sequence[float]]) -> "StandardScaler":
        columns = len(X[0])
        means = [0.0] * columns
        stds = [0.0] * columns
        for j in range(columns):
            column = [row[j] for row in X]
            means[j] = sum(column) / len(column)
            variance = sum((value - means[j]) ** 2 for value in column) / max(len(column) - 1, 1)
            stds[j] = math.sqrt(variance) if variance > 0 else 1.0
        return cls(means, stds)

    def transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        transformed: List[List[float]] = []
        for row in X:
            transformed.append(
                [
                    (value - mean) / std
                    for value, mean, std in zip(row, self.means, self.stds)
                ]
            )
        return transformed

    def transform_row(self, row: Sequence[float]) -> List[float]:
        return [
            (value - mean) / std
            for value, mean, std in zip(row, self.means, self.stds)
        ]


@dataclass
class LogisticRegressionModel:
    weights: List[float]
    bias: float

    @classmethod
    def train(
        cls,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        learning_rate: float = 0.05,
        epochs: int = 500,
    ) -> "LogisticRegressionModel":
        features = len(X[0])
        weights = [0.0] * features
        bias = 0.0

        for _ in range(epochs):
            grad_w = [0.0] * features
            grad_b = 0.0
            for row, target in zip(X, y):
                z = sum(w * x for w, x in zip(weights, row)) + bias
                if z >= 0:
                    pred = 1 / (1 + math.exp(-z))
                else:
                    exp_z = math.exp(z)
                    pred = exp_z / (1 + exp_z)
                error = pred - target
                for j in range(features):
                    grad_w[j] += error * row[j]
                grad_b += error

            for j in range(features):
                weights[j] -= learning_rate * grad_w[j] / len(X)
            bias -= learning_rate * grad_b / len(X)

        return cls(weights=weights, bias=bias)

    def predict_proba(self, row: Sequence[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, row)) + self.bias
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)

    def predict(self, row: Sequence[float]) -> int:
        return 1 if self.predict_proba(row) >= 0.5 else 0


@dataclass
class DirectionalModel:
    scaler: StandardScaler
    model: LogisticRegressionModel

    def predict_proba(self, row: Sequence[float]) -> float:
        scaled = self.scaler.transform_row(row)
        return self.model.predict_proba(scaled)

    def predict(self, row: Sequence[float]) -> int:
        scaled = self.scaler.transform_row(row)
        return self.model.predict(scaled)


def build_training_dataset(features: Iterable[FeatureRow]) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []
    for row in features:
        X.append(row.as_vector())
        y.append(row.target)
    return X, y


def _time_series_splits(length: int, splits: int) -> Iterable[Tuple[range, range]]:
    fold_size = length // (splits + 1)
    for i in range(splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        if test_end > length:
            test_end = length
        if train_end == 0 or train_end == test_end:
            continue
        yield range(0, train_end), range(train_end, test_end)


def evaluate_model(X: Sequence[Sequence[float]], y: Sequence[int], splits: int = 5) -> EvaluationResult:
    accuracies: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for train_idx, test_idx in _time_series_splits(len(X), splits):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        if not X_test:
            continue

        scaler = StandardScaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegressionModel.train(X_train_scaled, y_train)

        predictions = [model.predict(row) for row in X_test_scaled]

        tp = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 1)
        tn = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 0)
        fp = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 1)

        total = len(y_test)
        accuracies.append((tp + tn) / total if total else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)

    return EvaluationResult(
        accuracy=sum(accuracies) / len(accuracies) if accuracies else 0.0,
        precision=sum(precisions) / len(precisions) if precisions else 0.0,
        recall=sum(recalls) / len(recalls) if recalls else 0.0,
    )


def train_directional_model(X: Sequence[Sequence[float]], y: Sequence[int]) -> DirectionalModel:
    scaler = StandardScaler.fit(X)
    X_scaled = scaler.transform(X)
    model = LogisticRegressionModel.train(X_scaled, y)
    return DirectionalModel(scaler=scaler, model=model)


def predict_next_direction(model: DirectionalModel, latest_row: Sequence[float]) -> float:
    return model.predict_proba(latest_row)
