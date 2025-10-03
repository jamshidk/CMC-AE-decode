"""Feature engineering without external dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from .data import PriceBar


@dataclass
class FeatureRow:
    """Engineered features for a single trading day."""

    return_1d: float
    log_return_1d: float
    sma_5: float
    sma_21: float
    ema_5: float
    ema_21: float
    rsi_14: float
    volume_change: float
    target: int

    def as_vector(self) -> List[float]:
        return [
            self.return_1d,
            self.log_return_1d,
            self.sma_5,
            self.sma_21,
            self.ema_5,
            self.ema_21,
            self.rsi_14,
            self.volume_change,
        ]


def _sma(values: List[float], window: int) -> List[float]:
    sma: List[float] = []
    for i in range(len(values)):
        if i + 1 < window:
            sma.append(float("nan"))
            continue
        window_values = values[i + 1 - window : i + 1]
        sma.append(sum(window_values) / window)
    return sma


def _ema(values: List[float], span: int) -> List[float]:
    ema: List[float] = []
    alpha = 2 / (span + 1)
    last = None
    for value in values:
        if last is None:
            last = value
        else:
            last = alpha * value + (1 - alpha) * last
        ema.append(last)
    return ema


def _rsi(values: List[float], window: int) -> List[float]:
    rsis: List[float] = []
    gains: List[float] = []
    losses: List[float] = []
    for i in range(len(values)):
        if i == 0:
            gains.append(0.0)
            losses.append(0.0)
        else:
            change = values[i] - values[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        if i + 1 < window:
            rsis.append(float("nan"))
            continue

        avg_gain = sum(gains[i + 1 - window : i + 1]) / window
        avg_loss = sum(losses[i + 1 - window : i + 1]) / window
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))
    return rsis


def compute_features(bars: Iterable[PriceBar]) -> List[FeatureRow]:
    close_prices = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]

    if len(close_prices) < 25:
        raise ValueError("At least 25 data points are required to compute indicators.")

    returns = [0.0]
    log_returns = [0.0]
    for i in range(1, len(close_prices)):
        change = (close_prices[i] / close_prices[i - 1]) - 1
        returns.append(change)
        log_returns.append(math.log1p(change))

    sma_5 = _sma(close_prices, 5)
    sma_21 = _sma(close_prices, 21)
    ema_5 = _ema(close_prices, 5)
    ema_21 = _ema(close_prices, 21)
    rsi_14 = _rsi(close_prices, 14)

    volume_change = [0.0]
    for i in range(1, len(volumes)):
        prev = volumes[i - 1]
        volume_change.append(0.0 if prev == 0 else (volumes[i] / prev) - 1)

    features: List[FeatureRow] = []
    for i in range(len(close_prices) - 1):
        if any(math.isnan(value) for value in (
            sma_5[i],
            sma_21[i],
            ema_5[i],
            ema_21[i],
            rsi_14[i],
        )):
            continue

        target = 1 if close_prices[i + 1] > close_prices[i] else 0
        features.append(
            FeatureRow(
                return_1d=returns[i],
                log_return_1d=log_returns[i],
                sma_5=sma_5[i],
                sma_21=sma_21[i],
                ema_5=ema_5[i],
                ema_21=ema_21[i],
                rsi_14=rsi_14[i],
                volume_change=volume_change[i],
                target=target,
            )
        )

    return features
