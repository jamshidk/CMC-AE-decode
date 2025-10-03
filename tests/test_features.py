from datetime import date, timedelta

from spy_predictor.data import PriceBar
from spy_predictor.features import compute_features


def make_sample_data():
    start = date(2020, 1, 1)
    bars = []
    for i in range(40):
        current_date = start + timedelta(days=i)
        close = 300 + i
        bars.append(
            PriceBar(
                date=current_date,
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1_000_000 + i * 1000,
            )
        )
    return bars


def test_compute_features_returns_non_empty_dataset():
    bars = make_sample_data()
    features = compute_features(bars)
    assert len(features) > 0
    assert all(row.target in (0, 1) for row in features)


def test_feature_vectors_have_expected_length():
    bars = make_sample_data()
    features = compute_features(bars)
    vector_lengths = {len(row.as_vector()) for row in features}
    assert vector_lengths == {8}
