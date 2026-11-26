import numpy as np
import pandas as pd
import pytest

from cryptocast.features.groups.momentum import compute_momentum_features
from cryptocast.features.groups.returns import compute_returns_features
from cryptocast.features.groups.volatility import compute_volatility_features
from cryptocast.features.leakage_guard import assert_no_lookahead_leakage, compute_data_hash
from cryptocast.features.pipeline import FeaturePipeline


def make_sample_data(n: int = 80) -> pd.DataFrame:
    base_time = pd.date_range("2026-11-01 00:00:00", periods=n, freq="1h", tz="UTC")
    prices = 100.0 + np.cumsum(np.sin(np.linspace(0, 15, n)))
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "open_time": base_time,
            "open": prices,
            "high": prices + 1.5,
            "low": prices - 1.5,
            "close": prices + 0.5,
            "volume": [1000.0 + i * 20 for i in range(n)],
            "quote_volume": [100000.0] * n,
            "trades": [200 + i for i in range(n)],
            "taker_buy_base": [600.0] * n,
            "taker_buy_quote": [60000.0] * n,
            "close_time": base_time + pd.Timedelta(hours=1),
            "ingested_at": pd.Timestamp.now(tz="UTC"),
        }
    )


def test_leakage_guard_on_feature_groups():
    df = make_sample_data(70)

    # All legitimate feature groups must pass leakage guard
    assert_no_lookahead_leakage(compute_returns_features, df, cutoff_idx=40)
    assert_no_lookahead_leakage(compute_volatility_features, df, cutoff_idx=40)
    assert_no_lookahead_leakage(compute_momentum_features, df, cutoff_idx=40)


def test_leakage_guard_on_full_pipeline():
    df = make_sample_data(70)
    pipeline = FeaturePipeline()

    def run_pipeline(d: pd.DataFrame) -> pd.DataFrame:
        return pipeline.extract_features(d, symbol="BTCUSDT")

    # Full pipeline must pass leakage guard without lookahead
    assert_no_lookahead_leakage(run_pipeline, df, cutoff_idx=45)


def test_leakage_guard_catches_intentional_leakage():
    df = make_sample_data(70)

    # Intentionally leaky feature function (peeking into t+1)
    def leaky_feature_fn(d: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=d.index)
        feats["leaked_future_close"] = d["close"].shift(-1)  # LOOKAHEAD LEAK!
        return feats

    # Leakage guard MUST catch this and raise AssertionError
    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_no_lookahead_leakage(leaky_feature_fn, df, cutoff_idx=40)


def test_compute_data_hash_deterministic():
    df = make_sample_data(30)
    hash1 = compute_data_hash(df)
    hash2 = compute_data_hash(df)
    assert hash1 == hash2
    assert len(hash1) == 16

    # Modifying data changes hash
    df_mod = df.copy()
    df_mod.loc[0, "close"] += 1.0
    hash_mod = compute_data_hash(df_mod)
    assert hash1 != hash_mod
