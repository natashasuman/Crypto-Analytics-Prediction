from datetime import UTC, datetime

import pandas as pd

from cryptocast.quality.checks import (
    check_duplicates,
    check_freshness,
    check_gaps,
    check_ohlc_consistency,
    check_schema,
    run_klines_quality_suite,
)


def make_sample_klines(n: int = 10, freq: str = "1h") -> pd.DataFrame:
    base_time = pd.date_range("2026-11-26 00:00:00", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "open_time": base_time,
            "open": [90000.0 + i * 10 for i in range(n)],
            "high": [90050.0 + i * 10 for i in range(n)],
            "low": [89950.0 + i * 10 for i in range(n)],
            "close": [90020.0 + i * 10 for i in range(n)],
            "volume": [100.0] * n,
            "quote_volume": [9000000.0] * n,
            "trades": [500] * n,
            "taker_buy_base": [50.0] * n,
            "taker_buy_quote": [4500000.0] * n,
            "close_time": base_time + pd.Timedelta(hours=1),
            "ingested_at": pd.to_datetime(datetime.now(UTC)),
        }
    )


def test_check_schema_success_and_failure():
    df = make_sample_klines(5)
    res = check_schema(df)
    assert res.passed is True

    df_bad = df.drop(columns=["close"])
    res_bad = check_schema(df_bad)
    assert res_bad.passed is False
    assert "close" in res_bad.details


def test_check_duplicates():
    df = make_sample_klines(5)
    res = check_duplicates(df)
    assert res.passed is True

    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    res_dup = check_duplicates(df_dup)
    assert res_dup.passed is False
    assert "Found 1 duplicate" in res_dup.details


def test_check_ohlc_consistency():
    df = make_sample_klines(5)
    res = check_ohlc_consistency(df)
    assert res.passed is True

    # Violate high < max(open, close)
    df_bad = df.copy()
    df_bad.loc[0, "high"] = 80000.0
    res_bad = check_ohlc_consistency(df_bad)
    assert res_bad.passed is False

    # Violate volume < 0
    df_bad2 = df.copy()
    df_bad2.loc[1, "volume"] = -10.0
    res_bad2 = check_ohlc_consistency(df_bad2)
    assert res_bad2.passed is False


def test_check_gaps():
    df = make_sample_klines(5, freq="1h")
    res = check_gaps(df, expected_freq="1h")
    assert res.passed is True

    # Drop candle at index 2 to create a gap
    df_gap = df.drop(index=[2]).reset_index(drop=True)
    res_gap = check_gaps(df_gap, expected_freq="1h")
    assert res_gap.passed is False
    assert "Detected 1 gaps" in res_gap.details


def test_check_freshness():
    # Very old candle
    old_time = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")
    df_old = make_sample_klines(3)
    df_old["open_time"] = old_time
    res_old = check_freshness(df_old, max_delay_minutes=90)
    assert res_old.passed is False


def test_run_klines_quality_suite():
    df = make_sample_klines(5)
    results = run_klines_quality_suite(df, interval="1h")
    assert len(results) >= 5
    assert all(r.passed for r in results)
