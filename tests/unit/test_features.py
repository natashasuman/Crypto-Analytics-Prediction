import numpy as np
import pandas as pd

from cryptocast.features.groups.calendar import compute_calendar_features
from cryptocast.features.groups.cross_asset import compute_cross_asset_features
from cryptocast.features.groups.mean_reversion import compute_mean_reversion_features
from cryptocast.features.groups.momentum import compute_momentum_features
from cryptocast.features.groups.returns import compute_returns_features
from cryptocast.features.groups.volatility import compute_volatility_features
from cryptocast.features.groups.volume_flow import compute_volume_flow_features
from cryptocast.labels import compute_target_labels


def make_dummy_klines(n: int = 50) -> pd.DataFrame:
    base_time = pd.date_range("2026-11-01 00:00:00", periods=n, freq="1h", tz="UTC")
    prices = 100.0 + np.cumsum(np.sin(np.linspace(0, 10, n)))
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "open_time": base_time,
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices + 0.2,
            "volume": [1000.0 + i * 10 for i in range(n)],
            "quote_volume": [100000.0] * n,
            "trades": [100 + i for i in range(n)],
            "taker_buy_base": [500.0] * n,
            "taker_buy_quote": [50000.0] * n,
            "close_time": base_time + pd.Timedelta(hours=1),
            "ingested_at": pd.Timestamp.now(tz="UTC"),
        }
    )


def test_returns_features():
    df = make_dummy_klines(30)
    ret_df = compute_returns_features(df, windows=[1, 4])
    assert "ret_log_1" in ret_df.columns
    assert "ret_log_4" in ret_df.columns
    # Verify hand calculation
    expected_ret_1 = np.log(df["close"].iloc[5] / df["close"].iloc[4])
    assert np.isclose(ret_df["ret_log_1"].iloc[5], expected_ret_1)


def test_volatility_features():
    df = make_dummy_klines(40)
    vol_df = compute_volatility_features(df, windows=[6, 12], atr_window=14)
    assert "vol_realized_6" in vol_df.columns
    assert "vol_parkinson_12" in vol_df.columns
    assert "vol_garman_klass_12" in vol_df.columns
    assert "vol_atr" in vol_df.columns
    assert (vol_df["vol_parkinson_12"] >= 0).all()


def test_momentum_and_mean_reversion():
    df = make_dummy_klines(40)
    mom_df = compute_momentum_features(df)
    mr_df = compute_mean_reversion_features(df)

    assert "trend_rsi_14" in mom_df.columns
    assert "trend_macd_hist" in mom_df.columns
    assert "mr_bollinger_pct_b" in mr_df.columns
    assert "mr_close_zscore" in mr_df.columns
    assert (mom_df["trend_rsi_14"] >= 0.0).all() and (mom_df["trend_rsi_14"] <= 100.0).all()


def test_volume_flow_and_calendar():
    df = make_dummy_klines(30)
    flow_df = compute_volume_flow_features(df)
    cal_df = compute_calendar_features(df["open_time"])

    assert "flow_vol_zscore" in flow_df.columns
    assert "flow_taker_buy_ratio" in flow_df.columns
    assert "cal_hour_sin" in cal_df.columns
    assert "cal_dow_cos" in cal_df.columns
    assert (cal_df["cal_hour_sin"] >= -1.0).all() and (cal_df["cal_hour_sin"] <= 1.0).all()


def test_cross_asset_features():
    df_eth = make_dummy_klines(30)
    df_eth["symbol"] = "ETHUSDT"
    df_btc = make_dummy_klines(30)

    cross_df = compute_cross_asset_features(df_eth, df_btc, symbol="ETHUSDT", corr_windows=[12])
    assert "cross_btc_ret_1" in cross_df.columns
    assert "cross_corr_btc_12" in cross_df.columns
    assert (cross_df["cross_corr_btc_12"] >= -1.0).all() and (
        cross_df["cross_corr_btc_12"] <= 1.0
    ).all()


def test_compute_target_labels():
    df = make_dummy_klines(20)
    horizon = 4
    labels = compute_target_labels(df, horizon=horizon)

    assert f"target_ret_{horizon}" in labels.columns
    assert f"target_dir_{horizon}" in labels.columns

    # Verify that the last `horizon` rows are NaN
    assert labels[f"target_ret_{horizon}"].iloc[-horizon:].isna().all()
    assert labels[f"target_dir_{horizon}"].iloc[-horizon:].isna().all()

    # Verify forward return calculation at index 0: ln(close[4] / close[0])
    expected_ret = np.log(df["close"].iloc[4] / df["close"].iloc[0])
    assert np.isclose(labels[f"target_ret_{horizon}"].iloc[0], expected_ret)
    expected_dir = 1 if expected_ret > 0 else 0
    assert labels[f"target_dir_{horizon}"].iloc[0] == expected_dir
