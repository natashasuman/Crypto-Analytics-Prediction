import numpy as np
import pandas as pd


def compute_cross_asset_features(
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    symbol: str,
    benchmark_symbol: str = "BTCUSDT",
    corr_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute benchmark (BTC) returns and rolling correlations to the benchmark."""
    corr_windows = corr_windows or [24, 72]
    features = pd.DataFrame(index=asset_df.index)

    if symbol == benchmark_symbol or benchmark_df.empty:
        # If this asset is the benchmark itself, return neutral self-correlation
        features["cross_btc_ret_1"] = np.log(asset_df["close"] / asset_df["close"].shift(1)).fillna(
            0.0
        )
        features["cross_btc_ret_4"] = np.log(asset_df["close"] / asset_df["close"].shift(4)).fillna(
            0.0
        )
        for w in corr_windows:
            features[f"cross_corr_btc_{w}"] = 1.0
        return features

    # Align on open_time
    asset_ret = np.log(asset_df["close"] / asset_df["close"].shift(1)).fillna(0.0)
    merged = pd.merge(
        asset_df[["open_time"]].copy(),
        benchmark_df[["open_time", "close"]].copy(),
        on="open_time",
        how="left",
        suffixes=("", "_btc"),
    )
    btc_close = merged["close"]
    btc_ret1 = np.log(btc_close / btc_close.shift(1)).fillna(0.0)
    btc_ret4 = np.log(btc_close / btc_close.shift(4)).fillna(0.0)

    features["cross_btc_ret_1"] = btc_ret1.values
    features["cross_btc_ret_4"] = btc_ret4.values

    # Rolling correlations
    for w in corr_windows:
        rolling_cov = asset_ret.rolling(window=w, min_periods=w // 2).cov(btc_ret1)
        rolling_std_a = asset_ret.rolling(window=w, min_periods=w // 2).std().replace(0, np.nan)
        rolling_std_b = btc_ret1.rolling(window=w, min_periods=w // 2).std().replace(0, np.nan)
        corr = (rolling_cov / (rolling_std_a * rolling_std_b)).fillna(0.0)
        features[f"cross_corr_btc_{w}"] = corr.clip(-1.0, 1.0).values

    return features
