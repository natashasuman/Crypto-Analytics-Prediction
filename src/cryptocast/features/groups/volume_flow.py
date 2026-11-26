import numpy as np
import pandas as pd


def compute_volume_flow_features(
    df: pd.DataFrame,
    zscore_window: int = 24,
    trades_window: int = 4,
) -> pd.DataFrame:
    """Compute volume z-score, taker-buy ratio, trade count change, and divergence."""
    features = pd.DataFrame(index=df.index)
    volume = df["volume"]
    trades = df["trades"]
    taker_buy = df["taker_buy_base"]

    # Volume z-score
    vol_mean = volume.rolling(window=zscore_window, min_periods=zscore_window // 2).mean()
    vol_std = (
        volume.rolling(window=zscore_window, min_periods=zscore_window // 2)
        .std()
        .replace(0, np.nan)
    )
    vol_zscore = ((volume - vol_mean) / vol_std).fillna(0.0)
    features["flow_vol_zscore"] = vol_zscore

    # Taker buy ratio (share of aggressive market buy orders)
    features["flow_taker_buy_ratio"] = (taker_buy / volume.replace(0, np.nan)).fillna(0.5)

    # Trade count change
    features["flow_trades_change_4"] = (
        trades / trades.shift(trades_window).replace(0, np.nan) - 1.0
    ).fillna(0.0)

    # Volume-price divergence: price return sign * volume deviation
    ret1 = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    features["flow_vol_price_divergence"] = (np.sign(ret1) * vol_zscore).fillna(0.0)

    return features
