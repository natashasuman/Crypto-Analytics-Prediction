import numpy as np
import pandas as pd


def compute_mean_reversion_features(
    df: pd.DataFrame,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    zscore_window: int = 24,
) -> pd.DataFrame:
    """Compute Bollinger Bands (%B and bandwidth) and price z-score."""
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    # Bollinger Bands
    sma = close.rolling(window=bollinger_window, min_periods=bollinger_window // 2).mean()
    std = (
        close.rolling(window=bollinger_window, min_periods=bollinger_window // 2)
        .std()
        .replace(0, np.nan)
    )
    upper_band = sma + bollinger_std * std
    lower_band = sma - bollinger_std * std

    width = upper_band - lower_band
    features["mr_bollinger_pct_b"] = ((close - lower_band) / width.replace(0, np.nan)).fillna(0.5)
    features["mr_bollinger_bandwidth"] = (width / sma.replace(0, np.nan)).fillna(0.0)

    # Rolling price z-score
    z_mean = close.rolling(window=zscore_window, min_periods=zscore_window // 2).mean()
    z_std = (
        close.rolling(window=zscore_window, min_periods=zscore_window // 2).std().replace(0, np.nan)
    )
    features["mr_close_zscore"] = ((close - z_mean) / z_std).fillna(0.0)

    return features
