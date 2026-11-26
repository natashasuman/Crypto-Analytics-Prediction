import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Wilder's smoothing
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_momentum_features(
    df: pd.DataFrame,
    rsi_windows: list[int] | None = None,
    roc_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute trend and momentum features: RSI, MACD, EMA ratios, and ROC."""
    rsi_windows = rsi_windows or [14, 28]
    roc_windows = roc_windows or [6, 12, 24]
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    # RSI
    for w in rsi_windows:
        features[f"trend_rsi_{w}"] = compute_rsi(close, window=w)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Normalize MACD by close price for scale invariance across price regimes
    features["trend_macd_line"] = (macd_line / close).fillna(0.0)
    features["trend_macd_signal"] = (signal_line / close).fillna(0.0)
    features["trend_macd_hist"] = (macd_hist / close).fillna(0.0)

    # EMA Ratios
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    features["trend_ema_ratio_12_26"] = (ema12 / ema26.replace(0, np.nan) - 1.0).fillna(0.0)
    features["trend_ema_ratio_50_200"] = (ema50 / ema200.replace(0, np.nan) - 1.0).fillna(0.0)

    # Rate of Change (ROC)
    for w in roc_windows:
        features[f"trend_roc_{w}"] = (close / close.shift(w).replace(0, np.nan) - 1.0).fillna(0.0)

    return features
