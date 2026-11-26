import numpy as np
import pandas as pd


def compute_volatility_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Compute realized volatility, Parkinson volatility, Garman-Klass, and ATR."""
    windows = windows or [6, 12, 24, 72]
    features = pd.DataFrame(index=df.index)

    # 1-bar log returns
    ret1 = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

    # Realized volatility (rolling std of returns)
    for w in windows:
        features[f"vol_realized_{w}"] = (
            ret1.rolling(window=w, min_periods=max(2, w // 2)).std().fillna(0.0)
        )

    # Parkinson Volatility: sqrt( 1/(4*ln(2)) * mean(ln(high/low)^2) )
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan)).fillna(0.0)
    log_hl_sq = log_hl**2
    factor_p = 1.0 / (4.0 * np.log(2.0))
    for w in [12, 24]:
        features[f"vol_parkinson_{w}"] = np.sqrt(
            factor_p * log_hl_sq.rolling(window=w, min_periods=2).mean().fillna(0.0)
        )

    # Garman-Klass Volatility: sqrt( mean(0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2) )
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan)).fillna(0.0)
    gk_term = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    for w in [12, 24]:
        mean_gk = gk_term.rolling(window=w, min_periods=2).mean().clip(lower=0.0).fillna(0.0)
        features[f"vol_garman_klass_{w}"] = np.sqrt(mean_gk)

    # Average True Range (ATR)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)
    features["vol_atr"] = tr.rolling(window=atr_window, min_periods=2).mean().fillna(0.0)
    # Normalized ATR by close price
    features["vol_natr"] = (
        (features["vol_atr"] / df["close"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )

    return features
