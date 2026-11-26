import numpy as np
import pandas as pd


def compute_returns_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute backward-looking log returns: ln(close[t] / close[t-k])."""
    windows = windows or [1, 2, 4, 8, 24, 72]
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    for w in windows:
        features[f"ret_log_{w}"] = np.log(close / close.shift(w)).fillna(0.0)

    return features
