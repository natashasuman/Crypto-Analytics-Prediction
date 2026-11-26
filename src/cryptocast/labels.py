import numpy as np
import pandas as pd


def compute_target_labels(
    df: pd.DataFrame,
    horizon: int = 4,
    dead_zone_eps: float = 0.0,
) -> pd.DataFrame:
    """Compute dual prediction targets for bar close time t over horizon h:

    y_ret = ln(close[t+h] / close[t])
    y_dir = 1 if y_ret > 0 else 0 (with optional dead_zone filtering)

    Note: The last `horizon` rows in the dataset will have NaN labels because
    the future horizon has not yet matured.
    """
    labels = pd.DataFrame(index=df.index)
    close = df["close"]

    # Forward log return over horizon h
    y_ret = np.log(close.shift(-horizon) / close)
    labels[f"target_ret_{horizon}"] = y_ret

    # Direction binary classification
    y_dir = (y_ret > 0.0).astype(int)
    # Mask rows where future return is NaN (unmatured candles)
    y_dir = y_dir.where(y_ret.notna(), other=np.nan)
    labels[f"target_dir_{horizon}"] = y_dir

    if dead_zone_eps > 0.0:
        # Mask near-zero noise returns as NaN so training ignores them
        labels.loc[y_ret.abs() < dead_zone_eps, f"target_dir_{horizon}"] = np.nan

    return labels
