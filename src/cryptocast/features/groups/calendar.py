import numpy as np
import pandas as pd


def compute_calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    """Compute cyclical trigonometric encoding for hour-of-day and day-of-week."""
    ts = pd.to_datetime(timestamps, utc=True)
    features = pd.DataFrame(index=timestamps.index)

    hour = ts.dt.hour
    dayofweek = ts.dt.day_of_week  # 0 = Monday, 6 = Sunday

    # Hour of day (24h period)
    features["cal_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["cal_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Day of week (7 days period)
    features["cal_dow_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    features["cal_dow_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)

    return features
