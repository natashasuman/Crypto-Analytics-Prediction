import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 lineage hash of input market data."""
    if df.empty:
        return "empty"
    core = df[["open_time", "close", "volume"]].copy()
    core["open_time"] = core["open_time"].astype(str)
    raw_bytes = core.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def assert_no_lookahead_leakage(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    cutoff_idx: int = 40,
    tolerance: float = 1e-6,
) -> None:
    """Verify that features computed at time t are completely invariant to future perturbations.

    Raises AssertionError if any future alteration leaks backward into past features.
    """
    if len(df) <= cutoff_idx + 10:
        raise ValueError(
            f"Dataset length ({len(df)}) must exceed cutoff_idx + 10 ({cutoff_idx + 10})"
        )

    # 1. Compute features on unmodified historical data
    base_features = feature_fn(df.copy())
    past_base = base_features.iloc[:cutoff_idx]

    # 2. Create perturbed future dataset: corrupt all future prices and volumes
    # Any index >= cutoff_idx belongs to the future relative to past_base (indices 0..cutoff_idx-1)
    df_perturbed = df.copy()
    future_indices = df_perturbed.index[cutoff_idx:]

    rng = np.random.default_rng(42)
    # Alter future close by 50-200% and future volume by 10x
    df_perturbed.loc[future_indices, "close"] = df_perturbed.loc[future_indices, "close"] * rng.uniform(
        1.5, 3.0, size=len(future_indices)
    )
    df_perturbed.loc[future_indices, "high"] = df_perturbed.loc[future_indices, "close"] * 1.05
    df_perturbed.loc[future_indices, "low"] = df_perturbed.loc[future_indices, "close"] * 0.95
    df_perturbed.loc[future_indices, "volume"] = df_perturbed.loc[future_indices, "volume"] * rng.uniform(
        2.0, 10.0, size=len(future_indices)
    )

    # 3. Re-compute features on perturbed data
    perturbed_features = feature_fn(df_perturbed)
    past_perturbed = perturbed_features.iloc[:cutoff_idx]

    # 4. Compare past feature outputs row by row, column by column
    numeric_cols = [c for c in past_base.columns if pd.api.types.is_numeric_dtype(past_base[c])]
    for col in numeric_cols:
        diff = (past_base[col].fillna(0.0) - past_perturbed[col].fillna(0.0)).abs()
        max_diff = diff.max()
        if max_diff > tolerance:
            raise AssertionError(
                f"Data leakage detected in feature '{col}'! "
                f"Maximum difference in historical window was {max_diff:.8f} (tolerance: {tolerance})"
            )
