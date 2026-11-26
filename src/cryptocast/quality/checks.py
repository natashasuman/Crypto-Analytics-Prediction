from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cryptocast.quality.contracts import SILVER_KLINES_COLUMNS, DQResult


def check_schema(df: pd.DataFrame, expected_columns: list[str] | None = None) -> DQResult:
    """Verify that required columns are present in the DataFrame."""
    expected = expected_columns or SILVER_KLINES_COLUMNS
    now_str = datetime.now(UTC).isoformat()
    missing = [col for col in expected if col not in df.columns]

    if missing:
        return DQResult(
            check_name="schema_check",
            passed=False,
            severity="error",
            details=f"Missing columns: {missing}",
            timestamp=now_str,
        )
    return DQResult(
        check_name="schema_check",
        passed=True,
        severity="error",
        details="All expected columns present",
        timestamp=now_str,
    )


def check_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> DQResult:
    """Ensure no duplicate records exist for the primary key."""
    keys = subset or ["symbol", "open_time"]
    now_str = datetime.now(UTC).isoformat()

    dup_count = df.duplicated(subset=keys).sum()
    if dup_count > 0:
        return DQResult(
            check_name="duplicate_check",
            passed=False,
            severity="error",
            details=f"Found {dup_count} duplicate rows for keys {keys}",
            timestamp=now_str,
        )
    return DQResult(
        check_name="duplicate_check",
        passed=True,
        severity="error",
        details="No duplicates found",
        timestamp=now_str,
    )


def check_ohlc_consistency(df: pd.DataFrame) -> DQResult:
    """Validate high >= max(open, close), low <= min(open, close), volume >= 0."""
    now_str = datetime.now(UTC).isoformat()
    if df.empty:
        return DQResult(
            check_name="ohlc_consistency",
            passed=True,
            severity="error",
            details="Empty dataset, skipped",
            timestamp=now_str,
        )

    max_oc = np.maximum(df["open"].values, df["close"].values)
    min_oc = np.minimum(df["open"].values, df["close"].values)

    high_violations = (df["high"].values < max_oc - 1e-6).sum()
    low_violations = (df["low"].values > min_oc + 1e-6).sum()
    volume_violations = (df["volume"].values < 0).sum()

    total_violations = high_violations + low_violations + volume_violations
    if total_violations > 0:
        return DQResult(
            check_name="ohlc_consistency",
            passed=False,
            severity="error",
            details=(
                f"OHLC violations: high < max(open, close): {high_violations}, "
                f"low > min(open, close): {low_violations}, volume < 0: {volume_violations}"
            ),
            timestamp=now_str,
        )
    return DQResult(
        check_name="ohlc_consistency",
        passed=True,
        severity="error",
        details="OHLC and volume values consistent",
        timestamp=now_str,
    )


def check_gaps(df: pd.DataFrame, expected_freq: str = "1h") -> DQResult:
    """Detect missing candles in a contiguous time series."""
    now_str = datetime.now(UTC).isoformat()
    if len(df) <= 1:
        return DQResult(
            check_name="gap_detection",
            passed=True,
            severity="error",
            details="Dataset has <= 1 candle, no gap possible",
            timestamp=now_str,
        )

    sorted_times = pd.to_datetime(df["open_time"]).sort_values()
    diffs = sorted_times.diff().iloc[1:]
    expected_delta = pd.Timedelta(expected_freq)

    gaps = diffs[diffs > expected_delta]
    if len(gaps) > 0:
        max_gap = gaps.max()
        return DQResult(
            check_name="gap_detection",
            passed=False,
            severity="error",
            details=f"Detected {len(gaps)} gaps; maximum gap is {max_gap}",
            timestamp=now_str,
        )
    return DQResult(
        check_name="gap_detection",
        passed=True,
        severity="error",
        details="No gaps detected in the series",
        timestamp=now_str,
    )


def check_freshness(df: pd.DataFrame, max_delay_minutes: int = 90) -> DQResult:
    """Verify that the latest candle is not older than max_delay_minutes."""
    now_str = datetime.now(UTC).isoformat()
    if df.empty:
        return DQResult(
            check_name="freshness_check",
            passed=False,
            severity="error",
            details="DataFrame is empty, cannot verify freshness",
            timestamp=now_str,
        )

    latest_time = pd.to_datetime(df["open_time"]).max()
    if latest_time.tzinfo is None:
        latest_time = latest_time.replace(tzinfo=UTC)
    now = datetime.now(UTC)

    delay = (now - latest_time).total_seconds() / 60.0
    if delay > max_delay_minutes:
        return DQResult(
            check_name="freshness_check",
            passed=False,
            severity="error",
            details=f"Latest candle is {delay:.1f} minutes old (threshold: {max_delay_minutes}m)",
            timestamp=now_str,
        )
    return DQResult(
        check_name="freshness_check",
        passed=True,
        severity="error",
        details=f"Freshness OK: latest candle is {delay:.1f} minutes old",
        timestamp=now_str,
    )


def check_price_outliers(
    df: pd.DataFrame,
    rolling_window: int = 48,
    std_threshold: float = 5.0,
) -> DQResult:
    """Check for severe price spikes (outliers) beyond N standard deviations."""
    now_str = datetime.now(UTC).isoformat()
    if len(df) < rolling_window:
        return DQResult(
            check_name="price_outlier",
            passed=True,
            severity="warn",
            details="Insufficient data for rolling outlier check",
            timestamp=now_str,
        )

    log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    rolling_mean = log_returns.rolling(window=rolling_window).mean()
    rolling_std = log_returns.rolling(window=rolling_window).std().replace(0, np.nan)
    z_scores = np.abs((log_returns - rolling_mean) / rolling_std).fillna(0)

    outliers = (z_scores > std_threshold).sum()
    if outliers > 0:
        return DQResult(
            check_name="price_outlier",
            passed=False,
            severity="warn",
            details=f"Found {outliers} candles exceeding {std_threshold}x rolling std",
            timestamp=now_str,
        )
    return DQResult(
        check_name="price_outlier",
        passed=True,
        severity="warn",
        details="No statistical price outliers detected",
        timestamp=now_str,
    )


def run_klines_quality_suite(
    df: pd.DataFrame,
    interval: str = "1h",
    max_delay_minutes: int = 90,
    enforce_freshness: bool = False,
) -> list[DQResult]:
    """Execute complete data quality validation suite for klines."""
    results = [
        check_schema(df),
        check_duplicates(df),
        check_ohlc_consistency(df),
        check_gaps(df, expected_freq=interval),
        check_price_outliers(df),
    ]
    if enforce_freshness:
        results.append(check_freshness(df, max_delay_minutes=max_delay_minutes))
    return results
