import time
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import requests

from cryptocast.config import get_settings
from cryptocast.quality.checks import run_klines_quality_suite
from cryptocast.quality.contracts import SILVER_KLINES_COLUMNS
from cryptocast.quality.logger import MetadataLogger
from cryptocast.storage import Storage, get_storage

BINANCE_RAW_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class BinanceIngester:
    """Binance market data ingester with automatic partitioning and data quality gates."""

    def __init__(
        self,
        storage: Storage | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.storage = storage or get_storage()
        self.base_url = base_url or self.settings.binance_base_url
        self.meta_logger = MetadataLogger(self.storage)

    def fetch_klines_raw(
        self,
        symbol: str,
        interval: str = "1h",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
        max_retries: int = 3,
    ) -> list[list[Any]]:
        """Fetch raw klines from Binance API with exponential backoff."""
        endpoint = f"{self.base_url}/api/v3/klines"
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(endpoint, params=params, timeout=15)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return cast(list[list[Any]], response.json())
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch Binance klines after {max_retries} attempts: {e}"
                    ) from e
                time.sleep(delay)
                delay *= 2.0
        return []

    def transform_to_silver(self, raw_data: list[list[Any]], symbol: str) -> pd.DataFrame:
        """Convert raw Binance klines payload into standardized silver DataFrame."""
        if not raw_data:
            return pd.DataFrame(columns=SILVER_KLINES_COLUMNS)

        df = pd.DataFrame(raw_data, columns=BINANCE_RAW_COLS)
        df["symbol"] = symbol
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        float_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in float_cols:
            df[col] = df[col].astype(float)

        df["trades"] = df["trades"].astype("int64")
        df["ingested_at"] = pd.to_datetime(datetime.now(UTC))

        return df[SILVER_KLINES_COLUMNS]

    def _get_partition_path(self, layer: str, symbol: str, interval: str, month: str) -> str:
        return f"{layer}/klines/symbol={symbol}/interval={interval}/month={month}.parquet"

    def upsert_partition(
        self,
        layer: str,
        symbol: str,
        interval: str,
        month: str,
        new_df: pd.DataFrame,
    ) -> None:
        """Idempotently upsert klines for a given monthly partition."""
        rel_path = self._get_partition_path(layer, symbol, interval, month)
        if self.storage.exists(rel_path):
            existing_df = self.storage.read_parquet(rel_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "open_time"], keep="last")
            combined = combined.sort_values("open_time").reset_index(drop=True)
        else:
            combined = new_df.drop_duplicates(subset=["symbol", "open_time"], keep="last")
            combined = combined.sort_values("open_time").reset_index(drop=True)

        self.storage.write_parquet(rel_path, combined)

    def ingest_recent(
        self,
        symbol: str,
        interval: str = "1h",
        lookback_bars: int = 48,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Ingest recent bars (for hourly DAG), validate DQ, and update watermarks."""
        start_time = time.time()
        raw_klines = self.fetch_klines_raw(symbol=symbol, interval=interval, limit=lookback_bars)
        silver_df = self.transform_to_silver(raw_klines, symbol=symbol)

        if silver_df.empty:
            self.meta_logger.log_run(
                "ingest_klines", "success", 0, duration_seconds=time.time() - start_time
            )
            return silver_df

        # Run DQ validations
        if validate:
            dq_results = run_klines_quality_suite(silver_df, interval=interval)
            self.meta_logger.log_dq_results(dq_results, asset=symbol)
            errors = [r for r in dq_results if not r.passed and r.severity == "error"]
            if errors:
                error_msg = (
                    f"Data quality validation failed for {symbol}: {[e.details for e in errors]}"
                )
                self.meta_logger.log_run(
                    "ingest_klines",
                    "failed",
                    len(silver_df),
                    error_message=error_msg,
                    duration_seconds=time.time() - start_time,
                )
                raise ValueError(error_msg)

        # Partition by month and upsert
        silver_df["_month"] = silver_df["open_time"].dt.strftime("%Y-%m")
        for month, group in silver_df.groupby("_month"):
            clean_group = group.drop(columns=["_month"])
            # Save Bronze and Silver
            self.upsert_partition("bronze", symbol, interval, str(month), clean_group)
            self.upsert_partition("silver", symbol, interval, str(month), clean_group)

        # Update watermark
        latest_candle = silver_df["open_time"].max().isoformat()
        self.meta_logger.update_watermark(f"binance_{symbol}_{interval}", latest_candle)
        self.meta_logger.log_run(
            "ingest_klines",
            "success",
            len(silver_df),
            duration_seconds=time.time() - start_time,
        )
        return silver_df
