from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cryptocast.quality.contracts import DQResult
from cryptocast.storage.base import Storage


class MetadataLogger:
    """Manages metadata tables: runs.parquet, dq_results.parquet, and watermarks.json."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def log_run(
        self,
        dag_name: str,
        status: str,
        records_processed: int = 0,
        error_message: str | None = None,
        duration_seconds: float = 0.0,
    ) -> None:
        """Append pipeline execution entry to meta/runs.parquet."""
        run_record = {
            "timestamp": pd.to_datetime(datetime.now(UTC)),
            "dag_name": dag_name,
            "status": status,
            "records_processed": records_processed,
            "duration_seconds": duration_seconds,
            "error_message": error_message or "",
        }
        df_new = pd.DataFrame([run_record])

        rel_path = "meta/runs.parquet"
        if self.storage.exists(rel_path):
            try:
                df_existing = self.storage.read_parquet(rel_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        self.storage.write_parquet(rel_path, df_combined)

    def log_dq_results(self, results: list[DQResult], asset: str) -> None:
        """Append data quality check results to meta/dq_results.parquet."""
        if not results:
            return

        records = [
            {
                "timestamp": pd.to_datetime(r.timestamp),
                "asset": asset,
                "check_name": r.check_name,
                "passed": r.passed,
                "severity": r.severity,
                "details": r.details,
            }
            for r in results
        ]
        df_new = pd.DataFrame(records)

        rel_path = "meta/dq_results.parquet"
        if self.storage.exists(rel_path):
            try:
                df_existing = self.storage.read_parquet(rel_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        self.storage.write_parquet(rel_path, df_combined)

    def get_watermark(self, key: str) -> str | None:
        """Read watermark timestamp string for a given source key."""
        rel_path = "meta/watermarks.json"
        if not self.storage.exists(rel_path):
            return None
        try:
            watermarks = self.storage.read_json(rel_path)
            return watermarks.get(key)
        except Exception:
            return None

    def update_watermark(self, key: str, timestamp_str: str) -> None:
        """Update watermark timestamp string in meta/watermarks.json."""
        rel_path = "meta/watermarks.json"
        watermarks: dict[str, Any] = {}
        if self.storage.exists(rel_path):
            try:
                watermarks = self.storage.read_json(rel_path)
            except Exception:
                watermarks = {}
        watermarks[key] = timestamp_str
        self.storage.write_json(rel_path, watermarks)
