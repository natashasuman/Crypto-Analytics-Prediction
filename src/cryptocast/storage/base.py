from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Storage(ABC):
    """Abstract storage interface for data persistence."""

    @abstractmethod
    def read_parquet(self, relative_path: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Read Parquet file into a Pandas DataFrame."""
        pass

    @abstractmethod
    def write_parquet(
        self,
        relative_path: str,
        df: pd.DataFrame,
        compression: str = "zstd",
    ) -> None:
        """Write a Pandas DataFrame as Parquet."""
        pass

    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """Check if relative path exists in storage."""
        pass

    @abstractmethod
    def read_json(self, relative_path: str) -> dict[str, Any]:
        """Read JSON file from storage."""
        pass

    @abstractmethod
    def write_json(self, relative_path: str, data: dict[str, Any]) -> None:
        """Write dictionary to JSON file in storage."""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """List file paths under a prefix."""
        pass
