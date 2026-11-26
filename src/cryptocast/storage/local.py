import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from cryptocast.storage.base import Storage


class LocalStorage(Storage):
    """Local filesystem storage implementation."""

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        clean_path = relative_path.lstrip("/\\")
        full_path = self.base_dir / clean_path
        return full_path

    def read_parquet(self, relative_path: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self._resolve(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return pd.read_parquet(path, columns=columns)

    def write_parquet(
        self,
        relative_path: str,
        df: pd.DataFrame,
        compression: str = "zstd",
    ) -> None:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression=compression, index=False)

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self._resolve(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

    def write_json(self, relative_path: str, data: dict[str, Any]) -> None:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def list_files(self, prefix: str = "") -> list[str]:
        target_dir = self._resolve(prefix)
        if not target_dir.exists():
            return []
        if target_dir.is_file():
            return [prefix]
        results: list[str] = []
        for p in target_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.base_dir).as_posix()
                results.append(rel)
        return sorted(results)
