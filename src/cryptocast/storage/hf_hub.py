from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from cryptocast.storage.base import Storage
from cryptocast.storage.local import LocalStorage


class HFHubStorage(Storage):
    """Hugging Face Hub dataset storage implementation with local cache layer.

    Pulls from and pushes to a Hugging Face Hub Dataset repository.
    """

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        cache_dir: str | Path = "data",
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.local = LocalStorage(base_dir=cache_dir)
        self.api = HfApi(token=token)

    def _normalize_key(self, relative_path: str) -> str:
        return relative_path.lstrip("/\\").replace("\\", "/")

    def download_if_needed(self, relative_path: str) -> Path:
        key = self._normalize_key(relative_path)
        local_path = self.local._resolve(key)
        if not local_path.exists():
            try:
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=key,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=str(self.local.base_dir),
                )
                return Path(downloaded)
            except Exception as e:
                raise FileNotFoundError(f"Failed to download {key} from {self.repo_id}: {e}")
        return local_path

    def read_parquet(self, relative_path: str, columns: list[str] | None = None) -> pd.DataFrame:
        key = self._normalize_key(relative_path)
        self.download_if_needed(key)
        return self.local.read_parquet(key, columns=columns)

    def write_parquet(
        self,
        relative_path: str,
        df: pd.DataFrame,
        compression: str = "zstd",
    ) -> None:
        key = self._normalize_key(relative_path)
        self.local.write_parquet(key, df, compression=compression)
        local_path = self.local._resolve(key)
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=key,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update {key}",
        )

    def exists(self, relative_path: str) -> bool:
        key = self._normalize_key(relative_path)
        if self.local.exists(key):
            return True
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            return key in files
        except Exception:
            return False

    def read_json(self, relative_path: str) -> dict[str, Any]:
        key = self._normalize_key(relative_path)
        self.download_if_needed(key)
        return self.local.read_json(key)

    def write_json(self, relative_path: str, data: dict[str, Any]) -> None:
        key = self._normalize_key(relative_path)
        self.local.write_json(key, data)
        local_path = self.local._resolve(key)
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=key,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update {key}",
        )

    def list_files(self, prefix: str = "") -> list[str]:
        prefix_norm = self._normalize_key(prefix)
        try:
            repo_files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            filtered = [f for f in repo_files if f.startswith(prefix_norm)]
            return sorted(filtered)
        except Exception:
            return self.local.list_files(prefix)
