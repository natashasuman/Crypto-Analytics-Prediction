from cryptocast.config import get_settings
from cryptocast.storage.base import Storage
from cryptocast.storage.hf_hub import HFHubStorage
from cryptocast.storage.local import LocalStorage


def get_storage(backend: str | None = None) -> Storage:
    """Factory to get the configured storage provider."""
    settings = get_settings()
    chosen_backend = (backend or settings.storage_backend).lower()

    if chosen_backend == "hf_hub":
        if not settings.hf_repo_id:
            raise ValueError("HF_REPO_ID must be set when using hf_hub storage backend.")
        return HFHubStorage(
            repo_id=settings.hf_repo_id,
            token=settings.hf_token,
            cache_dir=settings.local_storage_dir,
        )
    return LocalStorage(base_dir=settings.local_storage_dir)


__all__ = ["Storage", "LocalStorage", "HFHubStorage", "get_storage"]
