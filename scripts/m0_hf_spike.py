"""M0 Feasibility Spike: Hugging Face Hub Storage Connectivity.

Tests Hugging Face Hub dataset repository interactions if HF_TOKEN and HF_REPO_ID are present,
or validates that the LocalStorage fallback operates cleanly.
"""

import os
import sys
import pandas as pd
from cryptocast.config import get_settings
from cryptocast.storage import LocalStorage, HFHubStorage


def test_hf_spike() -> bool:
    settings = get_settings()
    repo_id = os.getenv("HF_REPO_ID", settings.hf_repo_id)
    token = os.getenv("HF_TOKEN", settings.hf_token)

    sample_df = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT"],
        "price": [95000.0, 3500.0],
        "timestamp": pd.to_datetime(["2026-11-26 12:00:00", "2026-11-26 12:00:00"], utc=True),
    })

    if not token or not repo_id:
        print("[M0 HF Spike] HF_TOKEN or HF_REPO_ID not set. Testing LocalStorage fallback...")
        local_store = LocalStorage(base_dir="data/test_spike")
        test_file = "silver/test_sample.parquet"
        local_store.write_parquet(test_file, sample_df)
        read_back = local_store.read_parquet(test_file)
        assert len(read_back) == 2, "Row count mismatch in LocalStorage"
        print("[M0 HF Spike] LocalStorage fallback validated successfully.")
        return True

    print(f"[M0 HF Spike] Testing Hugging Face Hub connection to dataset {repo_id}...")
    try:
        hf_store = HFHubStorage(repo_id=repo_id, token=token, cache_dir="data/test_spike")
        test_file = "meta/test_spike.parquet"
        hf_store.write_parquet(test_file, sample_df)
        read_back = hf_store.read_parquet(test_file)
        assert len(read_back) == 2, "Row count mismatch in HFHubStorage"
        print(f"[M0 HF Spike] Successfully wrote and read back Parquet from Hugging Face repo {repo_id}!")
        return True
    except Exception as e:
        print(f"[M0 HF Spike] Hugging Face Hub check encountered error: {e}")
        return False


if __name__ == "__main__":
    success = test_hf_spike()
    sys.exit(0 if success else 1)
