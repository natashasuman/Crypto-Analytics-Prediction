from pathlib import Path

import pandas as pd

from cryptocast.storage.local import LocalStorage


def test_local_storage_parquet_roundtrip(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "close": [95000.5, 3600.2],
            "volume": [1200.0, 5400.0],
        }
    )

    relative_file = "silver/klines/test.parquet"
    assert not store.exists(relative_file)

    store.write_parquet(relative_file, df)
    assert store.exists(relative_file)

    loaded = store.read_parquet(relative_file)
    assert len(loaded) == 2
    assert list(loaded["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    assert loaded["close"].iloc[0] == 95000.5


def test_local_storage_json_roundtrip(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    data = {
        "last_updated": "2026-11-26T12:00:00Z",
        "champion_version": "v1.0.0",
        "mcc": 0.185,
    }

    relative_file = "meta/champion.json"
    store.write_json(relative_file, data)
    assert store.exists(relative_file)

    loaded = store.read_json(relative_file)
    assert loaded["champion_version"] == "v1.0.0"
    assert loaded["mcc"] == 0.185


def test_local_storage_list_files(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    store.write_parquet("bronze/file1.parquet", df)
    store.write_parquet("bronze/sub/file2.parquet", df)
    store.write_parquet("silver/file3.parquet", df)

    bronze_files = store.list_files("bronze")
    assert len(bronze_files) == 2
    assert "bronze/file1.parquet" in bronze_files
    assert "bronze/sub/file2.parquet" in bronze_files

    all_files = store.list_files()
    assert len(all_files) == 3
