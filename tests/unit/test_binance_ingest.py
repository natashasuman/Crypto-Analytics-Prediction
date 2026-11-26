from pathlib import Path
from unittest.mock import patch

from cryptocast.ingest.binance import BinanceIngester
from cryptocast.storage.local import LocalStorage


def make_mock_raw_klines(n: int = 5) -> list[list]:
    rows = []
    base_ts = 1732622400000  # ms timestamp
    for i in range(n):
        rows.append(
            [
                base_ts + i * 3600000,
                "90000.0",
                "90100.0",
                "89900.0",
                "90050.0",
                "150.0",
                base_ts + (i + 1) * 3600000 - 1,
                "13500000.0",
                400,
                "75.0",
                "6750000.0",
                "0",
            ]
        )
    return rows


def test_transform_to_silver(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    ingester = BinanceIngester(storage=store)

    raw_data = make_mock_raw_klines(5)
    silver_df = ingester.transform_to_silver(raw_data, symbol="BTCUSDT")

    assert len(silver_df) == 5
    assert list(silver_df["symbol"]) == ["BTCUSDT"] * 5
    assert silver_df["close"].dtype == "float64"
    assert silver_df["trades"].dtype == "int64"
    assert silver_df["open_time"].dt.tz is not None


def test_upsert_partition_idempotency(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    ingester = BinanceIngester(storage=store)

    raw_data = make_mock_raw_klines(5)
    silver_df = ingester.transform_to_silver(raw_data, symbol="BTCUSDT")

    month = "2026-11"
    # First write
    ingester.upsert_partition("silver", "BTCUSDT", "1h", month, silver_df)
    saved_1 = store.read_parquet(f"silver/klines/symbol=BTCUSDT/interval=1h/month={month}.parquet")
    assert len(saved_1) == 5

    # Second write with duplicate rows
    ingester.upsert_partition("silver", "BTCUSDT", "1h", month, silver_df)
    saved_2 = store.read_parquet(f"silver/klines/symbol=BTCUSDT/interval=1h/month={month}.parquet")
    assert len(saved_2) == 5  # No duplicates created


def test_ingest_recent_with_mock(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    ingester = BinanceIngester(storage=store)

    mock_raw = make_mock_raw_klines(10)
    with patch.object(ingester, "fetch_klines_raw", return_value=mock_raw):
        df = ingester.ingest_recent(
            symbol="BTCUSDT", interval="1h", lookback_bars=10, validate=True
        )
        assert len(df) == 10

        # Check watermark updated
        watermark = ingester.meta_logger.get_watermark("binance_BTCUSDT_1h")
        assert watermark is not None

        # Check runs log created
        assert store.exists("meta/runs.parquet")
        assert store.exists("meta/dq_results.parquet")
