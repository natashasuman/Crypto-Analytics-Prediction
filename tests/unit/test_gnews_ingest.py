from pathlib import Path

from cryptocast.ingest.gnews import GNewsIngester
from cryptocast.storage.local import LocalStorage


def test_gnews_transform_and_hashing(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    ingester = GNewsIngester(storage=store, api_key=None)

    mock_articles = [
        {
            "title": "Bitcoin reaches new high",
            "description": "BTC prices soar on ETF inflows",
            "url": "https://example.com/bitcoin-news-1",
            "publishedAt": "2026-11-26T12:00:00Z",
            "source": {"name": "CryptoNews", "url": "https://example.com"},
        },
        {
            "title": "Ethereum upgrade announced",
            "description": "Network improvements scheduled",
            "url": "https://example.com/ethereum-news-2",
            "publishedAt": "2026-11-26T13:00:00Z",
            "source": {"name": "ETHWorld", "url": "https://example.com"},
        },
    ]

    df = ingester.transform_to_silver(mock_articles, asset="BTC")
    assert len(df) == 2
    assert "url_hash" in df.columns
    assert len(df["url_hash"].iloc[0]) == 16
    assert df["asset"].iloc[0] == "BTC"


def test_gnews_graceful_empty_when_no_key(tmp_path: Path):
    store = LocalStorage(base_dir=tmp_path)
    ingester = GNewsIngester(storage=store, api_key=None)

    df = ingester.ingest_asset_news(asset="BTC", keywords=["bitcoin"])
    assert df.empty
    assert store.exists("meta/runs.parquet")
