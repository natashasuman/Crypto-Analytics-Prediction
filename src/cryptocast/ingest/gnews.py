import hashlib
import time
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import requests

from cryptocast.config import get_settings
from cryptocast.quality.contracts import SILVER_NEWS_COLUMNS
from cryptocast.quality.logger import MetadataLogger
from cryptocast.storage import Storage, get_storage


class GNewsIngester:
    """GNews headline ingester with graceful fallback when unauthenticated."""

    def __init__(
        self,
        storage: Storage | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.storage = storage or get_storage()
        self.api_key = api_key or self.settings.gnews_api_key
        self.meta_logger = MetadataLogger(self.storage)

    def fetch_news_raw(
        self,
        keyword: str,
        max_articles: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch headlines from GNews API."""
        if not self.api_key:
            return []

        endpoint = "https://gnews.io/api/v4/search"
        params: dict[str, Any] = {
            "q": keyword,
            "lang": "en",
            "max": max_articles,
            "token": self.api_key,
        }
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 429:
                return []
            response.raise_for_status()
            data = response.json()
            return cast(list[dict[str, Any]], data.get("articles", []))
        except Exception:
            return []

    def transform_to_silver(
        self,
        articles: list[dict[str, Any]],
        asset: str,
    ) -> pd.DataFrame:
        """Standardize GNews articles into silver news schema with dedup hash."""
        if not articles:
            return pd.DataFrame(columns=SILVER_NEWS_COLUMNS)

        records = []
        now_ts = pd.to_datetime(datetime.now(UTC))

        for item in articles:
            url = item.get("url", "")
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            pub_at_str = item.get("publishedAt")
            published_at = pd.to_datetime(pub_at_str, utc=True) if pub_at_str else now_ts

            records.append(
                {
                    "asset": asset,
                    "published_at": published_at,
                    "url_hash": url_hash,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "source": item.get("source", {}).get("name", "Unknown"),
                    "url": url,
                    "ingested_at": now_ts,
                }
            )

        df = pd.DataFrame(records)
        return df[SILVER_NEWS_COLUMNS]

    def ingest_asset_news(
        self,
        asset: str,
        keywords: list[str],
    ) -> pd.DataFrame:
        """Ingest news headlines for an asset, partition by date, and save."""
        start_time = time.time()
        all_articles: list[dict[str, Any]] = []

        for kw in keywords:
            articles = self.fetch_news_raw(kw, max_articles=5)
            all_articles.extend(articles)

        silver_df = self.transform_to_silver(all_articles, asset=asset)
        if silver_df.empty:
            self.meta_logger.log_run(
                "ingest_news", "success", 0, duration_seconds=time.time() - start_time
            )
            return silver_df

        silver_df["_date"] = silver_df["published_at"].dt.strftime("%Y-%m-%d")
        for date_str, group in silver_df.groupby("_date"):
            clean_group = group.drop(columns=["_date"])
            rel_path = f"silver/news/asset={asset}/date={date_str}.parquet"
            if self.storage.exists(rel_path):
                existing = self.storage.read_parquet(rel_path)
                combined = pd.concat([existing, clean_group], ignore_index=True)
                combined = combined.drop_duplicates(subset=["url_hash"], keep="last")
            else:
                combined = clean_group.drop_duplicates(subset=["url_hash"], keep="last")
            self.storage.write_parquet(rel_path, combined)

        self.meta_logger.log_run(
            "ingest_news",
            "success",
            len(silver_df),
            duration_seconds=time.time() - start_time,
        )
        return silver_df
