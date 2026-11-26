from datetime import UTC, datetime

import pandas as pd

from cryptocast.config import get_settings
from cryptocast.features.groups.calendar import compute_calendar_features
from cryptocast.features.groups.cross_asset import compute_cross_asset_features
from cryptocast.features.groups.mean_reversion import compute_mean_reversion_features
from cryptocast.features.groups.momentum import compute_momentum_features
from cryptocast.features.groups.returns import compute_returns_features
from cryptocast.features.groups.volatility import compute_volatility_features
from cryptocast.features.groups.volume_flow import compute_volume_flow_features
from cryptocast.features.leakage_guard import compute_data_hash
from cryptocast.labels import compute_target_labels
from cryptocast.sentiment.aggregate import SentimentAggregator
from cryptocast.storage import Storage, get_storage


class FeaturePipeline:
    """Builds the comprehensive Gold feature store from Silver klines and sentiment."""

    def __init__(self, storage: Storage | None = None) -> None:
        self.settings = get_settings()
        self.storage = storage or get_storage()
        self.sentiment_agg = SentimentAggregator(storage=self.storage)

    def extract_features(
        self,
        klines_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None = None,
        scored_news_df: pd.DataFrame | None = None,
        symbol: str = "BTCUSDT",
    ) -> pd.DataFrame:
        """Extract all technical, flow, calendar, cross-asset, and sentiment features for klines."""
        if klines_df.empty:
            return pd.DataFrame()

        df = klines_df.sort_values("open_time").reset_index(drop=True)

        # 1. Technical feature groups
        f_returns = compute_returns_features(df)
        f_vol = compute_volatility_features(df)
        f_mom = compute_momentum_features(df)
        f_mr = compute_mean_reversion_features(df)
        f_flow = compute_volume_flow_features(df)
        f_cal = compute_calendar_features(df["open_time"])

        # 2. Cross-asset features
        bench = benchmark_df if benchmark_df is not None else df
        f_cross = compute_cross_asset_features(df, bench, symbol=symbol)

        # 3. Sentiment features
        news = scored_news_df if scored_news_df is not None else pd.DataFrame()
        f_sent = self.sentiment_agg.compute_bar_sentiment_features(
            bar_times=df["open_time"],
            scored_news_df=news,
        ).drop(columns=["open_time"])

        # Combine all features along with key identifiers
        combined = pd.concat(
            [
                df[["symbol", "open_time", "close"]],
                f_returns,
                f_vol,
                f_mom,
                f_mr,
                f_flow,
                f_cross,
                f_cal,
                f_sent,
            ],
            axis=1,
        )
        return combined

    def build_and_save_features(
        self,
        symbol: str,
        horizon: int = 4,
    ) -> pd.DataFrame:
        """Load silver data, construct Gold features with target labels, and persist."""
        # Find all silver klines files for this symbol
        silver_prefix = f"silver/klines/symbol={symbol}/interval=1h"
        kline_files = self.storage.list_files(silver_prefix)

        if not kline_files:
            return pd.DataFrame()

        kline_dfs = [self.storage.read_parquet(f) for f in kline_files]
        klines_df = pd.concat(kline_dfs, ignore_index=True)
        klines_df = (
            klines_df.drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        # Load benchmark (BTC) if needed
        benchmark_df: pd.DataFrame | None = None
        if symbol != "BTCUSDT":
            btc_prefix = "silver/klines/symbol=BTCUSDT/interval=1h"
            btc_files = self.storage.list_files(btc_prefix)
            if btc_files:
                btc_dfs = [self.storage.read_parquet(f) for f in btc_files]
                benchmark_df = pd.concat(btc_dfs, ignore_index=True)
                benchmark_df = benchmark_df.drop_duplicates(subset=["open_time"]).sort_values(
                    "open_time"
                )

        # Load scored news if available
        news_df: pd.DataFrame | None = None
        assets_cfg = self.settings.get_assets_config()
        asset_info = assets_cfg.assets.get(symbol)
        if asset_info:
            news_prefix = f"silver/news/asset={asset_info.base_asset}"
            news_files = self.storage.list_files(news_prefix)
            if news_files:
                dfs = [self.storage.read_parquet(f) for f in news_files]
                raw_news = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["url_hash"])
                news_df = self.sentiment_agg.score_silver_news(raw_news)

        # Compute full features
        features_df = self.extract_features(
            klines_df=klines_df,
            benchmark_df=benchmark_df,
            scored_news_df=news_df,
            symbol=symbol,
        )

        # Append forward targets
        labels_df = compute_target_labels(klines_df, horizon=horizon)
        gold_df = pd.concat([features_df, labels_df], axis=1)

        # Attach metadata lineage hash
        gold_df.attrs["data_hash"] = compute_data_hash(klines_df)
        gold_df.attrs["created_at"] = datetime.now(UTC).isoformat()

        # Save to Gold storage
        out_path = f"gold/features/symbol={symbol}/horizon={horizon}.parquet"
        self.storage.write_parquet(out_path, gold_df)
        return gold_df
