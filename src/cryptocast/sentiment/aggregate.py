import numpy as np
import pandas as pd

from cryptocast.sentiment.finbert import FinBertSentimentScorer
from cryptocast.storage import Storage, get_storage


class SentimentAggregator:
    """Scores silver news articles and builds rolling sentiment aggregates for bar t."""

    def __init__(
        self,
        storage: Storage | None = None,
        use_finbert: bool = False,
    ) -> None:
        self.storage = storage or get_storage()
        self.scorer = FinBertSentimentScorer(force_fallback=not use_finbert)

    def score_silver_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add sent_pos, sent_neg, sent_neu, and sent_score columns to news DataFrame."""
        if news_df.empty:
            cols = list(news_df.columns) + ["sent_pos", "sent_neg", "sent_neu", "sent_score"]
            return pd.DataFrame(columns=cols)

        scored = news_df.copy()
        pos_list, neg_list, neu_list, score_list = [], [], [], []

        for title in scored["title"]:
            res = self.scorer.score_text(str(title))
            pos_list.append(res.pos)
            neg_list.append(res.neg)
            neu_list.append(res.neu)
            score_list.append(res.score)

        scored["sent_pos"] = pos_list
        scored["sent_neg"] = neg_list
        scored["sent_neu"] = neu_list
        scored["sent_score"] = score_list
        return scored

    def compute_bar_sentiment_features(
        self,
        bar_times: pd.Series,
        scored_news_df: pd.DataFrame,
        half_life_hours: float = 12.0,
    ) -> pd.DataFrame:
        """Compute point-in-time rolling sentiment features for each bar close time.

        Strictly enforces no-lookahead: only news where published_at <= bar_time is included.
        """
        results = []
        decay_constant = np.log(2) / half_life_hours

        if scored_news_df.empty:
            for bt in bar_times:
                results.append(
                    {
                        "open_time": bt,
                        "news_count_6h": 0,
                        "news_count_24h": 0,
                        "news_mean_sent_24h": 0.0,
                        "news_max_sent_24h": 0.0,
                        "news_decayed_sent": 0.0,
                        "news_negative_ratio_24h": 0.0,
                        "news_available": 0,
                    }
                )
            return pd.DataFrame(results)

        news = scored_news_df.copy()
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
        news = news.sort_values("published_at").reset_index(drop=True)

        for bt in bar_times:
            bt_ts = pd.to_datetime(bt, utc=True)
            # Strict point-in-time filter: news available at or before bar close
            past_news = news[news["published_at"] <= bt_ts]
            news_24h = past_news[past_news["published_at"] >= (bt_ts - pd.Timedelta(hours=24))]
            news_6h = past_news[past_news["published_at"] >= (bt_ts - pd.Timedelta(hours=6))]

            cnt_24 = len(news_24h)
            cnt_6 = len(news_6h)

            if cnt_24 == 0:
                results.append(
                    {
                        "open_time": bt,
                        "news_count_6h": 0,
                        "news_count_24h": 0,
                        "news_mean_sent_24h": 0.0,
                        "news_max_sent_24h": 0.0,
                        "news_decayed_sent": 0.0,
                        "news_negative_ratio_24h": 0.0,
                        "news_available": 0,
                    }
                )
            else:
                scores = news_24h["sent_score"].values
                mean_sent = float(np.mean(scores))
                max_sent = float(np.max(scores))
                neg_ratio = float(np.mean(news_24h["sent_neg"].values > 0.5))

                # Exponential decay calculation based on elapsed hours
                elapsed_hours = (
                    bt_ts - news_24h["published_at"]
                ).dt.total_seconds().values / 3600.0
                weights = np.exp(-decay_constant * elapsed_hours)
                decayed_sent = float(np.sum(scores * weights) / max(np.sum(weights), 1e-6))

                results.append(
                    {
                        "open_time": bt,
                        "news_count_6h": cnt_6,
                        "news_count_24h": cnt_24,
                        "news_mean_sent_24h": round(mean_sent, 4),
                        "news_max_sent_24h": round(max_sent, 4),
                        "news_decayed_sent": round(decayed_sent, 4),
                        "news_negative_ratio_24h": round(neg_ratio, 4),
                        "news_available": 1,
                    }
                )

        return pd.DataFrame(results)
