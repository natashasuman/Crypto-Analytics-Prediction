import pandas as pd

from cryptocast.sentiment.aggregate import SentimentAggregator
from cryptocast.sentiment.finbert import FinBertSentimentScorer
from cryptocast.sentiment.vader import VaderSentimentScorer


def test_vader_scorer():
    scorer = VaderSentimentScorer()

    bullish = scorer.score_text("Bitcoin surges to record high breakout rally!")
    assert bullish.score > 0.2
    assert bullish.pos > bullish.neg

    bearish = scorer.score_text("Market crashes in massive plunge and selloff decline.")
    assert bearish.score < -0.2
    assert bearish.neg > bearish.pos

    neutral = scorer.score_text("Weekly report on market volume and kline data.")
    assert abs(neutral.score) <= 0.2


def test_finbert_fallback():
    scorer = FinBertSentimentScorer(force_fallback=True)
    res = scorer.score_text("Ethereum rally brings huge gains.")
    assert res.score > 0.0
    assert 0.0 <= res.pos <= 1.0


def test_sentiment_aggregator_empty_news():
    agg = SentimentAggregator(use_finbert=False)
    bar_times = pd.date_range("2026-11-26", periods=5, freq="1h", tz="UTC")
    empty_news = pd.DataFrame()

    feats = agg.compute_bar_sentiment_features(bar_times, empty_news)
    assert len(feats) == 5
    assert (feats["news_available"] == 0).all()
    assert (feats["news_count_24h"] == 0).all()


def test_sentiment_aggregator_point_in_time():
    agg = SentimentAggregator(use_finbert=False)
    bar_times = pd.date_range("2026-11-26 12:00:00", periods=3, freq="1h", tz="UTC")

    news_df = pd.DataFrame(
        [
            {
                "published_at": pd.Timestamp("2026-11-26 11:30:00", tz="UTC"),
                "sent_pos": 0.8,
                "sent_neg": 0.1,
                "sent_neu": 0.1,
                "sent_score": 0.7,
            },
            {
                "published_at": pd.Timestamp(
                    "2026-11-26 13:30:00", tz="UTC"
                ),  # Future for first two bars!
                "sent_pos": 0.1,
                "sent_neg": 0.8,
                "sent_neu": 0.1,
                "sent_score": -0.7,
            },
        ]
    )

    feats = agg.compute_bar_sentiment_features(bar_times, news_df)
    # At 12:00 (first bar), only first news item was available
    assert feats.iloc[0]["news_count_24h"] == 1
    assert feats.iloc[0]["news_mean_sent_24h"] == 0.7
    # At 14:00 (third bar), both news items are available
    assert feats.iloc[2]["news_count_24h"] == 2
