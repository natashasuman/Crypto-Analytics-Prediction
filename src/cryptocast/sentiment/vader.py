import re
from dataclasses import dataclass

# Fallback financial lexicon dictionary for fast CPU sentiment scoring
POS_WORDS = {
    "bullish",
    "surge",
    "gain",
    "profit",
    "rally",
    "high",
    "breakout",
    "jump",
    "growth",
    "record",
    "boom",
    "soar",
    "positive",
    "win",
    "adoption",
    "inflow",
    "upward",
    "upgrade",
    "outperform",
    "success",
    "buy",
    "accumulate",
    "pump",
}
NEG_WORDS = {
    "bearish",
    "crash",
    "loss",
    "plunge",
    "drop",
    "low",
    "collapse",
    "dump",
    "decline",
    "hack",
    "scam",
    "fraud",
    "fall",
    "negative",
    "outflow",
    "downward",
    "ban",
    "liquidation",
    "selloff",
    "risk",
    "warning",
    "underperform",
    "fail",
}


@dataclass
class SentimentResult:
    pos: float
    neg: float
    neu: float
    score: float  # pos - neg in [-1.0, 1.0]


class VaderSentimentScorer:
    """Lightweight lexicon-based sentiment scorer with financial keyword weighting."""

    def score_text(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(pos=0.0, neg=0.0, neu=1.0, score=0.0)

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return SentimentResult(pos=0.0, neg=0.0, neu=1.0, score=0.0)

        pos_count = sum(1 for w in words if w in POS_WORDS)
        neg_count = sum(1 for w in words if w in NEG_WORDS)
        total = len(words)

        pos = min(pos_count / max(total, 1) * 3.0, 1.0)
        neg = min(neg_count / max(total, 1) * 3.0, 1.0)
        neu = max(1.0 - (pos + neg), 0.0)

        # Normalize so pos + neg + neu == 1.0
        s = pos + neg + neu
        if s > 0:
            pos /= s
            neg /= s
            neu /= s
        score = max(min(pos - neg, 1.0), -1.0)
        return SentimentResult(
            pos=round(pos, 4), neg=round(neg, 4), neu=round(neu, 4), score=round(score, 4)
        )
