from dataclasses import dataclass

SILVER_KLINES_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "close_time",
    "ingested_at",
]

SILVER_KLINES_DTYPES: dict[str, str] = {
    "symbol": "object",
    "open_time": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "trades": "int64",
    "taker_buy_base": "float64",
    "taker_buy_quote": "float64",
    "close_time": "datetime64[ns, UTC]",
    "ingested_at": "datetime64[ns, UTC]",
}

SILVER_NEWS_COLUMNS = [
    "asset",
    "published_at",
    "url_hash",
    "title",
    "description",
    "source",
    "url",
    "ingested_at",
]


@dataclass
class DQResult:
    """Data Quality check outcome."""

    check_name: str
    passed: bool
    severity: str  # "error" or "warn"
    details: str
    timestamp: str
