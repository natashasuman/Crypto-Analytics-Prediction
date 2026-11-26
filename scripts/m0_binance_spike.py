"""M0 Feasibility Spike: Binance Market Data Ingestion.

Tests connectivity to Binance's public market data endpoint (data-api.binance.vision)
which avoids US IP geo-blocking (HTTP 451) that occurs on api.binance.com.
"""

import sys
import requests
import pandas as pd


def test_binance_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 10) -> bool:
    host = "https://data-api.binance.vision"
    endpoint = f"{host}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    print(f"Connecting to {endpoint} for {symbol} ({interval})...")
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        status_code = response.status_code
        print(f"HTTP Status Code: {status_code}")

        if status_code == 451:
            print("ERROR: Received HTTP 451 Geo-blocked! The runner IP is blocked.")
            return False
        response.raise_for_status()

        raw_klines = response.json()
        print(f"Successfully fetched {len(raw_klines)} candles.")

        # Binance Kline schema:
        # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume,
        # 6: Close time, 7: Quote asset volume, 8: Number of trades,
        # 9: Taker buy base asset volume, 10: Taker buy quote asset volume, 11: Ignore
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(raw_klines, columns=columns)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        print("\nLatest candle preview:")
        print(df[["open_time", "close", "volume", "trades"]].tail(2))
        return True

    except Exception as e:
        print(f"Failed to fetch klines: {e}")
        return False


if __name__ == "__main__":
    success = test_binance_klines()
    sys.exit(0 if success else 1)
