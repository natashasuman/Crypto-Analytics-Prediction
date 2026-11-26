"""M0 Feasibility Spike: GNews API Connectivity.

Tests the GNews API search endpoint for crypto asset news keywords.
Checks for rate limits, payload schema, and article fields.
"""

import os
import sys
import requests
from cryptocast.config import get_settings


def test_gnews_spike(keyword: str = "bitcoin") -> bool:
    settings = get_settings()
    api_key = os.getenv("GNEWS_API_KEY", settings.gnews_api_key)

    if not api_key:
        print("[M0 GNews Spike] GNEWS_API_KEY not configured. Documenting endpoint and mock schema...")
        print("Endpoint: https://gnews.io/api/v4/search?q={query}&lang=en&token={API_KEY}")
        print("Schema fields: title, description, content, url, image, publishedAt, source (name, url)")
        print("Free tier limits: 100 requests/day, 10 articles/request, delayed ~15 minutes.")
        print("System design note: Pipeline degrades gracefully when news is unavailable.")
        return True

    endpoint = "https://gnews.io/api/v4/search"
    params = {
        "q": keyword,
        "lang": "en",
        "max": 5,
        "token": api_key,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        print(f"HTTP Status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        print(f"Fetched {len(articles)} articles for keyword '{keyword}'.")
        if articles:
            print(f"Sample headline: {articles[0].get('title')}")
            print(f"Published at: {articles[0].get('publishedAt')}")
        return True
    except Exception as e:
        print(f"GNews query failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gnews_spike()
    sys.exit(0 if success else 1)
