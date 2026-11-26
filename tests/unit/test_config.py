from pathlib import Path

from cryptocast.config import get_settings


def test_default_settings():
    settings = get_settings()
    assert settings.storage_backend in ["local", "hf_hub"]
    assert settings.binance_base_url == "https://data-api.binance.vision"


def test_assets_config(tmp_path: Path):
    settings = get_settings()
    assets_cfg = settings.get_assets_config()

    assert assets_cfg.interval == "1h"
    assert 4 in assets_cfg.horizons
    assert "BTCUSDT" in assets_cfg.assets
    assert "ETHUSDT" in assets_cfg.assets
    assert "SOLUSDT" in assets_cfg.assets
    assert assets_cfg.assets["BTCUSDT"].is_benchmark is True


def test_features_config():
    settings = get_settings()
    features_cfg = settings.get_features_config()
    assert "returns" in features_cfg.groups
    assert "volatility" in features_cfg.groups
    assert "trend" in features_cfg.groups


def test_model_and_promotion_config():
    settings = get_settings()
    model_cfg = settings.get_model_config()
    promotion_cfg = settings.get_promotion_config()

    assert "xgboost" in model_cfg.models
    assert promotion_cfg.primary_metric_direction == "mcc"
    assert promotion_cfg.primary_metric_return == "spearman_ic"
