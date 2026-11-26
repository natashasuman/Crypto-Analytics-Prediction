from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssetConfig(BaseModel):
    base_asset: str
    quote_asset: str
    news_keywords: list[str] = Field(default_factory=list)
    is_benchmark: bool = False


class AssetsSettings(BaseModel):
    interval: str = "1h"
    horizons: list[int] = Field(default_factory=lambda: [1, 4, 24])
    primary_horizon: int = 4
    fee_bps_per_side: int = 10
    assets: dict[str, AssetConfig] = Field(default_factory=dict)


class FeatureSettings(BaseModel):
    groups: dict[str, Any] = Field(default_factory=dict)


class ModelSettings(BaseModel):
    models: dict[str, Any] = Field(default_factory=dict)
    tuning: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    signals: dict[str, Any] = Field(default_factory=dict)


class PromotionSettings(BaseModel):
    primary_metric_direction: str = "mcc"
    primary_metric_return: str = "spearman_ic"
    bootstrap_confidence: float = 0.95
    min_beat_baseline_folds_frac: float = 0.70
    max_ece: float = 0.05
    fee_bps_per_side: int = 10
    retrain: dict[str, Any] = Field(default_factory=dict)
    rollback: dict[str, Any] = Field(default_factory=dict)


class DQSettings(BaseModel):
    checks: dict[str, Any] = Field(default_factory=dict)


class AppSettings(BaseSettings):
    """Global application settings combining env vars and YAML configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    storage_backend: str = "local"  # "local" or "hf_hub"
    local_storage_dir: str = "data"
    hf_repo_id: str | None = None
    hf_token: str | None = None

    # External APIs
    gnews_api_key: str | None = None
    dune_api_key: str | None = None
    alert_webhook_url: str | None = None
    binance_base_url: str = "https://data-api.binance.vision"

    # Tracking & paths
    mlflow_tracking_uri: str = "mlruns"
    config_dir: str = "config"

    def load_yaml(self, filename: str) -> dict[str, Any]:
        path = Path(self.config_dir) / filename
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_assets_config(self) -> AssetsSettings:
        data = self.load_yaml("assets.yaml")
        return AssetsSettings(**data)

    def get_features_config(self) -> FeatureSettings:
        data = self.load_yaml("features.yaml")
        return FeatureSettings(**data)

    def get_model_config(self) -> ModelSettings:
        data = self.load_yaml("model.yaml")
        return ModelSettings(**data)

    def get_promotion_config(self) -> PromotionSettings:
        data = self.load_yaml("promotion.yaml")
        return PromotionSettings(**data)

    def get_dq_config(self) -> DQSettings:
        data = self.load_yaml("dq.yaml")
        return DQSettings(**data)


_settings_instance: AppSettings | None = None


def get_settings() -> AppSettings:
    """Singleton getter for AppSettings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance
