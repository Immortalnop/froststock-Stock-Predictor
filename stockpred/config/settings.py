"""Typed configuration. Override any field with env var STOCKPRED_<NAME> or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STOCKPRED_", env_file=".env", extra="ignore"
    )

    # --- storage ---
    data_dir: Path = Path("data_store")
    artifacts_dir: Path = Path("artifacts")
    runs_dir: Path = Path("runs")

    # --- LLM ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 120.0
    llm_max_concurrent: int = 2
    llm_samples: int = 1  # >1 enables self-consistency (temperature is raised for samples>1)
    llm_max_retries: int = 3
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"

    # --- news ---
    user_agent: str = "stockpred/0.1 (research; set STOCKPRED_USER_AGENT with contact email)"
    news_interval_minutes: int = 5
    max_article_age_hours: float = 24.0
    max_articles_per_cycle: int = 60
    news_latency_minutes: int = 15  # assumed publish->available lag for backfilled articles
    dedup_threshold: float = 0.6  # MinHash Jaccard threshold for syndicated duplicates

    # --- modelling ---
    horizon_days: int = 5
    embargo_days: int = 2
    benchmark: str = "SPY"
    random_seed: int = 7

    # --- display ---
    dashboard_refresh_s: int = 30
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if self.llm_samples < 1:
            raise ValueError("llm_samples must be >= 1")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "stockpred.duckdb"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
