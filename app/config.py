from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://agent:agentpass@localhost:5432/agentdb"
    quality_threshold: float = 0.75
    max_retries: int = 3
    anthropic_model: str = "claude-3-5-sonnet-20241022"


@lru_cache
def get_settings() -> Settings:
    return Settings()
