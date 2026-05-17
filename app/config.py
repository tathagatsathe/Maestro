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
    anthropic_model: str = "claude-sonnet-4-6"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "maestro"
    langchain_endpoint: str = "https://api.smith.langchain.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
