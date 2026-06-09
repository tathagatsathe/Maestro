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
    database_url: str = "postgresql+asyncpg://agent:agentpass@localhost:5432/agentdb"
    api_key: str = "dev-api-key-change-me"
    rate_limit_runs_per_minute: int = 10
    quality_threshold: float = 0.75
    max_retries: int = 3
    anthropic_model: str = "claude-sonnet-4-6"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "maestro"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    llm_input_cost_per_million: float = 3.0
    llm_output_cost_per_million: float = 15.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    job_queue_key: str = "maestro:jobs"
    run_events_stream_prefix: str = "maestro:events"


@lru_cache
def get_settings() -> Settings:
    return Settings()
