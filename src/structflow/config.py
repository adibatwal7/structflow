from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="STRUCTFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM Server & Model Settings
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible vLLM server endpoint",
    )
    vllm_api_key: SecretStr = Field(
        default=SecretStr("EMPTY"),
        description="API key for authentication with the vLLM server",
    )
    model_name: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Identifier of the model to query",
    )

    # Phase 2 Operational & Batching Settings:
    # Units in names: request_timeout_s, microbatch_max_wait_ms
    request_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="HTTP request timeout in seconds: strictly greater than 0.0 (gt=0.0)",
    )
    max_inflight_generation: int = Field(
        default=64,
        gt=0,
        description="Maximum concurrent generation requests in flight: strictly greater than 0 (gt=0)",
    )
    microbatch_max_size: int = Field(
        default=8,
        gt=0,
        description="Maximum microbatch size: strictly greater than 0 (gt=0)",
    )
    microbatch_max_wait_ms: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Maximum wait time in ms before dispatching microbatch: at least 0 (ge=0), at most 100 (le=100)",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("vllm_base_url")
    @classmethod
    def validate_vllm_base_url(cls, v: str) -> str:
        """Validate and normalize vllm_base_url:

        1. Reject values that do not start with 'http://' or 'https://'.
        2. Strip any trailing '/' so downstream endpoints (e.g. /chat/completions)
           do not produce malformed double slashes ('//chat/completions').
        """
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                "vllm_base_url must start with 'http://' or 'https://'"
            )
        return v.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
