from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM Server & Model Settings
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the OpenAI-compatible vLLM server endpoint",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for authentication with the vLLM server",
    )
    model_name: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Identifier of the model to query",
    )

    # Number rules:
    # gt: strictly greater than (>)
    # ge: at least / greater than or equal to (>=)
    # le: at most / less than or equal to (<=)
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature: at least 0.0 (ge=0.0), at most 2.0 (le=2.0)",
    )
    top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability: greater than 0.0 (gt=0.0), at most 1.0 (le=1.0)",
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        le=32768,
        description="Maximum tokens to generate: greater than 0 (gt=0), at most 32768 (le=32768)",
    )
    timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="HTTP request timeout in seconds: greater than 0.0 (gt=0.0)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts: at least 0 (ge=0), at most 10 (le=10)",
    )

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


# Default singleton settings instance
settings = get_settings()
