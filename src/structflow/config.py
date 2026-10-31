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
    microbatch_max_queue: int = Field(
        default=256,
        gt=0,
        description="Requests allowed to wait for the batcher before new ones get a 503",
    )

    # Softmax head (Step 2.4)
    classifier_backend: Literal["fake", "hf"] = Field(
        default="fake",
        description="'fake' needs no model (tests, demos); 'hf' loads a real Hugging Face model",
    )
    classifier_model: str = Field(
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        description="Hugging Face model id (or local folder) for the softmax head",
    )
    classifier_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    classifier_max_length: int = Field(
        default=512, gt=0, description="Longer texts are truncated to this many tokens"
    )

    span_backend: Literal["fake", "hf"] = Field(
        default="fake",
        description="'fake' needs no model (tests, demos); 'hf' loads a real NER model",
    )
    span_model: str = Field(
        default="dslim/distilbert-NER",
        description="Hugging Face token-classification model id (or local folder)",
    )
    span_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    span_max_length: int = Field(
        default=512, gt=0, description="Longer texts are refused, never cut"
    )

    generator_backend: Literal["fake", "vllm"] = Field(
        default="fake",
        description="'fake' needs no GPU; 'vllm' calls vLLM's OpenAI-compatible server",
    )

    fake_generation_delay_ms: int = Field(
        default=0,
        ge=0,
        description="Make the fake LLM this slow, to see limits and timeouts locally",
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
            raise ValueError("vllm_base_url must start with 'http://' or 'https://'")
        return v.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
