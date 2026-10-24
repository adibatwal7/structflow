import pytest
from pydantic import ValidationError

from structflow.config import Settings, get_settings


class TestVllmBaseUrlValidator:
    def test_valid_http_url_without_trailing_slash(self):
        s = Settings(_env_file=None, vllm_base_url="http://localhost:8000/v1")
        assert s.vllm_base_url == "http://localhost:8000/v1"

    def test_valid_https_url_without_trailing_slash(self):
        s = Settings(_env_file=None, vllm_base_url="https://api.vllm.ai/v1")
        assert s.vllm_base_url == "https://api.vllm.ai/v1"

    def test_trailing_slash_is_stripped(self):
        s = Settings(_env_file=None, vllm_base_url="http://localhost:8000/v1/")
        assert s.vllm_base_url == "http://localhost:8000/v1"
        # Verify endpoint concatenation does not produce double slashes
        endpoint = f"{s.vllm_base_url}/chat/completions"
        assert endpoint == "http://localhost:8000/v1/chat/completions"

    def test_multiple_trailing_slashes_stripped(self):
        s = Settings(_env_file=None, vllm_base_url="https://api.vllm.ai/v1///")
        assert s.vllm_base_url == "https://api.vllm.ai/v1"

    def test_rejects_missing_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, vllm_base_url="localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)

    def test_rejects_non_http_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, vllm_base_url="ftp://localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)

    def test_rejects_ws_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, vllm_base_url="ws://localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)


class TestNumberRules:
    # request_timeout_s: gt=0.0
    def test_request_timeout_s_positive(self):
        s = Settings(_env_file=None, request_timeout_s=0.5)
        assert s.request_timeout_s == 0.5

    def test_request_timeout_s_zero_or_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, request_timeout_s=0.0)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, request_timeout_s=-1.0)

    # max_inflight_generation: gt=0, default=64
    def test_max_inflight_generation_default(self):
        s = Settings(_env_file=None)
        assert s.max_inflight_generation == 64

    def test_max_inflight_generation_positive(self):
        s = Settings(_env_file=None, max_inflight_generation=1)
        assert s.max_inflight_generation == 1

    def test_max_inflight_generation_zero_or_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_inflight_generation=0)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_inflight_generation=-5)

    # microbatch_max_size: gt=0
    def test_microbatch_max_size_positive(self):
        s = Settings(_env_file=None, microbatch_max_size=1)
        assert s.microbatch_max_size == 1

    def test_microbatch_max_size_zero_or_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, microbatch_max_size=0)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, microbatch_max_size=-2)

    # microbatch_max_wait_ms: ge=0, le=100, default=5
    def test_microbatch_max_wait_ms_default(self):
        s = Settings(_env_file=None)
        assert s.microbatch_max_wait_ms == 5

    def test_microbatch_max_wait_ms_boundaries(self):
        s0 = Settings(_env_file=None, microbatch_max_wait_ms=0)
        assert s0.microbatch_max_wait_ms == 0
        s100 = Settings(_env_file=None, microbatch_max_wait_ms=100)
        assert s100.microbatch_max_wait_ms == 100

    def test_microbatch_max_wait_ms_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, microbatch_max_wait_ms=-1)

    def test_microbatch_max_wait_ms_above_cap_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, microbatch_max_wait_ms=101)


class TestLogLevelValidation:
    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            s = Settings(_env_file=None, log_level=level)  # type: ignore[arg-type]
            assert s.log_level == level

    def test_invalid_log_level_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, log_level="LOUD")  # type: ignore[arg-type]


class TestSecretStrAndEnvPrefix:
    def test_api_key_is_secret_str(self):
        s = Settings(_env_file=None, vllm_api_key="super-secret-token")
        # Ensure secret value is retrieved via get_secret_value()
        assert s.vllm_api_key.get_secret_value() == "super-secret-token"
        # Ensure representation obscures the raw secret
        assert "super-secret-token" not in str(s.vllm_api_key)
        assert "super-secret-token" not in repr(s)

    def test_default_api_key_secret(self):
        s = Settings(_env_file=None)
        assert s.vllm_api_key.get_secret_value() == "EMPTY"
        assert "EMPTY" not in str(s.vllm_api_key)

    def test_structflow_env_prefix(self, monkeypatch):
        monkeypatch.setenv("STRUCTFLOW_REQUEST_TIMEOUT_S", "120.0")
        monkeypatch.setenv("STRUCTFLOW_VLLM_BASE_URL", "https://api.vllm.ai/v1/")
        monkeypatch.setenv("STRUCTFLOW_VLLM_API_KEY", "env-secret-key")
        s = Settings(_env_file=None)
        assert s.request_timeout_s == 120.0
        assert s.vllm_base_url == "https://api.vllm.ai/v1"
        assert s.vllm_api_key.get_secret_value() == "env-secret-key"

    def test_generic_env_names_are_ignored(self, monkeypatch):
        # Without prefix, generic env names should not overwrite defaults
        monkeypatch.setenv("REQUEST_TIMEOUT_S", "1.9")
        s = Settings(_env_file=None)
        assert s.request_timeout_s == 60.0

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("STRUCTFLOW_MODEL_NAME", "test-model")
        get_settings.cache_clear()
        assert get_settings().model_name == "test-model"
        get_settings.cache_clear()
