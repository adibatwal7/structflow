import pytest
from pydantic import ValidationError

from structflow.config import Settings


class TestVllmBaseUrlValidator:
    def test_valid_http_url_without_trailing_slash(self):
        s = Settings(vllm_base_url="http://localhost:8000/v1")
        assert s.vllm_base_url == "http://localhost:8000/v1"

    def test_valid_https_url_without_trailing_slash(self):
        s = Settings(vllm_base_url="https://api.vllm.ai/v1")
        assert s.vllm_base_url == "https://api.vllm.ai/v1"

    def test_trailing_slash_is_stripped(self):
        s = Settings(vllm_base_url="http://localhost:8000/v1/")
        assert s.vllm_base_url == "http://localhost:8000/v1"
        # Verify endpoint concatenation does not produce double slashes
        endpoint = f"{s.vllm_base_url}/chat/completions"
        assert endpoint == "http://localhost:8000/v1/chat/completions"

    def test_multiple_trailing_slashes_stripped(self):
        s = Settings(vllm_base_url="https://api.vllm.ai/v1///")
        assert s.vllm_base_url == "https://api.vllm.ai/v1"

    def test_rejects_missing_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(vllm_base_url="localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)

    def test_rejects_non_http_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(vllm_base_url="ftp://localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)

    def test_rejects_ws_protocol(self):
        with pytest.raises(ValidationError) as exc_info:
            Settings(vllm_base_url="ws://localhost:8000/v1")
        assert "must start with 'http://' or 'https://'" in str(exc_info.value)


class TestNumberRules:
    # temperature: ge=0.0, le=2.0
    def test_temperature_within_range(self):
        s = Settings(temperature=0.0)
        assert s.temperature == 0.0
        s = Settings(temperature=2.0)
        assert s.temperature == 2.0

    def test_temperature_below_minimum(self):
        with pytest.raises(ValidationError):
            Settings(temperature=-0.1)

    def test_temperature_above_maximum(self):
        with pytest.raises(ValidationError):
            Settings(temperature=2.1)

    # top_p: gt=0.0, le=1.0
    def test_top_p_within_range(self):
        s = Settings(top_p=0.01)
        assert s.top_p == 0.01
        s = Settings(top_p=1.0)
        assert s.top_p == 1.0

    def test_top_p_boundary_zero_rejected(self):
        # gt=0.0 means 0.0 is rejected
        with pytest.raises(ValidationError):
            Settings(top_p=0.0)

    def test_top_p_above_maximum(self):
        with pytest.raises(ValidationError):
            Settings(top_p=1.1)

    # max_tokens: gt=0
    def test_max_tokens_positive(self):
        s = Settings(max_tokens=1)
        assert s.max_tokens == 1

    def test_max_tokens_zero_or_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(max_tokens=0)
        with pytest.raises(ValidationError):
            Settings(max_tokens=-10)

    # timeout: gt=0.0
    def test_timeout_positive(self):
        s = Settings(timeout=0.5)
        assert s.timeout == 0.5

    def test_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            Settings(timeout=0.0)

    # max_retries: ge=0, le=10
    def test_max_retries_range(self):
        s = Settings(max_retries=0)
        assert s.max_retries == 0
        s = Settings(max_retries=10)
        assert s.max_retries == 10

    def test_max_retries_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(max_retries=-1)

    def test_max_retries_excessive_rejected(self):
        with pytest.raises(ValidationError):
            Settings(max_retries=11)


class TestRootConfigReExport:
    def test_root_config_imports(self):
        import config

        assert config.Settings is Settings
        assert config.settings is not None
