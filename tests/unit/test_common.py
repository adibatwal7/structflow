import json

import pytest
from pydantic import TypeAdapter, ValidationError

from structflow.schemas.common import (
    MAX_GENERATION_TOKENS,
    MAX_LABEL_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_STOP_CHARS,
    MAX_STOP_SEQUENCES,
    MAX_TEXT_CHARS,
    ChatMessage,
    FinishReason,
    NonBlankLabel,
    NonBlankText,
    Route,
    SamplingParams,
    StopSequence,
    StrictModel,
    TaskType,
)

text_adapter = TypeAdapter(NonBlankText)
label_adapter = TypeAdapter(NonBlankLabel)
stop_adapter = TypeAdapter(StopSequence)


class TestNonBlankText:
    def test_accepts_normal_text(self):
        assert text_adapter.validate_python("hello") == "hello"

    def test_preserves_surrounding_whitespace(self):
        assert text_adapter.validate_python("  hi  ") == "  hi  "

    def test_rejects_empty(self):
        with pytest.raises(ValidationError):
            text_adapter.validate_python("")

    @pytest.mark.parametrize("value", [" ", "\t", "\n  \n"])
    def test_rejects_whitespace_only(self, value):
        with pytest.raises(ValidationError) as exc_info:
            text_adapter.validate_python(value)
        assert "non-whitespace" in str(exc_info.value)

    def test_accepts_max_length(self):
        assert len(text_adapter.validate_python("a" * MAX_TEXT_CHARS)) == MAX_TEXT_CHARS

    def test_rejects_over_max_length(self):
        with pytest.raises(ValidationError):
            text_adapter.validate_python("a" * (MAX_TEXT_CHARS + 1))


class TestNonBlankLabel:
    def test_accepts_max_length(self):
        assert (
            label_adapter.validate_python("a" * MAX_LABEL_CHARS)
            == "a" * MAX_LABEL_CHARS
        )

    def test_rejects_over_max_length(self):
        with pytest.raises(ValidationError):
            label_adapter.validate_python("a" * (MAX_LABEL_CHARS + 1))

    def test_rejects_whitespace_only(self):
        with pytest.raises(ValidationError):
            label_adapter.validate_python("   ")


class TestStopSequence:
    def test_allows_whitespace_only(self):
        # Stop sequences like "\n" are legitimate, so blank is allowed
        assert stop_adapter.validate_python("\n") == "\n"

    def test_rejects_empty(self):
        with pytest.raises(ValidationError):
            stop_adapter.validate_python("")

    def test_rejects_over_max_length(self):
        with pytest.raises(ValidationError):
            stop_adapter.validate_python("a" * (MAX_STOP_CHARS + 1))


class TestStrictModel:
    class Sample(StrictModel):
        name: str

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            self.Sample(name="x", extra="nope")

    def test_is_frozen(self):
        obj = self.Sample(name="x")
        with pytest.raises(ValidationError):
            obj.name = "y"


class TestEnums:
    def test_task_type_values(self):
        assert {t.value for t in TaskType} == {
            "classify",
            "extract_spans",
            "structured",
            "generate",
        }

    def test_route_values(self):
        assert {r.value for r in Route} == {
            "softmax_head",
            "span_head",
            "llm_label_scoring",
            "constrained_generation",
            "unconstrained_generation",
        }

    def test_finish_reason_values(self):
        assert {f.value for f in FinishReason} == {"stop", "length"}

    def test_str_enum_compares_to_string(self):
        assert TaskType.CLASSIFY == "classify"
        assert TaskType("generate") is TaskType.GENERATE


class TestChatMessage:
    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_valid_roles(self, role):
        msg = ChatMessage(role=role, content="hi")
        assert msg.role == role

    def test_rejects_unknown_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="tool", content="hi")

    def test_rejects_blank_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="   ")

    def test_rejects_over_max_length(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="a" * (MAX_MESSAGE_CHARS + 1))

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="hi", name="bob")


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 0.0
        assert p.top_p == 1.0
        assert p.max_tokens == 512
        assert p.seed is None
        assert p.stop == ()

    @pytest.mark.parametrize("value", [-0.1, 2.1])
    def test_temperature_bounds(self, value):
        with pytest.raises(ValidationError):
            SamplingParams(temperature=value)

    def test_top_p_must_be_positive(self):
        with pytest.raises(ValidationError):
            SamplingParams(top_p=0.0)

    def test_top_p_upper_bound(self):
        with pytest.raises(ValidationError):
            SamplingParams(top_p=1.01)

    def test_max_tokens_bounds(self):
        assert (
            SamplingParams(max_tokens=MAX_GENERATION_TOKENS).max_tokens
            == MAX_GENERATION_TOKENS
        )
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=0)
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=MAX_GENERATION_TOKENS + 1)

    def test_seed_rejects_negative(self):
        with pytest.raises(ValidationError):
            SamplingParams(seed=-1)

    def test_stop_accepts_list_as_tuple(self):
        p = SamplingParams(stop=["\n", "END"])
        assert p.stop == ("\n", "END")

    def test_stop_rejects_too_many(self):
        with pytest.raises(ValidationError):
            SamplingParams(stop=["x"] * (MAX_STOP_SEQUENCES + 1))

    def test_stop_rejects_empty_sequence(self):
        with pytest.raises(ValidationError):
            SamplingParams(stop=[""])

    @pytest.mark.parametrize(
        "payload",
        ['{"seed": true}', '{"max_tokens": "512"}', '{"temperature": "0.7"}'],
    )
    def test_rejects_loosely_typed_numbers(self, payload: str) -> None:
        with pytest.raises(ValidationError):
            SamplingParams.model_validate_json(payload)

    def test_int_temperature_is_fine(self) -> None:
        assert (
            SamplingParams.model_validate_json('{"temperature": 1}').temperature == 1.0
        )

    def test_json_stop_becomes_tuple(self) -> None:
        assert SamplingParams.model_validate_json('{"stop": ["\\n"]}').stop == ("\n",)

    def test_enum_serializes_as_plain_string(self) -> None:
        assert json.dumps(TaskType.CLASSIFY) == '"classify"'
