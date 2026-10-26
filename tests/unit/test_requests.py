import typing

import pytest
from pydantic import BaseModel, ValidationError

from structflow.schemas.common import TaskType
from structflow.schemas.requests import (
    INFERENCE_REQUEST_ADAPTER,
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    StructuredRequest,
)


class Invoice(BaseModel):
    amount: float
    due_date: str


def test_structured_request_survives_json_round_trip() -> None:
    req = INFERENCE_REQUEST_ADAPTER.validate_python(
        {
            "task": "structured",
            "messages": [{"role": "user", "content": "Extract the invoice"}],
            "constraint": {
                "type": "json_schema",
                "schema": Invoice.model_json_schema(),
            },
        }
    )
    as_json = req.model_dump_json(by_alias=True)
    assert INFERENCE_REQUEST_ADAPTER.validate_json(as_json) == req


class TestTaskDiscrimination:
    def test_classify_request_parses_correctly(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "classify",
                "text": "Hello world",
                "labels": ["positive", "negative"],
            }
        )
        assert isinstance(req, ClassifyRequest)

    def test_extract_spans_request_parses_correctly(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "extract_spans",
                "text": "Alice met Bob",
                "entity_types": ["PERSON"],
            }
        )
        assert isinstance(req, ExtractSpansRequest)

    def test_structured_request_parses_correctly(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "structured",
                "messages": [{"role": "user", "content": "Extract the invoice"}],
                "constraint": {
                    "type": "json_schema",
                    "schema": Invoice.model_json_schema(),
                },
            }
        )
        assert isinstance(req, StructuredRequest)

    def test_generate_request_parses_correctly(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "generate",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        assert isinstance(req, GenerateRequest)

    def test_unknown_task_gives_union_tag_invalid(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {"task": "invalid", "text": "test"}
            )
        errors = exc_info.value.errors()
        assert errors[0]["type"] == "union_tag_invalid"
        assert "classify" in str(exc_info.value)

    def test_tags_match_task_type_values(self) -> None:
        models = (
            ClassifyRequest,
            ExtractSpansRequest,
            StructuredRequest,
            GenerateRequest,
        )
        tags = {typing.get_args(m.model_fields["task"].annotation)[0] for m in models}
        assert tags == {t.value for t in TaskType}


class TestClassifyRequest:
    def test_one_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {"task": "classify", "text": "test", "labels": ["only_one"]}
            )

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": ["a", "b", "a"],
                }
            )
        assert "unique" in str(exc_info.value)

    def test_257_labels_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": [str(i) for i in range(257)],
                }
            )

    def test_top_k_larger_than_labels_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": ["a", "b"],
                    "top_k": 3,
                }
            )
        assert "top_k" in str(exc_info.value)

    def test_top_k_equal_to_labels_passes(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "classify",
                "text": "test",
                "labels": ["a", "b"],
                "top_k": 2,
            }
        )
        assert req.top_k == 2

    def test_threshold_without_multi_label_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": ["a", "b"],
                    "threshold": 0.5,
                }
            )
        assert "threshold" in str(exc_info.value)

    def test_threshold_with_multi_label_passes(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "classify",
                "text": "test",
                "labels": ["a", "b"],
                "multi_label": True,
                "threshold": 0.5,
            }
        )
        assert req.threshold == 0.5


class TestChatRequests:
    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {"task": "generate", "messages": []}
            )

    def test_conversation_ending_with_assistant_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "generate",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                }
            )
        assert "last message" in str(exc_info.value)

    def test_system_message_in_middle_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "generate",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "system", "content": "Be helpful"},
                        {"role": "user", "content": "Hi"},
                    ],
                }
            )
        assert "system message" in str(exc_info.value)

    def test_system_then_user_passes(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "generate",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
            }
        )
        assert len(req.messages) == 2

    def test_default_sampling_temperature(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "generate",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        assert req.sampling.temperature == 0.0


class TestStructuredRequest:
    def test_bad_regex_reports_exact_location(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "structured",
                    "messages": [{"role": "user", "content": "test"}],
                    "constraint": {"type": "regex", "pattern": "(?unclosed"},
                }
            )
        errors = exc_info.value.errors()
        # The error location should include the path to the pattern field
        loc = errors[0]["loc"]
        assert loc == ("structured", "constraint", "regex", "pattern")


class TestMetadataAndRequestId:
    def test_request_id_with_space_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": ["a", "b"],
                    "request_id": "req 42",
                }
            )

    def test_17_metadata_items_rejected(self) -> None:
        metadata = {f"key{i}": "value" for i in range(17)}
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "classify",
                    "text": "test",
                    "labels": ["a", "b"],
                    "metadata": metadata,
                }
            )

    def test_valid_request_id_passes(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "classify",
                "text": "test",
                "labels": ["a", "b"],
                "request_id": "req-42.a_b",
            }
        )
        assert req.request_id == "req-42.a_b"


class TestExtractSpansRequest:
    def test_text_not_stripped(self) -> None:
        req = INFERENCE_REQUEST_ADAPTER.validate_python(
            {
                "task": "extract_spans",
                "text": "  Alice met Bob",
                "entity_types": ["PERSON"],
            }
        )
        assert req.text.index("Alice") == 2  # Leading space preserved

    def test_min_score_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(
                {
                    "task": "extract_spans",
                    "text": "test",
                    "entity_types": ["PERSON"],
                    "min_score": "0.5",
                }
            )
