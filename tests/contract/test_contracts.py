from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from structflow.schemas.export import CONTRACTS, render
from structflow.schemas.requests import INFERENCE_REQUEST_ADAPTER

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "docs" / "contracts"


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_committed_contract_is_up_to_date(name: str) -> None:
    committed = (CONTRACT_DIR / f"{name}.schema.json").read_text(encoding="utf-8")
    assert committed == render(CONTRACTS[name]()), (
        f"{name} changed. Run `python -m structflow.schemas.export` and commit the diff."
    )


class TestSchemaValidity:
    def test_inference_request_schema_is_valid(self) -> None:
        schema = CONTRACTS["inference_request"]()
        Draft202012Validator.check_schema(schema)

    def test_inference_response_schema_is_valid(self) -> None:
        schema = CONTRACTS["inference_response"]()
        Draft202012Validator.check_schema(schema)

    def test_problem_details_schema_is_valid(self) -> None:
        schema = CONTRACTS["problem_details"]()
        Draft202012Validator.check_schema(schema)


class TestSchemaPydanticAgreement:
    def test_classify_request_accepted_by_both(self) -> None:
        request = {
            "task": "classify",
            "text": "Hello world",
            "labels": ["positive", "negative"],
        }
        # Pydantic accepts it
        INFERENCE_REQUEST_ADAPTER.validate_python(request)
        # JSON Schema accepts it
        schema = CONTRACTS["inference_request"]()
        Draft202012Validator(schema).validate(request)

    def test_extract_spans_request_accepted_by_both(self) -> None:
        request = {
            "task": "extract_spans",
            "text": "Alice met Bob",
            "entity_types": ["PERSON"],
        }
        INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        Draft202012Validator(schema).validate(request)

    def test_structured_request_accepted_by_both(self) -> None:
        request = {
            "task": "structured",
            "messages": [{"role": "user", "content": "Extract the invoice"}],
            "constraint": {"type": "json_schema", "schema": {"type": "object"}},
        }
        INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        Draft202012Validator(schema).validate(request)

    def test_generate_request_accepted_by_both(self) -> None:
        request = {
            "task": "generate",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        Draft202012Validator(schema).validate(request)

    def test_no_task_rejected_by_both(self) -> None:
        request = {"text": "test", "labels": ["a", "b"]}
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(request)

    def test_unknown_task_rejected_by_both(self) -> None:
        request = {"task": "invalid", "text": "test", "labels": ["a", "b"]}
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(request)

    def test_duplicate_labels_rejected_by_both(self) -> None:
        request = {"task": "classify", "text": "test", "labels": ["a", "b", "a"]}
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(request)

    def test_unknown_extra_field_rejected_by_both(self) -> None:
        request = {
            "task": "classify",
            "text": "test",
            "labels": ["a", "b"],
            "unknown_field": "value",
        }
        with pytest.raises(ValidationError):
            INFERENCE_REQUEST_ADAPTER.validate_python(request)
        schema = CONTRACTS["inference_request"]()
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(request)
