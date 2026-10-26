import pytest
from pydantic import ValidationError

from structflow.schemas.errors import (
    ERROR_CATALOG,
    PROBLEM_JSON,
    ErrorCode,
    ProblemDetails,
)
from structflow.schemas.requests import INFERENCE_REQUEST_ADAPTER


def test_validation_problem_does_not_echo_input() -> None:
    with pytest.raises(ValidationError) as exc_info:
        INFERENCE_REQUEST_ADAPTER.validate_python(
            {"task": "classify", "text": "SSN 123-45-6789", "labels": ["only"]}
        )
    problem = ProblemDetails.from_validation_error(exc_info.value, request_id="req-9")
    body = problem.model_dump_json(exclude_none=True)
    assert "123-45-6789" not in body
    assert problem.status == 422
    assert problem.errors is not None and problem.errors[0].loc == (
        "classify",
        "labels",
    )


class TestFromCode:
    def test_backend_saturated_has_correct_status(self) -> None:
        problem = ProblemDetails.from_code(
            ErrorCode.BACKEND_SATURATED, detail="64 in flight"
        )
        assert problem.status == 503
        parsed = problem.model_dump(exclude_none=True)
        assert set(parsed.keys()) == {"type", "title", "status", "detail", "code"}


class TestErrorCatalog:
    def test_every_error_code_has_catalog_entry(self) -> None:
        for code in ErrorCode:
            assert code in ERROR_CATALOG


class TestProblemJsonConstant:
    def test_problem_json_constant(self) -> None:
        assert PROBLEM_JSON == "application/problem+json"
