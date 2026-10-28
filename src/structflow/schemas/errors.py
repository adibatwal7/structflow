from enum import StrEnum
from typing import Final, Self

from pydantic import Field, ValidationError

from .common import StrictModel


# Part 1: Error codes and their HTTP statuses
class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    CONSTRAINT_COMPILE_FAILED = "constraint_compile_failed"
    OUTPUT_TRUNCATED = "output_truncated"
    BACKEND_SATURATED = "backend_saturated"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    INTERNAL_ERROR = "internal_error"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    NOT_READY = "not_ready"
    UNSUPPORTED_REQUEST = "unsupported_request"
    CLIENT_DISCONNECTED = "client_disconnected"


ERROR_CATALOG: Final[dict[ErrorCode, tuple[int, str]]] = {
    ErrorCode.VALIDATION_ERROR: (422, "Request validation failed"),
    ErrorCode.CONSTRAINT_COMPILE_FAILED: (422, "Constraint could not be compiled"),
    ErrorCode.OUTPUT_TRUNCATED: (422, "Output was truncated before it was complete"),
    ErrorCode.BACKEND_SATURATED: (503, "Inference backend is at capacity"),
    ErrorCode.UPSTREAM_TIMEOUT: (504, "Inference backend timed out"),
    ErrorCode.INTERNAL_ERROR: (500, "Internal server error"),
    ErrorCode.NOT_FOUND: (404, "Resource not found"),
    ErrorCode.METHOD_NOT_ALLOWED: (405, "Method not allowed"),
    ErrorCode.NOT_READY: (503, "Inference backend is not ready yet"),
    ErrorCode.UNSUPPORTED_REQUEST: (422, "No available engine can serve this request"),
    ErrorCode.CLIENT_DISCONNECTED: (499, "Client closed the request"),
}

PROBLEM_JSON: Final = "application/problem+json"


# Part 2: The models
class FieldError(StrictModel):
    loc: tuple[str | int, ...]
    msg: str
    type: str


class ProblemDetails(StrictModel):
    type: str
    title: str
    status: int = Field(ge=400, le=599)
    detail: str | None = None
    instance: str | None = None
    code: ErrorCode
    request_id: str | None = None
    errors: tuple[FieldError, ...] | None = None

    # Part 3: Factory classmethods
    @classmethod
    def from_code(
        cls,
        code: ErrorCode,
        detail: str | None = None,
        *,
        request_id: str | None = None,
        errors: tuple[FieldError, ...] | None = None,
    ) -> Self:
        status, title = ERROR_CATALOG[code]
        return cls(
            type=f"/errors/{code.value}",
            title=title,
            status=status,
            detail=detail,
            code=code,
            request_id=request_id,
            errors=errors,
        )

    @classmethod
    def from_validation_error(
        cls,
        exc: ValidationError,
        *,
        request_id: str | None = None,
    ) -> Self:
        pydantic_errors = exc.errors(
            include_url=False, include_input=False, include_context=False
        )
        field_errors = tuple(
            FieldError(loc=e["loc"], msg=e["msg"], type=e["type"])
            for e in pydantic_errors
        )
        return cls.from_code(
            ErrorCode.VALIDATION_ERROR,
            detail="Request validation failed",
            request_id=request_id,
            errors=field_errors,
        )
