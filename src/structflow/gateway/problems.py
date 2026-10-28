import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from structflow.engines.base import UnsupportedRequestError
from structflow.engines.batcher import BatcherClosedError, BatcherFullError
from structflow.engines.limits import GenerationBusyError, GenerationTimeoutError
from structflow.schemas.errors import (
    PROBLEM_JSON,
    ErrorCode,
    FieldError,
    ProblemDetails,
)

from .disconnect import ClientDisconnectedError
from .request_id import HEADER

logger = logging.getLogger("structflow.gateway")

_HTTP_STATUS_TO_CODE = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}
_ENGINE_ERRORS: dict[type[Exception], tuple[ErrorCode, dict[str, str]]] = {
    BatcherFullError: (ErrorCode.BACKEND_SATURATED, {"Retry-After": "1"}),
    BatcherClosedError: (ErrorCode.NOT_READY, {"Retry-After": "5"}),
    UnsupportedRequestError: (ErrorCode.UNSUPPORTED_REQUEST, {}),
    GenerationBusyError: (ErrorCode.BACKEND_SATURATED, {"Retry-After": "1"}),
    GenerationTimeoutError: (ErrorCode.UPSTREAM_TIMEOUT, {}),
    ClientDisconnectedError: (ErrorCode.CLIENT_DISCONNECTED, {}),
}


class StructFlowError(Exception):
    """The only exception our own code raises for client-visible failures."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail or code.value)
        self.code = code
        self.detail = detail
        self.headers = headers or {}  # e.g. {"Retry-After": "1"} in Step 2.7


def _request_id(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    return rid if isinstance(rid, str) else None


def problem_response(
    request: Request, problem: ProblemDetails, headers: dict[str, str] | None = None
) -> JSONResponse:
    all_headers = dict(headers or {})
    if (rid := _request_id(request)) is not None:
        all_headers[HEADER] = (
            rid  # crash responses bypass the middleware, so add the ID here
        )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=problem.status,
        media_type=PROBLEM_JSON,
        headers=all_headers,
    )


async def _on_structflow_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StructFlowError)  # narrows the type for mypy
    problem = ProblemDetails.from_code(
        exc.code, exc.detail, request_id=_request_id(request)
    )
    return problem_response(request, problem, exc.headers)


async def _on_engine_error(request: Request, exc: Exception) -> JSONResponse:
    code, headers = next(
        entry for kind, entry in _ENGINE_ERRORS.items() if isinstance(exc, kind)
    )
    problem = ProblemDetails.from_code(code, str(exc), request_id=_request_id(request))
    return problem_response(request, problem, headers)


async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"request_id": _request_id(request)})
    problem = ProblemDetails.from_code(
        ErrorCode.INTERNAL_ERROR,
        "An unexpected error occurred.",
        request_id=_request_id(request),
    )
    return problem_response(request, problem)


async def _on_request_validation(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    field_errors = tuple(
        FieldError(loc=tuple(e["loc"]), msg=e["msg"], type=e["type"])
        for e in exc.errors()
    )
    problem = ProblemDetails.from_code(
        ErrorCode.VALIDATION_ERROR,
        f"{len(field_errors)} field(s) failed validation",
        request_id=_request_id(request),
        errors=field_errors,
    )
    return problem_response(request, problem)


async def _on_pydantic_validation(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ValidationError)
    problem = ProblemDetails.from_validation_error(exc, request_id=_request_id(request))
    return problem_response(request, problem)


async def _on_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    problem = ProblemDetails.from_code(code, None, request_id=_request_id(request))
    return problem_response(request, problem, dict(exc.headers or {}))


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StructFlowError, _on_structflow_error)
    app.add_exception_handler(RequestValidationError, _on_request_validation)
    app.add_exception_handler(ValidationError, _on_pydantic_validation)
    app.add_exception_handler(StarletteHTTPException, _on_http_exception)
    for kind in _ENGINE_ERRORS:
        app.add_exception_handler(kind, _on_engine_error)
    app.add_exception_handler(Exception, _on_unhandled)
