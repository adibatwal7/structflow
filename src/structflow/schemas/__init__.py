"""Public wire contract for StructFlow."""

from .common import ChatMessage, FinishReason, Route, SamplingParams, TaskType
from .constraints import CONSTRAINT_ADAPTER, Constraint
from .errors import ERROR_CATALOG, PROBLEM_JSON, ErrorCode, ProblemDetails
from .requests import (
    INFERENCE_REQUEST_ADAPTER,
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    InferenceRequest,
    StructuredRequest,
)
from .responses import (
    ALLOWED_ROUTES,
    INFERENCE_RESPONSE_ADAPTER,
    ClassifyResult,
    ExtractSpansResult,
    GenerateResult,
    InferenceResponse,
    StructuredResult,
)

__all__ = [
    "ALLOWED_ROUTES",
    "CONSTRAINT_ADAPTER",
    "ERROR_CATALOG",
    "INFERENCE_REQUEST_ADAPTER",
    "INFERENCE_RESPONSE_ADAPTER",
    "PROBLEM_JSON",
    "ChatMessage",
    "ClassifyRequest",
    "ClassifyResult",
    "Constraint",
    "ErrorCode",
    "ExtractSpansRequest",
    "ExtractSpansResult",
    "FinishReason",
    "GenerateRequest",
    "GenerateResult",
    "InferenceRequest",
    "InferenceResponse",
    "ProblemDetails",
    "Route",
    "SamplingParams",
    "StructuredRequest",
    "StructuredResult",
    "TaskType",
]
