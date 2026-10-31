from dataclasses import dataclass
from typing import Protocol

from structflow.schemas.requests import (
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    StructuredRequest,
)
from structflow.schemas.responses import (
    ClassifyResult,
    ExtractSpansResult,
    GenerateResult,
    StructuredResult,
    Usage,
)


@dataclass(frozen=True, slots=True)
class EngineOutput[R]:
    """What every engine returns: the result, what it cost, and how long it took.

    ttft_ms and tpot_ms are only known when the answer was streamed (the LLM).
    """

    result: R
    usage: Usage
    queue_ms: float = 0.0
    ttft_ms: float | None = None
    tpot_ms: float | None = None


class Classifier(Protocol):
    name: str

    async def classify(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]: ...


class SpanExtractor(Protocol):
    name: str

    async def extract(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]: ...


class GenerationBackend(Protocol):
    name: str

    async def generate_structured(
        self, request: StructuredRequest, *, constrained: bool = True
    ) -> EngineOutput[StructuredResult]: ...

    async def generate(
        self, request: GenerateRequest
    ) -> EngineOutput[GenerateResult]: ...

    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]: ...

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]: ...


class UnsupportedRequestError(RuntimeError):
    """This engine cannot serve this request (for example, labels it never learned)."""


class ConstraintRejectedError(Exception):
    """The backend refused the constraint (usually: it could not compile it)."""


class OutputInvalidError(Exception):
    """The answer broke its constraint. It must never reach the client."""
