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
    """What every engine returns: the result, what it cost, and how long it queued."""

    result: R
    usage: Usage
    queue_ms: float = 0.0


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
        self, request: StructuredRequest
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
