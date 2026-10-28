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


class Classifier(Protocol):
    name: str

    async def classify(
        self, request: ClassifyRequest
    ) -> tuple[ClassifyResult, Usage]: ...


class SpanExtractor(Protocol):
    name: str

    async def extract(
        self, request: ExtractSpansRequest
    ) -> tuple[ExtractSpansResult, Usage]: ...


class GenerationBackend(Protocol):
    name: str

    async def generate_structured(
        self, request: StructuredRequest
    ) -> tuple[StructuredResult, Usage]: ...

    async def generate(
        self, request: GenerateRequest
    ) -> tuple[GenerateResult, Usage]: ...


class UnsupportedRequestError(RuntimeError):
    """This engine cannot serve this request (for example, labels it never learned)."""
