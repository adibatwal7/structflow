import json
import re
from typing import Any

from structflow.engines.base import EngineOutput, UnsupportedRequestError
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
    LabelScore,
    Span,
    StructuredResult,
    Usage,
)


class FakeClassifier:
    name = "fake-classifier"

    async def classify(self, request: ClassifyRequest) -> EngineOutput[ClassifyResult]:
        text = request.text.lower()
        raw = [1.0 if label.lower() in text else 0.1 for label in request.labels]
        total = sum(raw)
        scored = [
            LabelScore(label=label, score=s / total)
            for label, s in zip(request.labels, raw, strict=True)
        ]
        scored.sort(
            key=lambda ls: ls.score, reverse=True
        )  # your envelope requires this order
        usage = Usage(prompt_tokens=len(request.text.split()), forward_passes=1)
        return EngineOutput(
            result=ClassifyResult(task="classify", labels=tuple(scored)), usage=usage
        )


class FakeSpanExtractor:
    name = "fake-span-extractor"

    def supports(self, request: ExtractSpansRequest) -> bool:
        return not request.entity_types or set(request.entity_types) == {"ENT"}

    async def extract(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        if not self.supports(request):
            raise UnsupportedRequestError(f"{self.name} only finds ENT")
        text = request.text
        spans = []
        for m in re.finditer(r"\b[A-Z][a-z]+\b", text):
            spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    label="ENT",
                    text=m.group(),
                    score=0.9,
                )
            )
        usage = Usage(prompt_tokens=len(text.split()), forward_passes=1)
        return EngineOutput(
            result=ExtractSpansResult(task="extract_spans", spans=tuple(spans)),
            usage=usage,
        )


class FakeGenerator:
    name = "fake-generator"

    def __init__(self, structured_output: dict[str, Any] | None = None) -> None:
        self._structured_output = structured_output or {}

    async def generate_structured(
        self, request: StructuredRequest
    ) -> EngineOutput[StructuredResult]:
        raw_text = json.dumps(self._structured_output)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(raw_text.split()),
            forward_passes=1 + len(raw_text.split()),
        )
        result = StructuredResult(
            task="structured", output=self._structured_output, raw_text=raw_text
        )
        return EngineOutput(result=result, usage=usage)

    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        text = f"(fake) you said: {request.messages[-1].content}"
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            forward_passes=1 + len(text.split()),
        )
        return EngineOutput(
            result=GenerateResult(task="generate", text=text, finish_reason="stop"),
            usage=usage,
        )

    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        """Pretend LLM label scoring: same idea as the fake classifier."""
        return await FakeClassifier().classify(request)

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        """Pretend LLM extraction: capitalised words, labelled with the first type asked for."""
        label = request.entity_types[0] if request.entity_types else "ENT"
        spans = tuple(
            Span(start=m.start(), end=m.end(), label=label, text=m.group(), score=0.8)
            for m in re.finditer(r"\b[A-Z][a-z]+\b", request.text)
        )
        usage = Usage(
            prompt_tokens=len(request.text.split()),
            completion_tokens=4 * len(spans),
            forward_passes=1 + 4 * len(spans),
        )
        return EngineOutput(
            result=ExtractSpansResult(task="extract_spans", spans=spans), usage=usage
        )
