import json
import re
from typing import Any

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

    async def classify(self, request: ClassifyRequest) -> tuple[ClassifyResult, Usage]:
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
        return ClassifyResult(task="classify", labels=tuple(scored)), usage


class FakeSpanExtractor:
    name = "fake-span-extractor"

    async def extract(
        self, request: ExtractSpansRequest
    ) -> tuple[ExtractSpansResult, Usage]:
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
        return ExtractSpansResult(task="extract_spans", spans=tuple(spans)), usage


class FakeGenerator:
    name = "fake-generator"

    def __init__(self, structured_output: dict[str, Any] | None = None) -> None:
        self._structured_output = structured_output or {}

    async def generate_structured(
        self, request: StructuredRequest
    ) -> tuple[StructuredResult, Usage]:
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
        return result, usage

    async def generate(self, request: GenerateRequest) -> tuple[GenerateResult, Usage]:
        text = f"(fake) you said: {request.messages[-1].content}"
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            forward_passes=1 + len(text.split()),
        )
        return GenerateResult(task="generate", text=text, finish_reason="stop"), usage
