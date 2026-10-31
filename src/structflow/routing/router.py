import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import assert_never

from structflow.engines.base import (
    Classifier,
    ConstraintRejectedError,
    EngineOutput,
    GenerationBackend,
    OutputInvalidError,
    SpanExtractor,
    UnsupportedRequestError,
)
from structflow.routing.grounding import ground
from structflow.schemas.common import Route
from structflow.schemas.requests import (
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    InferenceRequest,
    StructuredRequest,
)
from structflow.schemas.responses import (
    ExtractSpansResult,
    InferenceResponse,
    Result,
    Timings,
)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One way to serve a request: which route, which model, and how to call it."""

    route: Route
    model: str
    run: Callable[[], Awaitable[EngineOutput[Result]]]
    fall_back_on: tuple[type[Exception], ...] = (UnsupportedRequestError,)


class Router:
    def __init__(
        self,
        *,
        classifier: Classifier,
        span_extractor: SpanExtractor,
        generator: GenerationBackend,
    ) -> None:
        self._classifier = classifier
        self._span_extractor = span_extractor
        self._generator = generator

    def plan(self, request: InferenceRequest) -> list[Attempt]:
        """Every route that could serve this request, cheapest first."""
        match request:
            case ClassifyRequest() as classify:
                return [
                    Attempt(
                        Route.SOFTMAX_HEAD,
                        self._classifier.name,
                        lambda: self._classifier.classify(classify),
                    ),
                    Attempt(
                        Route.LLM_LABEL_SCORING,
                        self._generator.name,
                        lambda: self._generator.score_labels(classify),
                    ),
                ]
            case ExtractSpansRequest() as extract:
                return [
                    Attempt(
                        Route.SPAN_HEAD,
                        self._span_extractor.name,
                        lambda: self._span_extractor.extract(extract),
                    ),
                    Attempt(
                        Route.CONSTRAINED_GENERATION,
                        self._generator.name,
                        lambda: self._grounded_spans(extract),
                    ),
                ]
            case StructuredRequest() as structured:
                constrained = Attempt(
                    Route.CONSTRAINED_GENERATION,
                    self._generator.name,
                    lambda: self._generator.generate_structured(structured),
                    fall_back_on=(),  # its errors go straight to the client
                )
                if not structured.allow_unconstrained_fallback:
                    return [constrained]  # its errors go straight to the client
                return [
                    replace(
                        constrained,
                        fall_back_on=(ConstraintRejectedError, OutputInvalidError),
                    ),
                    Attempt(
                        Route.UNCONSTRAINED_GENERATION,
                        self._generator.name,
                        lambda: self._generator.generate_structured(
                            structured, constrained=False
                        ),
                        fall_back_on=(),
                    ),
                ]
            case GenerateRequest() as generate:
                return [
                    Attempt(
                        Route.UNCONSTRAINED_GENERATION,
                        self._generator.name,
                        lambda: self._generator.generate(generate),
                    ),
                ]
            case _:
                assert_never(request)

    async def handle(
        self, request: InferenceRequest, *, request_id: str
    ) -> InferenceResponse:
        started = time.perf_counter()
        trail: list[Route] = []
        for attempt in self.plan(request):
            trail.append(attempt.route)
            try:
                output = await attempt.run()
            except attempt.fall_back_on:
                continue  # this route can't serve it: try the next one

            total_ms = (time.perf_counter() - started) * 1000
            return InferenceResponse(
                request_id=request_id,
                model=attempt.model,
                route=attempt.route,
                route_trail=tuple(trail),
                degraded=isinstance(request, StructuredRequest)
                and attempt.route is Route.UNCONSTRAINED_GENERATION,
                result=output.result,
                usage=output.usage,
                timings=Timings(
                    total_ms=total_ms,
                    ttft_ms=output.ttft_ms,
                    tpot_ms=output.tpot_ms,
                    queue_ms=output.queue_ms,
                    inference_ms=max(total_ms - output.queue_ms, 0.0),
                ),
            )
        tried = ", ".join(trail)
        raise UnsupportedRequestError(
            f"no route can serve this request (tried {tried})"
        )

    async def _grounded_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        output = await self._generator.extract_spans(request)
        spans = ground(request.text, output.result.spans)
        return replace(
            output, result=ExtractSpansResult(task="extract_spans", spans=spans)
        )
