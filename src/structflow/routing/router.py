import time
from typing import assert_never

from structflow.engines.base import Classifier, GenerationBackend, SpanExtractor
from structflow.schemas.common import Route
from structflow.schemas.requests import (
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    InferenceRequest,
    StructuredRequest,
)
from structflow.schemas.responses import InferenceResponse, Result, Timings, Usage


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

    async def handle(
        self, request: InferenceRequest, *, request_id: str
    ) -> InferenceResponse:
        started = time.perf_counter()
        result: Result
        usage: Usage
        match request:
            case ClassifyRequest():
                route, model = Route.SOFTMAX_HEAD, self._classifier.name
                result, usage = await self._classifier.classify(request)
            case ExtractSpansRequest():
                route, model = Route.SPAN_HEAD, self._span_extractor.name
                result, usage = await self._span_extractor.extract(request)
            case StructuredRequest():
                route, model = Route.CONSTRAINED_GENERATION, self._generator.name
                result, usage = await self._generator.generate_structured(request)
            case GenerateRequest():
                route, model = Route.UNCONSTRAINED_GENERATION, self._generator.name
                result, usage = await self._generator.generate(request)
            case _:
                assert_never(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return InferenceResponse(
            route=route,
            model=model,
            route_trail=(route,),
            result=result,
            timings=Timings(queue_ms=0.0, inference_ms=elapsed_ms, total_ms=elapsed_ms),
            usage=usage,
            request_id=request_id,
        )

    # match with class patterns picks a branch by the request’s type, and inside each branch mypy knows exactly which request class it has
    # assert_never is an exhaustiveness check. If someone adds a fifth task to InferenceRequest and forgets the router, mypy fails the build here. The compiler reminds you, instead of a customer finding out
