from collections.abc import Mapping

import pytest

from structflow.engines.base import EngineOutput, UnsupportedRequestError
from structflow.engines.batcher import BatcherFullError
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.routing.router import Router
from structflow.schemas.common import Route
from structflow.schemas.requests import (
    INFERENCE_REQUEST_ADAPTER,
    ClassifyRequest,
    ExtractSpansRequest,
    InferenceRequest,
)
from structflow.schemas.responses import (
    ClassifyResult,
    ExtractSpansResult,
    InferenceResponse,
    Span,
    Usage,
)

pytestmark = pytest.mark.anyio

TEXT = "Tim Cook met Sundar Pichai in Paris"
CLASSIFY = {"task": "classify", "text": "a great film", "labels": ["great", "awful"]}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RefusingClassifier(FakeClassifier):
    async def classify(self, request: ClassifyRequest) -> EngineOutput[ClassifyResult]:
        raise UnsupportedRequestError("only knows POSITIVE, NEGATIVE")


class OverloadedClassifier(FakeClassifier):
    async def classify(self, request: ClassifyRequest) -> EngineOutput[ClassifyResult]:
        raise BatcherFullError("256 waiting")


class SpyGenerator(FakeGenerator):
    """Counts how often the expensive LLM route is used."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        self.calls += 1
        return await super().score_labels(request)


class RefusingGenerator(FakeGenerator):
    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        raise UnsupportedRequestError("no LLM is configured")


class SloppyGenerator(FakeGenerator):
    """An LLM that gets offsets wrong and invents a name."""

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        spans = (
            Span(start=0, end=5, label="LOC", text="Paris", score=0.9),
            Span(start=0, end=13, label="PER", text="Satya Nadella", score=0.9),
        )
        result = ExtractSpansResult(task="extract_spans", spans=spans)
        return EngineOutput(result=result, usage=Usage(forward_passes=9))


def make_router(
    classifier: FakeClassifier | None = None, generator: FakeGenerator | None = None
) -> Router:
    return Router(
        classifier=classifier or FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=generator or FakeGenerator(),
    )


async def handle(router: Router, body: Mapping[str, object]) -> InferenceResponse:
    request: InferenceRequest = INFERENCE_REQUEST_ADAPTER.validate_python(body)
    return await router.handle(request, request_id="req-1")


async def test_the_cheap_head_answers_when_it_can() -> None:
    response = await handle(make_router(), CLASSIFY)
    assert response.route == Route.SOFTMAX_HEAD
    assert response.route_trail == (Route.SOFTMAX_HEAD,)


async def test_unknown_labels_fall_back_to_llm_label_scoring() -> None:
    response = await handle(make_router(RefusingClassifier()), CLASSIFY)
    assert response.route == Route.LLM_LABEL_SCORING
    assert response.route_trail == (Route.SOFTMAX_HEAD, Route.LLM_LABEL_SCORING)
    assert response.model == "fake-generator"


async def test_overload_is_never_a_reason_to_fall_back() -> None:
    spy = SpyGenerator()
    with pytest.raises(BatcherFullError):
        await handle(make_router(OverloadedClassifier(), spy), CLASSIFY)
    assert spy.calls == 0  # the busy head says "busy"; the LLM is not dragged in


async def test_unknown_entity_types_fall_back_to_constrained_generation() -> None:
    body = {"task": "extract_spans", "text": TEXT, "entity_types": ["PER"]}
    response = await handle(make_router(), body)
    assert response.route == Route.CONSTRAINED_GENERATION
    assert response.route_trail == (Route.SPAN_HEAD, Route.CONSTRAINED_GENERATION)
    assert isinstance(response.result, ExtractSpansResult)
    assert all(TEXT[s.start : s.end] == s.text for s in response.result.spans)


async def test_spans_from_the_llm_are_grounded() -> None:
    body = {"task": "extract_spans", "text": TEXT, "entity_types": ["PER"]}
    response = await handle(make_router(generator=SloppyGenerator()), body)
    assert isinstance(response.result, ExtractSpansResult)
    assert [(s.text, s.start) for s in response.result.spans] == [("Paris", 30)]


async def test_when_no_route_is_left_the_request_is_refused() -> None:
    router = make_router(RefusingClassifier(), RefusingGenerator())
    with pytest.raises(
        UnsupportedRequestError, match="softmax_head, llm_label_scoring"
    ):
        await handle(router, CLASSIFY)


def test_the_plan_is_cheapest_first() -> None:
    router = make_router()
    plans = {
        task: [
            a.route
            for a in router.plan(INFERENCE_REQUEST_ADAPTER.validate_python(body))
        ]
        for task, body in {
            "classify": CLASSIFY,
            "extract_spans": {"task": "extract_spans", "text": TEXT},
            "generate": {
                "task": "generate",
                "messages": [{"role": "user", "content": "hi"}],
            },
        }.items()
    }
    assert plans == {
        "classify": [Route.SOFTMAX_HEAD, Route.LLM_LABEL_SCORING],
        "extract_spans": [Route.SPAN_HEAD, Route.CONSTRAINED_GENERATION],
        "generate": [Route.UNCONSTRAINED_GENERATION],
    }
