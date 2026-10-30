"""LLM label scoring and LLM span extraction through vLLM (against a fake vLLM)."""

import json
import math
from typing import Any

import httpx2
import pytest

from structflow.engines.base import UnsupportedRequestError
from structflow.engines.fake import FakeClassifier, FakeSpanExtractor
from structflow.engines.vllm import (
    UpstreamError,
    VLLMBackend,
    average_token_probability,
    entities_schema,
    locate,
    sequence_probability,
)
from structflow.routing.router import Router
from structflow.schemas.common import Route
from structflow.schemas.requests import (
    INFERENCE_REQUEST_ADAPTER,
    ClassifyRequest,
    ExtractSpansRequest,
)
from structflow.schemas.responses import ExtractSpansResult

pytestmark = pytest.mark.anyio

TEXT = "Tim Cook met Tim Cook's team in Paris."


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeVLLM:
    """Answers with fixed content and per-token logprobs; remembers what we sent."""

    def __init__(self, content: str, logprobs: list[float] | None) -> None:
        self.content = content
        self.logprobs = logprobs
        self.sent: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(json.loads(request.content))
        choice: dict[str, Any] = {
            "message": {"content": self.content},
            "finish_reason": "stop",
        }
        if self.logprobs is not None:
            choice["logprobs"] = {
                "content": [{"token": "t", "logprob": lp} for lp in self.logprobs]
            }
        return httpx2.Response(
            200,
            json={
                "choices": [choice],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3},
            },
        )


def backend(fake: FakeVLLM) -> VLLMBackend:
    return VLLMBackend(
        base_url="http://vllm.test/v1",
        model="llama",
        transport=httpx2.MockTransport(fake),
    )


def classify(**extra: Any) -> ClassifyRequest:
    return ClassifyRequest.model_validate(
        {"task": "classify", "text": "I loved it", "labels": ["happy", "sad"], **extra}
    )


def extract(**extra: Any) -> ExtractSpansRequest:
    return ExtractSpansRequest.model_validate(
        {"task": "extract_spans", "text": TEXT, **extra}
    )


def test_probabilities_from_logprobs() -> None:
    assert sequence_probability([math.log(0.5), math.log(0.5)]) == pytest.approx(0.25)
    assert average_token_probability([math.log(0.25), math.log(1.0)]) == pytest.approx(
        0.5
    )


async def test_label_scoring_uses_a_choice_constraint_and_logprobs() -> None:
    fake = FakeVLLM("happy", [math.log(0.9)])
    out = await backend(fake).score_labels(classify())
    sent = fake.sent[0]
    assert sent["structured_outputs"] == {"choice": ["happy", "sad"]}
    assert sent["logprobs"] is True and sent["temperature"] == 0.0
    assert sent["messages"][1] == {"role": "user", "content": "I loved it"}
    (only,) = out.result.labels
    assert only.label == "happy" and only.score == pytest.approx(0.9)


async def test_label_scoring_needs_the_logprobs() -> None:
    with pytest.raises(UpstreamError, match="logprobs"):
        await backend(FakeVLLM("happy", None)).score_labels(classify())


async def test_label_scoring_cannot_do_multi_label() -> None:
    with pytest.raises(UnsupportedRequestError):
        await backend(FakeVLLM("happy", [0.0])).score_labels(classify(multi_label=True))


def test_entities_schema_limits_the_labels() -> None:
    schema = entities_schema(["PER", "LOC"])
    item = schema["properties"]["entities"]["items"]
    assert item["properties"]["label"] == {"enum": ["PER", "LOC"]}


def test_locate_finds_repeats_and_drops_inventions() -> None:
    found = [
        ("Tim Cook", "PER"),
        ("Tim Cook", "PER"),
        ("Satya", "PER"),
        ("Paris", "LOC"),
    ]
    spans = locate(TEXT, found, 0.8)
    assert [(s.text, s.start) for s in spans] == [
        ("Tim Cook", 0),
        ("Tim Cook", 13),
        ("Paris", 32),
    ]
    assert all(TEXT[s.start : s.end] == s.text for s in spans)


async def test_span_extraction_asks_for_json_and_grounds_the_answer() -> None:
    answer = {"entities": [{"text": "Paris", "label": "LOC"}]}
    fake = FakeVLLM(json.dumps(answer), [math.log(0.9)] * 4)
    out = await backend(fake).extract_spans(extract(entity_types=["LOC"]))
    schema = fake.sent[0]["structured_outputs"]["json"]
    assert schema["properties"]["entities"]["items"]["properties"]["label"] == {
        "enum": ["LOC"]
    }
    (span,) = out.result.spans
    assert (span.text, span.label, span.start) == ("Paris", "LOC", 32)
    assert span.score == pytest.approx(0.9)


async def test_span_extraction_respects_min_score() -> None:
    answer = {"entities": [{"text": "Paris", "label": "LOC"}]}
    fake = FakeVLLM(json.dumps(answer), [math.log(0.3)])
    out = await backend(fake).extract_spans(extract(min_score=0.5))
    assert out.result.spans == ()


async def test_the_router_falls_back_to_vllm_for_unknown_entity_types() -> None:
    answer = {"entities": [{"text": "Paris", "label": "CITY"}]}
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),  # only knows ENT
        generator=backend(FakeVLLM(json.dumps(answer), [math.log(0.9)])),
    )
    request = INFERENCE_REQUEST_ADAPTER.validate_python(
        {"task": "extract_spans", "text": TEXT, "entity_types": ["CITY"]}
    )
    response = await router.handle(request, request_id="r1")
    assert response.route_trail == (Route.SPAN_HEAD, Route.CONSTRAINED_GENERATION)
    assert isinstance(response.result, ExtractSpansResult)
    assert [s.text for s in response.result.spans] == ["Paris"]


async def test_span_extraction_lets_the_model_skip_other_entities() -> None:
    fake = FakeVLLM('{"entities": []}', [math.log(0.9)])
    out = await backend(fake).extract_spans(extract(entity_types=["PRODUCT", "PRICE"]))
    system = fake.sent[0]["messages"][0]
    assert system["role"] == "system"
    assert "PRODUCT, PRICE" in system["content"]
    assert "Skip anything" in system["content"]
    assert out.result.spans == ()
