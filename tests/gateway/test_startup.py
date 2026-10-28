import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from structflow.config import Settings
from structflow.engines.base import (
    Classifier,
    EngineOutput,
    GenerationBackend,
    UnsupportedRequestError,
)
from structflow.engines.batcher import BatcherFullError
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.gateway.app import create_app
from structflow.routing.router import Router
from structflow.schemas.requests import ClassifyRequest
from structflow.schemas.responses import ClassifyResult

CLASSIFY = {"task": "classify", "text": "a great film", "labels": ["great", "awful"]}


class StubClassifier(FakeClassifier):
    """A fake classifier we can make fail, or report a queue time."""

    def __init__(self, error: Exception | None = None, queue_ms: float = 0.0) -> None:
        self.error = error
        self.queue_ms = queue_ms

    async def classify(self, request: ClassifyRequest) -> EngineOutput[ClassifyResult]:
        if self.error is not None:
            raise self.error
        await asyncio.sleep(0.02)
        out = await super().classify(request)
        return EngineOutput(out.result, out.usage, queue_ms=self.queue_ms)


class RefusingGenerator(FakeGenerator):
    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        raise UnsupportedRequestError("no LLM is configured")


def client_for(
    classifier: Classifier, generator: GenerationBackend | None = None
) -> TestClient:
    router = Router(
        classifier=classifier,
        span_extractor=FakeSpanExtractor(),
        generator=generator or FakeGenerator(),
    )
    app = create_app(Settings(_env_file=None), router=router)
    return TestClient(app, raise_server_exceptions=False)


def wait_for_startup(client: TestClient) -> str:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status: str = client.get("/readyz").json()["status"]
        if status != "starting":
            return status
        time.sleep(0.05)
    raise AssertionError("startup never finished")


def test_infer_before_startup_is_503_not_ready() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))  # no "with": no startup
    r = client.post("/v1/infer", json=CLASSIFY)
    assert r.status_code == 503
    assert r.json()["code"] == "not_ready"
    assert r.headers["retry-after"] == "5"


def test_a_model_that_fails_to_load_is_reported(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        classifier_backend="hf",
        classifier_model=str(tmp_path),  # an empty folder: there is no model in it
    )
    with TestClient(create_app(settings)) as client:
        assert wait_for_startup(client) == "failed"
        assert client.post("/v1/infer", json=CLASSIFY).status_code == 503


def test_a_full_queue_is_503_with_retry_after() -> None:
    with client_for(StubClassifier(error=BatcherFullError("256 waiting"))) as client:
        r = client.post("/v1/infer", json=CLASSIFY)
    assert r.status_code == 503
    assert r.json()["code"] == "backend_saturated"
    assert r.headers["retry-after"] == "1"


def test_no_route_left_is_422_with_the_routes_tried() -> None:
    error = UnsupportedRequestError("only knows POSITIVE, NEGATIVE")
    with client_for(StubClassifier(error=error), RefusingGenerator()) as client:
        r = client.post("/v1/infer", json=CLASSIFY)
    assert r.status_code == 422
    assert r.json()["code"] == "unsupported_request"
    assert "softmax_head, llm_label_scoring" in r.json()["detail"]


def test_queue_time_is_reported_separately() -> None:
    with client_for(StubClassifier(queue_ms=10.0)) as client:
        timings = client.post("/v1/infer", json=CLASSIFY).json()["timings"]
    assert timings["queue_ms"] == 10.0
    assert timings["inference_ms"] > 0
    assert timings["queue_ms"] + timings["inference_ms"] == pytest.approx(
        timings["total_ms"]
    )
