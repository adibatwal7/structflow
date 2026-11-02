"""Telemetry: every inference request leaves one log line and updates the metrics."""

import json
import logging
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from structflow.config import Settings
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.engines.stream import encode_chat_stream
from structflow.engines.vllm import VLLMBackend
from structflow.gateway.app import create_app
from structflow.gateway.telemetry import RequestRecord
from structflow.routing.router import Router

CLASSIFY = {"task": "classify", "text": "great film", "labels": ["great", "awful"]}


def app_with(generator: Any = None) -> TestClient:
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=generator or FakeGenerator(),
    )
    app = create_app(Settings(_env_file=None), router=router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="structflow.requests")
    return caplog


def records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "structflow.requests"
    ]


def metric(client: TestClient, line_start: str) -> float:
    """Read one sample from /metrics, e.g. 'structflow_requests_total{...}'.

    A sample that was never recorded doesn't appear at all: that reads as 0.
    """
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_a_good_request_leaves_one_json_log_line(
    log: pytest.LogCaptureFixture,
) -> None:
    with app_with() as client:
        r = client.post("/v1/infer", json=CLASSIFY, headers={"X-Request-ID": "abc"})
    (rec,) = records(log)
    assert r.status_code == 200
    assert rec["request_id"] == "abc"
    assert (rec["status"], rec["task"], rec["route"]) == (
        200,
        "classify",
        "softmax_head",
    )
    assert rec["total_ms"] >= 0
    assert "error" not in rec  # empty fields are left out


def test_requests_are_counted_by_task_route_and_outcome() -> None:
    with app_with() as client:
        client.post("/v1/infer", json=CLASSIFY)
        client.post("/v1/infer", json=CLASSIFY)
        client.post("/v1/infer", json={"task": "classify"})  # invalid
        ok = metric(
            client,
            'structflow_requests_total{outcome="ok",route="softmax_head",task="classify"}',
        )
        bad = metric(
            client,
            'structflow_requests_total{outcome="validation_error",route="none",task="unknown"}',
        )
        seconds = metric(
            client,
            'structflow_request_seconds_count{route="softmax_head",task="classify"}',
        )
    assert (ok, bad, seconds) == (2.0, 1.0, 2.0)


def test_a_fallback_is_counted() -> None:
    body = {"task": "extract_spans", "text": "Tim met Ann", "entity_types": ["PER"]}
    with app_with() as client:
        r = client.post("/v1/infer", json=body)  # the fake span head only knows ENT
        assert len(r.json()["route_trail"]) == 2
        client.post("/v1/infer", json=CLASSIFY)  # served by the first route tried
        assert metric(client, 'structflow_fallbacks_total{task="extract_spans"}') == 1.0
        assert metric(client, 'structflow_fallbacks_total{task="classify"}') == 0.0


def test_a_crash_is_recorded_as_internal_error(
    log: pytest.LogCaptureFixture,
) -> None:
    class Broken(FakeGenerator):
        async def generate(self, request: Any) -> Any:
            raise RuntimeError("bug")

    body = {"task": "generate", "messages": [{"role": "user", "content": "hi"}]}
    with app_with(Broken()) as client:
        r = client.post("/v1/infer", json=body)
    assert r.status_code == 500
    rec = records(log)[-1]
    assert (rec["status"], rec["error"], rec["task"]) == (
        500,
        "internal_error",
        "generate",
    )


def test_llm_answers_report_ttft_and_tokens(log: pytest.LogCaptureFixture) -> None:
    def vllm(request: httpx2.Request) -> httpx2.Response:
        answer = {
            "choices": [
                {"message": {"content": "2026-11-01"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6},
        }
        return httpx2.Response(
            200,
            content=encode_chat_stream(answer),
            headers={"content-type": "text/event-stream"},
        )

    backend = VLLMBackend(
        base_url="http://vllm.test/v1",
        model="llama",
        transport=httpx2.MockTransport(vllm),
    )
    body = {
        "task": "structured",
        "messages": [{"role": "user", "content": "date?"}],
        "constraint": {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
    }
    with app_with(backend) as client:
        assert client.post("/v1/infer", json=body).status_code == 200
        route = 'route="constrained_generation"'
        assert metric(client, f"structflow_ttft_seconds_count{{{route}}}") == 1.0
        assert metric(client, f"structflow_generated_tokens_total{{{route}}}") == 6.0
    (rec,) = records(log)
    assert rec["completion_tokens"] == 6 and rec["ttft_ms"] >= 0


def test_other_endpoints_are_not_recorded(log: pytest.LogCaptureFixture) -> None:
    with app_with() as client:
        client.get("/healthz")
        client.get("/metrics")
    assert records(log) == []


def test_record_json_leaves_out_empty_fields() -> None:
    rec = RequestRecord(ts="t", request_id="r", status=422, total_ms=1.5, error="x")
    assert json.loads(rec.to_json()) == {
        "ts": "t",
        "request_id": "r",
        "status": 422,
        "total_ms": 1.5,
        "degraded": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": "x",
    }
