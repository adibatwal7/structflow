"""vLLM failures, seen from the client: every one becomes a clear problem response."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from structflow.config import Settings
from structflow.engines.fake import FakeClassifier, FakeSpanExtractor
from structflow.engines.limits import LimitedGenerator
from structflow.engines.stream import encode_chat_stream
from structflow.engines.vllm import VLLMBackend
from structflow.gateway.app import create_app
from structflow.routing.router import Router

CONTRACTS = Path(__file__).parents[2] / "docs" / "contracts"
RESPONSE = Draft202012Validator(
    json.loads((CONTRACTS / "inference_response.schema.json").read_text())
)
PROBLEM = Draft202012Validator(
    json.loads((CONTRACTS / "problem_details.schema.json").read_text())
)
USER = [{"role": "user", "content": "What date is it?"}]
DATE = {
    "task": "structured",
    "messages": USER,
    "constraint": {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
}


def streamed(completion: dict[str, Any]) -> httpx2.Response:
    """Answer the way vLLM does when asked to stream: Server-Sent Events."""
    return httpx2.Response(
        200,
        content=encode_chat_stream(completion),
        headers={"content-type": "text/event-stream"},
    )


def vllm_says(status: int, payload: Any) -> TestClient:
    def answer(request: httpx2.Request) -> httpx2.Response:
        if status == 200:
            return streamed(payload)
        return httpx2.Response(status, json=payload)

    vllm = VLLMBackend(
        base_url="http://vllm.test/v1",
        model="llama",
        transport=httpx2.MockTransport(answer),
    )
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=LimitedGenerator(vllm, max_inflight=4, timeout_s=5),
    )
    app = create_app(Settings(_env_file=None), router=router)
    return TestClient(app, raise_server_exceptions=False)


def reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5},
    }


def test_a_good_answer_is_a_contract_valid_response() -> None:
    with vllm_says(200, reply("2026-10-29")) as client:
        r = client.post("/v1/infer", json=DATE)
    assert r.status_code == 200, r.text
    RESPONSE.validate(r.json())
    assert r.json()["result"]["output"] == "2026-10-29"
    assert r.json()["model"] == "llama"
    timings = r.json()["timings"]
    assert timings["ttft_ms"] is not None  # streamed, so the tokens were timed
    assert timings["tpot_ms"] is not None


@pytest.mark.parametrize(
    ("status", "payload", "expected_status", "code"),
    [
        (200, reply("2026-10-", "length"), 422, "output_truncated"),
        (200, reply("tomorrow"), 502, "output_invalid"),
        (
            400,
            {"error": {"message": "unsupported regex"}},
            422,
            "constraint_compile_failed",
        ),
        (500, {"message": "CUDA out of memory"}, 502, "upstream_error"),
    ],
    ids=["truncated", "broke-the-pattern", "constraint-rejected", "crashed"],
)
def test_every_vllm_failure_is_a_clear_problem(
    status: int, payload: Any, expected_status: int, code: str
) -> None:
    with vllm_says(status, payload) as client:
        r = client.post("/v1/infer", json=DATE)
    assert r.status_code == expected_status, r.text
    PROBLEM.validate(r.json())
    assert r.json()["code"] == code


def test_the_app_can_be_pointed_at_vllm() -> None:
    settings = Settings(
        _env_file=None,
        generator_backend="vllm",
        vllm_base_url="http://127.0.0.1:9/v1",  # nothing listens here
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        r = client.post("/v1/infer", json={"task": "generate", "messages": USER})
    assert r.status_code == 502
    assert r.json()["code"] == "upstream_error"
    assert r.headers["retry-after"] == "5"


def test_a_degraded_answer_is_contract_valid_and_marked() -> None:
    body = {**DATE, "allow_unconstrained_fallback": True}
    calls: list[bool] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent = json.loads(request.content)
        calls.append("structured_outputs" in sent)
        if "structured_outputs" in sent:
            return httpx2.Response(400, json={"error": {"message": "no regex support"}})
        return streamed(reply("2026-10-29"))

    vllm = VLLMBackend(
        base_url="http://vllm.test/v1",
        model="llama",
        transport=httpx2.MockTransport(answer),
    )
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=LimitedGenerator(vllm, max_inflight=4, timeout_s=5),
    )
    with TestClient(create_app(Settings(_env_file=None), router=router)) as client:
        r = client.post("/v1/infer", json=body)
    assert r.status_code == 200, r.text
    RESPONSE.validate(r.json())
    assert r.json()["degraded"] is True
    assert r.json()["route_trail"] == [
        "constrained_generation",
        "unconstrained_generation",
    ]
    assert calls == [True, False]
