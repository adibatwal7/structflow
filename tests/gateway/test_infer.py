from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from structflow.config import Settings
from structflow.engines.fake import (
    FakeClassifier,
    FakeGenerator,
    FakeSpanExtractor,
)
from structflow.gateway.app import create_app
from structflow.gateway.problems import StructFlowError
from structflow.routing.router import Router
from structflow.schemas.errors import ErrorCode
from structflow.schemas.requests import GenerateRequest
from structflow.schemas.responses import GenerateResult, Usage

USER = [{"role": "user", "content": "Extract the invoice"}]


# FIX 1: the fixture was missing. Every test that takes "client" needs it.
@pytest.fixture
def client() -> Iterator[TestClient]:
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=FakeGenerator(structured_output={"vendor": "ACME"}),
    )
    app = create_app(Settings(_env_file=None), router=router)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def post(client: TestClient, body: dict[str, Any], **headers: str) -> Any:
    return client.post("/v1/infer", json=body, headers=headers)


def test_classify_routes_to_softmax_head(client: TestClient) -> None:
    r = post(
        client,
        {"task": "classify", "text": "This is great", "labels": ["bad", "great"]},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "softmax_head"
    assert body["route_trail"] == ["softmax_head"]
    assert body["result"]["labels"][0]["label"] == "great"


def test_extracted_spans_are_grounded(client: TestClient) -> None:
    text = "  Alice met Bob in Paris"
    spans = post(client, {"task": "extract_spans", "text": text}).json()["result"][
        "spans"
    ]
    assert spans[0]["start"] == 2  # the leading spaces were kept
    assert all(text[s["start"] : s["end"]] == s["text"] for s in spans)


def test_engine_failure_becomes_problem_response() -> None:
    class Overloaded(FakeGenerator):
        async def generate(
            self, request: GenerateRequest
        ) -> tuple[GenerateResult, Usage]:
            raise StructFlowError(
                ErrorCode.BACKEND_SATURATED, "busy", headers={"Retry-After": "2"}
            )

    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=Overloaded(),
    )
    app = create_app(Settings(_env_file=None), router=router)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = post(c, {"task": "generate", "messages": USER})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "2"
    assert r.json()["code"] == "backend_saturated"


# FIX 2: a structured request needs "messages", with the schema inside "constraint".
def test_structured_and_generate(client: TestClient) -> None:
    r = post(
        client,
        {
            "task": "structured",
            "messages": USER,
            "constraint": {"type": "json_schema", "schema": {"type": "object"}},
        },
    )
    assert r.status_code == 200
    assert r.json()["route"] == "constrained_generation"
    assert r.json()["result"]["output"] == {"vendor": "ACME"}

    g = post(client, {"task": "generate", "messages": USER})
    assert g.json()["route"] == "unconstrained_generation"
    assert g.json()["result"]["finish_reason"] == "stop"


# FIX 3: real duplicates ("a", "a"), and the error type is "value_error".
def test_duplicate_labels_yields_422(client: TestClient) -> None:
    text = "SSN 123-45-6789"
    r = post(client, {"task": "classify", "text": text, "labels": ["a", "a"]})

    assert r.status_code == 422
    assert r.headers["content-type"] == "application/problem+json"
    body = r.json()
    assert body["code"] == "validation_error"
    assert body["errors"][0]["loc"][0] == "body"
    assert body["errors"][0]["type"] == "value_error"
    assert "123-45-6789" not in r.text


def test_union_tag_invalid_yields_422(client: TestClient) -> None:
    r = post(client, {"task": "summarize"})

    assert r.status_code == 422
    assert r.headers["content-type"] == "application/problem+json"
    body = r.json()
    assert body["code"] == "validation_error"
    assert body["errors"][0]["type"] == "union_tag_invalid"


def test_broken_json_yields_422(client: TestClient) -> None:
    r = client.post(
        "/v1/infer", content=b'{"task": ', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["errors"][0]["type"] == "json_invalid"


# FIX 4: the real header name has hyphens, so it has to be passed with **{...}.
def test_request_id_from_body_and_header(client: TestClient) -> None:
    r = post(
        client,
        {"task": "generate", "messages": USER, "request_id": "body-7"},
        **{"X-Request-ID": "header-7"},
    )

    assert r.status_code == 200
    assert r.json()["request_id"] == "body-7"
    assert r.headers["x-request-id"] == "body-7"
