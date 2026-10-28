"""Every route and every error, over HTTP, checked against the published contracts.

If the server ever sends something the JSON Schemas in docs/contracts don't allow,
a client generated from those schemas would break. These tests catch that.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from structflow.config import Settings
from structflow.gateway.app import create_app

CONTRACTS = Path(__file__).parents[2] / "docs" / "contracts"
RESPONSE = Draft202012Validator(
    json.loads((CONTRACTS / "inference_response.schema.json").read_text())
)
PROBLEM = Draft202012Validator(
    json.loads((CONTRACTS / "problem_details.schema.json").read_text())
)
USER = [{"role": "user", "content": "Extract the invoice vendor"}]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(Settings(_env_file=None))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def check_success(r: Any, route: str) -> dict[str, Any]:
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json"
    assert r.headers["x-request-id"]
    body: dict[str, Any] = r.json()
    RESPONSE.validate(body)
    assert body["route"] == route
    return body


def check_problem(r: Any, status: int, code: str) -> None:
    assert r.status_code == status, r.text
    assert r.headers["content-type"] == "application/problem+json"
    assert r.headers["x-request-id"]
    PROBLEM.validate(r.json())
    assert r.json()["code"] == code


@pytest.mark.parametrize(
    ("body", "route"),
    [
        (
            {"task": "classify", "text": "great film", "labels": ["great", "bad"]},
            "softmax_head",
        ),
        ({"task": "extract_spans", "text": "  Tim met Ana in Rome"}, "span_head"),
        (
            {"task": "extract_spans", "text": "Tim met Ana", "entity_types": ["PER"]},
            "constrained_generation",
        ),
        (
            {
                "task": "structured",
                "messages": USER,
                "constraint": {"type": "json_schema", "schema": {"type": "object"}},
            },
            "constrained_generation",
        ),
        ({"task": "generate", "messages": USER}, "unconstrained_generation"),
    ],
    ids=["classify", "spans", "spans-fallback", "structured", "generate"],
)
def test_every_route_answers_within_the_contract(
    client: TestClient, body: dict[str, Any], route: str
) -> None:
    check_success(client.post("/v1/infer", json=body), route)


@pytest.mark.parametrize(
    ("method", "path", "payload", "status", "code"),
    [
        (
            "POST",
            "/v1/infer",
            {"task": "classify", "text": "x", "labels": ["one"]},
            422,
            "validation_error",
        ),
        ("POST", "/v1/infer", {"task": "dance"}, 422, "validation_error"),
        ("GET", "/v1/infer", None, 405, "method_not_allowed"),
        ("GET", "/nowhere", None, 404, "not_found"),
    ],
    ids=["bad-labels", "unknown-task", "wrong-method", "unknown-path"],
)
def test_every_error_is_a_problem_within_the_contract(
    client: TestClient,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    status: int,
    code: str,
) -> None:
    check_problem(client.request(method, path, json=payload), status, code)


def test_broken_json_is_a_problem(client: TestClient) -> None:
    r = client.post(
        "/v1/infer", content=b"{not json", headers={"content-type": "application/json"}
    )
    check_problem(r, 422, "validation_error")


def test_not_ready_is_a_problem() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))  # no startup
    check_problem(
        client.post("/v1/infer", json={"task": "generate", "messages": USER}),
        503,
        "not_ready",
    )


def test_a_timeout_is_a_problem() -> None:
    settings = Settings(
        _env_file=None, fake_generation_delay_ms=500, request_timeout_s=0.05
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        r = client.post("/v1/infer", json={"task": "generate", "messages": USER})
    check_problem(r, 504, "upstream_timeout")
