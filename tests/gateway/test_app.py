from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from structflow.config import Settings
from structflow.gateway.app import create_app
from structflow.gateway.problems import StructFlowError
from structflow.schemas.errors import ErrorCode


@pytest.fixture
def app() -> FastAPI:
    return create_app(Settings(_env_file=None))  # no .env, no surprises


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # "with" runs the lifespan (startup and shutdown).
    # raise_server_exceptions=False lets us see the 500 *response* instead of the raw exception.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_readyz_is_503_before_startup(app: FastAPI) -> None:
    c = TestClient(app)  # no "with", so the lifespan never runs
    assert c.get("/readyz").status_code == 503


def test_crash_returns_generic_500_with_request_id(app: FastAPI) -> None:
    async def boom() -> None:
        raise RuntimeError("secret database password")

    app.add_api_route("/boom", boom)  # a route that exists only in this test
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"X-Request-ID": "trace-1"})
    assert r.status_code == 500
    assert "secret" not in r.text  # nothing leaks to the client
    assert (
        r.headers["x-request-id"] == "trace-1"
    )  # the handler added it, not the middleware
    assert r.json()["request_id"] == "trace-1"


def test_healthz_returns_ok(client: TestClient) -> None:
    """healthz returns 200 and {"status": "ok"}"""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# readyz returns 200 when you use the client fixture (startup has run).
def test_readyz_is_ready_after_startup(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


# GET /nope returns 404, content type application/problem+json, code == "not_found", and a request_id equal to the x-request-id header
def test_not_found_returns_problem(client: TestClient) -> None:
    r = client.get("/nope", headers={"X-Request-ID": "trace-2"})
    assert r.status_code == 404
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == "not_found"
    assert r.json()["request_id"] == "trace-2"


# POST /healthz returns 405, code == "method_not_allowed", and an allow header of "GET".
def test_method_not_allowed_returns_problem(client: TestClient) -> None:
    r = client.post("/healthz", headers={"X-Request-ID": "trace-3"})
    assert r.status_code == 405
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == "method_not_allowed"
    assert r.json()["request_id"] == "trace-3"
    assert r.headers["allow"] == "GET"


# Request IDs:
# "req-42.a_b" comes back unchanged
# No header gets a new 32-character ID
# "bad id with spaces" is replaced with a new 32-character ID
def test_request_ids(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "req-42.a_b"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert r.headers["x-request-id"] == "req-42.a_b"

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert len(r.headers["x-request-id"]) == 32
    assert r.headers["x-request-id"].isalnum()

    r = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert len(r.headers["x-request-id"]) == 32
    assert r.headers["x-request-id"].isalnum()


def test_saturated_backend_returns_503_with_retry_after(app: FastAPI) -> None:
    async def busy() -> None:
        raise StructFlowError(
            ErrorCode.BACKEND_SATURATED, "64 in flight", headers={"Retry-After": "1"}
        )

    app.add_api_route("/busy", busy)
    with TestClient(app) as c:
        r = c.get("/busy")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert r.json()["code"] == "backend_saturated"
