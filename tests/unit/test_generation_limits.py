from fastapi.testclient import TestClient

from structflow.config import Settings
from structflow.engines.base import EngineOutput, GenerationBackend
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.engines.limits import GenerationBusyError, LimitedGenerator
from structflow.gateway.app import create_app
from structflow.routing.router import Router
from structflow.schemas.requests import GenerateRequest
from structflow.schemas.responses import GenerateResult

GENERATE = {"task": "generate", "messages": [{"role": "user", "content": "hello"}]}


class BusyGenerator(FakeGenerator):
    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        raise GenerationBusyError("64 generation requests are already in flight")


def client_for(generator: GenerationBackend) -> TestClient:
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=generator,
    )
    app = create_app(Settings(_env_file=None), router=router)
    return TestClient(app, raise_server_exceptions=False)


def test_too_many_generations_is_503_with_retry_after() -> None:
    with client_for(BusyGenerator()) as client:
        r = client.post("/v1/infer", json=GENERATE)
    assert r.status_code == 503
    assert r.json()["code"] == "backend_saturated"
    assert r.headers["retry-after"] == "1"


def test_a_generation_past_its_deadline_is_504() -> None:
    slow = LimitedGenerator(FakeGenerator(delay_s=1.0), max_inflight=4, timeout_s=0.05)
    with client_for(slow) as client:
        r = client.post("/v1/infer", json=GENERATE)
    assert r.status_code == 504
    assert r.json()["code"] == "upstream_timeout"


def test_the_app_puts_limits_on_generation_by_default() -> None:
    settings = Settings(
        _env_file=None, fake_generation_delay_ms=1000, request_timeout_s=0.05
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        r = client.post("/v1/infer", json=GENERATE)
    assert r.status_code == 504
