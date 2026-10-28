"""Run the real load generator against the real app, all inside this process."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx2
import pytest

from structflow.bench.load import BODIES, run, summarize
from structflow.config import Settings
from structflow.gateway.app import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def client_for(settings: Settings) -> AsyncGenerator[httpx2.AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):  # startup and shutdown, like uvicorn
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_a_gentle_load_is_all_200s() -> None:
    async with client_for(Settings(_env_file=None)) as client:
        samples = await run(client, BODIES["classify"], rps=40, duration_s=0.5)
    report = summarize(samples, duration_s=0.5)
    assert report.statuses == {200: 20}
    assert report.routes == {"softmax_head": 20}


async def test_more_generations_than_slots_get_fast_503s() -> None:
    settings = Settings(
        _env_file=None, fake_generation_delay_ms=1000, max_inflight_generation=2
    )
    async with client_for(settings) as client:
        samples = await run(client, BODIES["generate"], rps=40, duration_s=0.25)
    report = summarize(samples, duration_s=0.25)
    assert report.statuses == {200: 2, 503: 8}
    rejected = [s.latency_ms for s in samples if s.status == 503]
    assert max(rejected) < 100  # turned away at once, not after waiting 1 s
