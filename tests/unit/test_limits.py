import asyncio

import pytest

from structflow.engines.base import EngineOutput
from structflow.engines.fake import FakeGenerator
from structflow.engines.limits import (
    GenerationBusyError,
    GenerationTimeoutError,
    LimitedGenerator,
)
from structflow.schemas.requests import GenerateRequest
from structflow.schemas.responses import GenerateResult

pytestmark = pytest.mark.anyio

HELLO = GenerateRequest.model_validate(
    {"task": "generate", "messages": [{"role": "user", "content": "hello"}]}
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class GatedGenerator(FakeGenerator):
    """Every call waits until the test opens the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        await self.gate.wait()
        return await super().generate(request)


class BrokenGenerator(FakeGenerator):
    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        raise RuntimeError("the LLM server crashed")


async def test_results_pass_straight_through() -> None:
    limited = LimitedGenerator(FakeGenerator(), max_inflight=2, timeout_s=5)
    out = await limited.generate(HELLO)
    assert out.result.text == "(fake) you said: hello"
    assert limited.name == "fake-generator"


async def test_one_call_too_many_is_turned_away_at_once() -> None:
    inner = GatedGenerator()
    limited = LimitedGenerator(inner, max_inflight=2, timeout_s=5)
    running = [asyncio.create_task(limited.generate(HELLO)) for _ in range(2)]
    await asyncio.sleep(0)
    assert limited.in_flight == 2
    with pytest.raises(GenerationBusyError):
        await limited.generate(HELLO)
    inner.gate.set()
    await asyncio.gather(*running)
    assert limited.in_flight == 0
    await limited.generate(HELLO)  # a free slot again


async def test_a_slow_call_times_out_and_frees_its_slot() -> None:
    limited = LimitedGenerator(
        FakeGenerator(delay_s=1.0), max_inflight=1, timeout_s=0.05
    )
    with pytest.raises(GenerationTimeoutError):
        await limited.generate(HELLO)
    assert limited.in_flight == 0


async def test_a_failing_call_frees_its_slot() -> None:
    limited = LimitedGenerator(BrokenGenerator(), max_inflight=1, timeout_s=5)
    with pytest.raises(RuntimeError):
        await limited.generate(HELLO)
    assert limited.in_flight == 0


async def test_a_cancelled_call_frees_its_slot() -> None:
    limited = LimitedGenerator(GatedGenerator(), max_inflight=1, timeout_s=5)
    task = asyncio.create_task(limited.generate(HELLO))
    await asyncio.sleep(0)
    assert limited.in_flight == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limited.in_flight == 0


@pytest.mark.parametrize(("max_inflight", "timeout_s"), [(0, 5.0), (1, 0.0)])
def test_bad_settings_are_rejected(max_inflight: int, timeout_s: float) -> None:
    with pytest.raises(ValueError):
        LimitedGenerator(
            FakeGenerator(), max_inflight=max_inflight, timeout_s=timeout_s
        )
