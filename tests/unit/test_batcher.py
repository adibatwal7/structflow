import asyncio
import threading
import time
from collections.abc import Sequence

import pytest

from structflow.engines.batcher import (
    BatcherClosedError,
    BatcherFullError,
    MicroBatcher,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Recorder:
    """A fake model. Remembers every batch it saw and multiplies by 10."""

    def __init__(
        self, delay_s: float = 0.0, gate: threading.Event | None = None
    ) -> None:
        self.batches: list[list[int]] = []
        self.delay_s = delay_s
        self.gate = gate
        self.started = threading.Event()

    def __call__(self, items: Sequence[int]) -> list[int]:
        self.batches.append(list(items))
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=5), "test forgot to open the gate"
        time.sleep(self.delay_s)
        if any(x < 0 for x in items):
            raise ValueError("boom")
        return [x * 10 for x in items]


def make(
    fn: Recorder, *, size: int = 8, wait_ms: float = 50, queue: int = 64
) -> MicroBatcher[int, int]:
    return MicroBatcher(
        fn, max_batch_size=size, max_wait_ms=wait_ms, max_queue_size=queue
    )


async def wait_until_started(fn: Recorder) -> None:
    while not fn.started.is_set():
        await asyncio.sleep(0.001)


async def test_concurrent_calls_share_one_batch() -> None:
    fn = Recorder()
    async with make(fn) as batcher:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(8)))
    assert [r.value for r in results] == [i * 10 for i in range(8)]
    assert len(fn.batches) == 1
    assert {r.batch_size for r in results} == {8}


async def test_max_batch_size_splits_the_work() -> None:
    fn = Recorder()
    async with make(fn, size=8) as batcher:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(20)))
    assert [len(b) for b in fn.batches] == [8, 8, 4]
    assert [r.value for r in results] == [i * 10 for i in range(20)]


async def test_lone_request_is_sent_when_the_window_closes() -> None:
    fn = Recorder()
    async with make(fn, wait_ms=20) as batcher:
        result = await batcher.submit(7)
    assert result.value == 70
    assert result.batch_size == 1
    assert 15 <= result.queue_ms < 1000


async def test_zero_wait_still_takes_what_is_already_queued() -> None:
    fn = Recorder()
    async with make(fn, wait_ms=0) as batcher:
        await asyncio.gather(*(batcher.submit(i) for i in range(5)))
    assert fn.batches == [[0, 1, 2, 3, 4]]


async def test_one_error_fails_its_whole_batch_but_not_the_batcher() -> None:
    fn = Recorder()
    async with make(fn) as batcher:
        results = await asyncio.gather(
            *(batcher.submit(i) for i in (1, -1, 2)), return_exceptions=True
        )
        assert all(isinstance(r, ValueError) for r in results)
        assert (await batcher.submit(3)).value == 30


async def test_wrong_number_of_outputs_is_an_error() -> None:
    batcher: MicroBatcher[int, int] = MicroBatcher(
        lambda items: [0], max_batch_size=4, max_wait_ms=20, max_queue_size=8
    )
    async with batcher:
        results = await asyncio.gather(
            batcher.submit(1), batcher.submit(2), return_exceptions=True
        )
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_full_queue_is_rejected_immediately() -> None:
    gate = threading.Event()
    fn = Recorder(gate=gate)
    async with make(fn, size=1, wait_ms=0, queue=2) as batcher:
        first = asyncio.create_task(batcher.submit(1))
        await wait_until_started(fn)
        waiting = [asyncio.create_task(batcher.submit(i)) for i in (2, 3)]
        await asyncio.sleep(0)
        with pytest.raises(BatcherFullError):
            await batcher.submit(4)
        gate.set()
        results = await asyncio.gather(first, *waiting)
    assert [r.value for r in results] == [10, 20, 30]


async def test_caller_cancelled_before_dispatch_is_skipped() -> None:
    fn = Recorder()
    async with make(fn, wait_ms=100) as batcher:
        gone = asyncio.create_task(batcher.submit(1))
        kept = asyncio.create_task(batcher.submit(2))
        await asyncio.sleep(0.005)
        gone.cancel()
        assert (await kept).value == 20
    assert fn.batches == [[2]]
    assert gone.cancelled()


async def test_caller_cancelled_during_inference_does_not_break_the_batcher() -> None:
    gate = threading.Event()
    fn = Recorder(gate=gate)
    async with make(fn, wait_ms=10) as batcher:
        gone = asyncio.create_task(batcher.submit(1))
        kept = asyncio.create_task(batcher.submit(2))
        await wait_until_started(fn)
        gone.cancel()
        gate.set()
        assert (await kept).value == 20
        assert (await batcher.submit(3)).value == 30
    assert gone.cancelled()


async def test_event_loop_stays_responsive_during_a_slow_batch() -> None:
    fn = Recorder(delay_s=0.2)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async with make(fn, wait_ms=0) as batcher:
        beat = asyncio.create_task(heartbeat())
        await batcher.submit(1)
        beat.cancel()
    assert ticks >= 10


async def test_stop_during_the_wait_window_fails_the_caller() -> None:
    fn = Recorder()
    batcher = make(fn, wait_ms=1000)
    batcher.start()
    waiting = asyncio.create_task(batcher.submit(1))
    await asyncio.sleep(0.01)
    await batcher.stop()
    with pytest.raises(BatcherClosedError):
        await waiting
    assert fn.batches == []


async def test_stop_fails_in_flight_and_queued_callers_and_rejects_new_ones() -> None:
    gate = threading.Event()
    fn = Recorder(gate=gate)
    batcher = make(fn, size=1, wait_ms=0)
    batcher.start()
    in_flight = asyncio.create_task(batcher.submit(1))
    await wait_until_started(fn)
    queued = asyncio.create_task(batcher.submit(2))
    await asyncio.sleep(0)
    await batcher.stop()
    gate.set()
    for task in (in_flight, queued):
        with pytest.raises(BatcherClosedError):
            await task
    with pytest.raises(BatcherClosedError):
        await batcher.submit(3)


# YOUR TURN: test_submit_before_start_is_rejected

# Build a batcher with make(Recorder()) but do NOT start it.

# Calling submit(1) must raise BatcherClosedError. (3 lines)


async def test_submit_before_start_is_rejected() -> None:
    fn = Recorder()
    batcher = make(fn)
    with pytest.raises(BatcherClosedError):
        await batcher.submit(1)


@pytest.mark.parametrize(
    ("size", "wait_ms", "queue"), [(0, 5, 8), (4, -1, 8), (4, 5, 0)]
)
def test_bad_settings_are_rejected(size: int, wait_ms: float, queue: int) -> None:
    with pytest.raises(ValueError):
        make(Recorder(), size=size, wait_ms=wait_ms, queue=queue)
