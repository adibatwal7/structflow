"""Async micro-batcher: many concurrent callers share one forward pass.

Callers ``await submit(item)``. A single collector task gathers items until the
batch is full or the wait window closes, runs the blocking batch function on a
dedicated worker thread, and hands each caller its own result.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Self

logger = logging.getLogger(__name__)


class BatcherError(Exception):
    """Base class for micro-batcher errors."""


class BatcherFullError(BatcherError):
    """The waiting line is full. The caller should back off and retry."""


class BatcherClosedError(BatcherError):
    """The batcher is not running (not started yet, or already stopped)."""


@dataclass(frozen=True, slots=True)
class Batched[R]:
    """One caller's result, plus where its time went."""

    value: R
    queue_ms: float
    inference_ms: float
    batch_size: int


@dataclass(slots=True)
class _Pending[T, R]:
    item: T
    future: asyncio.Future[Batched[R]]
    enqueued_at: float


def _fail_all[T, R](batch: Sequence[_Pending[T, R]], exc: BaseException) -> None:
    for p in batch:
        if not p.future.done():
            p.future.set_exception(exc)


class MicroBatcher[T, R]:
    def __init__(
        self,
        batch_fn: Callable[[Sequence[T]], Sequence[R]],
        *,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_size: int,
        name: str = "microbatcher",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000
        self._name = name
        self._queue: asyncio.Queue[_Pending[T, R]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError(f"{self._name} was already started")
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name=self._name
        )

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        leftovers: list[_Pending[T, R]] = []
        while not self._queue.empty():
            leftovers.append(self._queue.get_nowait())
        _fail_all(leftovers, BatcherClosedError(f"{self._name} stopped"))
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def submit(self, item: T) -> Batched[R]:
        if not self.running:
            raise BatcherClosedError(f"{self._name} is not running")
        future: asyncio.Future[Batched[R]] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Pending(item, future, time.perf_counter()))
        except asyncio.QueueFull:
            raise BatcherFullError(
                f"{self._name} has {self._queue.maxsize} requests waiting"
            ) from None
        return await future

    async def _run(self) -> None:
        while True:
            batch: list[_Pending[T, R]] = []
            try:
                await self._collect(batch)
                await self._dispatch(batch)
            except asyncio.CancelledError:
                _fail_all(batch, BatcherClosedError(f"{self._name} stopped"))
                raise

    async def _collect(self, batch: list[_Pending[T, R]]) -> None:
        batch.append(await self._queue.get())
        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self._max_batch_size:
            if not self._queue.empty():
                batch.append(self._queue.get_nowait())
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                async with asyncio.timeout(remaining):
                    batch.append(await self._queue.get())
            except TimeoutError:
                return

    async def _dispatch(self, batch: list[_Pending[T, R]]) -> None:
        live = [pending for pending in batch if not pending.future.done()]
        if not live:
            return
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(
                self._executor, self._batch_fn, [p.item for p in live]
            )
        except Exception as exc:
            logger.exception("%s: batch of %d failed", self._name, len(live))
            _fail_all(live, exc)
            return
        inference_ms = (time.perf_counter() - started) * 1000
        if len(outputs) != len(live):
            _fail_all(
                live,
                RuntimeError(
                    f"batch function returned {len(outputs)} outputs for {len(live)} inputs"
                ),
            )
            return
        for pending, output in zip(live, outputs, strict=True):
            if not pending.future.done():
                pending.future.set_result(
                    Batched(
                        value=output,
                        queue_ms=(started - pending.enqueued_at) * 1000,
                        inference_ms=inference_ms,
                        batch_size=len(live),
                    )
                )
