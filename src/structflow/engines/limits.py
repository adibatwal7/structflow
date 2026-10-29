"""Admission control for the slow, expensive path: LLM generation.

LimitedGenerator wraps any GenerationBackend. It keeps at most N calls in flight
(the rest are turned away at once) and gives every call a deadline.
"""

import asyncio
from collections.abc import Awaitable, Callable

from structflow.engines.base import EngineOutput, GenerationBackend
from structflow.schemas.requests import (
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    StructuredRequest,
)
from structflow.schemas.responses import (
    ClassifyResult,
    ExtractSpansResult,
    GenerateResult,
    StructuredResult,
)


class GenerationBusyError(Exception):
    """Too many generation calls are already running. Try again shortly."""


class GenerationTimeoutError(Exception):
    """A generation call ran past its deadline and was cancelled."""


class LimitedGenerator:
    def __init__(
        self, inner: GenerationBackend, *, max_inflight: int, timeout_s: float
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.name = inner.name
        self._inner = inner
        self._max_inflight = max_inflight
        self._timeout_s = timeout_s
        self.in_flight = 0

    async def _limited[T](self, call: Callable[[], Awaitable[T]]) -> T:
        if self.in_flight >= self._max_inflight:
            raise GenerationBusyError(
                f"{self._max_inflight} generation requests are already in flight"
            )
        self.in_flight += 1
        try:
            async with asyncio.timeout(self._timeout_s):
                return await call()
        except TimeoutError:
            raise GenerationTimeoutError(
                f"{self.name} did not answer within {self._timeout_s:g} s"
            ) from None
        finally:
            self.in_flight -= 1

    async def generate_structured(
        self, request: StructuredRequest, *, constrained: bool = True
    ) -> EngineOutput[StructuredResult]:
        return await self._limited(
            lambda: self._inner.generate_structured(request, constrained=constrained)
        )

    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        return await self._limited(lambda: self._inner.generate(request))

    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        return await self._limited(lambda: self._inner.score_labels(request))

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        return await self._limited(lambda: self._inner.extract_spans(request))
