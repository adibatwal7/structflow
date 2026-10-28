import asyncio

import pytest
from starlette.requests import Request
from starlette.types import Message

from structflow.gateway.disconnect import ClientDisconnectedError, unless_disconnected

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def request_that_hangs_up_when(event: asyncio.Event) -> Request:
    async def receive() -> Message:
        await event.wait()
        return {"type": "http.disconnect"}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


async def test_the_answer_comes_back_while_the_client_waits() -> None:
    async def quick() -> str:
        return "done"

    never = asyncio.Event()
    assert (
        await unless_disconnected(request_that_hangs_up_when(never), quick()) == "done"
    )


async def test_work_is_cancelled_when_the_client_hangs_up() -> None:
    started, cancelled, hang_up = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "too late"

    async def client_leaves() -> None:
        await started.wait()
        hang_up.set()

    leaving = asyncio.create_task(client_leaves())
    with pytest.raises(ClientDisconnectedError):
        await unless_disconnected(request_that_hangs_up_when(hang_up), slow())
    await leaving
    await asyncio.sleep(0)  # let the cancellation reach the work
    assert cancelled.is_set()


async def test_errors_from_the_work_still_reach_the_caller() -> None:
    async def broken() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await unless_disconnected(request_that_hangs_up_when(asyncio.Event()), broken())
