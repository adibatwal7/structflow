"""Stop working on a request when the client hangs up."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from starlette.requests import Request


class ClientDisconnectedError(Exception):
    """The client closed the connection before we answered."""


async def _wait_for_disconnect(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def unless_disconnected[T](request: Request, work: Coroutine[Any, Any, T]) -> T:
    """Run `work`, but cancel it if the client goes away first."""
    task = asyncio.create_task(work)
    watcher = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {task, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:  # we were cancelled ourselves: don't leave the work running
        task.cancel()
        raise
    finally:
        watcher.cancel()
    if task not in done:
        task.cancel()
        raise ClientDisconnectedError("the client closed the connection")
    return task.result()
