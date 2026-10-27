from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from structflow.config import Settings, get_settings

from .health import router as health_router
from .problems import install_problem_handlers
from .request_id import RequestIdMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # startup: loads the model heads here, before we report ready
        app.state.ready = True
        yield  # the server handles requests while we're paused here
        # shutdown: stop accepting traffic, then clean up
        app.state.ready = False

    app = FastAPI(title="StructFlow", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved
    app.state.ready = False
    app.add_middleware(RequestIdMiddleware)
    install_problem_handlers(app)
    app.include_router(health_router)
    return app
