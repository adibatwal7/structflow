from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from structflow.config import Settings, get_settings
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.routing.router import Router

from .health import router as health_router
from .infer import router as infer_router
from .problems import install_problem_handlers
from .request_id import RequestIdMiddleware


def create_app(
    settings: Settings | None = None, *, router: Router | None = None
) -> FastAPI:
    resolved = settings or get_settings()
    engine_router = router or Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=FakeGenerator(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="StructFlow", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved
    app.state.ready = False
    app.state.router = engine_router
    app.add_middleware(RequestIdMiddleware)
    install_problem_handlers(app)
    app.include_router(health_router)
    app.include_router(infer_router)
    return app
