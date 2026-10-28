import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from structflow.config import Settings, get_settings
from structflow.engines.base import Classifier
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.routing.router import Router

from .health import router as health_router
from .infer import router as infer_router
from .problems import install_problem_handlers
from .request_id import RequestIdMiddleware

if TYPE_CHECKING:  # only for type checkers: importing torch at startup is slow
    from structflow.engines.hf_classifier import HFClassifier

logger = logging.getLogger("structflow.gateway")


def _router_with(classifier: Classifier) -> Router:
    return Router(
        classifier=classifier,
        span_extractor=FakeSpanExtractor(),
        generator=FakeGenerator(),
    )


def _load_hf_classifier(settings: Settings) -> "HFClassifier":
    """Slow and blocking (imports torch, reads the weights), so it runs in a thread."""
    from structflow.engines.hf_classifier import HFClassifier

    return HFClassifier.load(
        settings.classifier_model,
        device=settings.classifier_device,
        max_length=settings.classifier_max_length,
        max_batch_size=settings.microbatch_max_size,
        max_wait_ms=settings.microbatch_max_wait_ms,
        max_queue_size=settings.microbatch_max_queue,
    )


async def _load_in_background(
    app: FastAPI, settings: Settings, stack: AsyncExitStack
) -> None:
    """The server answers /healthz while this runs."""
    try:
        classifier = await asyncio.to_thread(_load_hf_classifier, settings)
        await stack.enter_async_context(classifier)
    except Exception:
        logger.exception("could not load %s", settings.classifier_model)
        app.state.startup_error = f"could not load {settings.classifier_model}"
        return
    app.state.router = _router_with(classifier)
    app.state.ready = True
    logger.info("%s is ready on %s", classifier.name, classifier.device)


def create_app(
    settings: Settings | None = None, *, router: Router | None = None
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if router is not None or resolved.classifier_backend == "fake":
                app.state.router = router or _router_with(FakeClassifier())
                app.state.ready = True
            else:
                loading = asyncio.create_task(_load_in_background(app, resolved, stack))
                stack.callback(loading.cancel)
            yield
            app.state.ready = False

    app = FastAPI(title="StructFlow", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved
    app.state.ready = False
    app.state.router = None
    app.state.startup_error = None
    app.add_middleware(RequestIdMiddleware)
    install_problem_handlers(app)
    app.include_router(health_router)
    app.include_router(infer_router)
    return app
