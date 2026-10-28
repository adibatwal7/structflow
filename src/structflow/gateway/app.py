import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from structflow.config import Settings, get_settings
from structflow.engines.base import Classifier, SpanExtractor
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.engines.limits import LimitedGenerator
from structflow.routing.router import Router

from .health import router as health_router
from .infer import router as infer_router
from .problems import install_problem_handlers
from .request_id import RequestIdMiddleware

if TYPE_CHECKING:  # only for type checkers: importing torch at startup is slow
    from structflow.engines.hf_classifier import HFClassifier
    from structflow.engines.hf_spans import HFSpanExtractor

logger = logging.getLogger("structflow.gateway")


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


def _load_hf_span_extractor(settings: Settings) -> "HFSpanExtractor":
    """Slow and blocking, like the classifier, so it also runs in a thread."""
    from structflow.engines.hf_spans import HFSpanExtractor

    return HFSpanExtractor.load(
        settings.span_model,
        device=settings.span_device,
        max_length=settings.span_max_length,
        max_batch_size=settings.microbatch_max_size,
        max_wait_ms=settings.microbatch_max_wait_ms,
        max_queue_size=settings.microbatch_max_queue,
    )


def _uses_real_models(settings: Settings) -> bool:
    return settings.classifier_backend == "hf" or settings.span_backend == "hf"


async def _build_router(settings: Settings, stack: AsyncExitStack) -> Router:
    """Real models load in a worker thread, then join the exit stack to stop cleanly."""
    classifier: Classifier = FakeClassifier()
    span_extractor: SpanExtractor = FakeSpanExtractor()
    if settings.classifier_backend == "hf":
        loaded = await asyncio.to_thread(_load_hf_classifier, settings)
        classifier = await stack.enter_async_context(loaded)
    if settings.span_backend == "hf":
        loaded_spans = await asyncio.to_thread(_load_hf_span_extractor, settings)
        span_extractor = await stack.enter_async_context(loaded_spans)

    generator = LimitedGenerator(
        FakeGenerator(delay_s=settings.fake_generation_delay_ms / 1000),
        max_inflight=settings.max_inflight_generation,
        timeout_s=settings.request_timeout_s,
    )
    return Router(
        classifier=classifier,
        span_extractor=span_extractor,
        generator=generator,
    )


async def _load_in_background(
    app: FastAPI, settings: Settings, stack: AsyncExitStack
) -> None:
    """The server answers /healthz while this runs."""
    try:
        app.state.router = await _build_router(settings, stack)
    except Exception:
        logger.exception("could not load the models")
        app.state.startup_error = "could not load the models (see the server log)"
        return
    app.state.ready = True
    logger.info("models are ready")


def create_app(
    settings: Settings | None = None, *, router: Router | None = None
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            if router is not None:
                app.state.router = router
                app.state.ready = True
            elif not _uses_real_models(resolved):
                app.state.router = await _build_router(resolved, stack)
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
