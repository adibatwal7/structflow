"""The structured fallback: only when the caller allows it, and always marked degraded."""

from collections.abc import Mapping

import pytest

from structflow.engines.base import (
    ConstraintRejectedError,
    EngineOutput,
    OutputInvalidError,
)
from structflow.engines.fake import FakeClassifier, FakeGenerator, FakeSpanExtractor
from structflow.routing.router import Router
from structflow.schemas.common import Route
from structflow.schemas.requests import (
    INFERENCE_REQUEST_ADAPTER,
    InferenceRequest,
    StructuredRequest,
)
from structflow.schemas.responses import InferenceResponse, StructuredResult

pytestmark = pytest.mark.anyio

USER = [{"role": "user", "content": "Who sold this? ACME Corp, $40"}]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class PickyGenerator(FakeGenerator):
    """Fails the constrained call in a chosen way; the unconstrained one works."""

    def __init__(self, failure: Exception, unconstrained_fails: bool = False) -> None:
        super().__init__(structured_output={"vendor": "ACME"})
        self.failure = failure
        self.unconstrained_fails = unconstrained_fails
        self.calls: list[bool] = []

    async def generate_structured(
        self, request: StructuredRequest, *, constrained: bool = True
    ) -> EngineOutput[StructuredResult]:
        self.calls.append(constrained)
        if constrained or self.unconstrained_fails:
            raise self.failure
        return await super().generate_structured(request, constrained=False)


def body(allow: bool) -> Mapping[str, object]:
    return {
        "task": "structured",
        "messages": USER,
        "constraint": {"type": "json_schema", "schema": {"type": "object"}},
        "allow_unconstrained_fallback": allow,
    }


async def handle(generator: FakeGenerator, allow: bool) -> InferenceResponse:
    router = Router(
        classifier=FakeClassifier(),
        span_extractor=FakeSpanExtractor(),
        generator=generator,
    )
    request: InferenceRequest = INFERENCE_REQUEST_ADAPTER.validate_python(body(allow))
    return await router.handle(request, request_id="r1")


async def test_a_working_constraint_is_never_degraded() -> None:
    response = await handle(FakeGenerator(structured_output={"v": 1}), allow=True)
    assert response.route == Route.CONSTRAINED_GENERATION
    assert response.degraded is False


@pytest.mark.parametrize(
    "failure",
    [ConstraintRejectedError("bad regex"), OutputInvalidError("missing field")],
    ids=["refused", "invalid-answer"],
)
async def test_with_permission_it_falls_back_and_says_so(failure: Exception) -> None:
    generator = PickyGenerator(failure)
    response = await handle(generator, allow=True)
    assert response.route == Route.UNCONSTRAINED_GENERATION
    assert response.route_trail == (
        Route.CONSTRAINED_GENERATION,
        Route.UNCONSTRAINED_GENERATION,
    )
    assert response.degraded is True
    assert generator.calls == [True, False]


async def test_without_permission_the_error_reaches_the_client() -> None:
    generator = PickyGenerator(ConstraintRejectedError("bad regex"))
    with pytest.raises(ConstraintRejectedError):
        await handle(generator, allow=False)
    assert generator.calls == [True]  # never tried without the mask


async def test_if_the_fallback_answer_is_also_wrong_nothing_gets_through() -> None:
    generator = PickyGenerator(OutputInvalidError("no"), unconstrained_fails=True)
    with pytest.raises(OutputInvalidError):
        await handle(generator, allow=True)
