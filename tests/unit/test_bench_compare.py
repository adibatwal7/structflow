"""The benchmark runner, against a fake StructFlow."""

import asyncio
import json
from typing import Any

import httpx2
import pytest

from structflow.bench.compare import (
    RETRY_TEMPERATURE,
    Outcome,
    ask_baseline,
    ask_structflow,
    render,
    run_level,
    summarize,
)
from structflow.bench.tasks import all_tasks

pytestmark = pytest.mark.anyio
SENTIMENT, TOPIC, INVOICE = all_tasks(n=10)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def classified(label: str) -> dict[str, Any]:
    return {
        "route": "softmax_head",
        "result": {"task": "classify", "labels": [{"label": label, "score": 0.9}]},
        "usage": {"prompt_tokens": 9, "completion_tokens": 0, "forward_passes": 1},
    }


def generated(text: str, tokens: int = 5) -> dict[str, Any]:
    return {
        "route": "unconstrained_generation",
        "result": {"task": "generate", "text": text, "finish_reason": "stop"},
        "usage": {"prompt_tokens": 30, "completion_tokens": tokens},
    }


class FakeStructFlow:
    """Answers each request with the next reply in its list; remembers what it got."""

    def __init__(self, *replies: tuple[int, dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sent: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(json.loads(request.content))
        status, body = self.replies.pop(0)
        return httpx2.Response(status, json=body)


def client(fake: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url="http://structflow.test", transport=httpx2.MockTransport(fake)
    )


async def test_structflow_answers_once_and_is_scored() -> None:
    ex = SENTIMENT.examples[0]
    fake = FakeStructFlow((200, classified(ex.expected)))
    out = await ask_structflow(client(fake), SENTIMENT, ex)
    assert (out.ok, out.correct, out.attempts, out.route) == (
        True,
        True,
        1,
        "softmax_head",
    )
    assert fake.sent[0]["task"] == "classify"


async def test_a_structflow_error_is_recorded_not_raised() -> None:
    fake = FakeStructFlow((502, {"code": "upstream_error"}))
    out = await ask_structflow(client(fake), INVOICE, INVOICE.examples[0])
    assert (out.ok, out.correct, out.error) == (False, False, "upstream_error")


async def test_the_baseline_retries_until_it_can_parse() -> None:
    ex = SENTIMENT.examples[0]
    fake = FakeStructFlow(
        (200, generated("Hmm, hard to say.", tokens=6)),
        (200, generated(ex.expected, tokens=2)),
    )
    out = await ask_baseline(client(fake), SENTIMENT, ex)
    assert (out.ok, out.correct, out.attempts) == (True, True, 2)
    assert out.route == "unconstrained_generation"
    assert out.completion_tokens == 8  # both attempts are paid for
    first, retry = (sent["sampling"]["temperature"] for sent in fake.sent)
    assert (first, retry) == (0.0, RETRY_TEMPERATURE)


async def test_the_baseline_gives_up_after_max_attempts() -> None:
    fake = FakeStructFlow(*[(200, generated("no idea"))] * 3)
    out = await ask_baseline(client(fake), SENTIMENT, SENTIMENT.examples[0])
    assert (out.ok, out.attempts, out.error) == (False, 3, "unparseable")
    assert len(fake.sent) == 3


async def test_a_dropped_connection_is_an_outcome_too() -> None:
    def refused(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    out = await ask_baseline(client(refused), TOPIC, TOPIC.examples[0], max_attempts=2)
    assert (out.ok, out.attempts, out.error) == (False, 2, "no_answer")


async def test_run_level_keeps_at_most_n_requests_in_flight() -> None:
    in_flight = peak = 0

    async def answer(c: Any, task: Any, ex: Any) -> Outcome:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return Outcome(True, True, 1, 10.0, 0)

    outcomes, wall_s = await run_level(
        client(FakeStructFlow()), answer, SENTIMENT, SENTIMENT.examples, concurrency=3
    )
    assert len(outcomes) == len(SENTIMENT.examples)
    assert peak == 3
    assert wall_s > 0


def test_summarize_turns_outcomes_into_rates() -> None:
    outcomes = [
        Outcome(ok=True, correct=True, attempts=1, latency_ms=100, completion_tokens=4),
        Outcome(
            ok=True, correct=False, attempts=2, latency_ms=300, completion_tokens=9
        ),
        Outcome(
            ok=False,
            correct=False,
            attempts=3,
            latency_ms=900,
            completion_tokens=15,
            error="unparseable",
            route="unconstrained_generation",
        ),
        Outcome(ok=True, correct=True, attempts=1, latency_ms=200, completion_tokens=4),
    ]
    row = summarize(outcomes, task="t", system="baseline", concurrency=1, wall_s=2.0)
    assert (row.accuracy, row.valid_rate, row.retry_rate) == (0.5, 0.75, 0.5)
    assert row.mean_attempts == pytest.approx(1.75)
    assert row.tokens_per_request == pytest.approx(8.0)
    assert (row.p50_ms, row.p99_ms) == (200, 900)
    assert row.throughput_rps == pytest.approx(2.0)
    assert row.errors == {"unparseable": 1}
    assert "baseline" in render([row]) and "50%" in render([row])


async def test_the_last_failure_is_the_one_reported() -> None:
    fake = FakeStructFlow(
        (503, {"code": "backend_saturated"}),
        (200, generated("no idea")),
        (200, generated("still no idea")),
    )
    out = await ask_baseline(client(fake), SENTIMENT, SENTIMENT.examples[0])
    assert (out.ok, out.error) == (False, "unparseable")
