"""StructFlow vs the usual way: answer every benchmark task both ways and compare.

    python -m structflow.bench.compare --concurrency 1,8,32 --json results.json

- **structflow**: one request whose answer shape is guaranteed.
- **baseline**: ask the LLM in the prompt, parse its free text, retry if the parse
  fails (up to --max-attempts). The first try uses temperature 0; retries use 0.7,
  because asking again at temperature 0 would just repeat the same wrong answer.

Both go through the same StructFlow gateway (the baseline uses its plain `generate`
route), so the network, the GPU and the model are identical; only the method differs.

This is a *closed-loop* test: `concurrency` workers each send their next request as
soon as the last one finishes. It answers "how fast can N busy clients get through
this work?", while bench/load.py's open loop answers "what happens at a fixed rate?".
"""

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import httpx2

from structflow.bench.load import percentile
from structflow.bench.tasks import Example, Task, all_tasks

FIRST_TEMPERATURE = 0.0
RETRY_TEMPERATURE = 0.7
BASELINE_MAX_TOKENS = 200


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one example, answered one way."""

    ok: bool  # we got an answer in the right shape
    correct: bool  # ...and it was the right answer
    attempts: int  # LLM calls it took (StructFlow: always 1)
    latency_ms: float  # the whole thing, retries included
    completion_tokens: int  # generated tokens, all attempts together
    route: str | None = None
    error: str | None = None  # why it failed, if it did


@dataclass(frozen=True, slots=True)
class Summary:
    task: str
    system: str
    concurrency: int
    n: int
    accuracy: float  # share of examples answered correctly
    valid_rate: float  # share that got an answer in the right shape at all
    retry_rate: float  # share that needed more than one attempt
    mean_attempts: float
    tokens_per_request: float
    p50_ms: float
    p99_ms: float
    throughput_rps: float
    errors: dict[str, int]


Answerer = Callable[[httpx2.AsyncClient, Task, Example], Awaitable[Outcome]]


async def _post(
    client: httpx2.AsyncClient, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """POST to StructFlow. A dropped connection or a non-JSON reply counts as status 0."""
    try:
        response = await client.post("/v1/infer", json=body)
        data = response.json()
    except (httpx2.HTTPError, ValueError):
        return 0, {"code": "no_answer"}
    return response.status_code, data if isinstance(data, dict) else {}


async def ask_structflow(
    client: httpx2.AsyncClient, task: Task, ex: Example
) -> Outcome:
    started = time.perf_counter()
    status, body = await _post(client, task.structflow_body(ex))
    latency_ms = (time.perf_counter() - started) * 1000
    if status != 200:
        return Outcome(
            ok=False,
            correct=False,
            attempts=1,
            latency_ms=latency_ms,
            completion_tokens=0,
            error=str(body.get("code", status)),
        )
    answer = task.answer_from(body)
    return Outcome(
        ok=True,
        correct=task.correct(answer, ex.expected),
        attempts=1,
        latency_ms=latency_ms,
        completion_tokens=body["usage"]["completion_tokens"],
        route=body["route"],
    )


async def ask_baseline(
    client: httpx2.AsyncClient, task: Task, ex: Example, *, max_attempts: int = 3
) -> Outcome:
    """Ask in the prompt, parse the free text, and retry if the parse fails."""
    started = time.perf_counter()
    tokens = 0
    error = "unparseable"
    for attempt in range(1, max_attempts + 1):
        temperature = FIRST_TEMPERATURE if attempt == 1 else RETRY_TEMPERATURE
        status, body = await _post(
            client,
            {
                "task": "generate",
                "messages": task.baseline_messages(ex),
                "sampling": {
                    "temperature": temperature,
                    "max_tokens": BASELINE_MAX_TOKENS,
                },
            },
        )
        if status != 200:
            error = str(body.get("code", status))
            continue
        tokens += body["usage"]["completion_tokens"]
        answer = task.parse(body["result"]["text"])
        if answer is not None:
            return Outcome(
                ok=True,
                correct=task.correct(answer, ex.expected),
                attempts=attempt,
                latency_ms=(time.perf_counter() - started) * 1000,
                completion_tokens=tokens,
                route=body["route"],
            )
        error = "unparseable"
    return Outcome(
        ok=False,
        correct=False,
        attempts=max_attempts,
        latency_ms=(time.perf_counter() - started) * 1000,
        completion_tokens=tokens,
        error=error,
    )


async def run_level(
    client: httpx2.AsyncClient,
    answer: Answerer,
    task: Task,
    examples: Sequence[Example],
    *,
    concurrency: int,
) -> tuple[list[Outcome], float]:
    """Answer every example with `concurrency` workers. Returns (outcomes, wall seconds)."""
    queue = list(enumerate(examples))
    results: list[Outcome | None] = [None] * len(examples)

    async def worker() -> None:
        while queue:
            i, ex = queue.pop(0)
            results[i] = await answer(client, task, ex)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_s = time.perf_counter() - started
    return [r for r in results if r is not None], wall_s


def summarize(
    outcomes: Sequence[Outcome],
    *,
    task: str,
    system: str,
    concurrency: int,
    wall_s: float,
) -> Summary:
    n = len(outcomes)
    latencies = sorted(o.latency_ms for o in outcomes)
    errors: dict[str, int] = {}
    for o in outcomes:
        if o.error is not None:
            errors[o.error] = errors.get(o.error, 0) + 1
    return Summary(
        task=task,
        system=system,
        concurrency=concurrency,
        n=n,
        accuracy=sum(o.correct for o in outcomes) / n,
        valid_rate=sum(o.ok for o in outcomes) / n,
        retry_rate=sum(o.attempts > 1 for o in outcomes) / n,
        mean_attempts=mean(o.attempts for o in outcomes),
        tokens_per_request=mean(o.completion_tokens for o in outcomes),
        p50_ms=percentile(latencies, 50),
        p99_ms=percentile(latencies, 99),
        throughput_rps=n / wall_s if wall_s > 0 else 0.0,
        errors=errors,
    )


def render(rows: Sequence[Summary]) -> str:
    header = (
        f"{'task':<10}{'system':<12}{'conc':>5}{'acc':>7}{'valid':>7}{'retry':>7}"
        f"{'tokens':>8}{'p50 ms':>9}{'p99 ms':>9}{'req/s':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.task:<10}{r.system:<12}{r.concurrency:>5}{r.accuracy:>7.0%}"
            f"{r.valid_rate:>7.0%}{r.retry_rate:>7.0%}{r.tokens_per_request:>8.1f}"
            f"{r.p50_ms:>9.0f}{r.p99_ms:>9.0f}{r.throughput_rps:>8.1f}"
        )
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> list[Summary]:
    tasks = [t for t in all_tasks(args.n) if t.name in args.tasks]
    levels = [int(c) for c in args.concurrency.split(",")]
    systems: dict[str, Answerer] = {
        "structflow": ask_structflow,
        "baseline": lambda c, t, ex: ask_baseline(
            c, t, ex, max_attempts=args.max_attempts
        ),
    }
    limits = httpx2.Limits(max_connections=max(levels))
    rows = []
    print(render([]), flush=True)
    async with httpx2.AsyncClient(
        base_url=args.url, limits=limits, timeout=args.timeout
    ) as client:
        for task in tasks:
            for name, answer in systems.items():
                await answer(client, task, task.examples[0])  # warm-up: not measured
                for level in levels:
                    outcomes, wall_s = await run_level(
                        client, answer, task, task.examples, concurrency=level
                    )
                    row = summarize(
                        outcomes,
                        task=task.name,
                        system=name,
                        concurrency=level,
                        wall_s=wall_s,
                    )
                    print(render([row]).splitlines()[-1], flush=True)
                    rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--tasks", default="sentiment,topic,invoice")
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument("--n", type=int, default=60, help="examples per task")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args(argv)
    args.tasks = args.tasks.split(",")
    rows = asyncio.run(_main(args))
    print()
    print(render(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
