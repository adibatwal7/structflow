"""Open-loop load test: send requests on a fixed schedule and measure what comes back.

    python -m structflow.bench.load --task classify --rps 50 --duration 10

"Open loop" means the next request goes out on time even if earlier ones are still
waiting. Latency is measured from when a request was *due*, not from when it was
actually sent, so a slow server can't hide its delays by slowing the test down.
"""

import argparse
import asyncio
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import httpx2

BODIES: dict[str, dict[str, Any]] = {
    "classify": {
        "task": "classify",
        "text": "What a fantastic film, I loved every minute of it.",
        "labels": ["POSITIVE", "NEGATIVE"],
    },
    "extract_spans": {
        "task": "extract_spans",
        "text": "Tim Cook met Sundar Pichai in Paris.",
    },
    "structured": {
        "task": "structured",
        "messages": [
            {"role": "user", "content": "Extract the vendor from: ACME Corp, $40"}
        ],
        "constraint": {"type": "json_schema", "schema": {"type": "object"}},
    },
    "generate": {
        "task": "generate",
        "messages": [{"role": "user", "content": "Say hello"}],
    },
}


@dataclass(frozen=True, slots=True)
class Sample:
    status: int  # 0 means no answer at all (connection error or client timeout)
    latency_ms: float
    route: str | None


@dataclass(frozen=True, slots=True)
class Report:
    sent: int
    duration_s: float
    statuses: dict[int, int]
    routes: dict[str, int]
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest rank: the smallest value with at least p% of the values at or below it."""
    if not sorted_values:
        return 0.0
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def summarize(samples: Sequence[Sample], duration_s: float) -> Report:
    ok = sorted(s.latency_ms for s in samples if s.status == 200)
    return Report(
        sent=len(samples),
        duration_s=duration_s,
        statuses=dict(sorted(Counter(s.status for s in samples).items())),
        routes=dict(Counter(s.route for s in samples if s.route is not None)),
        p50_ms=percentile(ok, 50),
        p90_ms=percentile(ok, 90),
        p99_ms=percentile(ok, 99),
        max_ms=ok[-1] if ok else 0.0,
    )


async def run(
    client: httpx2.AsyncClient,
    body: Mapping[str, Any],
    *,
    rps: float,
    duration_s: float,
) -> list[Sample]:
    loop = asyncio.get_running_loop()
    start = loop.time()

    async def one(i: int) -> Sample:
        due = start + i / rps
        await asyncio.sleep(max(0.0, due - loop.time()))
        try:
            response = await client.post("/v1/infer", json=body)
        except httpx2.HTTPError:
            return Sample(status=0, latency_ms=(loop.time() - due) * 1000, route=None)
        latency_ms = (loop.time() - due) * 1000
        route = response.json().get("route") if response.status_code == 200 else None
        return Sample(status=response.status_code, latency_ms=latency_ms, route=route)

    return await asyncio.gather(*(one(i) for i in range(int(rps * duration_s))))


def render(report: Report) -> str:
    rate = report.sent / report.duration_s
    statuses = "   ".join(f"{code}: {n}" for code, n in report.statuses.items())
    routes = "   ".join(f"{route}: {n}" for route, n in report.routes.items()) or "-"
    latency = (
        f"p50 {report.p50_ms:.1f} ms   p90 {report.p90_ms:.1f} ms   "
        f"p99 {report.p99_ms:.1f} ms   max {report.max_ms:.1f} ms"
    )
    return (
        f"sent     {report.sent} requests in {report.duration_s:.1f} s ({rate:.1f} req/s)\n"
        f"status   {statuses}\n"
        f"routes   {routes}\n"
        f"latency  {latency}   (200s only)"
    )


async def _main(args: argparse.Namespace) -> Report:
    limits = httpx2.Limits(max_connections=args.connections)
    async with httpx2.AsyncClient(
        base_url=args.url, limits=limits, timeout=args.timeout
    ) as client:
        for _ in range(args.warmup):  # the first requests pay one-time costs: skip them
            await client.post("/v1/infer", json=BODIES[args.task])
        loop = asyncio.get_running_loop()
        started = loop.time()
        samples = await run(
            client, BODIES[args.task], rps=args.rps, duration_s=args.duration
        )
        return summarize(samples, loop.time() - started)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--task", choices=sorted(BODIES), default="classify")
    parser.add_argument("--rps", type=float, default=20.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--connections", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0, help="client timeout, s")
    parser.add_argument(
        "--warmup", type=int, default=5, help="unmeasured requests first"
    )
    parser.add_argument("--json", help="also write the report to this file")
    args = parser.parse_args(argv)
    report = asyncio.run(_main(args))
    print(render(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(report), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
