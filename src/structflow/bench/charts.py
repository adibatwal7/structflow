"""Turn benchmark results into the README's tables and charts.

    python -m structflow.bench.charts benchmarks/results/rtx-pro-6000.json \\
        --whitespace benchmarks/results/invoice-whitespace-allowed.json \\
                     benchmarks/results/invoice-whitespace-banned.json

Prints Markdown: a results table and Mermaid bar charts, which GitHub draws.
Every number in the README's results section comes from here, so the charts can't
drift away from the data.

One series per chart, with the category names saying what each bar is: Mermaid's
bar charts have no legend, and two series would be drawn on top of each other.
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]
BLUE = "#1565C0"
SYSTEM_NAMES = {"structflow": "StructFlow", "baseline": "baseline"}


def load(path: str | Path) -> list[Row]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise TypeError(f"{path} is not a list of result rows")
    return rows


def find(rows: Sequence[Row], task: str, system: str, concurrency: int) -> Row:
    for row in rows:
        if (row["task"], row["system"], row["concurrency"]) == (
            task,
            system,
            concurrency,
        ):
            return row
    raise KeyError(f"no result for {task} / {system} / concurrency {concurrency}")


def nice_max(value: float) -> float:
    """A round axis maximum just above `value`, like 75, 250 or 1000."""
    if value <= 0:
        return 1.0
    target = value * 1.05
    power = 10.0 ** math.floor(math.log10(target))
    for step in (1, 2, 2.5, 4, 5, 7.5, 10):
        if step * power >= target:
            return step * power
    raise AssertionError("unreachable")


def _number(value: float) -> str:
    return f"{value:g}" if value == int(value) else f"{value:.1f}"


def bar_chart(
    title: str,
    axis: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    horizontal: bool = False,
    top: float | None = None,
) -> str:
    if len(labels) != len(values):
        raise ValueError("one label per value")
    names = ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)
    bars = ", ".join(_number(round(v, 1)) for v in values)
    theme = json.dumps({"themeVariables": {"xyChart": {"plotColorPalette": BLUE}}})
    kind = "xychart-beta horizontal" if horizontal else "xychart-beta"
    return "\n".join(
        [
            "```mermaid",
            f"%%{{init: {theme}}}%%",
            kind,
            f"    title {json.dumps(title, ensure_ascii=False)}",
            f"    x-axis [{names}]",
            f'    y-axis "{axis}" 0 --> {_number(top or nice_max(max(values)))}',
            f"    bar [{bars}]",
            "```",
        ]
    )


def results_table(rows: Sequence[Row]) -> str:
    lines = [
        (
            "| Task | System | Clients | Correct | Valid | Retried "
            "| Tokens | p50 ms | p99 ms | Req/s |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {SYSTEM_NAMES.get(r['system'], r['system'])} "
            f"| {r['concurrency']} | {r['accuracy']:.0%} | {r['valid_rate']:.0%} "
            f"| {r['retry_rate']:.0%} | {r['tokens_per_request']:.1f} "
            f"| {r['p50_ms']:.0f} | {r['p99_ms']:.0f} | {r['throughput_rps']:.1f} |"
        )
    return "\n".join(lines)


def tasks_in(rows: Sequence[Row]) -> list[str]:
    return list(dict.fromkeys(r["task"] for r in rows))  # in first-seen order


def throughput_chart(rows: Sequence[Row], concurrency: int) -> str:
    labels, values = [], []
    for task in tasks_in(rows):
        for system in ("structflow", "baseline"):
            row = find(rows, task, system, concurrency)
            labels.append(f"{task.capitalize()} · {SYSTEM_NAMES[system]}")
            values.append(row["throughput_rps"])
    return bar_chart(
        f"Requests answered per second, {concurrency} concurrent clients "
        "(higher is better)",
        "requests/s",
        labels,
        values,
        horizontal=True,
    )


def latency_chart(rows: Sequence[Row], task: str) -> str:
    levels = sorted({r["concurrency"] for r in rows if r["task"] == task})
    labels, values = [], []
    for level in levels:
        for system in ("structflow", "baseline"):
            row = find(rows, task, system, level)
            client = "client" if level == 1 else "clients"
            labels.append(f"{level} {client} · {SYSTEM_NAMES[system]}")
            values.append(row["p50_ms"])
    return bar_chart(
        f"{task.capitalize()}: median latency (lower is better)",
        "ms",
        labels,
        values,
        horizontal=True,
    )


def whitespace_chart(allowed: Sequence[Row], banned: Sequence[Row]) -> str:
    level = allowed[0]["concurrency"]
    return bar_chart(
        f"Invoice JSON: share of {allowed[0]['n']} answers correct (higher is better)",
        "% correct",
        [
            "StructFlow, whitespace allowed",
            "StructFlow, whitespace banned",
            "Baseline (prompt + parse)",
        ],
        [
            100 * find(allowed, "invoice", "structflow", level)["accuracy"],
            100 * find(banned, "invoice", "structflow", level)["accuracy"],
            100 * find(allowed, "invoice", "baseline", level)["accuracy"],
        ],
        top=100,
    )


def report(rows: Sequence[Row], whitespace: Sequence[Sequence[Row]] = ()) -> str:
    top = max(r["concurrency"] for r in rows)
    parts = [
        results_table(rows),
        throughput_chart(rows, top),
        latency_chart(rows, "sentiment"),
    ]
    if whitespace:
        allowed, banned = whitespace
        parts.append(whitespace_chart(allowed, banned))
    return "\n\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="JSON written by structflow.bench.compare")
    parser.add_argument(
        "--whitespace",
        nargs=2,
        metavar=("ALLOWED", "BANNED"),
        help="the two invoice-only runs, with JSON whitespace allowed and banned",
    )
    args = parser.parse_args(argv)
    whitespace = [load(p) for p in args.whitespace] if args.whitespace else []
    print(report(load(args.results), whitespace), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
