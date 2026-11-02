"""Turning benchmark results into tables and Mermaid charts."""

import json
from pathlib import Path
from typing import Any

import pytest

from structflow.bench.charts import (
    bar_chart,
    find,
    latency_chart,
    main,
    nice_max,
    results_table,
    throughput_chart,
    whitespace_chart,
)


def row(task: str, system: str, concurrency: int, **numbers: float) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": task,
        "system": system,
        "concurrency": concurrency,
        "n": 60,
        "accuracy": 0.9,
        "valid_rate": 1.0,
        "retry_rate": 0.0,
        "mean_attempts": 1,
        "tokens_per_request": 3.0,
        "p50_ms": 40.0,
        "p99_ms": 80.0,
        "throughput_rps": 20.0,
        "errors": {},
    }
    return base | numbers


ROWS = [
    row("sentiment", "structflow", 1, p50_ms=10.2, throughput_rps=108.7),
    row("sentiment", "baseline", 1, p50_ms=44.0, throughput_rps=22.8),
    row("sentiment", "structflow", 32, p50_ms=35.0, throughput_rps=655.3),
    row("sentiment", "baseline", 32, p50_ms=90.0, throughput_rps=286.8),
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(655.3, 750), (90, 100), (48, 75), (0.4, 0.5), (230, 250), (0, 1)],
)
def test_nice_max_rounds_up_to_a_round_number(value: float, expected: float) -> None:
    assert nice_max(value) == pytest.approx(expected)


def test_bar_chart_is_a_mermaid_block() -> None:
    chart = bar_chart("Speed", "ms", ["a", "b"], [12.0, 3.25], horizontal=True)
    lines = chart.splitlines()
    assert lines[0] == "```mermaid" and lines[-1] == "```"
    assert "xychart-beta horizontal" in lines
    assert '    x-axis ["a", "b"]' in lines
    assert '    y-axis "ms" 0 --> 20' in lines
    assert "    bar [12, 3.2]" in lines


def test_bar_chart_needs_one_label_per_value() -> None:
    with pytest.raises(ValueError):
        bar_chart("t", "ms", ["only one"], [1.0, 2.0])


def test_find_picks_the_matching_row_or_says_what_is_missing() -> None:
    assert find(ROWS, "sentiment", "baseline", 32)["p50_ms"] == 90.0
    with pytest.raises(KeyError, match="invoice"):
        find(ROWS, "invoice", "baseline", 1)


def test_results_table_has_a_row_per_result() -> None:
    table = results_table(ROWS).splitlines()
    assert len(table) == 2 + len(ROWS)
    assert table[2].startswith("| sentiment | StructFlow | 1 | 90% | 100% | 0% |")


def test_throughput_chart_pairs_each_system_at_one_level() -> None:
    chart = throughput_chart(ROWS, 32)
    assert '["Sentiment · StructFlow", "Sentiment · baseline"]' in chart
    assert "bar [655.3, 286.8]" in chart


def test_latency_chart_covers_every_level() -> None:
    chart = latency_chart(ROWS, "sentiment")
    assert '"1 client · StructFlow"' in chart and '"32 clients · baseline"' in chart
    assert "bar [10.2, 44, 35, 90]" in chart


def test_whitespace_chart_compares_the_two_runs() -> None:
    allowed = [
        row("invoice", "structflow", 8, accuracy=0.9),
        row("invoice", "baseline", 8, accuracy=0.9),
    ]
    banned = [row("invoice", "structflow", 8, accuracy=0.4833)]
    chart = whitespace_chart(allowed, banned)
    assert "bar [90, 48.3, 90]" in chart
    assert '"% correct" 0 --> 100' in chart  # a percentage axis always ends at 100


def test_the_command_prints_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(ROWS))
    assert main([str(results)]) == 0
    out = capsys.readouterr().out
    assert out.count("```mermaid") == 2
    assert "| Task | System |" in out
