import pytest

from structflow.bench.load import Sample, percentile, render, summarize

HUNDRED = [float(v) for v in range(1, 101)]  # 1.0, 2.0, ..., 100.0


@pytest.mark.parametrize(
    ("p", "expected"), [(50, 50.0), (90, 90.0), (99, 99.0), (100, 100.0)]
)
def test_percentile_uses_nearest_rank(p: float, expected: float) -> None:
    assert percentile(HUNDRED, p) == expected


def test_percentile_of_a_tiny_sample() -> None:
    assert percentile([7.0], 99) == 7.0
    assert percentile([], 50) == 0.0


def test_summary_counts_statuses_and_routes() -> None:
    samples = [
        Sample(200, 10.0, "softmax_head"),
        Sample(200, 30.0, "softmax_head"),
        Sample(200, 20.0, "llm_label_scoring"),
        Sample(503, 1.0, None),
        Sample(0, 30000.0, None),
    ]
    report = summarize(samples, duration_s=2.0)
    assert report.sent == 5
    assert report.statuses == {0: 1, 200: 3, 503: 1}
    assert report.routes == {"softmax_head": 2, "llm_label_scoring": 1}
    assert (report.p50_ms, report.max_ms) == (20.0, 30.0)  # 200s only: 10, 20, 30


def test_render_mentions_every_number_that_matters() -> None:
    text = render(summarize([Sample(200, 12.5, "span_head")], duration_s=1.0))
    assert "1 requests" in text and "200: 1" in text and "span_head: 1" in text
    assert "p99 12.5 ms" in text
