from typing import Any

import pytest
from pydantic import ValidationError

from structflow.schemas.common import Route, TaskType
from structflow.schemas.responses import (
    ALLOWED_ROUTES,
    INFERENCE_RESPONSE_ADAPTER,
)


def envelope(**overrides: Any) -> dict[str, Any]:
    base: dict = {
        "request_id": "req-1",
        "model": "distilbert-sst2",
        "route": "softmax_head",
        "route_trail": ["softmax_head"],
        "result": {
            "task": "classify",
            "labels": [{"label": "pos", "score": 0.9}, {"label": "neg", "score": 0.1}],
        },
        "usage": {"prompt_tokens": 12, "forward_passes": 1},
        "timings": {"queue_ms": 1.0, "inference_ms": 8.0, "total_ms": 9.5},
    }
    base.update(overrides)
    return base


class TestEnvelopeRoundTrip:
    def test_valid_envelope_round_trips(self) -> None:
        env = envelope()
        resp = INFERENCE_RESPONSE_ADAPTER.validate_python(env)
        as_json = resp.model_dump_json()
        assert INFERENCE_RESPONSE_ADAPTER.validate_json(as_json) == resp


class TestRoutingValidation:
    def test_route_trail_must_end_with_route(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(route_trail=["llm_label_scoring"])
            )
        assert "route_trail" in str(exc_info.value)

    def test_route_must_match_task(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(
                    route="span_head",
                    route_trail=["span_head"],
                    result={
                        "task": "classify",
                        "labels": [{"label": "pos", "score": 0.9}],
                    },
                )
            )
        assert "not allowed" in str(exc_info.value)

    def test_degraded_true_on_normal_request_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_RESPONSE_ADAPTER.validate_python(envelope(degraded=True))
        assert "degraded" in str(exc_info.value)

    def test_structured_fallback_with_degraded_true_passes(self) -> None:
        env = envelope(
            route="unconstrained_generation",
            route_trail=["constrained_generation", "unconstrained_generation"],
            result={
                "task": "structured",
                "output": {"amount": 100},
                "raw_text": '{"amount": 100}',
            },
            degraded=True,
        )
        resp = INFERENCE_RESPONSE_ADAPTER.validate_python(env)
        assert resp.degraded is True

    def test_structured_fallback_with_degraded_false_fails(self) -> None:
        env = envelope(
            route="unconstrained_generation",
            route_trail=["constrained_generation", "unconstrained_generation"],
            result={
                "task": "structured",
                "output": {"amount": 100},
                "raw_text": '{"amount": 100}',
            },
            degraded=False,
        )
        with pytest.raises(ValidationError):
            INFERENCE_RESPONSE_ADAPTER.validate_python(env)


class TestLabelSorting:
    def test_labels_sorted_lowest_first_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(
                    result={
                        "task": "classify",
                        "labels": [
                            {"label": "neg", "score": 0.1},
                            {"label": "pos", "score": 0.9},
                        ],
                    }
                )
            )


class TestSpanValidation:
    def test_span_with_end_equal_start_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(
                    route="span_head",
                    route_trail=["span_head"],
                    result={
                        "task": "extract_spans",
                        "spans": [
                            {
                                "start": 2,
                                "end": 2,
                                "label": "PER",
                                "text": "",
                                "score": 0.9,
                            }
                        ],
                    },
                )
            )
        assert "end" in str(exc_info.value)

    def test_span_text_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(
                    route="span_head",
                    route_trail=["span_head"],
                    result={
                        "task": "extract_spans",
                        "spans": [
                            {
                                "start": 2,
                                "end": 7,
                                "label": "PER",
                                "text": "Alic",
                                "score": 0.9,
                            }
                        ],
                    },
                )
            )
        assert "text" in str(exc_info.value)

    def test_valid_span_passes(self) -> None:
        env = envelope(
            route="span_head",
            route_trail=["span_head"],
            result={
                "task": "extract_spans",
                "spans": [
                    {
                        "start": 2,
                        "end": 7,
                        "label": "PER",
                        "text": "Alice",
                        "score": 0.9,
                    }
                ],
            },
        )
        resp = INFERENCE_RESPONSE_ADAPTER.validate_python(env)
        assert resp.result.spans[0].text == "Alice"


class TestStructuredResult:
    def test_structured_result_with_finish_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            INFERENCE_RESPONSE_ADAPTER.validate_python(
                envelope(
                    result={
                        "task": "structured",
                        "output": {"amount": 100},
                        "raw_text": '{"amount": 100}',
                        "finish_reason": "stop",
                    }
                )
            )


class TestDriftProtection:
    def test_allowed_routes_keys_match_task_type(self) -> None:
        assert set(ALLOWED_ROUTES.keys()) == set(TaskType)

    def test_allowed_routes_values_cover_all_routes(self) -> None:
        all_routes = set()
        for routes in ALLOWED_ROUTES.values():
            all_routes.update(routes)
        assert all_routes == set(Route)
