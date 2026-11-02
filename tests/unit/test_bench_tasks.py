"""The benchmark's datasets, and how the baseline reads free-text answers."""

import json
from typing import Any

import pytest

from structflow.bench.tasks import (
    INVOICE_VALIDATOR,
    SENTIMENTS,
    all_tasks,
    invoice_correct,
    invoice_examples,
    parse_json,
    parse_label,
    sentiment_examples,
)
from structflow.schemas.requests import INFERENCE_REQUEST_ADAPTER

INVOICE = {
    "vendor": "Orchard Labs",
    "amount": 604.45,
    "currency": "GBP",
    "date": "2026-05-15",
}


def test_datasets_are_the_same_every_run() -> None:
    assert sentiment_examples() == sentiment_examples()
    assert invoice_examples() == invoice_examples()


def test_every_task_has_unique_examples_with_answers() -> None:
    for task in all_tasks(n=30):
        assert len(task.examples) == 30
        assert len({ex.id for ex in task.examples}) == 30
        assert all(ex.expected is not None for ex in task.examples)


def test_every_expected_invoice_fits_the_schema() -> None:
    for ex in invoice_examples():
        assert INVOICE_VALIDATOR.is_valid(ex.expected), ex.text
        assert invoice_correct(ex.expected, ex.expected)


def test_requests_follow_the_published_contract() -> None:
    for task in all_tasks(n=5):
        ex = task.examples[0]
        INFERENCE_REQUEST_ADAPTER.validate_python(task.structflow_body(ex))
        INFERENCE_REQUEST_ADAPTER.validate_python(
            {"task": "generate", "messages": task.baseline_messages(ex)}
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("POSITIVE", "POSITIVE"),
        (" negative.\n", "NEGATIVE"),
        ("**Positive**", "POSITIVE"),
        ("The sentiment is POSITIVE.", "POSITIVE"),
        ("It's mixed: partly positive, partly negative.", None),
        ("I'd say it is upbeat.", None),
        ("", None),
    ],
)
def test_parse_label(text: str, expected: str | None) -> None:
    assert parse_label(text, SENTIMENTS) == expected


GOOD = json.dumps(INVOICE)  # {"vendor": "Orchard Labs", "amount": 604.45, ...}


@pytest.mark.parametrize(
    "text",
    [GOOD, f"```json\n{GOOD}\n```", f"Sure! Here it is: {GOOD} Hope that helps."],
    ids=["plain", "fenced", "chatty"],
)
def test_parse_json_finds_the_object(text: str) -> None:
    assert parse_json(text, INVOICE_VALIDATOR) == INVOICE


@pytest.mark.parametrize(
    "text",
    [
        "I could not find an invoice.",
        GOOD[:40],
        json.dumps({**INVOICE, "amount": "604.45"}),
        json.dumps({**INVOICE, "date": "15 May 2026"}),
        "[1, 2, 3]",
    ],
    ids=["no-json", "cut-off", "amount-as-text", "wrong-date-format", "not-an-object"],
)
def test_parse_json_refuses_what_breaks_the_schema(text: str) -> None:
    assert parse_json(text, INVOICE_VALIDATOR) is None


@pytest.mark.parametrize(
    ("answer", "right"),
    [
        (INVOICE, True),
        ({**INVOICE, "vendor": "orchard labs "}, True),
        ({**INVOICE, "amount": 604.46}, False),
        ({**INVOICE, "date": "2026-05-16"}, False),
        ({**INVOICE, "amount": "604.45"}, False),
        ("not a dict", False),
    ],
)
def test_invoice_correct_checks_every_field(answer: Any, right: bool) -> None:
    assert invoice_correct(answer, INVOICE) is right


def test_answers_are_read_from_structflow_responses() -> None:
    sentiment, _, invoice = all_tasks(n=1)
    classify = {"result": {"labels": [{"label": "NEGATIVE", "score": 0.9}]}}
    assert sentiment.answer_from(classify) == "NEGATIVE"
    assert invoice.answer_from({"result": {"output": INVOICE}}) == INVOICE
