"""Benchmark tasks: small labelled datasets, and two ways to answer each one.

Each task can be answered by
- **StructFlow**: one request whose answer shape is guaranteed (a label, or JSON
  that matches a schema), and
- **the baseline**: what most apps do today: ask the LLM nicely in the prompt, then
  parse its free text, and try again if the parse fails.

The datasets are generated from templates with a fixed seed, so every run uses the
same examples and the right answer is always known. Some examples are written to be
awkward (negations, "nothing special", European number formats, odd date styles):
an easy benchmark proves nothing.
"""

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from jsonschema import Draft202012Validator

# --- What a task is ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    expected: Any  # a label, or the dict the JSON answer must match


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    examples: tuple[Example, ...]
    structflow_body: Callable[[Example], dict[str, Any]]
    answer_from: Callable[[Mapping[str, Any]], Any]  # StructFlow response -> answer
    baseline_messages: Callable[[Example], list[dict[str, str]]]
    parse: Callable[[str], Any]  # the baseline's free text -> answer, or None
    correct: Callable[[Any, Any], bool]  # (answer, expected) -> right or wrong


# --- Parsing free text, the way a careful engineer would ------------------------------


def parse_label(text: str, labels: Sequence[str]) -> str | None:
    """Find which label a free-text answer means, or None if we can't tell.

    Accepts "POSITIVE", " positive. ", or "The sentiment is POSITIVE." (exactly one
    label mentioned). Refuses answers that mention no label, or more than one.
    """
    # YOUR TURN (about 10 lines):
    # 1. Clean the text: strip spaces, newlines and the characters  . ! " ' ` *
    #    then compare it (lowercased) with each label (lowercased). A match wins.
    # 2. Otherwise, list the labels that appear in the text as whole words, ignoring
    #    case. Hint: re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE)
    # 3. Return the label if exactly one was mentioned; otherwise return None.
    clean = text.strip().strip(" .!\"'`*")
    for label in labels:
        if clean.lower() == label.lower():
            return label
    found: list[str] = []
    for label in labels:
        pattern = rf"\b{re.escape(label)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(label)
    return found[0] if len(found) == 1 else None


def parse_json(text: str, validator: Draft202012Validator) -> dict[str, Any] | None:
    """Pull a JSON object out of free text and check it, or None if that fails.

    Copes with ```json fences and chatty text around the object, by taking
    everything from the first "{" to the last "}".
    """
    # YOUR TURN (about 10 lines):
    # 1. Find the first "{" and the last "}" in the text. If either is missing, or
    #    they are in the wrong order, return None.
    # 2. json.loads the part between them (both included). A JSONDecodeError
    #    means None.
    # 3. Return the value only if it is a dict AND validator.is_valid(value).
    open_brace = text.find("{")
    close_brace = text.rfind("}")
    if open_brace == -1 or close_brace == -1 or close_brace < open_brace:
        return None
    try:
        parsed = json.loads(text[open_brace : close_brace + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    errors = list(validator.iter_errors(parsed))
    if errors:
        return None
    return parsed


# --- Task 1: sentiment (the small softmax head knows these labels) --------------------

SENTIMENTS = ("POSITIVE", "NEGATIVE")
_PRODUCTS = (
    "headphones",
    "blender",
    "laptop stand",
    "backpack",
    "coffee maker",
    "phone case",
    "desk lamp",
    "running shoes",
)
_GOOD = (
    "is absolutely fantastic",
    "works perfectly",
    "exceeded my expectations",
    "is worth every penny",
    "made my mornings so much better",
)
_BAD = (
    "broke after two days",
    "is a complete waste of money",
    "stopped working within a week",
    "feels cheap and flimsy",
    "arrived damaged and support never replied",
)
_TRICKY = (  # negations and mixed feelings: harder for everyone
    ("Not bad at all, the {p} honestly surprised me.", "POSITIVE"),
    ("I expected nothing from this {p} and it turned out great.", "POSITIVE"),
    ("Not as good as the reviews said, my {p} died in a month.", "NEGATIVE"),
    ("I really wanted to love this {p}, but it is just disappointing.", "NEGATIVE"),
)


def sentiment_examples(n: int = 60, seed: int = 7) -> tuple[Example, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        product = rng.choice(_PRODUCTS)
        if i % 5 == 4:
            template, label = rng.choice(_TRICKY)
            text = template.format(p=product)
        else:
            label = rng.choice(SENTIMENTS)
            phrase = rng.choice(_GOOD if label == "POSITIVE" else _BAD)
            text = f"The {product} {phrase}."
            if rng.random() < 0.3:
                text = text.lower().rstrip(".") + "!!"
        out.append(Example(id=f"sentiment-{i}", text=text, expected=label))
    return tuple(out)


# --- Task 2: topic (no small head knows these labels: the LLM must answer) ------------

TOPICS = ("sports", "politics", "technology", "business", "health")
_TOPIC_LINES = {
    "sports": (
        "The home team won in overtime after a late goal from the captain.",
        "She broke the national record in the 400 metres on Saturday.",
        "The coach was fired after a run of seven straight defeats.",
    ),
    "politics": (
        "The senate passed the budget bill by a narrow margin last night.",
        "Voters head to the polls next week in a tight mayoral race.",
        "The minister resigned after weeks of pressure from her own party.",
    ),
    "technology": (
        "The new phone ships with a faster chip and a brighter screen.",
        "A bug in the update left thousands of laptops unable to boot.",
        "The startup released an open-source tool for training small models.",
    ),
    "business": (
        "Shares fell eight percent after the company cut its yearly forecast.",
        "The two retailers agreed to merge in a deal worth four billion.",
        "Quarterly profit doubled thanks to strong sales in Asia.",
    ),
    "health": (
        "A new study links poor sleep to a higher risk of heart disease.",
        "The clinic is offering free flu shots to everyone over sixty.",
        "Doctors warn that the measles outbreak is spreading in schools.",
    ),
}


def topic_examples(n: int = 60, seed: int = 11) -> tuple[Example, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        text = rng.choice(_TOPIC_LINES[topic])
        out.append(Example(id=f"topic-{i}", text=text, expected=topic))
    return tuple(out)


# --- Task 3: invoice extraction to JSON (masked generation vs prompt and parse) -----

INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "amount": {"type": "number"},
        "currency": {"enum": ["USD", "EUR", "GBP"]},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
    "required": ["vendor", "amount", "currency", "date"],
    "additionalProperties": False,
}
INVOICE_VALIDATOR = Draft202012Validator(INVOICE_SCHEMA)
INVOICE_INSTRUCTION = (
    "Extract the invoice from the user's message. Use the vendor's name as written, "
    "the amount as a plain number, the currency as USD, EUR or GBP, and the date as "
    "YYYY-MM-DD."
)
_VENDORS = (
    "Northwind Traders",
    "Bluebird Bakery",
    "Kestrel Logistics",
    "Orchard Labs",
    "Pinecrest Supply",
    "Harbor & Co",
    "Lumen Analytics",
    "Redwood Tools",
)
_SENTENCES = (
    "Invoice from {vendor}: {amount}, issued {date}.",
    "{vendor} billed us {amount} on {date}. Please pay by the end of the month.",
    "hey, just got the bill from {vendor} - {amount}, dated {date}. thx",
    "Re: payment. Supplier: {vendor}. Total due: {amount}. Invoice date: {date}.",
)


def _money(amount: float, currency: str, style: int) -> str:
    whole, cents = f"{amount:,.2f}".split(".")
    if currency == "EUR" and style % 2 == 0:  # European style: 1.240,50 €
        return f"{whole.replace(',', '.')},{cents} €"
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}[currency]
    return f"{symbol}{whole}.{cents}" if style % 3 else f"{whole}.{cents} {currency}"


def _date_text(day: date, style: int) -> str:
    match style % 3:
        case 0:
            return day.isoformat()
        case 1:
            return f"{day.day} {day:%b} {day.year}"
        case _:
            return f"{day:%B} {day.day}, {day.year}"


def invoice_examples(n: int = 60, seed: int = 13) -> tuple[Example, ...]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        vendor = rng.choice(_VENDORS)
        currency = rng.choice(("USD", "EUR", "GBP"))
        amount = round(rng.uniform(20, 5000), 2)
        day = date(2026, 1, 1) + timedelta(days=rng.randrange(300))
        text = rng.choice(_SENTENCES).format(
            vendor=vendor,
            amount=_money(amount, currency, rng.randrange(6)),
            date=_date_text(day, rng.randrange(3)),
        )
        expected = {
            "vendor": vendor,
            "amount": amount,
            "currency": currency,
            "date": day.isoformat(),
        }
        out.append(Example(id=f"invoice-{i}", text=text, expected=expected))
    return tuple(out)


def invoice_correct(answer: Any, expected: Mapping[str, Any]) -> bool:
    """Every field must match. Vendor ignores case; amount must be right to the cent."""
    if not isinstance(answer, Mapping):
        return False
    return (
        str(answer.get("vendor", "")).strip().lower() == expected["vendor"].lower()
        and isinstance(answer.get("amount"), int | float)
        and abs(float(answer["amount"]) - expected["amount"]) < 0.005
        and answer.get("currency") == expected["currency"]
        and answer.get("date") == expected["date"]
    )


# --- The three tasks, ready to run ----------------------------------------------------


def _top_label(response: Mapping[str, Any]) -> str:
    return str(response["result"]["labels"][0]["label"])


def _label_task(
    name: str, labels: Sequence[str], examples: tuple[Example, ...], what: str
) -> Task:
    choices = ", ".join(labels)
    return Task(
        name=name,
        examples=examples,
        structflow_body=lambda ex: {
            "task": "classify",
            "text": ex.text,
            "labels": list(labels),
        },
        answer_from=_top_label,
        baseline_messages=lambda ex: [
            {
                "role": "system",
                "content": f"Classify the {what} of the user's text. "
                f"Answer with exactly one of: {choices}.",
            },
            {"role": "user", "content": ex.text},
        ],
        parse=lambda text: parse_label(text, labels),
        correct=lambda answer, expected: answer == expected,
    )


def all_tasks(n: int = 60) -> tuple[Task, ...]:
    invoice = Task(
        name="invoice",
        examples=invoice_examples(n),
        structflow_body=lambda ex: {
            "task": "structured",
            "messages": [
                {"role": "system", "content": INVOICE_INSTRUCTION},
                {"role": "user", "content": ex.text},
            ],
            "constraint": {"type": "json_schema", "schema": INVOICE_SCHEMA},
            "sampling": {"temperature": 0, "max_tokens": 200},
        },
        answer_from=lambda response: response["result"]["output"],
        baseline_messages=lambda ex: [
            {
                "role": "system",
                "content": f"{INVOICE_INSTRUCTION} Reply with only a JSON object "
                f"that matches this JSON Schema:\n{json.dumps(INVOICE_SCHEMA)}",
            },
            {"role": "user", "content": ex.text},
        ],
        parse=lambda text: parse_json(text, INVOICE_VALIDATOR),
        correct=invoice_correct,
    )
    return (
        _label_task("sentiment", SENTIMENTS, sentiment_examples(n), "sentiment"),
        _label_task("topic", TOPICS, topic_examples(n), "topic"),
        invoice,
    )
