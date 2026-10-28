"""Turn per-word BIO tags into grounded spans.

Pure Python on purpose: no torch here, so this logic is tested on every machine.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from structflow.schemas.responses import Span


@dataclass(frozen=True, slots=True)
class WordTag:
    """One word of the input: where it sits in the text, and what the model called it."""

    start: int
    end: int
    tag: str  # "B-PER", "I-PER" or "O"
    score: float


def split_tag(tag: str) -> tuple[str, str]:
    """'B-PER' -> ('B', 'PER'), 'O' -> ('O', ''), and a bare 'PER' -> ('I', 'PER')."""
    if tag == "O":
        return "O", ""
    prefix, dash, entity = tag.partition("-")
    if dash and prefix in ("B", "I"):
        return prefix, entity
    return "I", tag


def decode_bio(text: str, words: Sequence[WordTag]) -> list[Span]:
    groups: list[tuple[str, list[WordTag]]] = []
    inside = False
    for word in words:
        prefix, label = split_tag(word.tag)
        # YOUR TURN: three cases, in this order
        if prefix == "O":
            inside = False
        elif inside and prefix == "I" and groups[-1][0] == label:
            groups[-1][1].append(word)
        else:
            groups.append((label, [word]))
            inside = True

    return [_to_span(text, label, group) for label, group in groups]


def _to_span(text: str, label: str, words: list[WordTag]) -> Span:
    start, end = words[0].start, words[-1].end
    score = sum(word.score for word in words) / len(words)
    return Span(start=start, end=end, label=label, text=text[start:end], score=score)
