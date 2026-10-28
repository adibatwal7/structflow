"""Never trust a model's offsets: check every span against the text itself."""

from collections.abc import Sequence

from structflow.schemas.responses import Span


def ground(text: str, spans: Sequence[Span]) -> tuple[Span, ...]:
    """Keep spans that really are in the text, fixing offsets that point elsewhere.

    A span whose text does not appear anywhere in the input is dropped:
    that is a made-up entity, and it must never reach the client.
    """
    kept: dict[tuple[int, int, str], Span] = {}
    for span in spans:
        start = span.start
        if text[span.start : span.end] != span.text:
            start = text.find(span.text)
            if start == -1:
                continue

        fixed = Span(
            start=start,
            end=start + len(span.text),
            label=span.label,
            text=span.text,
            score=span.score,
        )
        kept.setdefault((fixed.start, fixed.end, fixed.label), fixed)
    return tuple(sorted(kept.values(), key=lambda s: (s.start, s.end)))
