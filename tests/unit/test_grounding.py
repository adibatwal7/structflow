from structflow.routing.grounding import ground
from structflow.schemas.responses import Span

TEXT = "Tim Cook met Sundar Pichai in Paris"


def span(text: str, start: int, label: str = "PER") -> Span:
    return Span(start=start, end=start + len(text), label=label, text=text, score=0.8)


def test_correct_spans_are_kept() -> None:
    spans = (span("Tim Cook", 0), span("Paris", 30, "LOC"))
    assert ground(TEXT, spans) == spans


def test_wrong_offsets_are_fixed() -> None:
    (fixed,) = ground(TEXT, [span("Paris", 3, "LOC")])
    assert (fixed.start, fixed.end) == (30, 35)
    assert TEXT[fixed.start : fixed.end] == "Paris"


def test_made_up_entities_are_dropped() -> None:
    assert ground(TEXT, [span("Satya Nadella", 0)]) == ()


def test_duplicates_are_merged_and_sorted() -> None:
    spans = [span("Paris", 30, "LOC"), span("Tim Cook", 0), span("Paris", 1, "LOC")]
    assert [s.text for s in ground(TEXT, spans)] == ["Tim Cook", "Paris"]
