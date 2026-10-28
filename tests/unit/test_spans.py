import pytest

from structflow.engines.spans import WordTag, decode_bio, split_tag

TEXT = "Tim Cook met Sundar Pichai in Paris"


def words(*tags: str) -> list[WordTag]:
    """One WordTag per word of TEXT, with the given tags (score 0.9 each)."""
    out: list[WordTag] = []
    start = 0
    for word, tag in zip(TEXT.split(" "), tags, strict=True):
        out.append(WordTag(start=start, end=start + len(word), tag=tag, score=0.9))
        start += len(word) + 1
    return out


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("B-PER", ("B", "PER")),
        ("I-LOC", ("I", "LOC")),
        ("O", ("O", "")),
        ("PER", ("I", "PER")),
    ],
)
def test_split_tag(tag: str, expected: tuple[str, str]) -> None:
    assert split_tag(tag) == expected


def test_b_then_i_is_one_span() -> None:
    spans = decode_bio(
        TEXT, words("B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC")
    )
    assert [(s.text, s.label) for s in spans] == [
        ("Tim Cook", "PER"),
        ("Sundar Pichai", "PER"),
        ("Paris", "LOC"),
    ]


def test_every_span_is_a_slice_of_the_text() -> None:
    spans = decode_bio(
        TEXT, words("B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC")
    )
    assert all(TEXT[s.start : s.end] == s.text for s in spans)


def test_b_after_b_starts_a_new_span() -> None:
    spans = decode_bio(TEXT, words("B-PER", "B-PER", "O", "O", "O", "O", "O"))
    assert [s.text for s in spans] == ["Tim", "Cook"]


def test_an_outside_word_ends_the_span() -> None:
    spans = decode_bio(TEXT, words("B-PER", "O", "I-PER", "O", "O", "O", "O"))
    assert [s.text for s in spans] == ["Tim", "met"]


def test_i_without_b_still_starts_a_span() -> None:
    spans = decode_bio(TEXT, words("O", "I-PER", "O", "O", "O", "O", "O"))
    assert [s.text for s in spans] == ["Cook"]


def test_a_change_of_type_splits_the_span() -> None:
    spans = decode_bio(TEXT, words("B-PER", "I-ORG", "O", "O", "O", "O", "O"))
    assert [(s.text, s.label) for s in spans] == [("Tim", "PER"), ("Cook", "ORG")]


def test_bare_labels_without_b_or_i_work_too() -> None:
    spans = decode_bio(TEXT, words("PER", "PER", "O", "O", "O", "O", "LOC"))
    assert [(s.text, s.label) for s in spans] == [
        ("Tim Cook", "PER"),
        ("Paris", "LOC"),
    ]


def test_score_is_the_average_of_its_words() -> None:
    tagged = [
        WordTag(start=0, end=3, tag="B-PER", score=0.8),
        WordTag(start=4, end=8, tag="I-PER", score=0.6),
    ]
    assert decode_bio(TEXT, tagged)[0].score == pytest.approx(0.7)


def test_all_outside_means_no_spans() -> None:
    assert decode_bio(TEXT, words(*["O"] * 7)) == []
