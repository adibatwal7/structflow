import math

import pytest

from structflow.engines.scoring import rank_labels, softmax


def test_softmax_sums_to_one() -> None:
    assert math.isclose(sum(softmax([2.0, 1.0, 0.1])), 1.0)


def test_softmax_ignores_a_constant_shift() -> None:
    a = softmax([2.0, 1.0, 0.1])
    b = softmax([102.0, 101.0, 100.1])
    assert a == pytest.approx(b)


def test_softmax_survives_huge_logits() -> None:
    # math.exp(1000) raises OverflowError. Subtracting the max avoids it.
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


def test_rank_labels_sorts_highest_first() -> None:
    ranked = rank_labels({"neg": -1.0, "pos": 2.0, "neutral": 0.0})
    assert [ls.label for ls in ranked] == ["pos", "neutral", "neg"]


def test_rank_labels_top_k_keeps_the_best() -> None:
    ranked = rank_labels({"a": 0.0, "b": 3.0, "c": 1.0}, top_k=2)
    assert [ls.label for ls in ranked] == ["b", "c"]


def test_a_subset_of_labels_is_renormalised() -> None:
    full = {ls.label: ls.score for ls in rank_labels({"a": 1.0, "b": 2.0, "c": 3.0})}
    sub = {ls.label: ls.score for ls in rank_labels({"a": 1.0, "b": 2.0})}
    assert sub["a"] == pytest.approx(full["a"] / (full["a"] + full["b"]))
    assert math.isclose(sum(sub.values()), 1.0)
