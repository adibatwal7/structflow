"""Turn raw model scores (logits) into ranked label probabilities.

Pure Python on purpose: no torch here, so this math is tested on every machine.
"""

import math
from collections.abc import Mapping, Sequence

from structflow.schemas.responses import LabelScore


def softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rank_labels(
    logits: Mapping[str, float], *, top_k: int | None = None
) -> tuple[LabelScore, ...]:
    labels = list(logits)
    probs = softmax([logits[label] for label in labels])
    ranked = sorted(zip(labels, probs, strict=True), key=lambda p: p[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return tuple(LabelScore(label=label, score=prob) for label, prob in ranked)
