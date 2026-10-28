"""Tests for the real softmax head, using a tiny model built on the spot.

No internet needed: the fixture creates a 1-layer DistilBERT with random
weights and a 9-word vocabulary. The numbers it predicts are meaningless,
but the plumbing (tokenize, pad, batch, score) is exactly the real one.
"""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from structflow.config import Settings

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from structflow.engines.base import UnsupportedRequestError
from structflow.engines.hf_classifier import HFClassifier
from structflow.gateway.app import create_app
from structflow.schemas.requests import ClassifyRequest

pytestmark = pytest.mark.anyio

LABELS = ("negative", "neutral", "positive")
WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "movie", "very", "ok"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-classifier")
    vocab = {word: i for i, word in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(path)
    config = DistilBertConfig(
        vocab_size=len(WORDS),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=64,
        initializer_range=0.5,  # big random weights, so mistakes show up as big differences
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    torch.manual_seed(0)
    DistilBertForSequenceClassification(config).save_pretrained(path)
    return path


@pytest.fixture
async def head(tiny_model_dir: Path) -> AsyncIterator[HFClassifier]:
    classifier = HFClassifier.load(
        str(tiny_model_dir), device="cpu", max_length=32, max_wait_ms=20
    )
    async with classifier:
        yield classifier


def request(
    text: str, labels: tuple[str, ...] = LABELS, **extra: object
) -> ClassifyRequest:
    return ClassifyRequest.model_validate(
        {"task": "classify", "text": text, "labels": labels, **extra}
    )


async def test_labels_come_from_the_model_config(head: HFClassifier) -> None:
    assert head.labels == LABELS


async def test_scores_are_a_probability_distribution(head: HFClassifier) -> None:
    out = await head.classify(request("very good movie"))
    assert {ls.label for ls in out.result.labels} == set(LABELS)
    assert math.isclose(sum(ls.score for ls in out.result.labels), 1.0, rel_tol=1e-6)
    assert out.usage.forward_passes == 1
    assert out.usage.prompt_tokens == 5  # [CLS] very good movie [SEP]
    assert out.queue_ms >= 15  # it waited for the 20 ms batching window


async def test_batched_answers_match_one_at_a_time(head: HFClassifier) -> None:
    texts = ["good", "very bad movie", "ok ok ok ok ok ok", "movie"]
    alone = [await head.classify(request(t)) for t in texts]
    together = await asyncio.gather(*(head.classify(request(t)) for t in texts))
    for a, b in zip(alone, together, strict=True):
        assert [ls.label for ls in a.result.labels] == [
            ls.label for ls in b.result.labels
        ]
        for x, y in zip(a.result.labels, b.result.labels, strict=True):
            assert x.score == pytest.approx(y.score, abs=1e-5)


async def test_a_subset_of_known_labels_is_allowed(head: HFClassifier) -> None:
    out = await head.classify(request("good movie", ("positive", "negative")))
    result = out.result
    assert {ls.label for ls in result.labels} == {"positive", "negative"}
    assert math.isclose(sum(ls.score for ls in result.labels), 1.0, rel_tol=1e-6)


async def test_top_k_trims_the_answer(head: HFClassifier) -> None:
    out = await head.classify(request("good movie", top_k=1))
    assert len(out.result.labels) == 1


async def test_long_text_is_truncated_not_rejected(head: HFClassifier) -> None:
    out = await head.classify(request("good " * 500))
    assert out.usage.prompt_tokens == 32


@pytest.mark.parametrize(
    "bad",
    [
        request("good movie", ("positive", "sarcastic")),  # a label it never learned
        request("good movie", multi_label=True),  # softmax can't do multi-label
    ],
)
async def test_unsupported_requests_are_refused(
    head: HFClassifier, bad: ClassifyRequest
) -> None:
    assert not head.supports(bad)
    with pytest.raises(UnsupportedRequestError):
        await head.classify(bad)


def wait_for_startup(client: TestClient) -> str:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status: str = client.get("/readyz").json()["status"]
        if status != "starting":
            return status
        time.sleep(0.05)
    raise AssertionError("the model never finished loading")


def test_the_app_serves_the_real_head(tiny_model_dir: Path) -> None:
    settings = Settings(
        _env_file=None,
        classifier_backend="hf",
        classifier_model=str(tiny_model_dir),
        classifier_device="cpu",
    )
    with TestClient(create_app(settings)) as client:
        assert wait_for_startup(client) == "ready"
        ok = client.post(
            "/v1/infer",
            json={
                "task": "classify",
                "text": "good movie",
                "labels": ["positive", "negative"],
            },
        )
        assert ok.status_code == 200
        body = ok.json()
        assert body["route"] == "softmax_head"
        assert body["model"] == str(tiny_model_dir)
        assert body["timings"]["queue_ms"] > 0

        unknown = client.post(
            "/v1/infer",
            json={
                "task": "classify",
                "text": "good movie",
                "labels": ["positive", "sarcastic"],
            },
        )
        assert unknown.status_code == 200  # the tiny model never learned "sarcastic",
        assert (
            unknown.json()["route"] == "llm_label_scoring"
        )  # so the LLM route answers
        assert unknown.json()["route_trail"] == ["softmax_head", "llm_label_scoring"]
