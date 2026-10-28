"""Tests for the real span head, using tiny taggers built on the spot.

No internet needed. Two kinds of tiny model:
- "random": random weights, so its tags are nonsense, but the plumbing is real;
- "forced": every token gets the same tag, so we know exactly what to expect.
The vocabulary has "Jo" and "##hn" but not "John", so "John" becomes two pieces.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from structflow.config import Settings
from structflow.engines.base import UnsupportedRequestError
from structflow.engines.hf_spans import HFSpanExtractor
from structflow.gateway.app import create_app
from structflow.schemas.requests import ExtractSpansRequest

pytestmark = pytest.mark.anyio

TAGS = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC")
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "Alice",
    "met",
    "Jo",
    "##hn",
    "in",
    "Paris",
    ".",
]
TEXT = "Alice met John in Paris."


def build_tagger(path: Path, forced_tag: str | None = None) -> Path:
    tok = Tokenizer(
        models.WordPiece(
            {piece: i for i, piece in enumerate(VOCAB)},
            unk_token="[UNK]",
            continuing_subword_prefix="##",
        )
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
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
        vocab_size=len(VOCAB),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=64,
        initializer_range=0.5,
        id2label=dict(enumerate(TAGS)),
        label2id={tag: i for i, tag in enumerate(TAGS)},
    )
    torch.manual_seed(0)
    model = DistilBertForTokenClassification(config)
    if forced_tag is not None:  # make one tag win for every token, whatever the input
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.bias[TAGS.index(forced_tag)] = 10.0
    model.save_pretrained(path)
    return path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def random_tagger(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tagger(tmp_path_factory.mktemp("random-tagger"))


def load(path: Path, max_length: int = 512) -> HFSpanExtractor:
    return HFSpanExtractor.load(
        str(path), device="cpu", max_length=max_length, max_wait_ms=20
    )


@pytest.fixture
async def random_head(random_tagger: Path) -> AsyncIterator[HFSpanExtractor]:
    async with load(random_tagger) as head:
        yield head


def request(text: str = TEXT, **extra: object) -> ExtractSpansRequest:
    return ExtractSpansRequest.model_validate(
        {"task": "extract_spans", "text": text, **extra}
    )


async def spans_with(
    tmp_path: Path, forced_tag: str, text: str = TEXT, **extra: object
) -> list[tuple[str, str, int, int]]:
    """Build a tagger that always says `forced_tag`, and run one request through it."""
    async with load(build_tagger(tmp_path, forced_tag)) as head:
        out = await head.extract(request(text, **extra))
    return [(s.text, s.label, s.start, s.end) for s in out.result.spans]


async def test_entity_types_come_from_the_model(random_head: HFSpanExtractor) -> None:
    assert random_head.entity_types == {"PER", "LOC"}


async def test_every_span_is_a_slice_of_the_input(random_head: HFSpanExtractor) -> None:
    text = "  Alice met John in Paris."
    out = await random_head.extract(request(text, min_score=0.0))
    assert all(text[s.start : s.end] == s.text for s in out.result.spans)
    assert out.usage.prompt_tokens == 9  # [CLS] Alice met Jo ##hn in Paris . [SEP]


async def test_a_word_split_into_pieces_stays_one_word(tmp_path: Path) -> None:
    found = await spans_with(tmp_path, "B-LOC")
    assert [text for text, *_ in found] == ["Alice", "met", "John", "in", "Paris", "."]


async def test_inside_tags_join_into_one_span(tmp_path: Path) -> None:
    found = await spans_with(tmp_path, "I-PER")
    assert found == [(TEXT, "PER", 0, len(TEXT))]


async def test_leading_spaces_keep_their_offsets(tmp_path: Path) -> None:
    found = await spans_with(tmp_path, "B-PER", text="  Alice met John")
    assert found[0] == ("Alice", "PER", 2, 7)


async def test_filters_by_entity_type_and_score(tmp_path: Path) -> None:
    assert await spans_with(tmp_path, "B-LOC", entity_types=["PER"]) == []
    assert await spans_with(tmp_path, "B-LOC", min_score=1.0) == []


async def test_batched_answers_match_one_at_a_time(
    random_head: HFSpanExtractor,
) -> None:
    texts = ["Alice met John", "Paris .", "John in Paris met Alice in Paris", "Alice"]
    alone = [await random_head.extract(request(t, min_score=0.0)) for t in texts]
    together = await asyncio.gather(
        *(random_head.extract(request(t, min_score=0.0)) for t in texts)
    )
    for a, b in zip(alone, together, strict=True):
        assert [(s.text, s.label) for s in a.result.spans] == [
            (s.text, s.label) for s in b.result.spans
        ]
        assert [s.score for s in a.result.spans] == pytest.approx(
            [s.score for s in b.result.spans], abs=1e-5
        )


async def test_unknown_entity_types_are_refused(random_head: HFSpanExtractor) -> None:
    with pytest.raises(UnsupportedRequestError):
        await random_head.extract(request(entity_types=["DATE"]))


def test_max_length_is_capped_by_the_model(random_tagger: Path) -> None:
    assert load(random_tagger).max_length == 64  # the tiny model has 64 positions


async def test_text_too_long_to_read_is_refused_not_cut(random_tagger: Path) -> None:
    async with load(random_tagger, max_length=16) as head:
        with pytest.raises(UnsupportedRequestError):
            await head.extract(request("Alice met John . " * 10))


def test_the_app_serves_the_span_head(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        span_backend="hf",
        span_model=str(build_tagger(tmp_path, "B-PER")),
        span_device="cpu",
    )
    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 10
        while client.get("/readyz").json()["status"] == "starting":
            assert time.monotonic() < deadline, "the model never finished loading"
            time.sleep(0.05)
        r = client.post(
            "/v1/infer", json={"task": "extract_spans", "text": "  Alice met John"}
        )
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "span_head"
    assert [s["text"] for s in body["result"]["spans"]] == ["Alice", "met", "John"]
    assert body["result"]["spans"][0]["start"] == 2
