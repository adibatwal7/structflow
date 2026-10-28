"""Span head: a Hugging Face token classifier (NER) behind the micro-batcher.

Importing this module imports torch, so only import it when the HF backend is used.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Self

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from structflow.engines.base import EngineOutput, UnsupportedRequestError
from structflow.engines.batcher import MicroBatcher
from structflow.engines.hf_classifier import Device, pick_device
from structflow.engines.spans import WordTag, decode_bio, split_tag
from structflow.schemas.requests import ExtractSpansRequest
from structflow.schemas.responses import ExtractSpansResult, Usage


@dataclass(frozen=True, slots=True)
class Tagged:
    words: tuple[WordTag, ...]
    tokens: int


def first_piece_per_word(
    word_ids: Sequence[int | None],
    offsets: Sequence[Sequence[int]],
    tag_ids: Sequence[int],
    probs: Sequence[float],
    tags: Sequence[str],
) -> list[WordTag]:
    """Collapse tokens into words. A word's first piece decides its tag."""
    words: list[WordTag] = []
    previous: int | None = None
    for i, word_id in enumerate(word_ids):
        if word_id is None:  # [CLS], [SEP] and padding belong to no word
            continue
        start, end = offsets[i]
        if word_id == previous:  # a later piece of the same word, like "##hn"
            words[-1] = replace(words[-1], end=end)
        else:
            words.append(
                WordTag(start=start, end=end, tag=tags[tag_ids[i]], score=probs[i])
            )
        previous = word_id
    return words


class HFSpanExtractor:
    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        model: torch.nn.Module,
        tags: Sequence[str],
        device: torch.device,
        *,
        max_length: int = 512,
        max_batch_size: int = 8,
        max_wait_ms: float = 5,
        max_queue_size: int = 256,
    ) -> None:
        self.name = name
        self.tags = tuple(tags)
        self.entity_types = frozenset(split_tag(t)[1] for t in self.tags if t != "O")
        self.device = device
        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self.max_length = max_length
        self._batcher: MicroBatcher[str, Tagged] = MicroBatcher(
            self._predict,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
            name=f"batcher:{name}",
        )

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: Device = "auto",
        max_length: int = 512,
        max_batch_size: int = 8,
        max_wait_ms: float = 5,
        max_queue_size: int = 256,
    ) -> Self:
        """Download (or read from cache) and load the model. Slow and blocking."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError(
                f"{model_id} needs a fast tokenizer (for character offsets)"
            )
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        positions = getattr(model.config, "max_position_embeddings", None)
        if isinstance(positions, int):  # never promise more than the model can read
            max_length = min(max_length, positions)
        id2label = model.config.id2label or {}
        tags = [tag for _, tag in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
        return cls(
            model_id,
            tokenizer,
            model,
            tags,
            pick_device(device),
            max_length=max_length,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
        )

    @property
    def running(self) -> bool:
        return self._batcher.running

    async def __aenter__(self) -> Self:
        self._batcher.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._batcher.stop()

    def token_count(self, text: str) -> int:
        return len(self._tokenizer(text)["input_ids"])

    def supports(self, request: ExtractSpansRequest) -> bool:
        wanted = set(request.entity_types or ())
        fits = self.token_count(request.text) <= self.max_length
        return wanted <= self.entity_types and fits

    async def extract(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        if not self.supports(request):
            raise UnsupportedRequestError(
                f"{self.name} finds {sorted(self.entity_types)} "
                f"in texts of up to {self.max_length} tokens"
            )
        batched = await self._batcher.submit(request.text)
        wanted = set(request.entity_types or self.entity_types)
        spans = tuple(
            span
            for span in decode_bio(request.text, batched.value.words)
            if span.label in wanted and span.score >= request.min_score
        )
        return EngineOutput(
            result=ExtractSpansResult(task="extract_spans", spans=spans),
            usage=Usage(prompt_tokens=batched.value.tokens, forward_passes=1),
            queue_ms=batched.queue_ms,
        )

    def _predict(self, texts: Sequence[str]) -> list[Tagged]:
        """Runs on the batcher's worker thread: one forward pass for the batch."""
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            logits = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        probs, tag_ids = logits.float().softmax(dim=-1).max(dim=-1)
        probs_rows = probs.cpu().tolist()
        tag_rows = tag_ids.cpu().tolist()
        offsets = encoded["offset_mapping"].tolist()
        tokens = attention_mask.sum(dim=1).cpu().tolist()
        return [
            Tagged(
                words=tuple(
                    first_piece_per_word(
                        encoded.word_ids(row),
                        offsets[row],
                        tag_rows[row],
                        probs_rows[row],
                        self.tags,
                    )
                ),
                tokens=int(tokens[row]),
            )
            for row in range(len(texts))
        ]
