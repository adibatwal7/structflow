"""Softmax head: a Hugging Face sequence classifier behind the micro-batcher.

Importing this module imports torch, so only import it when the HF backend is used.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from structflow.engines.base import UnsupportedRequestError
from structflow.engines.batcher import MicroBatcher
from structflow.engines.scoring import rank_labels
from structflow.schemas.requests import ClassifyRequest
from structflow.schemas.responses import ClassifyResult, Usage

Device = Literal["auto", "cpu", "cuda", "mps"]


def pick_device(requested: Device = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class Prediction:
    logits: tuple[float, ...]
    tokens: int


class HFClassifier:
    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        model: torch.nn.Module,
        labels: Sequence[str],
        device: torch.device,
        *,
        max_length: int = 512,
        max_batch_size: int = 8,
        max_wait_ms: float = 5,
        max_queue_size: int = 256,
    ) -> None:
        self.name = name
        self.labels = tuple(labels)
        self.device = device
        self._known = frozenset(self.labels)
        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self._max_length = max_length
        self._batcher: MicroBatcher[str, Prediction] = MicroBatcher(
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
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        config = model.config
        if config.problem_type == "multi_label_classification":
            raise ValueError(f"{model_id} is a multi-label model; not supported yet")
        id2label = config.id2label or {}
        labels = [
            label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))
        ]
        return cls(
            model_id,
            tokenizer,
            model,
            labels,
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

    def supports(self, request: ClassifyRequest) -> bool:
        return not request.multi_label and set(request.labels) <= self._known

    async def classify(self, request: ClassifyRequest) -> tuple[ClassifyResult, Usage]:
        if not self.supports(request):
            raise UnsupportedRequestError(
                f"{self.name} only knows the labels {list(self.labels)} "
                "and single-label classification"
            )
        batched = await self._batcher.submit(request.text)
        by_label = dict(zip(self.labels, batched.value.logits, strict=True))
        scores = rank_labels(
            {label: by_label[label] for label in request.labels}, top_k=request.top_k
        )
        usage = Usage(prompt_tokens=batched.value.tokens, forward_passes=1)
        return ClassifyResult(task="classify", labels=scores), usage

    def _predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Runs on the batcher's worker thread: one forward pass for the batch."""
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            logits = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        rows = logits.float().cpu().tolist()
        tokens = attention_mask.sum(dim=1).cpu().tolist()
        return [
            Prediction(logits=tuple(row), tokens=int(n))
            for row, n in zip(rows, tokens, strict=True)
        ]
