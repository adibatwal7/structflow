from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

MAX_TEXT_CHARS: Final = 20_000
MAX_LABEL_CHARS: Final = 64
MAX_MESSAGE_CHARS: Final = 32_000
MAX_GENERATION_TOKENS: Final = 4_096
MAX_STOP_SEQUENCES: Final = 4
MAX_STOP_CHARS: Final = 32


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain at least one non-whitespace character")
    return value  # return the ORIGINAL value, never the stripped one


NonBlankText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS),
    AfterValidator(_require_non_blank),
]

NonBlankLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_LABEL_CHARS),
    AfterValidator(_require_non_blank),
]

NonBlankMessage = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_MESSAGE_CHARS),
    AfterValidator(_require_non_blank),
]

StopSequence = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_STOP_CHARS),
]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and is frozen by default."""

    model_config = ConfigDict(
        extra="forbid",  # unknown fields are errors
        frozen=True,  # validated objects can't be changed
    )


class TaskType(StrEnum):
    CLASSIFY = "classify"
    EXTRACT_SPANS = "extract_spans"
    STRUCTURED = "structured"
    GENERATE = "generate"


class Route(StrEnum):
    SOFTMAX_HEAD = "softmax_head"
    SPAN_HEAD = "span_head"
    LLM_LABEL_SCORING = "llm_label_scoring"
    CONSTRAINED_GENERATION = "constrained_generation"
    UNCONSTRAINED_GENERATION = "unconstrained_generation"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: NonBlankMessage


class SamplingParams(StrictModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, strict=True)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, strict=True)
    max_tokens: int = Field(default=512, ge=1, le=MAX_GENERATION_TOKENS, strict=True)
    seed: int | None = Field(default=None, ge=0, strict=True)
    stop: tuple[StopSequence, ...] = Field(default=(), max_length=MAX_STOP_SEQUENCES)
