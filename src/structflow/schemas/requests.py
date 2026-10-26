from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    TypeAdapter,
    model_validator,
)

from .common import (
    ChatMessage,
    NonBlankLabel,
    NonBlankText,
    SamplingParams,
    StrictModel,
    require_unique,
)
from .constraints import Constraint

# Part 1: Limits
MAX_LABELS: Final = 256
MAX_ENTITY_TYPES: Final = 64
MAX_MESSAGES: Final = 64
MAX_METADATA_ITEMS: Final = 16

# Part 2: Reusable pieces
RequestId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9._-]{1,64}$"),
]

MetadataKey = Annotated[str, Field(min_length=1, max_length=64)]
MetadataValue = Annotated[str, Field(max_length=256)]


# Part 3: The shared base
class _RequestBase(StrictModel):
    request_id: RequestId | None = None
    metadata: dict[MetadataKey, MetadataValue] = Field(
        default_factory=dict, max_length=MAX_METADATA_ITEMS
    )


# Part 4: ClassifyRequest
class ClassifyRequest(_RequestBase):
    task: Literal["classify"]
    text: NonBlankText
    labels: Annotated[
        tuple[NonBlankLabel, ...],
        Field(
            min_length=2, max_length=MAX_LABELS, json_schema_extra={"uniqueItems": True}
        ),
        AfterValidator(require_unique),
    ]
    multi_label: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, strict=True)
    top_k: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def _check_options(self) -> Self:
        if self.top_k is not None and self.top_k > len(self.labels):
            raise ValueError(
                f"top_k ({self.top_k}) is larger than the number of labels ({len(self.labels)})"
            )
        if self.threshold is not None and not self.multi_label:
            raise ValueError("threshold can only be used when multi_label is true")
        return self


# Part 5: ExtractSpansRequest
class ExtractSpansRequest(_RequestBase):
    task: Literal["extract_spans"]
    text: NonBlankText
    entity_types: (
        Annotated[
            tuple[NonBlankLabel, ...],
            Field(
                min_length=1,
                max_length=MAX_ENTITY_TYPES,
                json_schema_extra={"uniqueItems": True},
            ),
            AfterValidator(require_unique),
        ]
        | None
    ) = None
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, strict=True)


# Part 6: Chat request base
class _ChatRequestBase(_RequestBase):
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=MAX_MESSAGES)
    sampling: SamplingParams = Field(default_factory=SamplingParams)

    @model_validator(mode="after")
    def _check_conversation(self) -> Self:
        # System message only allowed as first message
        for i, msg in enumerate(self.messages):
            if msg.role == "system" and i != 0:
                raise ValueError("system message is only allowed as the first message")

        # Last message must be from user
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")

        return self


# Part 6 continued: StructuredRequest and GenerateRequest
class StructuredRequest(_ChatRequestBase):
    task: Literal["structured"]
    constraint: Constraint
    allow_unconstrained_fallback: bool = False


class GenerateRequest(_ChatRequestBase):
    task: Literal["generate"]


# Part 7: The union
InferenceRequest = Annotated[
    ClassifyRequest | ExtractSpansRequest | StructuredRequest | GenerateRequest,
    Field(discriminator="task"),
]

INFERENCE_REQUEST_ADAPTER: Final = TypeAdapter(InferenceRequest)
