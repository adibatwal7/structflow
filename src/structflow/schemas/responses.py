from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from .common import FinishReason, NonBlankLabel, Route, StrictModel, TaskType
from .requests import RequestId

# Part 1: Reusable number types
Score = Annotated[float, Field(ge=0.0, le=1.0)]
Count = Annotated[int, Field(ge=0)]
Millis = Annotated[float, Field(ge=0.0)]


# Part 2: Usage and timings
class Usage(StrictModel):
    prompt_tokens: Count = 0
    completion_tokens: Count = 0
    forward_passes: Count


class Timings(StrictModel):
    queue_ms: Millis
    inference_ms: Millis
    total_ms: Millis
    compile_ms: Millis | None = None
    ttft_ms: Millis | None = None


# Part 3: Results
class LabelScore(StrictModel):
    label: NonBlankLabel
    score: Score


class ClassifyResult(StrictModel):
    task: Literal["classify"]
    labels: tuple[LabelScore, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_labels_sorted(self) -> Self:
        scores = [label.score for label in self.labels]
        if scores != sorted(scores, reverse=True):
            raise ValueError("labels must be sorted by score, highest first")
        return self


class Span(StrictModel):
    start: Count
    end: Count
    label: NonBlankLabel
    text: str
    score: Score

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("text length must equal end - start")
        return self


class ExtractSpansResult(StrictModel):
    task: Literal["extract_spans"]
    spans: tuple[Span, ...] = ()


class StructuredResult(StrictModel):
    task: Literal["structured"]
    output: dict[str, Any]
    raw_text: str
    # No finish_reason - truncated output is an error, not a result


class GenerateResult(StrictModel):
    task: Literal["generate"]
    text: str
    finish_reason: FinishReason


Result = Annotated[
    ClassifyResult | ExtractSpansResult | StructuredResult | GenerateResult,
    Field(discriminator="task"),
]

# Part 4: The envelope
ALLOWED_ROUTES: Final[dict[TaskType, frozenset[Route]]] = {
    TaskType.CLASSIFY: frozenset({Route.SOFTMAX_HEAD, Route.LLM_LABEL_SCORING}),
    TaskType.EXTRACT_SPANS: frozenset({Route.SPAN_HEAD, Route.CONSTRAINED_GENERATION}),
    TaskType.STRUCTURED: frozenset(
        {Route.CONSTRAINED_GENERATION, Route.UNCONSTRAINED_GENERATION}
    ),
    TaskType.GENERATE: frozenset({Route.UNCONSTRAINED_GENERATION}),
}


class InferenceResponse(StrictModel):
    request_id: RequestId
    model: str = Field(min_length=1)
    route: Route
    route_trail: tuple[Route, ...] = Field(min_length=1)
    degraded: bool = False
    result: Result
    usage: Usage
    timings: Timings

    @model_validator(mode="after")
    def _check_routing(self) -> Self:
        # Last entry of route_trail must equal route
        if self.route_trail[-1] != self.route:
            raise ValueError("last entry of route_trail must equal route")

        # Route must be allowed for the result's task
        result_task = TaskType(self.result.task)
        if self.route not in ALLOWED_ROUTES[result_task]:
            raise ValueError(
                f"route {self.route} is not allowed for task {result_task.value}"
            )

        # degraded is True exactly when structured task was served by unconstrained_generation
        is_structured = result_task == TaskType.STRUCTURED
        is_unconstrained = self.route == Route.UNCONSTRAINED_GENERATION
        if self.degraded != (is_structured and is_unconstrained):
            raise ValueError(
                "degraded must be True only when structured task is served by unconstrained_generation"
            )

        return self


INFERENCE_RESPONSE_ADAPTER: Final = TypeAdapter(InferenceResponse)
