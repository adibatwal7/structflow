"""Generation backend: vLLM's OpenAI-compatible server.

The grammar engine inside vLLM (XGrammar) makes invalid tokens impossible while the
model writes. We still check every answer here, because a grammar engine may not
enforce every JSON Schema keyword, and a constraint is a promise to our client.
"""

import json
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from types import TracebackType
from typing import Any, Self, assert_never

import httpx2
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from structflow.config import Settings
from structflow.engines.base import (
    ConstraintRejectedError,
    EngineOutput,
    OutputInvalidError,
    UnsupportedRequestError,
)
from structflow.engines.limits import GenerationTimeoutError
from structflow.schemas.common import ChatMessage, SamplingParams
from structflow.schemas.constraints import (
    ChoiceConstraint,
    Constraint,
    GrammarConstraint,
    JsonSchemaConstraint,
    RegexConstraint,
)
from structflow.schemas.requests import (
    ClassifyRequest,
    ExtractSpansRequest,
    GenerateRequest,
    StructuredRequest,
)
from structflow.schemas.responses import (
    ClassifyResult,
    ExtractSpansResult,
    GenerateResult,
    LabelScore,
    Span,
    StructuredResult,
    Usage,
)

DEFAULT_ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")  # the same four as the span head


class UpstreamError(Exception):
    """vLLM could not be reached, or failed, or answered something we can't read."""


class UpstreamRejectedError(Exception):
    """vLLM said 400: it will not run this request (vLLM's reason is the message)."""


class OutputTruncatedError(Exception):
    """The answer hit max_tokens before it was finished."""


# --- What vLLM sends back. We only declare the fields we use. ---------------------


class _Message(BaseModel):
    content: str | None = None


class _TokenLogprob(BaseModel):
    logprob: float


class _Logprobs(BaseModel):
    content: list[_TokenLogprob] | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None
    logprobs: _Logprobs | None = None


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _ChatCompletion(BaseModel):
    choices: list[_Choice]
    usage: _Usage = _Usage()


# --- Constraints: what to ask vLLM for, what to tell the model, how to check. -----


def structured_outputs(constraint: Constraint) -> dict[str, Any]:
    """Our constraint, in the shape vLLM's `structured_outputs` field expects."""
    match constraint:
        case JsonSchemaConstraint():
            return {"json": constraint.json_schema}
        case RegexConstraint():
            return {"regex": constraint.pattern}
        case ChoiceConstraint():
            return {"choice": list(constraint.choices)}
        case GrammarConstraint():
            return {"grammar": constraint.grammar}
        case _:
            assert_never(constraint)


def format_hint(constraint: Constraint) -> str:
    """Tell the model the format too, so the mask rarely has to fight it."""
    match constraint:
        case JsonSchemaConstraint():
            schema = json.dumps(constraint.json_schema)
            return f"Reply with only a JSON object that matches this JSON Schema:\n{schema}"
        case RegexConstraint():
            return (
                f"Reply with only text that matches this pattern: {constraint.pattern}"
            )
        case ChoiceConstraint():
            return "Reply with exactly one of: " + ", ".join(constraint.choices)
        case GrammarConstraint():
            return (
                f"Reply with only text that follows this grammar:\n{constraint.grammar}"
            )
        case _:
            assert_never(constraint)


@lru_cache(maxsize=256)
def _validator(schema_text: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads(schema_text))


def parse_output(constraint: Constraint, text: str) -> dict[str, Any] | str:
    """Check the answer against its constraint. Raise OutputInvalidError if it fails."""
    match constraint:
        case JsonSchemaConstraint():
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OutputInvalidError(f"not valid JSON: {exc}") from None
            schema_text = json.dumps(constraint.json_schema, separators=(",", ":"))
            problem = next(iter(_validator(schema_text).iter_errors(value)), None)
            if problem is not None:
                raise OutputInvalidError(f"breaks the schema: {problem.message}")
            if not isinstance(value, dict):
                raise OutputInvalidError("the JSON is not an object")
            return value
        case RegexConstraint():
            if re.fullmatch(constraint.pattern, text) is None:
                raise OutputInvalidError("does not match the pattern")
            return text
        case ChoiceConstraint():
            if text not in constraint.choices:
                raise OutputInvalidError("is not one of the choices")
            return text
        case GrammarConstraint():
            return text  # no grammar parser on our side: we trust the engine here
        case _:
            assert_never(constraint)


def with_hint(messages: tuple[ChatMessage, ...], hint: str) -> list[dict[str, str]]:
    """Put the hint in the system message (merging it into one if there already is one)."""
    out = [m.model_dump() for m in messages]
    if out[0]["role"] == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{hint}"
    else:
        out.insert(0, {"role": "system", "content": hint})
    return out


# --- The backend. ------------------------------------------------------------------


class VLLMBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_connections: int = 64,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key != "EMPTY" else {}
        self._client = httpx2.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout_s,
            limits=httpx2.Limits(max_connections=max_connections),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            base_url=settings.vllm_base_url,
            model=settings.model_name,
            api_key=settings.vllm_api_key.get_secret_value(),
            timeout_s=settings.request_timeout_s,
            max_connections=settings.max_inflight_generation,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def generate(self, request: GenerateRequest) -> EngineOutput[GenerateResult]:
        messages = [m.model_dump() for m in request.messages]
        reply, usage = await self._chat(self._body(messages, request.sampling))
        finish = "length" if reply.finish_reason == "length" else "stop"
        result = GenerateResult(
            task="generate", text=reply.message.content or "", finish_reason=finish
        )
        return EngineOutput(result=result, usage=usage)

    async def generate_structured(
        self, request: StructuredRequest, *, constrained: bool = True
    ) -> EngineOutput[StructuredResult]:
        """With constrained=False there is no mask: only the hint, and our check after."""
        constraint = request.constraint
        messages = with_hint(request.messages, format_hint(constraint))
        body = self._body(messages, request.sampling)
        if constrained:
            body["structured_outputs"] = structured_outputs(constraint)
        try:
            reply, usage = await self._chat(body)
        except UpstreamRejectedError as exc:
            if not constrained:
                raise
            raise ConstraintRejectedError(
                f"vLLM refused the constraint: {exc}"
            ) from None
        if reply.finish_reason == "length":
            raise OutputTruncatedError(
                f"the answer hit max_tokens ({request.sampling.max_tokens}) before it was complete"
            )
        text = reply.message.content or ""
        output = parse_output(constraint, text)
        result = StructuredResult(task="structured", output=output, raw_text=text)
        return EngineOutput(result=result, usage=usage)

    async def score_labels(
        self, request: ClassifyRequest
    ) -> EngineOutput[ClassifyResult]:
        """Let the model pick one label under a choice constraint.
        The score is NOT the model's belief that the label is true. vLLM reports
        logprobs after the grammar mask (measured on vLLM 0.32: blocked tokens come
        back at -9999 and the allowed ones sum to 1), so the score only compares the
        given labels with each other. It also follows one tokenization of the label,
        so it is a lower bound on the label's probability.
        """
        if request.multi_label:
            raise UnsupportedRequestError(f"{self.name} picks exactly one label")
        constraint = ChoiceConstraint(type="choice", choices=request.labels)
        hint = "Classify the user's text. " + format_hint(constraint)
        body = self._body(
            [
                {"role": "system", "content": hint},
                {"role": "user", "content": request.text},
            ],
            SamplingParams(temperature=0.0, max_tokens=64),
        )
        body["structured_outputs"] = structured_outputs(constraint)
        body["logprobs"] = True
        reply, usage = await self._chat(body)
        label = parse_output(constraint, reply.message.content or "")
        assert isinstance(label, str)
        score = sequence_probability(_token_logprobs(reply))
        result = ClassifyResult(
            task="classify", labels=(LabelScore(label=label, score=score),)
        )
        return EngineOutput(result=result, usage=usage)

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        """The model lists entities as JSON; we find each one in the text ourselves."""
        types = request.entity_types or DEFAULT_ENTITY_TYPES
        constraint = JsonSchemaConstraint.model_validate(
            {"type": "json_schema", "schema": entities_schema(types)}
        )
        hint = (
            f"Find entities of these types only: {', '.join(types)}. "
            "Skip anything that is not one of these types. "
            "Copy each entity exactly as it is written in the text. "
            'If there are none, reply with {"entities": []}. ' + format_hint(constraint)
        )
        body = self._body(
            [
                {"role": "system", "content": hint},
                {"role": "user", "content": request.text},
            ],
            SamplingParams(temperature=0.0, max_tokens=1024),
        )
        body["structured_outputs"] = structured_outputs(constraint)
        body["logprobs"] = True
        reply, usage = await self._chat(body)
        if reply.finish_reason == "length":
            raise OutputTruncatedError("the entity list was cut off at max_tokens")
        answer = parse_output(constraint, reply.message.content or "")
        assert isinstance(answer, dict)
        found = [(e["text"], e["label"]) for e in answer["entities"]]
        score = average_token_probability(_token_logprobs(reply))
        spans = tuple(
            span
            for span in locate(request.text, found, score)
            if span.score >= request.min_score
        )
        result = ExtractSpansResult(task="extract_spans", spans=spans)
        return EngineOutput(result=result, usage=usage)

    def _body(
        self, messages: list[dict[str, str]], sampling: SamplingParams
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            body["seed"] = sampling.seed
        return body

    async def _chat(self, body: dict[str, Any]) -> tuple[_Choice, Usage]:
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx2.TimeoutException:
            raise GenerationTimeoutError(
                f"{self.name} did not answer in time"
            ) from None
        except httpx2.TransportError as exc:
            raise UpstreamError(f"cannot reach vLLM: {exc}") from None
        if response.status_code == 400:
            raise UpstreamRejectedError(_error_message(response))
        if response.status_code >= 400:
            raise UpstreamError(
                f"vLLM answered {response.status_code}: {_error_message(response)}"
            )
        try:
            completion = _ChatCompletion.model_validate_json(response.content)
            reply = completion.choices[0]
        except (ValidationError, IndexError):
            raise UpstreamError("vLLM sent a response we could not read") from None
        tokens = completion.usage.completion_tokens
        usage = Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=tokens,
            # the prefill pass makes the first token; each later token needs one more
            forward_passes=max(tokens, 1),
        )
        return reply, usage


def _token_logprobs(reply: _Choice) -> list[float]:
    if reply.logprobs is None or not reply.logprobs.content:
        raise UpstreamError("vLLM did not send the logprobs we asked for")
    return [t.logprob for t in reply.logprobs.content]


def sequence_probability(logprobs: Sequence[float]) -> float:
    """P(whole answer) = product of each token's probability = exp(sum of logprobs)."""
    logprob_sum = sum(logprobs)
    return min(1.0, math.exp(logprob_sum))


def average_token_probability(logprobs: Sequence[float]) -> float:
    """A rough, answer-level confidence: the geometric mean of token probabilities."""
    return min(1.0, math.exp(sum(logprobs) / len(logprobs)))


def entities_schema(types: Sequence[str]) -> dict[str, Any]:
    entity = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "label": {"enum": list(types)}},
        "required": ["text", "label"],
    }
    return {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": entity}},
        "required": ["entities"],
    }


def locate(text: str, found: Sequence[tuple[str, str]], score: float) -> list[Span]:
    """Turn (entity text, label) pairs into spans by finding them in the text.

    The model only says *what* it saw. Where it is, we look up ourselves; anything
    that isn't in the text at all is dropped. A name mentioned twice is found twice.
    """
    spans: list[Span] = []
    next_search: dict[str, int] = {}
    for entity, label in found:
        start = text.find(entity, next_search.get(entity, 0)) if entity else -1
        if start < 0:
            continue
        next_search[entity] = start + len(entity)
        spans.append(
            Span(
                start=start,
                end=start + len(entity),
                label=label,
                text=entity,
                score=score,
            )
        )
    return spans


def _error_message(response: httpx2.Response) -> str:
    """vLLM has used two error shapes over the years; read either."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or "no details"
    if isinstance(data, dict):
        nested = data.get("error")
        message = (
            nested.get("message") if isinstance(nested, dict) else data.get("message")
        )
        if isinstance(message, str):
            return message
    return "no details"
