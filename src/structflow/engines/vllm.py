"""Generation backend: vLLM's OpenAI-compatible server.

The grammar engine inside vLLM (XGrammar) makes invalid tokens impossible while the
model writes. We still check every answer here, because a grammar engine may not
enforce every JSON Schema keyword, and a constraint is a promise to our client.
"""

import json
import re
from functools import lru_cache
from types import TracebackType
from typing import Any, Self, assert_never

import httpx2
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from structflow.config import Settings
from structflow.engines.base import EngineOutput, UnsupportedRequestError
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
    StructuredResult,
    Usage,
)


class UpstreamError(Exception):
    """vLLM could not be reached, or failed, or answered something we can't read."""


class UpstreamRejectedError(Exception):
    """vLLM said 400: it will not run this request (vLLM's reason is the message)."""


class OutputTruncatedError(Exception):
    """The answer hit max_tokens before it was finished."""


class OutputInvalidError(Exception):
    """The answer broke its constraint. It must never reach the client."""


# --- What vLLM sends back. We only declare the fields we use. ---------------------


class _Message(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


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
        self, request: StructuredRequest
    ) -> EngineOutput[StructuredResult]:
        constraint = request.constraint
        messages = with_hint(request.messages, format_hint(constraint))
        body = self._body(messages, request.sampling)
        body["structured_outputs"] = structured_outputs(constraint)
        reply, usage = await self._chat(body)
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
        raise UnsupportedRequestError(f"{self.name}: label scoring comes in Step 3.2")

    async def extract_spans(
        self, request: ExtractSpansRequest
    ) -> EngineOutput[ExtractSpansResult]:
        raise UnsupportedRequestError(f"{self.name}: span extraction comes in Step 3.2")

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
