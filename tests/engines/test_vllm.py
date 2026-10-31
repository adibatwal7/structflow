"""The vLLM backend, tested against a fake vLLM: no GPU, no network.

httpx2.MockTransport sends every request to a Python function instead of the
network. Our fake records what we sent and answers whatever the test wants.
"""

import json
from typing import Any

import httpx2
import pytest

from structflow.engines.base import ConstraintRejectedError, OutputInvalidError
from structflow.engines.limits import GenerationTimeoutError
from structflow.engines.stream import encode_chat_stream
from structflow.engines.vllm import (
    OutputTruncatedError,
    UpstreamError,
    UpstreamRejectedError,
    VLLMBackend,
)
from structflow.schemas.requests import GenerateRequest, StructuredRequest

pytestmark = pytest.mark.anyio

USER = [{"role": "user", "content": "Who sold this? ACME Corp, $40"}]
VENDOR_SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": "string"}},
    "required": ["vendor"],
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeVLLM:
    """Pretends to be vLLM: remembers what we sent, answers what the test says."""

    def __init__(
        self,
        content: str = "hello",
        *,
        finish_reason: str = "stop",
        status: int = 200,
        error: Any = None,
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.status = status
        self.error = error
        self.sent: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[httpx2.Headers] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(json.loads(request.content))
        self.paths.append(request.url.path)
        self.headers.append(request.headers)
        if self.status != 200:
            return httpx2.Response(self.status, json=self.error)
        return streamed(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self.content},
                        "finish_reason": self.finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )


def streamed(completion: dict[str, Any]) -> httpx2.Response:
    """Answer the way vLLM does when asked to stream: Server-Sent Events."""
    return httpx2.Response(
        200,
        content=encode_chat_stream(completion),
        headers={"content-type": "text/event-stream"},
    )


def backend(fake: Any, api_key: str = "EMPTY") -> VLLMBackend:
    return VLLMBackend(
        base_url="http://vllm.test/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=api_key,
        transport=httpx2.MockTransport(fake),
    )


def structured(constraint: dict[str, Any], **extra: Any) -> StructuredRequest:
    return StructuredRequest.model_validate(
        {"task": "structured", "messages": USER, "constraint": constraint, **extra}
    )


def generate(**extra: Any) -> GenerateRequest:
    return GenerateRequest.model_validate(
        {"task": "generate", "messages": USER, **extra}
    )


async def test_sends_an_openai_style_chat_request() -> None:
    fake = FakeVLLM()
    await backend(fake).generate(generate(sampling={"max_tokens": 50, "seed": 7}))
    (sent,) = fake.sent
    assert fake.paths == ["/v1/chat/completions"]
    assert sent["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert sent["messages"] == USER
    assert (sent["max_tokens"], sent["seed"], sent["temperature"]) == (50, 7, 0.0)
    assert "structured_outputs" not in sent
    assert "authorization" not in fake.headers[0]


async def test_the_api_key_is_sent_as_a_bearer_token() -> None:
    fake = FakeVLLM()
    await backend(fake, api_key="sk-test").generate(generate())
    assert fake.headers[0]["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize(
    ("constraint", "vllm_field", "reply", "output"),
    [
        (
            {"type": "json_schema", "schema": VENDOR_SCHEMA},
            {"json": VENDOR_SCHEMA},
            '{"vendor": "ACME Corp"}',
            {"vendor": "ACME Corp"},
        ),
        (
            {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
            {"regex": r"\d{4}-\d{2}-\d{2}"},
            "2026-10-29",
            "2026-10-29",
        ),
        (
            {"type": "choice", "choices": ["yes", "no"]},
            {"choice": ["yes", "no"]},
            "no",
            "no",
        ),
        (
            {"type": "grammar", "grammar": 'root ::= "yes" | "no"'},
            {"grammar": 'root ::= "yes" | "no"'},
            "yes",
            "yes",
        ),
    ],
    ids=["json_schema", "regex", "choice", "grammar"],
)
async def test_each_constraint_becomes_a_vllm_structured_output(
    constraint: dict[str, Any], vllm_field: dict[str, Any], reply: str, output: Any
) -> None:
    fake = FakeVLLM(reply)
    out = await backend(fake).generate_structured(structured(constraint))
    assert fake.sent[0]["structured_outputs"] == vllm_field
    assert out.result.output == output
    assert out.result.raw_text == reply


async def test_the_format_is_also_explained_to_the_model() -> None:
    fake = FakeVLLM('{"vendor": "ACME"}')
    constraint = {"type": "json_schema", "schema": VENDOR_SCHEMA}
    await backend(fake).generate_structured(structured(constraint))
    first = fake.sent[0]["messages"][0]
    assert first["role"] == "system" and "JSON Schema" in first["content"]

    fake = FakeVLLM('{"vendor": "ACME"}')
    with_system = [{"role": "system", "content": "You read invoices."}, *USER]
    request = StructuredRequest.model_validate(
        {"task": "structured", "messages": with_system, "constraint": constraint}
    )
    await backend(fake).generate_structured(request)
    roles = [m["role"] for m in fake.sent[0]["messages"]]
    assert roles == ["system", "user"]  # merged into the one system message
    assert fake.sent[0]["messages"][0]["content"].startswith("You read invoices.")


async def test_usage_comes_from_vllm() -> None:
    out = await backend(FakeVLLM()).generate(generate())
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (12, 7)
    assert out.usage.forward_passes == 7


async def test_a_truncated_structured_answer_is_an_error() -> None:
    fake = FakeVLLM('{"vendor": "AC', finish_reason="length")
    constraint = {"type": "json_schema", "schema": VENDOR_SCHEMA}
    with pytest.raises(OutputTruncatedError):
        await backend(fake).generate_structured(structured(constraint))


async def test_a_truncated_free_answer_is_still_an_answer() -> None:
    out = await backend(FakeVLLM("Once upon a", finish_reason="length")).generate(
        generate()
    )
    assert (out.result.text, out.result.finish_reason) == ("Once upon a", "length")


@pytest.mark.parametrize(
    ("constraint", "reply"),
    [
        ({"type": "json_schema", "schema": VENDOR_SCHEMA}, '{"vendor": '),
        ({"type": "json_schema", "schema": VENDOR_SCHEMA}, '{"amount": 40}'),
        ({"type": "regex", "pattern": r"\d+"}, "forty"),
        ({"type": "choice", "choices": ["yes", "no"]}, "maybe"),
    ],
    ids=["not-json", "missing-field", "regex-miss", "not-a-choice"],
)
async def test_an_answer_that_breaks_its_constraint_never_gets_through(
    constraint: dict[str, Any], reply: str
) -> None:
    with pytest.raises(OutputInvalidError):
        await backend(FakeVLLM(reply)).generate_structured(structured(constraint))


@pytest.mark.parametrize(
    "error",
    [
        {"error": {"message": "max_tokens is too large", "type": "BadRequestError"}},
        {"object": "error", "message": "max_tokens is too large", "code": 400},
    ],
    ids=["new-shape", "old-shape"],
)
async def test_a_400_is_a_rejection_with_vllms_reason(error: dict[str, Any]) -> None:
    fake = FakeVLLM(status=400, error=error)
    with pytest.raises(UpstreamRejectedError, match="max_tokens is too large"):
        await backend(fake).generate(generate())


async def test_a_server_error_is_an_upstream_error() -> None:
    fake = FakeVLLM(status=500, error={"message": "CUDA out of memory"})
    with pytest.raises(UpstreamError, match="500"):
        await backend(fake).generate(generate())


async def test_a_reply_we_cannot_read_is_an_upstream_error() -> None:
    def garbage(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"surprise": True})

    with pytest.raises(UpstreamError):
        await backend(garbage).generate(generate())


async def test_unreachable_vllm_is_an_upstream_error() -> None:
    def refused(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(UpstreamError, match="cannot reach vLLM"):
        await backend(refused).generate(generate())


async def test_a_slow_vllm_is_a_timeout() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    with pytest.raises(GenerationTimeoutError):
        await backend(slow).generate(generate())


async def test_without_the_mask_only_the_hint_and_the_check_remain() -> None:
    fake = FakeVLLM('{"vendor": "ACME"}')
    constraint = {"type": "json_schema", "schema": VENDOR_SCHEMA}
    out = await backend(fake).generate_structured(
        structured(constraint), constrained=False
    )
    assert "structured_outputs" not in fake.sent[0]
    assert "JSON Schema" in fake.sent[0]["messages"][0]["content"]
    assert out.result.output == {"vendor": "ACME"}

    with pytest.raises(OutputInvalidError):  # still checked, mask or not
        await backend(FakeVLLM("sure! here you go")).generate_structured(
            structured(constraint), constrained=False
        )


async def test_a_refused_constraint_is_a_constraint_error() -> None:
    fake = FakeVLLM(status=400, error={"error": {"message": "unsupported regex"}})
    constraint = {"type": "regex", "pattern": r"\d+"}
    with pytest.raises(ConstraintRejectedError, match="unsupported regex"):
        await backend(fake).generate_structured(structured(constraint))


async def test_every_request_is_streamed_with_usage() -> None:
    fake = FakeVLLM()
    await backend(fake).generate(generate())
    (sent,) = fake.sent
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


async def test_streaming_gives_us_token_timings() -> None:
    out = await backend(FakeVLLM("one two three")).generate(generate())
    assert out.result.text == "one two three"
    assert out.ttft_ms is not None and out.ttft_ms >= 0
    assert out.tpot_ms is not None and out.tpot_ms >= 0


async def test_a_stream_that_breaks_is_an_upstream_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b'data: {"choices": [{"delta": {"con')

    with pytest.raises(UpstreamError):
        await backend(broken).generate(generate())


async def test_compact_json_forbids_whitespace_in_json_only() -> None:
    fake = FakeVLLM('{"vendor":"ACME"}')
    compact = VLLMBackend(
        base_url="http://vllm.test/v1",
        model="llama",
        compact_json=True,
        transport=httpx2.MockTransport(fake),
    )
    await compact.generate_structured(
        structured({"type": "json_schema", "schema": VENDOR_SCHEMA})
    )
    await compact.generate_structured(structured({"type": "regex", "pattern": ".+"}))
    json_sent, regex_sent = fake.sent
    assert json_sent["structured_outputs"]["disable_any_whitespace"] is True
    assert "disable_any_whitespace" not in regex_sent["structured_outputs"]


async def test_whitespace_is_allowed_by_default() -> None:
    fake = FakeVLLM('{"vendor": "ACME"}')
    await backend(fake).generate_structured(
        structured({"type": "json_schema", "schema": VENDOR_SCHEMA})
    )
    assert "disable_any_whitespace" not in fake.sent[0]["structured_outputs"]
