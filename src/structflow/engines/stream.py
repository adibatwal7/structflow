"""Read vLLM's streamed chat answer, and time it.

With "stream": true, vLLM sends the answer a few tokens at a time as Server-Sent
Events: lines that start with "data: ", each holding one JSON chunk, and finally
"data: [DONE]". Streaming is how we see *when* each token arrives:

- TTFT (time to first token): queueing + reading the prompt (prefill) + the
  network. It is what a user waits before anything happens.
- TPOT (time per output token): the time between tokens after the first one.
  It is the steady speed of decoding, where a grammar mask would show its cost.

ChatStream puts the chunks back together into the same shape as a normal
(non-streamed) answer, so the rest of the backend doesn't care how it arrived.
"""

import json
from dataclasses import dataclass
from typing import Any


class StreamError(Exception):
    """The stream broke: bad JSON, an error event, or it ended before [DONE]."""


@dataclass(frozen=True, slots=True)
class StreamTimes:
    ttft_ms: float | None
    tpot_ms: float | None


class ChatStream:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._logprobs: list[dict[str, Any]] | None = None
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] = {}
        self._first_s: float | None = None
        self._last_s: float | None = None
        self.done = False

    def feed(self, line: str, at_s: float) -> None:
        """Take one line of the stream, received `at_s` seconds after we sent it."""
        if not line.startswith("data:"):
            return  # blank lines and SSE comments separate events; skip them
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            raise StreamError("vLLM streamed a chunk that is not JSON") from None
        if not isinstance(chunk, dict):
            raise StreamError("vLLM streamed a chunk that is not a JSON object")
        if "error" in chunk:
            raise StreamError(f"vLLM failed mid-answer: {chunk['error']}")
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            self._take(choice, at_s)

    def _take(self, choice: dict[str, Any], at_s: float) -> None:
        content = (choice.get("delta") or {}).get("content")
        if content:  # the first chunk often carries only the role, with ""
            self._parts.append(content)
            if self._first_s is None:
                self._first_s = at_s
            self._last_s = at_s
        logprobs = (choice.get("logprobs") or {}).get("content")
        if logprobs:
            self._logprobs = (self._logprobs or []) + logprobs
        if choice.get("finish_reason") is not None:
            self._finish_reason = choice["finish_reason"]

    def completion(self) -> dict[str, Any]:
        """The whole answer, shaped like vLLM's non-streamed reply."""
        if not self.done:
            raise StreamError("vLLM's stream ended before [DONE]")
        choice: dict[str, Any] = {
            "message": {"content": "".join(self._parts)},
            "finish_reason": self._finish_reason,
        }
        if self._logprobs is not None:
            choice["logprobs"] = {"content": self._logprobs}
        return {"choices": [choice], "usage": self._usage}

    def times(self) -> StreamTimes:
        if self._first_s is None or self._last_s is None:
            return StreamTimes(ttft_ms=None, tpot_ms=None)
        tokens = int(self._usage.get("completion_tokens", 0))
        if tokens <= 1:
            return StreamTimes(ttft_ms=self._first_s * 1000, tpot_ms=None)
        return StreamTimes(
            ttft_ms=self._first_s * 1000,
            tpot_ms=(self._last_s - self._first_s) * 1000 / (tokens - 1),
        )


def encode_chat_stream(completion: dict[str, Any]) -> bytes:
    """The opposite of ChatStream: turn a whole answer into vLLM-style SSE bytes.

    Test fakes use this to speak the same streaming protocol as the real vLLM.
    """
    (choice,) = completion["choices"]
    content: str = choice["message"]["content"] or ""
    words = content.split(" ")
    pieces = [w + " " for w in words[:-1]] + [words[-1]]
    chunks: list[dict[str, Any]] = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
    ]
    chunks += [
        {"choices": [{"index": 0, "delta": {"content": p}}]} for p in pieces if p
    ]
    last: dict[str, Any] = {
        "index": 0,
        "delta": {},
        "finish_reason": choice.get("finish_reason"),
    }
    if choice.get("logprobs") is not None:
        last["logprobs"] = choice["logprobs"]
    chunks.append({"choices": [last]})
    chunks.append({"choices": [], "usage": completion.get("usage", {})})
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()
