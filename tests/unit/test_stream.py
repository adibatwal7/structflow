"""Reading vLLM's streamed answers, and timing their tokens."""

import json
from typing import Any

import pytest

from structflow.engines.stream import ChatStream, StreamError, encode_chat_stream

ANSWER: dict[str, Any] = {
    "choices": [
        {
            "message": {"content": "Paris is lovely"},
            "finish_reason": "stop",
            "logprobs": {"content": [{"token": "Paris", "logprob": -0.1}]},
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3},
}


def data(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk)}"


def delta(content: str) -> str:
    return data({"choices": [{"index": 0, "delta": {"content": content}}]})


def test_a_streamed_answer_comes_back_whole() -> None:
    stream = ChatStream()
    for line in encode_chat_stream(ANSWER).decode().splitlines():
        stream.feed(line, at_s=0.0)
    assert stream.completion() == ANSWER


def test_ttft_is_the_first_real_token_and_tpot_the_gap_after_it() -> None:
    stream = ChatStream()
    stream.feed(
        data({"choices": [{"delta": {"role": "assistant", "content": ""}}]}), 0.05
    )
    stream.feed(delta("Par"), 0.20)  # first token: 200 ms
    stream.feed(delta("is"), 0.22)
    stream.feed(delta("!"), 0.24)
    stream.feed(data({"choices": [], "usage": {"completion_tokens": 3}}), 0.25)
    stream.feed("data: [DONE]", 0.25)
    times = stream.times()
    assert times.ttft_ms == pytest.approx(200.0)
    assert times.tpot_ms == pytest.approx(20.0)  # 40 ms over 2 gaps


def test_no_tokens_means_no_timings() -> None:
    stream = ChatStream()
    stream.feed(data({"choices": [], "usage": {"completion_tokens": 0}}), 0.1)
    assert stream.times().ttft_ms is None
    assert stream.times().tpot_ms is None


def test_blank_lines_and_comments_are_skipped() -> None:
    stream = ChatStream()
    for line in ["", ": keep-alive", delta("hi"), "", "data: [DONE]"]:
        stream.feed(line, 0.0)
    assert stream.completion()["choices"][0]["message"]["content"] == "hi"


@pytest.mark.parametrize(
    "line",
    ["data: {not json", "data: [1, 2]", data({"error": {"message": "CUDA OOM"}})],
    ids=["bad-json", "not-an-object", "error-event"],
)
def test_a_broken_stream_is_an_error(line: str) -> None:
    with pytest.raises(StreamError):
        ChatStream().feed(line, 0.0)


def test_a_stream_cut_off_before_done_is_an_error() -> None:
    stream = ChatStream()
    stream.feed(delta("half an ans"), 0.0)
    with pytest.raises(StreamError, match="DONE"):
        stream.completion()
