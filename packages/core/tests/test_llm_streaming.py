"""Offline checks for streamed OpenAI-compatible chat completions."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from wenyi_core.llm.retrying import (
    EmptyResponseError,
    IncompleteStreamError,
    TruncatedResponseError,
    retry_reason,
)
from wenyi_core.llm.router import RoutedLLMClient
from wenyi_core.llm.routing import resolve_routes

from tests.model_fixtures import FakeStream, model_config, stream_chunk

USAGE = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
MESSAGES = [{"role": "user", "content": "source"}]


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(monkeypatch, outcomes, *, events=None, max_retries=2, **connection):
    config = model_config(max_retries=max_retries, **connection)
    client = RoutedLLMClient(config)
    completions = _Completions(outcomes)
    client.adapter("default")._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(client.limits, "wait_for_retry", lambda delay: None)
    if events is not None:
        client.set_event_sink(lambda event, **data: events.append({"event": event, **data}))
    return client, completions


def _complete_stream(*parts, finish_reason="stop", usage=USAGE):
    chunks = [stream_chunk(content=part) for part in parts]
    chunks.append(stream_chunk(finish_reason=finish_reason))
    if usage is not None:
        chunks.append(stream_chunk(usage=usage))
    return FakeStream(chunks)


def _totals(client):
    return client.usage_summary()["totals"]


def test_default_request_streams_with_usage_and_joins_chunks(monkeypatch):
    stream = _complete_stream("Hel", "lo")
    client, completions = _client(monkeypatch, [stream])
    assert client.complete(MESSAGES, operation="translation.body") == "Hello"
    kwargs = completions.calls[0]
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert stream.closed
    assert (_totals(client)["calls"], _totals(client)["total_tokens"]) == (1, 10)


def test_stream_false_uses_single_response_path(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=USAGE,
    )
    client, completions = _client(monkeypatch, [response], stream=False)
    assert client.complete(MESSAGES, operation="translation.body") == "ok"
    assert completions.calls[0]["stream"] is False
    assert "stream_options" not in completions.calls[0]
    assert _totals(client)["total_tokens"] == 10


def test_stream_without_usage_records_nothing(monkeypatch):
    client, _ = _client(monkeypatch, [_complete_stream("ok", usage=None)])
    assert client.complete(MESSAGES, operation="translation.body") == "ok"
    assert _totals(client)["calls"] == 0


def test_stream_ignores_empty_choices_and_missing_delta(monkeypatch):
    stream = FakeStream(
        [
            SimpleNamespace(choices=[], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason=None)], usage=None),
            stream_chunk(content="ok"),
            stream_chunk(finish_reason="stop"),
            SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason=None)], usage=USAGE),
        ]
    )
    client, _ = _client(monkeypatch, [stream])
    assert client.complete(MESSAGES, operation="translation.body") == "ok"
    assert _totals(client)["total_tokens"] == 10


def test_stream_length_stop_is_truncated_after_recording_usage(monkeypatch):
    stream = _complete_stream("Part", finish_reason="length")
    client, completions = _client(monkeypatch, [stream])
    with pytest.raises(TruncatedResponseError):
        client.complete(MESSAGES, operation="translation.body")
    assert len(completions.calls) == 1
    assert stream.closed
    assert _totals(client)["total_tokens"] == 10


def test_stream_without_finish_reason_is_retried_and_discarded(monkeypatch):
    events: list[dict] = []
    incomplete = FakeStream([stream_chunk(content="Part"), stream_chunk(usage=USAGE)])
    client, completions = _client(
        monkeypatch, [incomplete, _complete_stream("Done")], events=events
    )
    assert client.complete(MESSAGES, operation="translation.body") == "Done"
    assert len(completions.calls) == 2
    assert incomplete.closed
    assert [event["reason"] for event in events if event["event"] == "llm_retry_wait"] == [
        "incomplete_stream"
    ]
    assert (_totals(client)["calls"], _totals(client)["total_tokens"]) == (1, 10)


def test_interrupted_stream_is_retried_and_usage_counted_once(monkeypatch):
    events: list[dict] = []
    broken = FakeStream(
        [stream_chunk(content="Part"), httpx.RemoteProtocolError("peer closed connection")]
    )
    client, completions = _client(monkeypatch, [broken, _complete_stream("Done")], events=events)
    assert client.complete(MESSAGES, operation="translation.body") == "Done"
    assert len(completions.calls) == 2
    assert broken.closed
    assert [event["reason"] for event in events if event["event"] == "llm_retry_wait"] == [
        "connection"
    ]
    assert (_totals(client)["calls"], _totals(client)["total_tokens"]) == (1, 10)


def test_streamed_reasoning_content_feeds_json_fallback(monkeypatch):
    stream = FakeStream(
        [
            stream_chunk(reasoning_content='{"translations":'),
            stream_chunk(reasoning_content='["ok"]}'),
            stream_chunk(finish_reason="stop"),
        ]
    )
    config = model_config(
        kind="openai-compatible",
        base_url="https://example.invalid/v1",
        max_retries=0,
        profiles={
            "strong": {"model": "m", "options": {"json_response_fallback": "reasoning_content"}}
        },
    )
    client = RoutedLLMClient(config)
    client.adapter("default")._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions([stream]))
    )
    result = client.complete(MESSAGES, json_mode=True, operation="translation.body")
    assert result == '{"translations":["ok"]}'


def test_streamed_empty_content_without_fallback_is_empty(monkeypatch):
    client, _ = _client(monkeypatch, [_complete_stream(" ")], max_retries=0, stream=True)
    with pytest.raises(EmptyResponseError):
        client.complete(MESSAGES, operation="translation.body")


def test_incomplete_stream_reason():
    assert retry_reason(IncompleteStreamError("cut")) == "incomplete_stream"


@pytest.mark.parametrize(
    "kind",
    ["deepseek", "openai", "openrouter", "opencode-go", "openai-compatible", "orcarouter"],
)
def test_stream_option_is_accepted_and_excluded_from_identity(kind):
    base_url = {"openai-compatible": {"base_url": "https://example.invalid/v1"}}.get(kind, {})
    routes = [
        resolve_routes(model_config(kind=kind, **base_url, **extra))["translation.body"]
        for extra in ({}, {"stream": True}, {"stream": False})
    ]
    assert len({route.model_identity for route in routes}) == 1
    assert len({route.fingerprint for route in routes}) == 1


def test_identity_matches_values_before_stream_option():
    compatible = resolve_routes(
        model_config(kind="openai-compatible", base_url="https://example.invalid/v1", stream=False)
    )["translation.body"]
    deepseek = resolve_routes(model_config(stream=False))["translation.body"]
    assert compatible.model_identity == (
        "0862812ee077cc60a3a9614932da521991a50280eca6d1605b989f51bc0dc6aa"
    )
    assert compatible.fingerprint == (
        "544955cb1ae4c1280da5f30fa685ee2e4cb099300755e3d48dc621e456b1c6d9"
    )
    assert deepseek.model_identity == (
        "3c7a59c406e22451bea4dc2acea891c50640abafb20433aa73ade34f6271edf3"
    )
    assert deepseek.fingerprint == (
        "fb6382872aa09bd143001ea0d9533b03d32d8aacc65c906ad535ae26f78e7adb"
    )


def test_gemini_rejects_stream_connection_option():
    with pytest.raises(ValueError):
        model_config(kind="gemini", stream=False)
