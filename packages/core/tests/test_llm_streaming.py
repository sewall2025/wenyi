"""Offline checks for streamed OpenAI-compatible chat completions."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from wenyi_core.llm.limits import RequestCancelled
from wenyi_core.llm.providers import _openai_compatible
from wenyi_core.llm.retrying import (
    ContextLengthExceeded,
    EmptyResponseError,
    IncompleteStreamError,
    StreamErrorEvent,
    StreamTimeoutError,
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
    with pytest.raises(TruncatedResponseError) as caught:
        client.complete(MESSAGES, operation="translation.body")
    assert len(completions.calls) == 1
    assert stream.closed
    assert _totals(client)["total_tokens"] == 10
    # The reported prompt plus completion total lets callers learn the context window.
    assert caught.value.usage_total == 10


def test_stream_without_finish_reason_is_retried_and_its_usage_recorded(monkeypatch):
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
    # Each attempt is billed by the provider, so the cut-off attempt is recorded once too.
    assert (_totals(client)["calls"], _totals(client)["total_tokens"]) == (2, 20)


def test_json_stream_without_finish_reason_accepts_complete_json(monkeypatch):
    events: list[dict] = []
    stream = FakeStream(
        [
            stream_chunk(content='{"translations":'),
            stream_chunk(content=' ["ok"]}\n'),
            stream_chunk(usage=USAGE),
        ]
    )
    client, completions = _client(monkeypatch, [stream], events=events, max_retries=0)
    result = client.complete(MESSAGES, json_mode=True, operation="translation.body")
    assert result == '{"translations": ["ok"]}\n'
    assert len(completions.calls) == 1
    missing = [event for event in events if event["event"] == "stream_missing_finish_reason"]
    assert len(missing) == 1
    assert (missing[0]["chunks"], missing[0]["content_chars"]) == (3, 25)
    assert not any(event["event"] == "llm_retry_wait" for event in events)
    assert _totals(client)["total_tokens"] == 10


def test_json_stream_without_finish_reason_rejects_truncated_json(monkeypatch):
    events: list[dict] = []
    stream = FakeStream(
        [
            stream_chunk(content='{"translations": ["first", "sec'),
            stream_chunk(usage=USAGE),
        ]
    )
    client, _ = _client(monkeypatch, [stream], events=events, max_retries=0)
    with pytest.raises(IncompleteStreamError) as raised:
        client.complete(MESSAGES, json_mode=True, operation="translation.body")
    message = str(raised.value)
    for field in ("chunks=2", "content_chars=31", "reasoning_chars=0", "usage=yes"):
        assert field in message
    assert 'tail=\'{"translations": ["first", "sec\'' in message
    exhausted = [event for event in events if event["event"] == "llm_retry_exhausted"]
    assert exhausted[0]["reason"] == "incomplete_stream"
    assert exhausted[0]["diagnostic"] == raised.value.diagnostic
    assert _totals(client)["total_tokens"] == 10


def test_stream_diagnostic_tail_is_short_and_redacted(monkeypatch):
    text = "x" * 200 + "sk-test-secret"
    events: list[dict] = []
    client, _ = _client(
        monkeypatch, [FakeStream([stream_chunk(content=text)])], events=events, max_retries=0
    )
    client._redact = lambda value: value.replace("sk-test-secret", "[redacted]")
    with pytest.raises(IncompleteStreamError) as raised:
        client.complete(MESSAGES, operation="translation.body")
    assert "usage=no" in str(raised.value)
    assert len(raised.value.diagnostic) < 200
    exhausted = next(event for event in events if event["event"] == "llm_retry_exhausted")
    assert "sk-test-secret" not in exhausted["diagnostic"]
    assert _totals(client)["calls"] == 0


def test_plain_stream_without_finish_reason_fails_even_when_text_is_json(monkeypatch):
    client, _ = _client(monkeypatch, [FakeStream([stream_chunk(content="{}")])], max_retries=0)
    with pytest.raises(IncompleteStreamError):
        client.complete(MESSAGES, operation="translation.body")


def test_json_stream_without_finish_reason_uses_reasoning_fallback(monkeypatch):
    stream = FakeStream([stream_chunk(reasoning_content='{"translations":["ok"]}')])
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


def _stream_error(code, error_type="server_error", message="secret prompt text sk-test"):
    from openai import APIError

    body = {"code": code, "type": error_type, "message": message}
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return FakeStream([stream_chunk(content="Part"), APIError(body["message"], request, body=body)])


def test_stream_error_event_is_retried_with_a_message_free_diagnostic(monkeypatch):
    events: list[dict] = []
    broken = _stream_error("overloaded")
    client, completions = _client(monkeypatch, [broken, _complete_stream("Done")], events=events)
    assert client.complete(MESSAGES, operation="translation.body") == "Done"
    assert len(completions.calls) == 2
    assert broken.closed
    waits = [event for event in events if event["event"] == "llm_retry_wait"]
    assert [event["reason"] for event in waits] == ["stream_error"]
    assert waits[0]["diagnostic"] == "code='overloaded' type='server_error' message_chars=26"
    assert "secret" not in repr(events)


@pytest.mark.parametrize(("code", "error_type"), [("bad", "invalid_request_error")])
def test_permanent_stream_error_event_is_not_retried(monkeypatch, code, error_type):
    from openai import APIError

    events: list[dict] = []
    client, completions = _client(monkeypatch, [_stream_error(code, error_type)], events=events)
    with pytest.raises(StreamErrorEvent) as raised:
        client.complete(MESSAGES, operation="translation.body")
    assert len(completions.calls) == 1
    assert retry_reason(raised.value) is None
    assert isinstance(raised.value.__cause__, APIError)
    assert "secret" not in str(raised.value)
    failed = [event for event in events if event["event"] == "llm_request_failed"]
    assert failed[0]["error_category"] == "stream_error"
    assert failed[0]["error_message"] == "The provider reported an error during streaming."


@pytest.mark.parametrize(
    ("code", "message", "window"),
    [
        ("context_length_exceeded", "secret prompt text sk-test", None),
        # Magpie forwards only the upstream text, cut off after the parameter hint.
        (
            "",
            '上游 400: {"error":{"message":"This model\'s maximum context length is 24576 '
            "tokens. However, you requested 0 output tokens and your prompt contains at least "
            "24577 input tokens. Please reduce the length of the input prompt. (paramet",
            24576,
        ),
    ],
)
def test_context_overflow_stream_event_is_classified_and_teaches_the_window(
    monkeypatch, code, message, window
):
    events: list[dict] = []
    client, completions = _client(
        monkeypatch, [_stream_error(code, "", message)], events=events, max_retries=3
    )
    with pytest.raises(ContextLengthExceeded) as raised:
        client.complete(MESSAGES, operation="translation.body")
    assert len(completions.calls) == 1  # Resending the same prompt cannot fit it.
    assert raised.value.window == window
    assert retry_reason(raised.value) is None
    assert client.context_window("translation.body") == window
    failed = [event for event in events if event["event"] == "llm_request_failed"]
    assert failed[0]["error_category"] == "context_length"
    assert "secret" not in repr(events)
    assert not [event for event in events if event["event"] == "llm_retry_wait"]


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


SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "object"}},
    "required": ["translations"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("stream", [True, False])
def test_structured_output_sends_strict_schema_on_both_paths(monkeypatch, stream):
    outcome = (
        _complete_stream("{}")
        if stream
        else SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            usage=USAGE,
        )
    )
    client, completions = _client(
        monkeypatch, [outcome], stream=stream, structured_output="json_schema"
    )
    client.complete(MESSAGES, operation="translation.body", json_mode=True, json_schema=SCHEMA)
    assert completions.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "wenyi_output", "strict": True, "schema": SCHEMA},
    }


@pytest.mark.parametrize(
    ("connection", "json_mode", "schema", "expected"),
    [
        ({}, True, SCHEMA, {"type": "json_object"}),
        ({"structured_output": "none"}, True, SCHEMA, {"type": "json_object"}),
        ({"structured_output": "json_schema"}, True, None, {"type": "json_object"}),
        ({"structured_output": "json_schema"}, False, SCHEMA, None),
    ],
)
def test_structured_output_falls_back_to_json_mode(
    monkeypatch, connection, json_mode, schema, expected
):
    client, completions = _client(monkeypatch, [_complete_stream("{}")], **connection)
    client.complete(MESSAGES, operation="translation.body", json_mode=json_mode, json_schema=schema)
    assert completions.calls[0].get("response_format") == expected


@pytest.mark.parametrize(
    "kind",
    ["deepseek", "openai", "openrouter", "opencode-go", "openai-compatible", "orcarouter"],
)
def test_structured_output_is_excluded_from_identity(kind):
    base_url = {"openai-compatible": {"base_url": "https://example.invalid/v1"}}.get(kind, {})
    routes = [
        resolve_routes(model_config(kind=kind, **base_url, **extra))["translation.body"]
        for extra in ({}, {"structured_output": "none"}, {"structured_output": "json_schema"})
    ]
    assert len({route.model_identity for route in routes}) == 1
    assert len({route.fingerprint for route in routes}) == 1


@pytest.mark.parametrize("kind", ["gemini", "fake"])
def test_non_openai_connections_reject_structured_output(kind):
    with pytest.raises(ValueError):
        model_config(kind=kind, structured_output="json_schema")


def test_structured_output_rejects_unknown_mode():
    with pytest.raises(ValueError):
        model_config(structured_output="json_object")


class _SlowStream(FakeStream):
    """Yield ``chunk`` every ``delay`` seconds until closed, like a never-ending SSE stream."""

    def __init__(self, chunk, delay=0.01):
        super().__init__([])
        self.chunk, self.delay = chunk, delay

    def close(self):
        self.closed = True

    def __iter__(self):
        while not self.closed:
            time.sleep(self.delay)
            yield self.chunk


@pytest.fixture
def fast_watchdog(monkeypatch):
    monkeypatch.setattr(_openai_compatible, "_STREAM_WATCHDOG_POLL_SECONDS", 0.01)


def test_stream_total_timeout_closes_attempt_and_retries(monkeypatch, fast_watchdog):
    endless = _SlowStream(stream_chunk(content="x"))
    events: list[dict] = []
    client, completions = _client(
        monkeypatch, [endless, _complete_stream("ok")], events=events, timeout=0.2
    )
    assert client.complete(MESSAGES, operation="translation.body") == "ok"
    assert endless.closed and len(completions.calls) == 2
    retry = [event for event in events if event["event"] == "llm_retry_wait"]
    assert retry and retry[0]["reason"] == "timeout"


def test_stream_heartbeats_without_content_trigger_idle_timeout(monkeypatch, fast_watchdog):
    monkeypatch.setattr(_openai_compatible, "_STREAM_IDLE_SECONDS", 0.1)
    heartbeat = _SlowStream(SimpleNamespace(choices=[], usage=None))
    client, completions = _client(monkeypatch, [heartbeat], max_retries=0, timeout=30)
    started = time.monotonic()
    with pytest.raises(StreamTimeoutError, match="no content"):
        client.complete(MESSAGES, operation="translation.body")
    assert time.monotonic() - started < 5 and heartbeat.closed
    assert retry_reason(StreamTimeoutError("idle")) == "timeout"


def test_cancel_closes_open_stream_and_is_not_retried(monkeypatch):
    endless = _SlowStream(stream_chunk(content="x"))
    client, completions = _client(monkeypatch, [endless, _complete_stream("never")], timeout=30)
    timer = threading.Timer(0.1, client.cancel)
    timer.start()
    started = time.monotonic()
    with pytest.raises(RequestCancelled):
        client.complete(MESSAGES, operation="translation.body")
    timer.join()
    assert time.monotonic() - started < 5
    assert endless.closed and len(completions.calls) == 1
    assert not client.limits._streams


def test_stream_progress_events_report_chunks_and_characters(monkeypatch):
    monkeypatch.setattr(_openai_compatible, "_STREAM_PROGRESS_SECONDS", 0)
    events: list[dict] = []
    client, _ = _client(monkeypatch, [_complete_stream("ab", "cde")], events=events)
    assert client.complete(MESSAGES, operation="translation.body") == "abcde"
    progress = [event for event in events if event["event"] == "llm_stream_progress"]
    assert progress[-1]["chunks"] == 4 and progress[-1]["chars"] == 5
    assert progress[0]["operation"] == "translation.body"
