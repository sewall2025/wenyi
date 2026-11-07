"""OpenAI-compatible wire protocol and single-attempt response handling."""

from __future__ import annotations

import json
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from ..limits import interrupt_stream
from ..retrying import (
    EmptyResponseError,
    IncompleteStreamError,
    StreamErrorEvent,
    StreamTimeoutError,
    TruncatedResponseError,
)
from ..transport import (
    ConnectionOptions,
    Messages,
    ProviderAdapter,
    RequestContext,
    ResolvedModel,
)
from ..usage import (
    UsageSample,
    make_usage_sample,
    read_usage_counter,
    read_usage_int,
    read_usage_value,
)

OptionsT = TypeVar("OptionsT", bound=BaseModel)
_JSON_MODE_INSTRUCTION = "Output must be valid json."


def base_request_kwargs(
    model: str,
    messages: Messages,
    *,
    json_mode: bool,
) -> dict[str, Any]:
    """Build base Chat Completions arguments and add explicit JSON instructions when requested."""
    request_messages = messages
    if json_mode:
        request_messages = [dict(message) for message in messages]
        for message in request_messages:
            if message.get("role") == "system":
                message["content"] = f"{message.get('content', '')}\n\n{_JSON_MODE_INSTRUCTION}"
                break
        else:
            request_messages.insert(
                0,
                {"role": "system", "content": _JSON_MODE_INSTRUCTION},
            )
        # Some gateways validate only user content, including when mapping to Responses input.
        # Mentioning JSON only in the system message may be insufficient,
        # so also append the instruction to the final user message.
        for message in reversed(request_messages):
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if "json" not in content.lower():
                    message["content"] = f"{content}\n\n{_JSON_MODE_INSTRUCTION}"
                break
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "stream": False,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge provider request bodies, preferring user values."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def normalize_openai_usage(usage: Any, *, separate_cache_writes: bool = True) -> UsageSample | None:
    """Normalize inclusive Chat Completions totals, not native Anthropic input_tokens.

    Gateways may expose OpenRouter-style nested cache_write_tokens or top-level
    cache_creation_input_tokens/cache_creation. Absence is not evidence of zero
    writes on a generic endpoint. Native OpenAI has no separate write counter.
    """
    if usage is None or read_usage_counter(usage, "prompt_tokens") is None:
        return None
    prompt_tokens = read_usage_int(usage, "prompt_tokens")
    details = read_usage_value(usage, "prompt_tokens_details")
    cache_hit_tokens = read_usage_counter(details, "cached_tokens")
    if cache_hit_tokens is None:
        cache_hit_tokens = read_usage_counter(usage, "cache_read_input_tokens")
    cache_write_tokens = read_usage_counter(details, "cache_write_tokens")
    if cache_write_tokens is None:
        cache_write_tokens = read_usage_counter(usage, "cache_creation_input_tokens")
    creation = read_usage_value(usage, "cache_creation")
    write_5m = read_usage_counter(creation, "ephemeral_5m_input_tokens")
    write_1h = read_usage_counter(creation, "ephemeral_1h_input_tokens")
    writes_complete = cache_write_tokens is not None or (
        write_5m is not None and write_1h is not None
    )
    if cache_write_tokens is None and (write_5m is not None or write_1h is not None):
        # A partial TTL breakdown proves some writes, but not the ordinary remainder.
        cache_write_tokens = (write_5m or 0) + (write_1h or 0)
    if cache_write_tokens is None and not separate_cache_writes:
        cache_write_tokens = 0
        writes_complete = True
    ordinary = None
    if cache_hit_tokens is not None and cache_write_tokens is not None and writes_complete:
        ordinary = max(0, prompt_tokens - cache_hit_tokens - cache_write_tokens)
    return make_usage_sample(
        usage,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=max(0, prompt_tokens - cache_hit_tokens)
        if cache_hit_tokens is not None
        else 0,
        cache_write_tokens=cache_write_tokens,
        uncached_tokens=ordinary,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
    )


_STREAM_TAIL_CHARS = 80
_STREAM_ERROR_FIELD_CHARS = 40
# Error events that asking again with the same request cannot fix.
_PERMANENT_STREAM_ERRORS = frozenset({"context_length_exceeded", "invalid_request_error"})


def _stream_error_event(error: Any, *, provider: str, operation: str) -> StreamErrorEvent:
    """Classify an SDK ``data: {"error": ...}`` event by its code and type, never its message."""
    body = error.body if isinstance(error.body, dict) else {}
    fields = {key: str(body.get(key) or "")[:_STREAM_ERROR_FIELD_CHARS] for key in ("code", "type")}
    message = body.get("message")
    diagnostic = (
        f"code={fields['code']!r} type={fields['type']!r} "
        f"message_chars={len(message) if isinstance(message, str) else 0}"
    )
    return StreamErrorEvent(
        provider=provider,
        operation=operation,
        retryable=not _PERMANENT_STREAM_ERRORS.intersection(fields.values()),
        diagnostic=diagnostic,
    )


# ponytail: SSE keep-alives reset httpx's read timeout, so a watchdog thread closes the stream
# when no useful chunk arrived for this long (capped by the connection timeout). Hidden-reasoning
# models that think silently longer than this are cut off; raise the cap if one needs it.
_STREAM_IDLE_SECONDS = 300.0
_STREAM_WATCHDOG_POLL_SECONDS = 1.0
_STREAM_PROGRESS_SECONDS = 30.0


class _StreamWatchdog:
    """Close ``stream`` once the attempt outlives ``total`` or goes ``idle`` seconds quiet."""

    def __init__(self, stream: Any, *, total: float, idle: float):
        self.stream = stream
        self.total = total
        self.idle = idle
        self.started = self.last_useful = time.monotonic()
        self.expired: str | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="wenyi-stream-watchdog")
        self._thread.daemon = True

    def __enter__(self) -> _StreamWatchdog:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._done.set()
        self._thread.join()

    def _watch(self) -> None:
        while not self._done.wait(_STREAM_WATCHDOG_POLL_SECONDS):
            now = time.monotonic()
            if now - self.started >= self.total:
                self.expired = f"stream exceeded the {self.total:g}s total request limit"
            elif now - self.last_useful >= self.idle:
                self.expired = f"stream sent no content for {self.idle:g}s"
            else:
                continue
            interrupt_stream(self.stream)
            return


def _read_stream(
    stream: Any,
    *,
    timeout: float | None = None,
    emit: Callable[..., None] | None = None,
) -> tuple[Any, Any, Any, int]:
    """Join one streamed completion into ``(message, finish_reason, usage, chunks)``.

    ``finish_reason`` stays None when the stream ended without one; the caller decides whether
    the joined text is still a complete answer. With ``timeout``, the attempt is closed after
    that many seconds in total, or after an idle stretch without content, finish or usage
    chunks, and raises a retryable :class:`StreamTimeoutError`.
    """
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    usage = None
    chunks = 0
    chars = 0
    watchdog = (
        _StreamWatchdog(stream, total=timeout, idle=min(timeout, _STREAM_IDLE_SECONDS))
        if timeout
        else None
    )
    reported = time.monotonic()
    try:
        with watchdog or nullcontext(), stream:
            for chunk in stream:
                chunks += 1
                useful = False
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                    useful = True
                for choice in getattr(chunk, "choices", None) or []:
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                        useful = True
                    delta = getattr(choice, "delta", None)
                    for parts, field in ((content, "content"), (reasoning, "reasoning_content")):
                        text = getattr(delta, field, None)
                        if isinstance(text, str) and text:
                            parts.append(text)
                            chars += len(text)
                            useful = True
                now = time.monotonic()
                if useful and watchdog is not None:
                    watchdog.last_useful = now
                if emit is not None and now - reported >= _STREAM_PROGRESS_SECONDS:
                    emit("llm_stream_progress", chunks=chunks, chars=chars)
                    reported = now
    except Exception as error:
        if watchdog is not None and watchdog.expired:
            raise StreamTimeoutError(watchdog.expired) from error
        raise
    if watchdog is not None and watchdog.expired:
        raise StreamTimeoutError(watchdog.expired)
    message = SimpleNamespace(
        content="".join(content), reasoning_content="".join(reasoning) if reasoning else None
    )
    return message, finish_reason, usage, chunks


def _is_complete_json(text: str | None) -> bool:
    """Strictly parse a whole JSON object or array; a cut-off document never parses."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        return isinstance(json.loads(text.strip()), (dict, list))
    except json.JSONDecodeError:
        return False


class OpenAIConnectionOptions(ConnectionOptions):
    """Chat Completions transport choices shared by every OpenAI-compatible connection."""

    # Streaming keeps gateways from closing long idle requests; it never changes model output.
    stream: bool = True
    # Strict schemas constrain reply shape where the gateway supports them; callers still validate.
    structured_output: Literal["none", "json_schema"] = "none"


class OpenAICompatibleBaseClient(ProviderAdapter, Generic[OptionsT]):
    """Reuse one SDK connection; route selection and usage belong to the caller."""

    connection_options: type[OpenAIConnectionOptions] = OpenAIConnectionOptions

    @property
    def stream(self) -> bool:
        return self.connection_options.model_validate(self.cfg.model_extra or {}).stream

    @property
    def structured_output(self) -> str:
        options = self.connection_options.model_validate(self.cfg.model_extra or {})
        return options.structured_output

    def _ensure_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                from openai import OpenAI

                self.validate_credentials()
                api_key = self.api_key()
                self._client = OpenAI(
                    api_key=api_key or "no-key",
                    base_url=self.base_url,
                    timeout=self.cfg.timeout,
                    max_retries=0,
                )
        return self._client

    def list_models(self) -> list[str]:
        """Read ``GET /models``; the OpenAI SDK pages through gateways that paginate."""
        return [str(item.id) for item in self._ensure_client().models.list() if item.id]

    @classmethod
    def output_limit(cls, options: BaseModel, hint: int | None, explicit: int | None) -> int | None:
        thinking = bool(getattr(options, "thinking", False))
        if explicit is not None:
            if thinking and explicit < 4096:
                raise ValueError(
                    "Thinking mode requires max_output_tokens >= 4096; disable thinking or increase the explicit limit"
                )
            return explicit
        return max(hint, 4096) if thinking and hint is not None else hint

    def _normalize_usage(self, usage: Any) -> UsageSample | None:
        return normalize_openai_usage(usage)

    def _json_response_fallback(
        self, model_config: ResolvedModel[OptionsT], message: Any
    ) -> str | None:
        return None

    def _unterminated_stream(
        self,
        model_config: ResolvedModel[OptionsT],
        message: Any,
        usage: Any,
        chunks: int,
        *,
        json_mode: bool,
        context: RequestContext,
    ) -> str:
        """Accept a JSON reply whose stream only lost its final stop chunk; reject the rest.

        Some gateways end the stream cleanly without sending ``finish_reason``. A strictly
        complete JSON document cannot be a mid-reply cut, so it is treated as ``stop``.
        """
        content = message.content
        reasoning = message.reasoning_content or ""
        stats = {
            "chunks": chunks,
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "usage_received": usage is not None,
        }
        candidate = (
            content
            if content.strip()
            else self._json_response_fallback(model_config, message)
            if json_mode
            else None
        )
        if json_mode and _is_complete_json(candidate):
            context.emit("stream_missing_finish_reason", **stats)
            return "stop"
        # Each attempt is billed by the provider, so record it like a length-stopped attempt.
        context.record_usage(self._normalize_usage(usage))
        tail = (content or reasoning)[-_STREAM_TAIL_CHARS:]
        if context.redact_diagnostic is not None:
            tail = context.redact_diagnostic(tail)
        diagnostic = (
            f"chunks={chunks} content_chars={len(content)} reasoning_chars={len(reasoning)} "
            f"usage={'yes' if usage is not None else 'no'} tail={tail!r}"
        )
        raise IncompleteStreamError(
            f"OpenAI-compatible stream ended without a finish reason ({diagnostic})",
            diagnostic=diagnostic,
        )

    @abstractmethod
    def _build_request_kwargs(
        self,
        model_config: ResolvedModel[OptionsT],
        messages: Messages,
        *,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _request(
        self,
        messages: Messages,
        model: ResolvedModel[OptionsT],
        *,
        json_mode: bool,
        context: RequestContext,
    ) -> str:
        model_config = model
        kwargs = self._build_request_kwargs(
            model, messages, json_mode=json_mode, max_tokens=context.max_tokens
        )
        if json_mode and context.json_schema and self.structured_output == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "wenyi_output",
                    "strict": True,
                    "schema": context.json_schema,
                },
            }
        completions = self._ensure_client().chat.completions
        if self.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            stream = completions.create(**kwargs)
            track = context.track_stream or nullcontext
            try:
                with track(stream):
                    message, finish_reason, usage, chunks = _read_stream(
                        stream, timeout=self.cfg.timeout, emit=context.emit
                    )
            except Exception as error:
                from openai import APIError

                # HTTP and connection failures use APIError subclasses; the SDK raises the
                # base class only for an error event inside an already started stream.
                if type(error) is not APIError:
                    raise
                raise _stream_error_event(
                    error, provider=self.cfg.kind, operation=context.operation
                ) from error
            if finish_reason is None:
                finish_reason = self._unterminated_stream(
                    model_config, message, usage, chunks, json_mode=json_mode, context=context
                )
        else:
            response = completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            choice = response.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", "")
        context.record_usage(self._normalize_usage(usage))
        raw_content = getattr(message, "content", None)
        content = raw_content if isinstance(raw_content, str) else ""
        if str(finish_reason).lower() == "length":
            # Truncated output is not a complete answer; never treat a mid-sentence cut as success.
            raise TruncatedResponseError(
                "OpenAI-compatible response was truncated at the token limit"
            )
        if not content.strip():
            fallback = self._json_response_fallback(model_config, message) if json_mode else None
            if fallback is None or not fallback.strip():
                raise EmptyResponseError(f"{self.cfg.kind} response content is empty")
            try:
                json.loads(fallback)
            except json.JSONDecodeError as error:
                raise EmptyResponseError(
                    f"{self.cfg.kind} configured JSON fallback response is invalid JSON"
                ) from error
            content = fallback
        return content
