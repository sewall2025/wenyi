"""Stable abstraction for LLM providers."""

from __future__ import annotations

import logging
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..ingest.tokens import count_tokens
from .json_parser import parse_json_loose
from .usage import UsageTracker

Messages = list[dict[str, str]]
EventSink = Callable[..., None]
_LOGGER = logging.getLogger(__name__)

# ponytail: reserve when the route has no max_output_tokens; CJK output can exceed three
# cl100k tokens per source token and hidden reasoning tokens are not covered. Configure
# max_output_tokens for an exact reserve.
_OUTPUT_TOKENS_PER_SOURCE_TOKEN = 3
# Role and separator tokens added by chat templates per message (approximate).
_MESSAGE_OVERHEAD_TOKENS = 4
# ponytail: fixed 5% headroom over the calibrated estimate for tokenizer drift between
# requests; a provider-side token-count endpoint would replace it.
_WINDOW_SAFETY_MARGIN = 0.05
# ponytail: the reported/estimated ratio only grows and is capped, so one odd usage report
# (cached or multimodal accounting) cannot shrink every later request below a few segments.
_MAX_PROMPT_RATIO = 2.0
# Tiny prompts are dominated by chat-template overhead and would skew the ratio.
_MIN_CALIBRATION_TOKENS = 256


def estimate_prompt_tokens(messages: Messages) -> int:
    """Return the local cl100k estimate of a chat prompt."""
    return sum(count_tokens(message["content"]) + _MESSAGE_OVERHEAD_TOKENS for message in messages)


class LLMClient(ABC):
    """Interface implemented by every provider."""

    def __init__(self) -> None:
        """Initialize independent usage accounting and an optional event sink for the provider."""
        self.usage = UsageTracker()
        self._event_sink: EventSink | None = None
        self._event_sink_lock = threading.Lock()
        self._event_listeners: dict[object, EventSink] = {}
        self._event_observer: ContextVar[EventSink | None] = ContextVar(
            "llm_event_observer", default=None
        )
        # ponytail: learned windows are per operation (one primary route per client) and live
        # only for this process; configure the model's context_window to persist them.
        self._learned_windows: dict[str, int] = {}
        # Reported prompt tokens per local estimate token, per operation, same lifetime.
        self._prompt_ratios: dict[str, float] = {}
        self._learned_windows_lock = threading.Lock()

    def set_event_sink(self, sink: EventSink | None) -> None:
        """Bind the run event sink so the pipeline can append retry events to the book log."""
        with self._event_sink_lock:
            self._event_sink = sink

    @contextmanager
    def capture_events(self, observer: EventSink) -> Iterator[None]:
        """Capture events in this context only; nested scopes restore their parent observer."""
        token = self._event_observer.set(observer)
        try:
            yield
        finally:
            self._event_observer.reset(token)

    @contextmanager
    def observe_events(self, observer: EventSink) -> Iterator[None]:
        """Observe this client's threads independently of ledger sinks and scoped capture."""
        identity = object()
        with self._event_sink_lock:
            self._event_listeners[identity] = observer
        try:
            yield
        finally:
            with self._event_sink_lock:
                del self._event_listeners[identity]

    def _emit_event(self, event: str, **data: Any) -> None:
        """Emit provider events safely across threads; logging failures must not hide model
        exceptions.
        """
        with self._event_sink_lock:
            sink = self._event_sink
            if sink is not None:
                try:
                    sink(event, **data)
                except Exception:  # noqa: BLE001 - Logging must not change model-call semantics.
                    self._log_event_sink_error(event)
            listeners = tuple(self._event_listeners.values())
        for listener in listeners:
            try:
                listener(event, **data)
            except Exception:  # noqa: BLE001 - Observability must not change model results.
                self._log_event_sink_error(event)
        observer = self._event_observer.get()
        if observer is not None:
            try:
                observer(event, **data)
            except Exception:  # noqa: BLE001 - Capture must not change model-call semantics.
                self._log_event_sink_error(event)

    def _log_event_sink_error(self, event: str) -> None:
        _LOGGER.exception("Failed to write LLM event: %s", event)

    def usage_summary(self) -> dict[str, Any]:
        """Return cumulative token usage with totals, tiers and cache hit rates."""
        return self.usage.summary()

    def context_window(self, operation: str) -> int | None:
        """Return the prompt plus output token limit known for this operation's route."""
        with self._learned_windows_lock:
            return self._learned_windows.get(operation)

    def learn_context_window(self, operation: str, total_tokens: int) -> None:
        """Remember a reported context limit (or a truncated reply's total) for this run."""
        if total_tokens <= 0:
            return
        with self._learned_windows_lock:
            known = self._learned_windows.get(operation)
            self._learned_windows[operation] = min(known or total_tokens, total_tokens)

    def calibrate_prompt_tokens(self, operation: str, messages: Messages, reported: int) -> None:
        """Raise the operation's estimate ratio when the provider counted more than cl100k."""
        estimated = estimate_prompt_tokens(messages)
        if reported <= 0 or estimated < _MIN_CALIBRATION_TOKENS:
            return
        ratio = min(reported / estimated, _MAX_PROMPT_RATIO)
        with self._learned_windows_lock:
            if ratio > self._prompt_ratios.get(operation, 1.0):
                self._prompt_ratios[operation] = ratio

    def fits_context_window(
        self, operation: str, messages: Messages, *, source_tokens: int
    ) -> bool:
        """Estimate prompt plus reserved output locally; unknown windows always fit."""
        window = self.context_window(operation)
        if window is None:
            return True
        with self._learned_windows_lock:
            ratio = self._prompt_ratios.get(operation, 1.0)
        reserve = (
            self.max_output_tokens(operation) or _OUTPUT_TOKENS_PER_SOURCE_TOKEN * source_tokens
        )
        prompt = estimate_prompt_tokens(messages) * ratio
        return (prompt + reserve) * (1 + _WINDOW_SAFETY_MARGIN) <= window

    def emit_event(self, event: str, **data: Any) -> None:
        """Append a workflow diagnostic to the bound run event sink, when any."""
        self._emit_event(event, **data)

    def max_output_tokens(self, operation: str) -> int | None:
        """Return the configured output limit for this operation's route, when known."""
        return None

    def batch_output(self, operation: str) -> bool:
        """Return False when the route's model takes one plain-text paragraph per request."""
        return True

    def request_snapshot(self, operation: str) -> dict[str, Any] | None:
        """Return authoritative replay metadata when supported, never an inferred route."""
        return None

    def validate_credentials(self, operations: Iterable[str] | None = None) -> None:
        """Validate provider credentials; local and test providers are exempt by default."""

    def probe_profiles(
        self, profiles: Iterable[str] | None = None, *, timeout: float | None = None
    ) -> list:
        """Send live connectivity probes; only routed provider clients support this."""
        raise RuntimeError("This model client does not support connectivity tests")

    def cancel(self) -> None:
        """Stop waiting and future requests when supported by the client."""

    @contextmanager
    def interrupt_scope(self):
        """Cancel queued model work before a thread pool joins after Ctrl+C."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = signal.getsignal(signal.SIGINT)

        def stop(signum, frame):
            self.cancel()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, stop)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)

    @abstractmethod
    def complete(
        self,
        messages: Messages,
        *,
        operation: str,
        json_mode: bool = False,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return model text using the registered operation route.

        ``json_schema`` is an optional hint for providers configured for strict structured
        output; callers must still validate the reply locally.
        """
        raise NotImplementedError

    def complete_json(
        self,
        messages: Messages,
        *,
        operation: str,
        max_tokens: int | None = None,
    ) -> Any:
        """Request and parse JSON output."""
        text = self.complete(messages, operation=operation, json_mode=True, max_tokens=max_tokens)
        return parse_json_loose(text)
