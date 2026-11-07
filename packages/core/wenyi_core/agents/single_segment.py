"""Pure helpers for the single-segment plain-text fallback chain.

The chain translates one paragraph as plain text, then sentence by sentence, and finally
leaves it untranslated (``refused``) when every model refuses. These helpers keep the
text handling and the service-health rule deterministic and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..errors import WenyiError
from ..llm.errors import describe_provider_failure
from ..llm.retrying import ProviderRequestError

# Segments in a row that failed with provider errors on every route before the run stops.
SERVICE_UNAVAILABLE_AFTER = 3

# Failures the client cannot fix by moving on to the next segment.
_FATAL_CATEGORIES = frozenset(
    {
        "authentication_failed",
        "insufficient_balance",
        "permission_denied",
        "model_or_endpoint_not_found",
    }
)
_CLOSERS = "\"'”’」』）)]】》"
_SENTENCE = re.compile(
    r".*?(?:[。．！？!?]+|…+|\.{3,}|\.(?=\s|$|[" + re.escape(_CLOSERS) + r"]))"
    r"[" + re.escape(_CLOSERS) + r"]*\s*|.+",
    re.DOTALL,
)
_FENCE = re.compile(r"\A```[^\n]*\n(.*?)\n?```\Z", re.DOTALL)
_QUOTE_PAIRS = ('""', "''", "“”", "‘’", "「」", "『』")
_UNSPACED_LANGUAGES = frozenset({"zh", "ja"})


class TranslationServiceUnavailable(WenyiError, RuntimeError):
    """Consecutive segments failed with provider errors on every configured route."""

    code = "translation_service_unavailable"


@dataclass
class SegmentResult:
    """Outcome of the plain-text chain for one segment; ``target is None`` means refused."""

    target: str | None
    tried: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    reason: str = ""
    provider_failed: bool = False


def unwrap_reply(text: str, source: str = "") -> str:
    """Strip whitespace, one code fence and one pair of quotes around the whole reply.

    Quotes stay when the source itself opens with a quote, as in a line of dialogue.
    """
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if source.lstrip()[:1] in {pair[0] for pair in _QUOTE_PAIRS}:
        return text
    for pair in _QUOTE_PAIRS:
        inner = text[1:-1]
        if len(text) >= 2 and text[0] == pair[0] and text[-1] == pair[1]:
            if pair[0] not in inner and pair[1] not in inner:
                return inner.strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split after sentence terminators; ``"".join(result) == text`` always holds."""
    return _SENTENCE.findall(text)


def join_sentences(parts: list[str], target_lang: str) -> str:
    """Join translated sentences without spaces for Chinese/Japanese, with spaces otherwise."""
    pieces = [part.strip() for part in parts if part.strip()]
    base = re.split(r"[-_]", target_lang.lower(), maxsplit=1)[0]
    return ("" if base in _UNSPACED_LANGUAGES else " ").join(pieces)


def is_fatal_provider_error(error: BaseException) -> bool:
    """Authentication, balance, permission and missing-model errors stop the run."""
    if describe_provider_failure(error).category in _FATAL_CATEGORIES:
        return True
    message = str(error).lower()
    return "insufficient balance" in message or "insufficient_quota" in message


def failure_reason(error: BaseException) -> str | None:
    """Classify a provider failure for refusal markers; ``None`` for non-provider errors.

    Unclassified exceptions (bugs, storage or interruption signals) are not refusals and
    must propagate so the run pauses instead of silently skipping segments.
    """
    category = describe_provider_failure(error).category
    if category != "unknown_error":
        return category
    if isinstance(error, ProviderRequestError):
        return error.reason or "provider_error"
    return None


def is_outage(reason: str) -> bool:
    """Count provider failures except request rejections, truncation and context overflow."""
    return reason not in {"invalid_request", "truncated_response", "context_length"}


class ServiceHealth:
    """Tell one refused segment from a provider outage.

    ponytail: counts consecutive provider-failed segments in one run; an outage that only
    hits every other segment, or a run of genuinely refused segments, is misjudged. A
    time window or per-route health probe would be the upgrade.
    """

    def __init__(self, limit: int = SERVICE_UNAVAILABLE_AFTER) -> None:
        self.limit = limit
        self.failures = 0

    def ok(self) -> None:
        """Reset the streak after any accepted model reply."""
        self.failures = 0

    def record(self, result: SegmentResult) -> None:
        if result.target is not None or not result.provider_failed:
            self.failures = 0
            return
        self.failures += 1
        if self.failures >= self.limit:
            raise TranslationServiceUnavailable(
                f"The translation service failed on {self.failures} segments in a row "
                f"({result.reason}); stopping so the run can resume later"
            )
