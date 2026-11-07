"""Translation agent using the strong tier.
Guarantee paragraph alignment: N source paragraphs must produce N corresponding
translations. Request an equal-length JSON array, retry count mismatches up to
align_retry_limit, then translate paragraphs individually as plain text, splitting a
paragraph into sentences when the whole paragraph fails. A paragraph that every route refuses
stays ``None`` (refused) instead of stopping the run or being replaced by its source.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..glossary.store import GlossaryStore
from ..ingest.tokens import count_tokens
from ..llm.json_parser import JsonParseError
from ..llm.retrying import ContextLengthExceeded, TruncatedResponseError
from ..models.glossary import GlossaryTerm
from . import prompts
from .base import Agent, Messages
from .single_segment import (
    SegmentResult,
    ServiceHealth,
    failure_reason,
    is_fatal_provider_error,
    is_outage,
    join_sentences,
    split_sentences,
    unwrap_reply,
)


class AlignmentError(Exception):
    pass


class _BatchOverflow(Exception):
    """A multi-paragraph batch does not fit the model context; the caller halves it."""


class Translator(Agent):
    """Body translator.

    After a successful single-shot batch call, ``last_batch_turn`` holds the
    ``system`` / ``user`` / ``assistant`` messages so polishing can append another
    user turn instead of opening a new conversation. Per-paragraph fallback clears
    that transcript.
    """

    def __init__(self, client, config):
        super().__init__(client, config)
        self.last_batch_turn: Messages | None = None
        self.last_batch_indices: list[int] | None = None
        # Paragraph index -> chain outcome for paragraphs refused by the last batch call.
        self.last_refused: dict[int, SegmentResult] = {}

    @staticmethod
    def _needs_translation(source: str) -> bool:
        """Send only nonempty paragraphs containing language characters to the model.
        PDF tables often yield separate hyphens, numbers or placeholders. Models may return
        them empty and trigger alignment errors; preserve those paragraphs unchanged.
        str.isalpha covers Unicode letters including Latin, Chinese, Japanese and Korean.
        """
        stripped = source.strip()
        return bool(stripped) and any(character.isalpha() for character in stripped)

    @staticmethod
    def _validate_annotation_contexts(
        sources: list[str],
        annotation_contexts: list[list[dict[str, str]]] | None,
    ) -> list[list[dict[str, str]]]:
        """Validate paragraph annotation references and retain only stable fields used by the
        prompt.
        """
        if annotation_contexts is None:
            return [[] for _ in sources]
        if not isinstance(annotation_contexts, list) or len(annotation_contexts) != len(sources):
            actual = (
                len(annotation_contexts) if isinstance(annotation_contexts, list) else "not a list"
            )
            raise ValueError(
                f"Annotation context count mismatch: expected {len(sources)} groups, got {actual}"
            )

        normalized: list[list[dict[str, str]]] = []
        for segment_index, items in enumerate(annotation_contexts):
            if not isinstance(items, list):
                raise ValueError(f"Annotation context for paragraph {segment_index} must be a list")
            segment_items: list[dict[str, str]] = []
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Annotation {item_index} in paragraph {segment_index} must be an object"
                    )
                target_key = item.get("target_key")
                source = item.get("source")
                if not isinstance(target_key, str) or not target_key.strip():
                    raise ValueError(
                        f"Annotation {item_index} in paragraph {segment_index} has no valid target_key"
                    )
                if not isinstance(source, str):
                    raise ValueError(
                        f"Annotation {item_index} in paragraph {segment_index} has no string source"
                    )
                segment_items.append({"target_key": target_key, "source": source})
            normalized.append(segment_items)
        return normalized

    def _call_batch(
        self,
        sources: list[str],
        glossary_terms: list[GlossaryTerm],
        style: str,
        context: str,
        book_synopsis: str = "",
        chapter_digest: str = "",
        annotation_contexts: list[list[dict[str, str]]] | None = None,
        next_source: str = "",
        *,
        allow_empty_translations: bool = False,
    ) -> tuple[list[str], Messages]:
        """Translate one batch and validate output types, count and (by default) nonempty content.

        When ``allow_empty_translations`` is true (MinerU PDF path), blank strings are kept as
        formal targets so VLM OCR junk that the model refuses to translate does not abort the run.
        Returns the translations and the three-turn transcript for optional polish continuation.
        """
        n = len(sources)
        system = self.render(
            "translator_system",
            src=self.src,
            tgt=self.tgt,
        )
        user = self.render(
            "translator_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "(none)",
            book_synopsis=book_synopsis or "(none)",
            glossary=prompts.render_glossary(
                glossary_terms,
                source_lang=self.src,
                note_source="\n".join(sources)
                if self.config.pipeline.terminology_context
                else None,
            ),
            annotation_contexts=prompts.render_annotation_contexts(
                annotation_contexts or [[] for _ in sources]
            ),
            chapter_digest=chapter_digest or "(none)",
            context=context or "(none)",
            n=n,
            n_minus_1=n - 1,
            numbered_source=prompts.numbered(sources),
            next_source=prompts.render_source_reference(next_source),
        )
        messages: Messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        source_tokens = sum(count_tokens(source) for source in sources)
        if not self.client.fits_context_window(
            "translation.body", messages, source_tokens=source_tokens
        ):
            raise _BatchOverflow("context_window")
        # Transient provider errors are retried only by the transport. Only JSON protocol errors in
        # successful responses enter alignment recovery, avoiding duplicate retries for 401/403/5xx errors.
        try:
            data, raw = self._complete_json_turn(messages, operation="translation.body")
        except ContextLengthExceeded as error:
            raise _BatchOverflow("context_length") from error
        except TruncatedResponseError as error:
            raise AlignmentError(
                "The translation response reached the output token limit"
            ) from error
        except JsonParseError as error:
            raise AlignmentError(
                "Cannot parse the translation JSON returned by the model"
            ) from error
        items = data.get("translations") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise AlignmentError("The model did not return a translation array")
        if len(items) != n:
            raise AlignmentError(
                f"Translation count mismatch: expected {n} paragraphs, got {len(items)}"
            )
        if any(not isinstance(item, str) for item in items):
            raise AlignmentError("The model returned a non-string translation")
        if not allow_empty_translations and any(not item.strip() for item in items):
            raise AlignmentError("The model returned an empty or non-string translation")
        turn = [
            *messages,
            {"role": "assistant", "content": raw},
        ]
        return items, turn

    def _plain_messages(
        self,
        source: str,
        glossary_terms: list[GlossaryTerm],
        style: str,
        context: str,
        book_synopsis: str,
        chapter_digest: str,
        annotation_context: list[dict[str, str]],
        next_source: str,
    ) -> Messages:
        """Build one plain-text translation request without any JSON protocol."""
        system = self.render("segment_translator_system", src=self.src, tgt=self.tgt)
        user = self.render(
            "segment_translator_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "(none)",
            book_synopsis=book_synopsis or "(none)",
            glossary=prompts.render_glossary(
                glossary_terms,
                source_lang=self.src,
                note_source=source if self.config.pipeline.terminology_context else None,
            ),
            annotation_contexts=prompts.render_annotation_contexts([annotation_context]),
            chapter_digest=chapter_digest or "(none)",
            context=context or "(none)",
            source=source,
            next_source=prompts.render_source_reference(next_source),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _route_names(self) -> list[str]:
        """Name the primary and fallback routes the router may try for body translation."""
        snapshot = self.client.request_snapshot("translation.body")
        if not snapshot:
            return []
        routes = [snapshot.get("primary") or {}, *(snapshot.get("fallbacks") or [])]
        return [str(route.get("profile") or route.get("model") or "") for route in routes]

    def translate_segment(
        self,
        source: str,
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        context: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        annotation_context: list[dict[str, str]] | None = None,
        next_source: str = "",
        allow_empty_translations: bool = False,
    ) -> SegmentResult:
        """Translate one paragraph as plain text, then sentence by sentence; never the source.

        Any nonblank reply is accepted, even when it equals the source. A context overflow
        (local estimate or provider) first drops optional context step by step. Truncation,
        blank replies and other non-fatal provider failures (after the router's own fallback
        routes) move on to sentence splitting; if any sentence still fails the result is refused
        (``target is None``). Authentication, balance and run-stop errors propagate.
        """
        result = SegmentResult(None, tried=["plain_text"], routes=self._route_names())

        overflow = {"trigger": ""}

        def ask(
            text: str,
            following: str,
            level: _SegmentContext,
            *,
            allow_empty: bool,
            preflight: bool = True,
        ) -> str | None:
            messages = self._plain_messages(
                text,
                level.terms,
                style,
                level.context,
                book_synopsis,
                level.chapter_digest,
                annotation_context or [],
                following,
            )
            if preflight and not self.client.fits_context_window(
                "translation.body", messages, source_tokens=count_tokens(text)
            ):
                # Local estimate: shrinking first is cheaper than a provider 400.
                result.reason, result.provider_failed = "context_length", False
                overflow["trigger"] = "context_window"
                return None
            try:
                reply = self.client.complete(messages, operation="translation.body")
            except Exception as error:
                reason = failure_reason(error)
                if reason is None or is_fatal_provider_error(error):
                    raise
                result.reason, result.provider_failed = reason, is_outage(reason)
                overflow["trigger"] = "context_length" if reason == "context_length" else ""
                return None
            reply = unwrap_reply(reply, text)
            if reply or allow_empty:
                return reply
            result.reason, result.provider_failed = "empty_reply", False
            return None

        sentences = split_sentences(source)
        splittable = sum(bool(sentence.strip()) for sentence in sentences) >= 2
        levels = _segment_levels(source, glossary_terms or [], context, chapter_digest, next_source)
        trigger = ""
        for level in levels:
            if trigger:
                self._log_trim(level.step, trigger)
            target = ask(
                source,
                level.next_source,
                level,
                allow_empty=allow_empty_translations,
                # Without a sentence fallback the provider, not the estimate, has the last word.
                preflight=splittable or level is not levels[-1],
            )
            if target is not None:
                return SegmentResult(target, tried=result.tried, routes=result.routes)
            trigger = overflow["trigger"]
            if not trigger:
                break
        if not splittable:
            return result
        if trigger:
            self._log_trim("sentences", trigger)
        result.tried.append("sentences")
        parts: list[str] = []
        for position, sentence in enumerate(sentences):
            if not self._needs_translation(sentence):
                parts.append(sentence)
                continue
            following = "".join(sentences[position + 1 :]).strip() or level.next_source
            # Sentences are the last step: let the provider decide instead of the estimate.
            part = ask(sentence, following, level, allow_empty=False, preflight=False)
            if part is None:
                # ponytail: accepted sentences are not checkpointed; a rerun buys them again.
                return result
            parts.append(part)
        return SegmentResult(
            join_sentences(parts, self.tgt), tried=result.tried, routes=result.routes
        )

    def _log_trim(self, step: str, trigger: str) -> None:
        """Record one context reduction; ``trigger`` is the local estimate or the provider."""
        self.client.emit_event(
            "context_trimmed", operation="translation.body", step=step, trigger=trigger
        )

    def _translate_halves(
        self,
        sources: list[str],
        translated_indices: list[int],
        trigger: str,
        **options,
    ) -> list[str | None]:
        """Translate both halves of an oversized batch; each half may split again."""
        self._log_trim("split", trigger)
        annotation_contexts = options.pop("annotation_contexts")
        next_source = options.pop("next_source")
        targets: list[str | None] = list(sources)
        refused: dict[int, SegmentResult] = {}
        middle = len(translated_indices) // 2
        for half in (translated_indices[:middle], translated_indices[middle:]):
            start, end = half[0], half[-1] + 1
            targets[start:end] = self.translate_batch(
                sources[start:end],
                annotation_contexts=annotation_contexts[start:end],
                next_source=sources[end] if end < len(sources) else next_source,
                **options,
            )
            refused.update({start + index: outcome for index, outcome in self.last_refused.items()})
        # Halves have separate transcripts, so polishing cannot continue one conversation.
        self.last_batch_turn = None
        self.last_batch_indices = None
        self.last_refused = refused
        return targets

    def translate_batch(
        self,
        sources: list[str],
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        context: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        annotation_contexts: list[list[dict[str, str]]] | None = None,
        next_source: str = "",
        allow_empty_translations: bool = False,
        health: ServiceHealth | None = None,
    ) -> list[str | None]:
        """Translate aligned paragraphs with one following source segment as reference only.

        Paragraphs refused by every fallback are ``None`` and listed in ``last_refused``;
        ``health`` stops the run when consecutive paragraphs fail with provider errors.
        """
        self.last_batch_turn = None
        self.last_batch_indices = None
        self.last_refused = {}
        glossary_terms = glossary_terms or []
        n = len(sources)
        annotation_contexts = self._validate_annotation_contexts(sources, annotation_contexts)
        if n == 0:
            return []

        translated_indices = [
            index for index, source in enumerate(sources) if self._needs_translation(source)
        ]
        if not translated_indices:
            return list(sources)
        translated_sources = [sources[index] for index in translated_indices]
        translated_annotation_contexts = [
            annotation_contexts[index] for index in translated_indices
        ]
        # A filtered trailing number or symbol remains the immediate source neighbor.
        following_index = translated_indices[-1] + 1
        batch_next_source = sources[following_index] if following_index < n else next_source

        # A model without batch JSON output goes straight to the per-paragraph chain below.
        attempts = (
            self.config.pipeline.align_retry_limit + 1
            if self.client.batch_output("translation.body")
            else 0
        )
        for _ in range(attempts):
            try:
                translated, turn = self._call_batch(
                    translated_sources,
                    glossary_terms,
                    style,
                    context,
                    book_synopsis,
                    chapter_digest,
                    translated_annotation_contexts,
                    next_source=batch_next_source,
                    allow_empty_translations=allow_empty_translations,
                )
                targets = list(sources)
                for index, target in zip(translated_indices, translated):
                    targets[index] = target
                self.last_batch_turn = turn
                self.last_batch_indices = list(translated_indices)
                if health is not None:
                    health.ok()
                return targets
            except AlignmentError:
                # Recover output protocol/alignment errors; the provider handles transport retries.
                continue
            except _BatchOverflow as overflow:
                if len(translated_indices) == 1:
                    break  # The single-paragraph chain below shrinks its context instead.
                return self._translate_halves(
                    sources,
                    translated_indices,
                    str(overflow),
                    glossary_terms=glossary_terms,
                    style=style,
                    context=context,
                    book_synopsis=book_synopsis,
                    chapter_digest=chapter_digest,
                    annotation_contexts=annotation_contexts,
                    next_source=next_source,
                    allow_empty_translations=allow_empty_translations,
                    health=health,
                )
            except Exception as error:
                # A provider failure after transport retries and route failover may be one
                # refused paragraph; isolate it below instead of retrying the whole batch.
                if failure_reason(error) is None or is_fatal_provider_error(error):
                    raise
                break

        # Fall back to individual plain-text paragraphs. Without allow_empty_translations, empty
        # placeholders are never accepted; MinerU may persist "" when the model returns blank.
        self.last_batch_turn = None
        self.last_batch_indices = None
        health = health or ServiceHealth()
        targets: list[str | None] = list(sources)
        for index, source, annotation_context in zip(
            translated_indices,
            translated_sources,
            translated_annotation_contexts,
        ):
            outcome = self.translate_segment(
                source,
                glossary_terms=glossary_terms,
                style=style,
                context=context,
                book_synopsis=book_synopsis,
                chapter_digest=chapter_digest,
                annotation_context=annotation_context,
                next_source=sources[index + 1] if index + 1 < n else next_source,
                allow_empty_translations=allow_empty_translations,
            )
            health.record(outcome)
            targets[index] = outcome.target
            if outcome.target is None:
                self.last_refused[index] = outcome
        return targets


@dataclass(frozen=True)
class _SegmentContext:
    """Optional context sent with one segment; later levels drop more of it."""

    step: str
    terms: list[GlossaryTerm]
    context: str
    chapter_digest: str
    next_source: str


def _segment_levels(
    source: str,
    terms: list[GlossaryTerm],
    context: str,
    chapter_digest: str,
    next_source: str,
) -> list[_SegmentContext]:
    """Full context, then without next_source, prior context/digest, and unmatched terms.

    Steps that would not remove anything are skipped.
    """
    levels = [_SegmentContext("full", terms, context, chapter_digest, next_source)]
    for level in (
        _SegmentContext("next_source", terms, context, chapter_digest, ""),
        _SegmentContext("context", terms, "", "", ""),
        _SegmentContext("glossary", GlossaryStore.terms_in(terms, source), "", "", ""),
    ):
        previous = levels[-1]
        if (level.terms, level.context, level.chapter_digest, level.next_source) != (
            previous.terms,
            previous.context,
            previous.chapter_digest,
            previous.next_source,
        ):
            levels.append(level)
    return levels
