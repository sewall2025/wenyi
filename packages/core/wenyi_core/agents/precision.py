"""Three independent drafts and one source-aware synthesis with per-segment output recovery."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace

from ..errors import WenyiError
from ..glossary.store import GlossaryStore
from ..ingest.tokens import count_tokens
from ..llm.base import Messages
from ..llm.json_parser import JsonParseError, parse_json_loose
from ..llm.retrying import ContextLengthExceeded, TruncatedResponseError
from ..models.glossary import GlossaryTerm
from . import prompts
from .base import Agent
from .single_segment import (
    SegmentResult,
    ServiceHealth,
    failure_reason,
    is_fatal_provider_error,
    is_outage,
    unwrap_reply,
)
from .translator import Translator, _segment_levels

# ``validate`` returns the stage's full-length targets with ``None`` for unresolved segments.
PrecisionCallRecorder = Callable[
    [str, Messages, Callable[[], str], Callable[[str], list[str | None]]],
    tuple[list[str | None], str],
]


class PrecisionError(WenyiError, ValueError):
    """A precision response cannot safely be used for document backfill."""


class _SegmentUnresolved(PrecisionError):
    """The plain-text chain left one segment unresolved; the reason names why."""

    def __init__(self, reason: str):
        super().__init__(f"Plain-text segment unresolved: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class PrecisionInputs:
    sources: tuple[str, ...]
    terms: tuple[GlossaryTerm, ...]
    context: str
    style: str
    book_synopsis: str
    chapter_digest: str
    annotation_contexts: list[list[dict[str, str]]]
    next_source: str
    allow_empty_translations: bool = False


class PrecisionAgent(Agent):
    """Keep a stable source prefix; synthesis receives drafts as data, not prior chat turns."""

    def __init__(self, client, config, *, recorder: PrecisionCallRecorder | None = None):
        super().__init__(client, config)
        self._recorder = recorder
        self.last_call_ref: str | None = None
        # Segment index -> reason for ids that the last stage call left unresolved.
        self.failures: dict[int, str] = {}

    def _packet(self, inputs: PrecisionInputs) -> str:
        annotations = Translator._validate_annotation_contexts(
            list(inputs.sources), inputs.annotation_contexts
        )
        return self.render(
            "precision_packet",
            packet=json.dumps(
                {
                    "style": inputs.style,
                    "book_synopsis": inputs.book_synopsis,
                    "glossary": prompts.render_glossary(
                        list(inputs.terms),
                        source_lang=self.src,
                        note_source="\n".join(inputs.sources)
                        if self.config.pipeline.terminology_context
                        else None,
                    ),
                    "chapter_digest": inputs.chapter_digest,
                    "context": inputs.context,
                    "sources": inputs.sources,
                    "annotations": json.loads(prompts.render_annotation_contexts(annotations)),
                    "next_source": inputs.next_source,
                    "allow_empty_translations": inputs.allow_empty_translations,
                },
                ensure_ascii=False,
            ),
        )

    def _call(
        self,
        inputs: PrecisionInputs,
        task: str,
        operation: str,
        drafts: list[list[str | None]] | None = None,
        *,
        resume: Sequence[str | None] | None = None,
        requested: Collection[int] | None = None,
    ) -> list[str | None]:
        """Request numbered segments, halving failed requests down to single segments.

        Returns full-length targets; ``None`` marks segments that stayed unresolved, with the
        reason in ``self.failures``. ``resume`` restores segments accepted by an earlier run.
        """
        output: list[str | None] = [
            (resume[index] if resume is not None else None)
            if Translator._needs_translation(source)
            else source
            for index, source in enumerate(inputs.sources)
        ]
        self.failures = {}
        prefix = self._prefix(inputs)
        # Protected positions stay in the first request so reply ids match packet positions.
        ids = [
            index
            for index, target in enumerate(output)
            if (target is None or not Translator._needs_translation(inputs.sources[index]))
            and (requested is None or index in requested)
        ]
        if all(output[index] is not None for index in ids):
            return output
        if not self.client.batch_output(operation):
            # Models without batch JSON output get one plain-text request per segment.
            self._solve_plain(inputs, task, operation, drafts, ids, output)
            return output
        # Halves run sequentially, so accepted segments merge in source order.
        self._solve(inputs, prefix, task, operation, drafts, ids, output, quote_sources=False)
        return output

    def _solve(
        self,
        inputs: PrecisionInputs,
        prefix: Messages,
        task: str,
        operation: str,
        drafts: list[list[str | None]] | None,
        ids: list[int],
        output: list[str | None],
        *,
        quote_sources: bool,
    ) -> None:
        """Whole request first; missing ids are re-requested, failed requests split in half."""
        if not ids:
            return
        if len(ids) == 1 and output[ids[0]] is None:
            reason = self._solve_single(inputs, prefix, task, operation, drafts, ids[0], output)
            if reason is not None:
                self.failures[ids[0]] = reason
            return
        messages = self._messages(inputs, prefix, task, drafts, ids, quote_sources)
        reason = (
            "context_window"
            if self._exceeds_window(inputs, operation, messages, ids)
            else self._attempt(
                inputs, prefix, task, operation, drafts, ids, output, 0, quote_sources
            )
        )
        remaining = [index for index in ids if output[index] is None]
        if reason is None:  # Progress: accepted ids merged, re-request only the missing ones.
            self._solve(
                inputs, prefix, task, operation, drafts, remaining, output, quote_sources=True
            )
            return
        middle = len(remaining) // 2
        for half in (remaining[:middle], remaining[middle:]):
            self._solve(inputs, prefix, task, operation, drafts, half, output, quote_sources=True)

    def _solve_plain(
        self,
        inputs: PrecisionInputs,
        task: str,
        operation: str,
        drafts: list[list[str | None]] | None,
        ids: list[int],
        output: list[str | None],
    ) -> None:
        """Resolve ids one by one in source order; each accepted segment is its own receipt.

        Receipts hold the stage's merged targets, so ``progress`` resumes after the last
        accepted segment. Unresolved segments keep ``None`` with the reason in ``failures``.
        """
        health = ServiceHealth()
        for index in ids:
            if output[index] is not None:
                continue
            if task == "translate":
                messages = self._plain_translation_messages(inputs, index)
            else:
                assert drafts is not None
                messages = self._plain_synthesis_messages(
                    inputs, drafts, index, self._plain_levels(inputs, index)[0]
                )
            outcome: dict[str, SegmentResult] = {}

            def invoke(index: int = index) -> str:
                result = (
                    self.translate_standard(inputs, index)
                    if task == "translate"
                    else self._synthesize_segment(inputs, drafts or [], index)
                )
                outcome["result"] = result
                if result.target is None:
                    raise _SegmentUnresolved(result.reason or "unresolved")
                return result.target

            def validate(raw: str, index: int = index) -> list[str | None]:
                merged = list(output)
                merged[index] = raw
                return merged

            try:
                if self._recorder:
                    merged, self.last_call_ref = self._recorder(
                        operation, messages, invoke, validate
                    )
                else:
                    merged = validate(invoke())
                output[:] = merged
            except _SegmentUnresolved as error:
                self.failures[index] = error.reason
            health.record(outcome.get("result") or SegmentResult(output[index]))

    def _plain_levels(self, inputs: PrecisionInputs, index: int):
        return _segment_levels(
            inputs.sources[index],
            list(inputs.terms),
            inputs.context,
            inputs.chapter_digest,
            self._next_source(inputs, index),
        )

    @staticmethod
    def _next_source(inputs: PrecisionInputs, index: int) -> str:
        return inputs.sources[index + 1] if index + 1 < len(inputs.sources) else inputs.next_source

    def _plain_translation_messages(self, inputs: PrecisionInputs, index: int) -> Messages:
        """The first plain-text request ``translate_standard`` sends, for the call receipt."""
        annotations = Translator._validate_annotation_contexts(
            list(inputs.sources), inputs.annotation_contexts
        )
        return Translator(self.client, self.config)._plain_messages(
            inputs.sources[index],
            list(inputs.terms),
            inputs.style,
            inputs.context,
            inputs.book_synopsis,
            inputs.chapter_digest,
            annotations[index],
            self._next_source(inputs, index),
        )

    def _plain_synthesis_messages(
        self,
        inputs: PrecisionInputs,
        drafts: list[list[str | None]],
        index: int,
        level,
    ) -> Messages:
        """One plain-text synthesis request: the source paragraph plus its distinct drafts."""
        source = inputs.sources[index]
        annotations = Translator._validate_annotation_contexts(
            list(inputs.sources), inputs.annotation_contexts
        )
        texts = list(dict.fromkeys(draft[index] for draft in drafts if draft[index] is not None))
        system = self.render("segment_translator_system", src=self.src, tgt=self.tgt)
        user = self.render(
            "segment_synthesizer_user",
            style=inputs.style or "(none)",
            book_synopsis=inputs.book_synopsis or "(none)",
            glossary=prompts.render_glossary(
                level.terms,
                source_lang=self.src,
                note_source=source if self.config.pipeline.terminology_context else None,
            ),
            annotation_contexts=prompts.render_annotation_contexts([annotations[index]]),
            chapter_digest=level.chapter_digest or "(none)",
            context=level.context or "(none)",
            source=source,
            next_source=prompts.render_source_reference(level.next_source),
            drafts=json.dumps(texts, ensure_ascii=False, indent=1),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _synthesize_segment(
        self, inputs: PrecisionInputs, drafts: list[list[str | None]], index: int
    ) -> SegmentResult:
        """Synthesize one segment as plain text, dropping optional context on overflow.

        ponytail: no sentence split and no drafts_2 step here; an unresolved synthesis adopts
        the first valid draft in ``degrade``, which is the same quality floor.
        """
        source = inputs.sources[index]
        result = SegmentResult(None, tried=["plain_text"])
        levels = self._plain_levels(inputs, index)
        trigger = ""
        for level in levels:
            if trigger:
                self._log_trim("polish.body", level.step, trigger, index)
            messages = self._plain_synthesis_messages(inputs, drafts, index, level)
            # The last level lets the provider, not the local estimate, decide.
            if level is not levels[-1] and not self.client.fits_context_window(
                "polish.body", messages, source_tokens=count_tokens(source)
            ):
                result.reason, result.provider_failed = "context_length", False
                trigger = "context_window"
                continue
            try:
                reply = self.client.complete(messages, operation="polish.body")
            except Exception as error:
                reason = failure_reason(error)
                if reason is None or is_fatal_provider_error(error):
                    raise
                result.reason, result.provider_failed = reason, is_outage(reason)
                if reason != "context_length":
                    return result
                trigger = "context_length"
                continue
            reply = unwrap_reply(reply, source)
            if reply or inputs.allow_empty_translations:
                return SegmentResult(reply, tried=result.tried)
            result.reason, result.provider_failed = "empty_reply", False
            return result
        return result

    def _messages(
        self,
        inputs: PrecisionInputs,
        prefix: Messages,
        task: str,
        drafts: list[list[str | None]] | None,
        ids: list[int],
        quote_sources: bool,
    ) -> Messages:
        """With ``quote_sources`` each requested id carries its source text: models miscount
        packet positions in repair requests, while the frozen packet prefix stays cacheable.
        """
        segments: list[object] = list(ids)
        if quote_sources:
            segments = [{"id": index, "source": inputs.sources[index]} for index in ids]
        payload: dict[str, object] = {"segments": segments}
        if drafts is not None:
            # Identical text is not additional evidence; drafts are keyed by segment id and a
            # draft that failed a segment simply omits that id.
            requested = dict.fromkeys(
                tuple((str(index), draft[index]) for index in ids if draft[index] is not None)
                for draft in drafts
            )
            payload["drafts"] = [dict(items) for items in requested if items]
        return prefix + [
            {
                "role": "user",
                "content": self.render(
                    f"precision_{task}", task=json.dumps(payload, ensure_ascii=False)
                ),
            }
        ]

    def _prefix(self, inputs: PrecisionInputs) -> Messages:
        return [
            {"role": "system", "content": self.render("precision_generation_system")},
            {"role": "user", "content": self._packet(inputs)},
        ]

    def _exceeds_window(
        self, inputs: PrecisionInputs, operation: str, messages: Messages, ids: list[int]
    ) -> bool:
        """Estimate prompt plus reserved output locally before sending a request."""
        source_tokens = sum(count_tokens(inputs.sources[index]) for index in ids)
        return not self.client.fits_context_window(operation, messages, source_tokens=source_tokens)

    def _solve_single(
        self,
        inputs: PrecisionInputs,
        prefix: Messages,
        task: str,
        operation: str,
        drafts: list[list[str | None]] | None,
        index: int,
        output: list[str | None],
    ) -> str | None:
        """Shrink optional context step by step while one segment exceeds the model context.

        Steps: drop ``next_source``, then prior context and chapter digest, then keep only the
        glossary terms this segment cites; synthesis then keeps two drafts. With one draft left
        the caller adopts that draft, and a failed draft falls back to standard translation.
        """
        retries = self.config.pipeline.align_retry_limit
        trigger = ""
        for step, level_prefix, level_drafts in self._shrink_steps(inputs, prefix, drafts, index):
            if trigger:
                self._log_trim(operation, step, trigger, index)
            messages = self._messages(inputs, level_prefix, task, level_drafts, [index], True)
            if self._exceeds_window(inputs, operation, messages, [index]):
                trigger = "context_window"  # Local estimate: shrink before sending.
                continue
            reason = self._attempt(
                inputs, level_prefix, task, operation, level_drafts, [index], output, retries, True
            )
            if reason != "context_length":
                return reason
            trigger = "context_length"  # The provider rejected the prompt size.
        if drafts is not None:
            # Synthesis over one draft would only restate it; the caller adopts that draft.
            self._log_trim(operation, "drafts_1", trigger, index)
        return "context_length"

    def _shrink_steps(
        self,
        inputs: PrecisionInputs,
        prefix: Messages,
        drafts: list[list[str | None]] | None,
        index: int,
    ) -> list[tuple[str, Messages, list[list[str | None]] | None]]:
        """Full request first, then each distinct smaller variant; unchanged steps are skipped."""
        steps: list[tuple[str, Messages, list[list[str | None]] | None]] = [
            ("full", prefix, drafts)
        ]
        current = inputs
        for step, smaller in (
            ("next_source", replace(inputs, next_source="")),
            ("context", replace(inputs, next_source="", context="", chapter_digest="")),
            (
                "glossary",
                replace(
                    inputs,
                    next_source="",
                    context="",
                    chapter_digest="",
                    terms=tuple(GlossaryStore.terms_in(list(inputs.terms), inputs.sources[index])),
                ),
            ),
        ):
            if smaller != current:
                steps.append((step, self._prefix(smaller), drafts))
                current = smaller
        if drafts is not None:
            valid = [draft for draft in drafts if draft[index] is not None]
            if len(valid) > 2:
                steps.append(("drafts_2", steps[-1][1], valid[:2]))
        return steps

    def _log_trim(self, operation: str, step: str, trigger: str, index: int) -> None:
        """Record one context reduction; ``trigger`` is the local estimate or the provider."""
        self.client.emit_event(
            "context_trimmed", operation=operation, step=step, trigger=trigger, position=index
        )

    def _attempt(
        self,
        inputs: PrecisionInputs,
        prefix: Messages,
        task: str,
        operation: str,
        drafts: list[list[str | None]] | None,
        ids: list[int],
        output: list[str | None],
        retries_remaining: int,
        quote_sources: bool,
    ) -> str | None:
        """Send one request for ``ids`` with bounded feedback retries; merge accepted segments.

        Returns ``None`` once a reply was accepted, otherwise the failure reason.
        """
        messages = self._messages(inputs, prefix, task, drafts, ids, quote_sources)
        schema = _translations_schema(ids)
        feedback: list[dict[str, str]] = []
        last_raw: dict[str, str] = {}
        previous_raw: str | None = None
        while True:
            attempt = messages + feedback
            last_raw.pop("raw", None)

            def invoke(attempt: Messages = attempt) -> str:
                # Match standard translation: leave output limits to the model/provider configuration.
                text = self.client.complete(
                    attempt, operation=operation, json_mode=True, json_schema=schema
                )
                last_raw["raw"] = text
                return text

            def validate(raw: str) -> list[str | None]:
                data, duplicate_keys = _parse_reply(raw)
                found = accepted_segments(
                    inputs.sources,
                    ids,
                    data,
                    duplicate_keys=duplicate_keys,
                    allow_empty=inputs.allow_empty_translations,
                    copy_is_untranslated=_language(self.src) != _language(self.tgt),
                )
                accepted = {
                    index: target for index, target in found.items() if output[index] is None
                }
                if not accepted:
                    raise PrecisionError(
                        f"Precision output contained no valid translation for segments {ids}"
                    )
                merged = list(output)
                for index, target in accepted.items():
                    merged[index] = target
                return merged

            try:
                if self._recorder:
                    merged, self.last_call_ref = self._recorder(
                        operation, attempt, invoke, validate
                    )
                else:
                    merged = validate(invoke())
                output[:] = merged
                return None
            except ContextLengthExceeded:
                # Only a smaller request helps; the router already learned a reported window.
                return "context_length"
            except TruncatedResponseError as error:
                # Asking again cannot raise an output cap: the caller splits the request instead.
                if error.usage_total:
                    self.client.learn_context_window(operation, error.usage_total)
                return "truncated"
            except (PrecisionError, JsonParseError) as error:
                # Output recovery is separate from provider transport retries and archival failures.
                raw = last_raw.get("raw")
                if raw is not None and raw == previous_raw:
                    # The same reply again: further feedback will not change the model's answer.
                    return "repeated_reply"
                if retries_remaining <= 0:
                    return "invalid_output"
                retries_remaining -= 1
                if raw is None:
                    # Without a received reply the feedback would be empty.
                    continue
                previous_raw = raw
                # Quote the rejected reply and restate the exact structural contract.
                keys = ", ".join(f'"{index}"' for index in ids)
                feedback.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"The previous reply failed local validation: {error}. "
                                "Return only a JSON object like "
                                '{"translations": {"<id>": "<translation>"}} '
                                f"with exactly these keys: {keys}, each mapped to the translation "
                                "string of that requested source segment"
                                + (
                                    " (quoted with its id in task.segments)"
                                    if quote_sources
                                    else ""
                                )
                                + ". Do not merge, split, omit or add segments, and do not "
                                "translate context or next_source."
                            ),
                        },
                    ]
                )

    @staticmethod
    def _targets(inputs: PrecisionInputs, targets: object) -> list[str | None]:
        """Validate structure and restore protected sources; ``None`` marks unresolved ids."""
        if not isinstance(targets, list):
            raise PrecisionError("Precision output did not include a translations array")
        if len(targets) != len(inputs.sources):
            raise PrecisionError(
                "Precision output count does not match the document segments: "
                f"expected {len(inputs.sources)}, got {len(targets)}"
            )
        output: list[str | None] = []
        for index, (source, target) in enumerate(zip(inputs.sources, targets)):
            if not Translator._needs_translation(source):
                target = source
            elif target is None:
                pass
            elif not isinstance(target, str):
                raise PrecisionError(f"Precision output {index} must be a string")
            elif not inputs.allow_empty_translations and not target.strip():
                raise PrecisionError(f"Precision output {index} must not be blank")
            output.append(target)
        return output

    def translate(
        self, inputs: PrecisionInputs, *, resume: Sequence[str | None] | None = None
    ) -> list[str | None]:
        return self._call(inputs, "translate", "translation.body", resume=resume)

    def synthesize(
        self,
        inputs: PrecisionInputs,
        drafts: list[list[str | None]],
        *,
        resume: Sequence[str | None] | None = None,
    ) -> list[str | None]:
        """Synthesize ids with at least two valid drafts; other ids stay ``None``."""
        if len(drafts) != 3:
            raise PrecisionError("Precision synthesis requires three initial drafts")
        # Keep all draws in checkpoints; ``_messages`` collapses identical text per request.
        drafts = [self._targets(inputs, draft) for draft in drafts]
        requested = {
            index
            for index in range(len(inputs.sources))
            if sum(draft[index] is not None for draft in drafts) >= 2
        }
        return self._call(
            inputs, "synthesize", "polish.body", drafts, resume=resume, requested=requested
        )

    def translate_standard(self, inputs: PrecisionInputs, index: int) -> SegmentResult:
        """Translate one segment with the plain-text single-segment chain."""
        annotations = Translator._validate_annotation_contexts(
            list(inputs.sources), inputs.annotation_contexts
        )
        # ponytail: the standard fallback sees the batch's frozen context, not sibling targets.
        return Translator(self.client, self.config).translate_segment(
            inputs.sources[index],
            glossary_terms=list(inputs.terms),
            style=inputs.style,
            context=inputs.context,
            book_synopsis=inputs.book_synopsis,
            chapter_digest=inputs.chapter_digest,
            annotation_context=annotations[index],
            next_source=inputs.sources[index + 1]
            if index + 1 < len(inputs.sources)
            else inputs.next_source,
            allow_empty_translations=inputs.allow_empty_translations,
        )


def _comparable(text: str) -> str:
    """NFKC, casefold and keep letters/digits only, so punctuation and spacing never decide."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _language(code: str) -> str:
    return re.split(r"[-_]", code.lower(), maxsplit=1)[0]


def _parse_reply(raw: str) -> tuple[object, set[str]]:
    """Parse a reply and collect object keys that occur more than once in one object."""
    duplicates: set[str] = set()

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        seen: set[str] = set()
        for key, _ in items:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        return dict(items)

    try:
        return json.loads(raw.strip(), object_pairs_hook=pairs), duplicates
    except (ValueError, TypeError):
        # ponytail: json-repair returns plain dicts, so repaired replies (fences, truncation)
        # cannot report duplicate keys; their later value wins as before.
        return parse_json_loose(raw), set()


def accepted_segments(
    sources: Sequence[str],
    ids: Sequence[int],
    data: object,
    *,
    duplicate_keys: Collection[str] = (),
    allow_empty: bool = False,
    copy_is_untranslated: bool = True,
) -> dict[int, str]:
    """Return the requested ids whose reply entry can be trusted; every other id stays pending.

    Entries map id strings to translation strings. A legacy bare array is keyed by request
    position and trusted only when it is well formed: neither longer than the request nor
    missing a translatable id. Unrequested ids are always dropped. Rejected ids:

    - text that is not a string (including old ``{"src", "text"}`` objects), or blank unless
      ``allow_empty``;
    - keys that occur twice in the reply;
    - when ids are omitted, the highest returned id that is directly followed (in request
      order of translatable ids) by an omitted id: a reply that stops early often truncates
      its last entry, so it is re-requested together with the omitted ids;
    - several ids whose comparable text is identical although their sources differ (a
      repetition loop);
    - with ``copy_is_untranslated`` and more than one translatable id requested, text whose
      comparable form equals its source. A single-id request accepts such copies, so names,
      codes or URLs cannot pause a batch.

    Ids whose sources need no translation never affect gaps, repeats or copies.
    """
    items = data.get("translations") if isinstance(data, dict) else data
    translatable = [index for index in ids if Translator._needs_translation(sources[index])]
    if isinstance(items, list):
        if len(items) > len(ids):
            return {}
        items = dict(zip(map(str, ids), items))
        if any(str(index) not in items for index in translatable):
            return {}
    elif not isinstance(items, dict):
        return {}
    candidates: dict[int, str] = {}
    for index in ids:
        text = items.get(str(index))
        if isinstance(text, str) and (allow_empty or text.strip()):
            candidates[index] = text

    suspicious = {index for index in ids if str(index) in duplicate_keys}
    before_gap = [
        previous
        for previous, following in zip(translatable, translatable[1:])
        if str(previous) in items and str(following) not in items
    ]
    if before_gap:
        suspicious.add(before_gap[-1])

    repeats: defaultdict[str, list[int]] = defaultdict(list)
    for index, text in candidates.items():
        comparable = _comparable(text)
        if not comparable or index not in translatable:
            continue
        repeats[comparable].append(index)
        if (
            copy_is_untranslated
            and len(translatable) > 1
            and comparable == _comparable(sources[index])
        ):
            suspicious.add(index)
    for group in repeats.values():
        if len({_comparable(sources[index]) for index in group}) > 1:
            suspicious.update(group)

    return {index: text for index, text in candidates.items() if index not in suspicious}


def _translations_schema(ids: list[int]) -> dict[str, object]:
    """Strict schema for providers that enforce structured output; local validation still runs."""
    keys = [str(index) for index in ids]
    entry = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "object",
                "properties": {key: entry for key in keys},
                "required": keys,
                "additionalProperties": False,
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }
