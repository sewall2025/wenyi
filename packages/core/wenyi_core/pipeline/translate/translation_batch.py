"""Explicit batch inputs and translate/polish results without formal state mutation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...agents.polisher import Polisher
from ...agents.single_segment import ServiceHealth
from ...agents.translator import Translator
from ...glossary.store import GlossaryStore
from ...ingest.segmenter import batch_segments
from ...markup.ruby import strip_ruby_markers
from ...models.document import Segment
from ...models.glossary import GlossaryTerm


@dataclass(frozen=True)
class BatchPlan:
    """Capture the source positions and context used for exactly one serial batch."""

    chapter: int
    start_index: int
    segment_indices: tuple[int, ...]
    sources: tuple[str, ...]
    terms: tuple[GlossaryTerm, ...]
    context: str
    style: str
    book_synopsis: str
    chapter_digest: str
    annotation_contexts: list[list[dict[str, str]]]
    next_source: str
    allow_empty_translations: bool = False

    @classmethod
    def capture(
        cls,
        chapter: int,
        start_index: int,
        segments: list[Segment],
        terms: list[GlossaryTerm],
        context: str,
        style: str,
        book_synopsis: str,
        chapter_digest: str,
        annotation_contexts: list[list[dict[str, str]]],
        next_source: str,
        *,
        allow_empty_translations: bool = False,
    ) -> BatchPlan:
        """Detach mutable term and annotation metadata from subsequent chapter updates.

        Only glossary entries cited by this batch's source text (or its annotation
        sources) are kept, in the snapshot's stable insertion order, so prompt size
        tracks the batch instead of the whole book and fingerprints stay reproducible.
        """
        cited = "\n".join(
            [s.source for s in segments]
            + [item.get("source", "") for contexts in annotation_contexts for item in contexts]
        )
        return cls(
            chapter,
            start_index,
            tuple(s.index for s in segments),
            tuple(s.source for s in segments),
            tuple(deepcopy(GlossaryStore.terms_in(terms, cited))),
            context,
            style,
            book_synopsis,
            chapter_digest,
            deepcopy(annotation_contexts),
            next_source,
            allow_empty_translations,
        )


@dataclass(frozen=True)
class BatchResult:
    # ``None`` marks a segment every fallback refused; the next run retries it.
    targets: tuple[str | None, ...]
    before_polish: tuple[str | None, ...]
    precision_key: str | None = None
    # One marker per segment that fell back to a draft or standard mode, or was refused.
    degraded: tuple[dict[str, Any], ...] = ()


class TranslationBatchExecutor:
    def __init__(self, translator: Translator, polisher: Polisher):
        self._translator = translator
        self._polisher = polisher

    def execute(
        self, plan: BatchPlan, *, polish: bool, health: ServiceHealth | None = None
    ) -> BatchResult:
        """Translate and optionally polish; the chapter service alone applies the result."""
        terms = list(plan.terms)
        targets = self._translator.translate_batch(
            list(plan.sources),
            glossary_terms=terms,
            style=plan.style,
            context=plan.context,
            book_synopsis=plan.book_synopsis,
            chapter_digest=plan.chapter_digest,
            annotation_contexts=plan.annotation_contexts,
            next_source=plan.next_source,
            allow_empty_translations=plan.allow_empty_translations,
            health=health,
        )
        refused = [
            {
                "segment": plan.segment_indices[index],
                "position": plan.start_index + index,
                "stage": "batch",
                "fallback": "refused",
                "reason": outcome.reason,
                "tried": ["batch", *outcome.tried],
                "routes": outcome.routes,
            }
            for index, outcome in sorted(self._translator.last_refused.items())
        ]
        targets = [None if target is None else strip_ruby_markers(target) for target in targets]
        before_polish: tuple[str | None, ...] = (None,) * len(targets)
        if polish and not self._translator.client.batch_output("polish.body"):
            # Standard polish needs aligned JSON lists; keep the translation and say why.
            self._translator.client.emit_event(
                "polish_skipped", operation="polish.body", reason="batch_output_disabled"
            )
            polish = False
        if polish:
            before_polish = tuple(targets)
            turn = self._translator.last_batch_turn
            indices = self._translator.last_batch_indices
            polished: list[str] | None = None
            if turn is not None and indices is not None:
                # Continue the translation conversation and restore filtered source positions.
                continued = self._polisher.polish_continue(
                    turn,
                    n=len(indices),
                    next_source=plan.next_source,
                )
                if continued is not None and len(continued) == len(indices):
                    polished = list(targets)
                    for index, text in zip(indices, continued):
                        polished[index] = strip_ruby_markers(text)
            if polished is None and any(target is not None for target in targets):
                # Refused segments stay ``None``; polish only the translated ones.
                kept = [index for index, target in enumerate(targets) if target is not None]
                subset = self._polisher.polish(
                    [targets[index] for index in kept],
                    glossary_terms=terms,
                    style=plan.style,
                    next_source=plan.next_source,
                    sources=[plan.sources[index] for index in kept],
                )
                if len(subset) == len(kept):
                    polished = list(targets)
                    for index, text in zip(kept, subset):
                        polished[index] = text
            if polished is not None and len(polished) == len(targets):
                targets = polished
        return BatchResult(tuple(targets), before_polish, degraded=tuple(refused))


def resume_batches(segments: list[Segment], max_tokens: int) -> list[list[Segment]]:
    """Split token-budget batches again at completed/pending boundaries.
    A changed budget may mix saved translations and unset targets in one batch. Group by
    completion state to translate only missing paragraphs and avoid overwriting confirmed
    content. A saved blank string is complete; only None means pending.
    """
    batches: list[list[Segment]] = []
    for raw_batch in batch_segments(segments, max_tokens):
        current: list[Segment] = []
        current_done: bool | None = None
        for segment in raw_batch:
            done = segment.target is not None
            if current and done != current_done:
                batches.append(current)
                current = []
            current.append(segment)
            current_done = done
        if current:
            batches.append(current)
    return batches
