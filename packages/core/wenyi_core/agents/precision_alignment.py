"""Pure checks that catch precision replies whose entries belong to another segment.

Every function here is deterministic and model-free. Scripts come from the language
profiles (``scripts`` in ``i18n/data/languages``), so no target language is special-cased.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from difflib import SequenceMatcher

# A reply whose first requested entry is this similar to the last context paragraph restated
# that paragraph: re-renderings of one paragraph differ only in punctuation or a word or two
# (ratio ≥ 0.9 in practice), while distinct paragraphs of one chapter stay far below 0.6.
CONTEXT_ECHO_RATIO = 0.8
# Shorter texts (comparable characters) are too generic for the context echo check.
CONTEXT_ECHO_MIN_CHARS = 8
# Drift checks skip sources shorter than this (comparable characters): titles, short dialogue
# and interjections carry too few anchors to judge.
DRIFT_MIN_CHARS = 20
# Targets with fewer letters than this are not judged for a missing target script: "OK",
# initials or a kept short name are legitimate in any target language.
SCRIPT_MIN_LETTERS = 5
# A neighbour must beat the segment's own source by this much anchor-plus-length score.
DRIFT_MARGIN = 0.5
# Consecutive shifted entries in one direction needed before the run is flagged.
DRIFT_MIN_RUN = 2

_QUOTES = frozenset('"“”„«»「」『』‹›')
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+(?![A-Za-z])")
_SENTENCE_START = frozenset(".!?:;") | _QUOTES


def comparable(text: str) -> str:
    """NFKC, casefold and keep letters/digits only, so punctuation and spacing never decide."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def char_script(char: str) -> str | None:
    """Return the Unicode script family of one letter, or ``None`` for non-letters."""
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    if name.startswith(("CJK ", "IDEOGRAPHIC")):
        return "Han"
    for script in ("HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "LATIN", "GREEK", "ARABIC"):
        if script in name.replace("-", " ").split():
            return script.capitalize()
    return name.split(" ", 1)[0].capitalize() or None


def scripts_of(text: str) -> set[str]:
    return {script for char in text if (script := char_script(char)) is not None}


def lacks_target_script(target: str, source: str, target_scripts: Collection[str]) -> bool:
    """True when ``target`` has letters, none of them in a target script, and is not ``source``.

    Undecidable cases return False: no known target scripts, a source without letters, or a
    source already written in every target script (same-script pairs such as en→fr). A target
    with fewer than ``SCRIPT_MIN_LETTERS`` letters (digits, symbols, "OK") is not judged either.
    """
    wanted = set(target_scripts)
    source_scripts = scripts_of(source)
    if not wanted or not source_scripts or wanted <= source_scripts:
        return False
    letters = [char for char in target if char.isalpha()]
    if len(letters) < SCRIPT_MIN_LETTERS:
        return False
    found = scripts_of(target)
    return not found & wanted and comparable(target) != comparable(source)


def copies_neighbor_source(
    target: str, sources: Sequence[str], index: int, next_source: str
) -> bool:
    """True when ``target`` equals the source of the previous or next segment, not its own."""
    text = comparable(target)
    if not text or text == comparable(sources[index]):
        return False
    neighbors = [sources[index - 1]] if index > 0 else []
    neighbors.append(sources[index + 1] if index + 1 < len(sources) else next_source)
    return any(text == comparable(neighbor) for neighbor in neighbors)


def echoes_context(target: str, context: str) -> bool:
    """True when ``target`` restates the last paragraph of the prior-context translation."""
    lines = [line for line in context.splitlines() if line.strip()]
    if not lines:
        return False
    last, text = comparable(lines[-1]), comparable(target)
    if min(len(last), len(text)) < CONTEXT_ECHO_MIN_CHARS:
        return False
    return SequenceMatcher(None, last, text, autojunk=False).ratio() >= CONTEXT_ECHO_RATIO


def _anchors(text: str) -> set[str]:
    """Signals that survive translation: numbers, ?/!, quotes and capitalized Latin names.

    Sentence-initial capitals are not names; names only count when the other side keeps
    Latin letters (see ``_score``), since many targets transliterate them.
    """
    normalized = unicodedata.normalize("NFKC", text)
    anchors = {f"#{number}" for number in _NUMBER.findall(normalized)}
    anchors.update(mark for mark in "?!" if mark in normalized)
    if any(char in _QUOTES for char in normalized):
        anchors.add("quote")
    for match in _WORD.finditer(normalized):
        before = normalized[: match.start()].rstrip()
        if before and before[-1] not in _SENTENCE_START:
            anchors.add(f"@{match.group()}")
    return anchors


def _score(text: str, source: str, median: float) -> tuple[float, float]:
    """Return (anchor Jaccard, anchor plus length closeness) of ``text`` against ``source``."""
    target_anchors, source_anchors = _anchors(text), _anchors(source)
    if "Latin" not in scripts_of(text):
        source_anchors = {anchor for anchor in source_anchors if not anchor.startswith("@")}
    union = target_anchors | source_anchors
    anchor = len(target_anchors & source_anchors) / len(union) if union else 0.0
    closeness = max(0.0, 1.0 - abs(math.log((len(text) / len(source)) / median)))
    return anchor, anchor + closeness


def drifted_segments(
    sources: Sequence[str],
    candidates: Mapping[int, str],
    ids: Sequence[int],
    next_source: str = "",
) -> set[int]:
    """Ids of a shifted stretch: ≥2 consecutive entries matching one neighbour's source better.

    An entry is shifted ``-1``/``+1`` when that neighbour's source (``next_source`` after the
    last segment) has more anchor overlap and scores ``DRIFT_MARGIN`` higher than its own.
    It is aligned when its own source has anchor overlap and no neighbour beats it, and
    unknown otherwise (short sources, no anchors). Length closeness is measured against the
    reply's median length ratio, which absorbs script differences, so length alone never
    decides. A run of ``DRIFT_MIN_RUN`` shifted entries in one direction is flagged together
    with the unknown entries before it and every unknown or shifted entry after it, since a
    shift persists until the reply ends; aligned entries stop the extension.
    """
    ratios = sorted(
        len(text) / len(sources[index])
        for index, text in candidates.items()
        if len(comparable(sources[index])) >= DRIFT_MIN_CHARS
    )
    if not ratios:
        return set()
    median = ratios[len(ratios) // 2]

    def neighbor_source(index: int) -> str | None:
        if 0 <= index < len(sources):
            return sources[index]
        return next_source if index == len(sources) else None

    states: list[tuple[int, int | None]] = []  # (id, ±1 shifted, 0 aligned, None unknown)
    for index in ids:
        text = candidates.get(index)
        state: int | None = None
        if text and len(comparable(sources[index])) >= DRIFT_MIN_CHARS:
            own_anchor, own = _score(text, sources[index], median)
            for step in (-1, 1):
                neighbor = neighbor_source(index + step)
                if neighbor is None or len(comparable(neighbor)) < DRIFT_MIN_CHARS:
                    continue
                anchor, total = _score(text, neighbor, median)
                if anchor > own_anchor and total >= own + DRIFT_MARGIN:
                    state = step
                    break
            if state is None and own_anchor > 0:
                state = 0
        states.append((index, state))

    flagged: set[int] = set()
    position = 0
    while position < len(states):
        index, state = states[position]
        end = position
        while state and end + 1 < len(states) and states[end + 1] == (states[end][0] + 1, state):
            end += 1
        if not state or end - position + 1 < DRIFT_MIN_RUN:
            position += 1
            continue
        first = position
        while first > 0 and states[first - 1][1] is None:
            first -= 1
        last = end
        while last + 1 < len(states) and states[last + 1][1] != 0:
            last += 1
        flagged.update(item for item, _state in states[first : last + 1])
        position = last + 1
    return flagged
