"""Deterministic admission rules for automatically extracted glossary candidates.

Model extraction proposes entries; these pure checks decide which ones may enter the
glossary and which non-name entries stay inactive candidates until the book's source
repeats them. Manual edits and imports never pass through this module.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from ..models.glossary import (
    TYPE_APPELLATION,
    TYPE_FIXED_EXPR,
    TYPE_HONORIFIC,
    TYPE_PERSON,
    TYPE_SPEECH,
    TYPE_TERM,
    GlossaryTerm,
)
from .store import GlossaryOccurrenceMatcher, match_text

TYPE_PLACE = "place"
TYPE_ORGANIZATION = "organization"
TYPE_TECHNIQUE = "technique"
TYPE_ITEM = "item"
TYPE_SETTING = "setting"

# The enum the extraction prompt offers. Anything else ("statement", "action", "other")
# signals that the model filed prose, not terminology, so it is rejected, not remapped.
EXTRACTION_TYPES = frozenset(
    {
        TYPE_PERSON,
        TYPE_PLACE,
        TYPE_ORGANIZATION,
        TYPE_TERM,
        TYPE_TECHNIQUE,
        TYPE_ITEM,
        TYPE_SETTING,
        TYPE_APPELLATION,
        TYPE_HONORIFIC,
        TYPE_SPEECH,
        TYPE_FIXED_EXPR,
    }
)
# Proper names get the longer word limit and, as single words, skip the whole-paragraph
# rule (a one-word paragraph "Eliza" is still a name).
NAME_TYPES = frozenset({TYPE_PERSON, TYPE_PLACE, TYPE_ORGANIZATION, TYPE_ITEM})
# Entities take effect on first sight; every other type must recur in the book source.
IMMEDIATE_TYPES = frozenset({TYPE_PERSON, TYPE_PLACE, TYPE_ORGANIZATION})

# Word limits for space-delimited scripts. Full names and titles ("The Royal Society of
# London for Improving Natural Knowledge") fit 8 words; memoQ caps candidates at 4 words
# and pyate at 6, so 5 words bounds terms, slogans and catchphrases.
MAX_NAME_WORDS = 8
MAX_PHRASE_WORDS = 5
# CJK has no word spaces: 16 characters covers long transliterated names and four-character
# idiom pairs while excluding sentences (KeywordGacha displays at most 32).
MAX_CJK_CHARS = 16
# Non-name entries need this many source occurrences in the whole book (memoQ uses 3).
MIN_BOOK_OCCURRENCES = 2

STATUS_CANDIDATE = "candidate"
STATUS_OK = "ok"

_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
_TRAILING_CLOSERS = "\"'”’»」』）)]】"
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+")
_ABBREVIATIONS = frozenset(
    "jr. sr. mr. mrs. ms. dr. st. mt. ft. inc. ltd. co. corp. bros. prof. gen. col. capt. "
    "lt. sgt. rev. hon. esq. no. vol.".split()
)


def _is_cjk(char: str) -> bool:
    name = unicodedata.name(char, "")
    return name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL"))


def _is_abbreviation(text: str) -> bool:
    """`Jr.`, `D.C.` and similar end in a period without ending a sentence."""
    last = text.split()[-1].lstrip("(,")
    return last.casefold() in _ABBREVIATIONS or _INITIALISM.fullmatch(last) is not None


def _paragraph_key(text: str) -> str:
    return " ".join(match_text(text).split())


def admit_candidate(term: GlossaryTerm, batch_sources: Iterable[str] = ()) -> str | None:
    """Return a rejection reason for an extracted candidate, or None to admit it.

    ``batch_sources`` are the source paragraphs the candidate was extracted from; a
    multi-word candidate equal to one of them is a copied paragraph, not a term.
    """
    source = term.source.strip()
    if not source:
        return "empty"
    if "\n" in source or "\r" in source:
        return "newline"
    if any(unicodedata.category(char) == "Cc" for char in source):
        return "control_character"
    if term.type not in EXTRACTION_TYPES:
        return "unknown_type"
    if source.rstrip(_TRAILING_CLOSERS).endswith(_SENTENCE_END) and not (
        source.endswith(".") and _is_abbreviation(source)
    ):
        return "sentence_punctuation"
    words = source.split()
    letters = [char for char in source if char.isalpha()]
    if (
        len(words) >= 2
        and letters
        and all("LATIN" in unicodedata.name(char, "") and char.isupper() for char in letters)
    ):
        return "all_caps_heading"
    if any(_is_cjk(char) for char in source):
        if len(source.replace(" ", "")) > MAX_CJK_CHARS:
            return "too_long"
    elif len(words) > (MAX_NAME_WORDS if term.type in NAME_TYPES else MAX_PHRASE_WORDS):
        return "too_long"
    single_word = len(words) == 1
    if not (single_word and term.type in NAME_TYPES):
        key = _paragraph_key(source)
        if any(_paragraph_key(paragraph) == key for paragraph in batch_sources):
            return "whole_paragraph"
    return None


def admission_status(term: GlossaryTerm, matcher: GlossaryOccurrenceMatcher) -> str:
    """Entities take effect at once; other entries stay candidates until the book repeats them."""
    if term.type in IMMEDIATE_TYPES:
        return STATUS_OK
    recurring = matcher.recurring_terms([term], min_occurrences=MIN_BOOK_OCCURRENCES)
    return STATUS_OK if recurring else STATUS_CANDIDATE


def active_terms(terms: Iterable[GlossaryTerm]) -> list[GlossaryTerm]:
    """Drop inactive candidates before a glossary reaches any translation prompt."""
    return [term for term in terms if term.status != STATUS_CANDIDATE]


__all__ = [
    "EXTRACTION_TYPES",
    "IMMEDIATE_TYPES",
    "MAX_CJK_CHARS",
    "MAX_NAME_WORDS",
    "MAX_PHRASE_WORDS",
    "MIN_BOOK_OCCURRENCES",
    "NAME_TYPES",
    "STATUS_CANDIDATE",
    "STATUS_OK",
    "active_terms",
    "admission_status",
    "admit_candidate",
]
