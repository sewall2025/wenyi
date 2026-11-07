"""Translate aligned title batches without accessing book state."""

from ..errors import WenyiError
from ..ingest.tokens import count_tokens
from ..llm.retrying import ContextLengthExceeded
from . import prompts
from .base import Agent


class TitleOutputError(WenyiError, RuntimeError):
    """Expose response counts so the caller can retain its rejection event contract."""

    def __init__(self, expected: int, actual: int | None):
        self.expected = expected
        self.actual = actual
        super().__init__(
            "Chapter/TOC title translation returned an invalid number of items: "
            f"expected {expected}, got {actual if actual is not None else 'non-list'}"
        )


class TitleContextOverflow(WenyiError):
    """The title request cannot fit the model context; ``trigger`` names the detector."""

    def __init__(self, trigger: str):
        self.trigger = trigger
        super().__init__(f"Title translation request exceeds the model context ({trigger})")


class TitleTranslator(Agent):
    """Own title prompts, the existing operation route and response alignment."""

    def translate(self, titles: list[str], glossary_text: str) -> list[str]:
        """Return stripped titles; empty-string fallback belongs to the title plan."""
        system = self.render("title_translator_system", src=self.src, tgt=self.tgt, n=len(titles))
        user = self.render(
            "title_translator_user",
            src=self.src,
            tgt=self.tgt,
            glossary=glossary_text,
            n=len(titles),
            numbered_titles=prompts.numbered(titles),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        source_tokens = sum(count_tokens(title) for title in titles)
        if not self.client.fits_context_window(
            "translation.title", messages, source_tokens=source_tokens
        ):
            raise TitleContextOverflow("context_window")
        try:
            data = self.client.complete_json(messages, operation="translation.title")
        except ContextLengthExceeded as error:
            raise TitleContextOverflow("context_length") from error
        out = data.get("titles") if isinstance(data, dict) else data
        if not isinstance(out, list) or len(out) != len(titles):
            raise TitleOutputError(len(titles), len(out) if isinstance(out, list) else None)
        return [str(title).strip() for title in out]
