"""Explicit routing graphs for transport tests with injected SDK responses."""

from types import SimpleNamespace

from wenyi_core.llm.configuration import LLMConfig
from wenyi_core.llm.registry import provider_spec


def model_config(*, kind="deepseek", profiles=None, **connection):
    factory = getattr(provider_spec(kind)._module(), "preset_models", None)
    defaults = factory() if factory else {}
    profiles = profiles or {}
    models = {}
    for tier in ("strong", "cheap", "fast"):
        default = defaults.get(tier)
        profile = profiles.get(tier, {})
        models[tier] = {
            "provider": "default",
            "model": profile.get("model") or (default.model if default else "test-model"),
            "options": profile.get("options", default.options.model_dump() if default else {}),
        }
    return LLMConfig.model_validate(
        {
            "providers": {"default": {"kind": kind, **connection}},
            "models": models,
            "tiers": {tier: tier for tier in models},
        }
    )


class FakeStream:
    """Context-managed chunk iterator mirroring ``openai.Stream`` for offline tests."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def stream_chunk(*, content=None, reasoning_content=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choices = (
        [] if usage is not None else [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )
    return SimpleNamespace(choices=choices, usage=usage)


def stream_chunks(response):
    """Split a non-streaming completion into equivalent stream chunks; missing stops become stop."""
    choice = response.choices[0]
    content = getattr(choice.message, "content", None) or ""
    middle = len(content) // 2
    return [
        stream_chunk(
            content=content[:middle],
            reasoning_content=getattr(choice.message, "reasoning_content", None),
        ),
        stream_chunk(content=content[middle:]),
        stream_chunk(finish_reason=getattr(choice, "finish_reason", None) or "stop"),
        SimpleNamespace(choices=[], usage=getattr(response, "usage", None)),
    ]


def streamed(response, kwargs):
    """Return ``response`` in the shape the adapter requested through ``stream``."""
    return FakeStream(stream_chunks(response)) if kwargs.get("stream") else response
