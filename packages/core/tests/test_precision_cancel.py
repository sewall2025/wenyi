"""Cancelling a precision run closes the three concurrent draft streams and returns promptly."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from wenyi_core.config import Config
from wenyi_core.llm.limits import RequestCancelled
from wenyi_core.llm.router import RoutedLLMClient
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.model_fixtures import FakeStream, model_config, stream_chunk
from tests.precision_fixtures import precision_store


class _EndlessStream(FakeStream):
    def __init__(self):
        super().__init__([])

    def close(self):
        self.closed = True

    def __iter__(self):
        while not self.closed:
            time.sleep(0.01)
            yield stream_chunk(content="x")


class _Completions:
    def __init__(self):
        self.streams: list[_EndlessStream] = []
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.streams.append(_EndlessStream())
            return self.streams[-1]


def test_cancel_ends_concurrent_precision_drafts(tmp_path):
    llm = model_config(timeout=60).model_dump(exclude_none=True)
    config = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": llm,
            "pipeline": {
                "translation_mode": "best_of_three",
                "review": False,
                "book_understanding": False,
                "annotation_alignment": False,
            },
            "paths": {"state_dir": str(tmp_path)},
        }
    )
    client = RoutedLLMClient(config.llm)
    completions = _Completions()
    client.adapter("default")._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    store = precision_store(tmp_path, ("one", "two"))

    def cancel_when_drafting():
        deadline = time.monotonic() + 5
        while len(completions.streams) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        client.cancel()

    canceller = threading.Thread(target=cancel_when_drafting)
    canceller.start()
    started = time.monotonic()
    with pytest.raises(RequestCancelled):
        Orchestrator(config, client)._translation.run(store, book_synopsis="synopsis")
    canceller.join()
    assert time.monotonic() - started < 10
    assert len(completions.streams) >= 3 and all(s.closed for s in completions.streams)
    assert not client.limits._streams
