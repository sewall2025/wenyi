"""Pure alignment checks for precision replies, plus Yellowface-shaped regressions."""

import json

import pytest
from wenyi_core.agents.precision import (
    PrecisionAgent,
    PrecisionInputs,
    accepted_segments,
    misaligned_reply,
    target_scripts,
)
from wenyi_core.agents.precision_alignment import (
    char_script,
    copies_neighbor_source,
    drifted_segments,
    echoes_context,
    lacks_target_script,
)
from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient

from tests.precision_fixtures import segment_ids

ZH, EN, JA, FR = (target_scripts(code) for code in ("zh", "en", "ja", "fr"))


def test_target_scripts_come_from_language_profiles():
    assert ZH == ("Han",)
    assert set(JA) == {"Han", "Hiragana", "Katakana"}
    assert target_scripts("zh-Hant") == ("Han",)
    assert target_scripts("en-GB") == ("Latin",)
    assert target_scripts("ru") == ("Cyrillic",)
    assert target_scripts("ko") == ("Hangul",)
    assert target_scripts("auto") == ()
    assert [char_script(c) for c in "a中あア한Дé1"] == [
        "Latin",
        "Han",
        "Hiragana",
        "Katakana",
        "Hangul",
        "Cyrillic",
        "Latin",
        None,
    ]


@pytest.mark.parametrize(
    ("target", "source", "scripts", "expected"),
    [
        # en→zh: an English sentence is not a Chinese translation.
        ("She walked out into the rain.", "He stayed inside.", ZH, True),
        ("她走进了雨里。", "She walked out into the rain.", ZH, False),
        # zh→en
        ("还没有翻译这一段。", "她走进了雨里。", EN, True),
        ("She walked out into the rain.", "她走进了雨里。", EN, False),
        # en→ja: kana or kanji both count as Japanese.
        ("She walked out into the rain.", "He stayed.", JA, True),
        ("彼女は雨の中へ出ていった。", "She walked out into the rain.", JA, False),
        ("カタカナだけ", "Katakana only", JA, False),
        # en→fr shares the Latin script: undecidable, never flagged.
        ("He stayed inside the house.", "She walked out into the rain.", FR, False),
        # Untranslated copies of the own source are judged by other rules.
        ("He stayed inside.", "He stayed inside.", ZH, False),
        # Short, numeric and symbol-only targets are not judged.
        ("OK", "Okay then, fine.", ZH, False),
        ("1984", "Chapter 1984", ZH, False),
        ("* * *", "* * *", ZH, False),
        ("Chapter", "", ZH, False),
        # Unknown target scripts never flag.
        ("She walked out into the rain.", "He stayed.", (), False),
    ],
)
def test_lacks_target_script(target, source, scripts, expected):
    assert lacks_target_script(target, source, scripts) is expected


def test_copies_neighbor_source():
    sources = ("Chapter One", "She opened the door.", "He was waiting outside.")
    assert copies_neighbor_source("He was waiting outside.", sources, 1, "")
    assert copies_neighbor_source("Chapter One", sources, 1, "")
    assert copies_neighbor_source("Next one.", sources, 2, "Next one.")
    assert not copies_neighbor_source("她打开了门。", sources, 1, "")
    # Equal to its own source (a repeated line) is not a neighbour copy.
    assert not copies_neighbor_source("Yes.", ("Yes.", "Yes."), 0, "")


def test_echoes_context_uses_only_the_last_context_paragraph():
    context = "很久以前的一段。\n她把那份手稿锁进了抽屉里，然后关上了灯。"
    assert echoes_context("她把那份手稿锁进了抽屉里，然后关上了灯", context)
    assert echoes_context("她把那份手稿锁进抽屉里，然后关了灯。", context)
    assert not echoes_context("很久以前的一段。", context)
    assert not echoes_context("第二天早上，她去了出版社。", context)
    # Short texts such as titles are too generic to compare.
    assert not echoes_context("第一章", "第一章")
    assert not echoes_context("anything", "")


DRIFT_SOURCES = (
    "Chapter 3",
    "On May 4 Athena called me at 9 to talk about the novel.",
    "“Are you coming tonight?” she asked, and I said I was not sure.",
    "I spent the afternoon reading reviews of Candice Lee and her debut book.",
    "The flat was silent except for the hum of the old refrigerator in the corner.",
)
DRIFT_ALIGNED = {
    0: "第3章",
    1: "5月4日，Athena在9点给我打电话，聊那本小说。",
    2: "“你今晚来吗？”她问，我说我不确定。",
    3: "我整个下午都在读关于Candice Lee和她处女作的书评。",
    4: "公寓里一片寂静，只有角落里那台旧冰箱嗡嗡作响。",
}


def test_drift_ignores_aligned_dialogue_titles_and_short_segments():
    ids = list(range(len(DRIFT_SOURCES)))
    assert drifted_segments(DRIFT_SOURCES, DRIFT_ALIGNED, ids) == set()
    short = ("Yes.", "No.", "Why?", "Fine!")
    assert (
        drifted_segments(short, {0: "不。", 1: "为什么？", 2: "好！", 3: "好"}, [0, 1, 2, 3])
        == set()
    )


def test_drift_flags_two_consecutive_entries_shifted_the_same_way():
    # Entries 2 and 3 carry the translations of sources 3 and 4 (shift forward).
    shifted = {**DRIFT_ALIGNED, 1: DRIFT_ALIGNED[2], 2: DRIFT_ALIGNED[3]}
    assert drifted_segments(DRIFT_SOURCES, shifted, [1, 2, 3, 4]) == {1, 2}
    # A single shifted entry is not enough evidence.
    single = {**DRIFT_ALIGNED, 2: DRIFT_ALIGNED[3]}
    assert drifted_segments(DRIFT_SOURCES, single, [1, 2, 3, 4]) == set()


def ids_reply(texts: dict[int, str]) -> dict:
    return {"translations": {str(i): t for i, t in texts.items()}}


def test_misaligned_reply_reasons():
    sources = ("She left.", "He stayed home all day.", "The rain did not stop.")
    ids = [0, 1, 2]
    good = ids_reply({0: "她走了。", 1: "他整天待在家里。", 2: "雨一直没停。"})
    assert misaligned_reply(sources, ids, good, context="之前的段落。", target_scripts=ZH) is None
    extra = ids_reply({0: "她走了。", 1: "他整天待在家里。", 2: "雨。", 3: "多"})
    assert misaligned_reply(sources, ids, extra, context="", target_scripts=ZH) == "unrequested_id"
    echo = ids_reply({0: "之前那一段的完整译文在这里", 1: "他。", 2: "雨。"})
    context = "更早。\n之前那一段的完整译文在这里。"
    assert misaligned_reply(sources, ids, echo, context=context, target_scripts=ZH) == (
        "context_echo"
    )
    foreign = ids_reply({0: "He stayed home all day.", 1: "The rain did not stop.", 2: "雨。"})
    assert misaligned_reply(sources, ids, foreign, context="", target_scripts=ZH) == (
        "missing_target_script"
    )
    # Same-script pairs cannot judge scripts, so the majority rule never fires for en→fr.
    assert misaligned_reply(sources, ids, foreign, context="", target_scripts=FR) is None
    # Exactly half is not a majority.
    half = ids_reply({0: "He stayed home all day.", 1: "他整天待在家里。"})
    assert misaligned_reply(sources[:2], [0, 1], half, context="", target_scripts=ZH) is None


def test_accepted_segments_marks_only_suspicious_entries_pending():
    sources = ("Chapter Two", "She opened the door slowly.", "He was waiting outside.")
    data = ids_reply({0: "第二章", 1: "He was waiting outside.", 2: "他在外面等着。"})
    found = accepted_segments(sources, [0, 1, 2], data, target_scripts=ZH)
    assert found == {0: "第二章", 2: "他在外面等着。"}
    # zh→en: an untranslated Chinese entry is pending, the English one is kept.
    zh_sources = ("她打开了门。", "安娜在外面。")
    data = ids_reply({0: "她慢慢地打开了门。", 1: "Anna was outside."})
    assert accepted_segments(zh_sources, [0, 1], data, target_scripts=EN) == {
        1: "Anna was outside.",
    }
    # en→fr: a correct translation is never flagged for its script.
    fr = ids_reply({0: "Chapitre deux", 1: "Elle ouvrit lentement la porte.", 2: "Il attendait."})
    assert accepted_segments(sources, [0, 1, 2], fr, target_scripts=FR) == {
        0: "Chapitre deux",
        1: "Elle ouvrit lentement la porte.",
        2: "Il attendait.",
    }


# Yellowface-shaped regressions with synthetic text.
BOOK = (
    "Chapter Four",
    "On March 3 I went back to her apartment alone, carrying 2 empty boxes.",
    "“Did you keep any of her notes?” her mother asked me at the door.",
    "I told myself I would only read the first 10 pages, nothing more!",
    "By noon I had read all 300 pages, and my hands would not stop shaking.",
    "Nobody knew the book existed except for me.",
)
CONTEXT = "我在葬礼上一句话也没说。\n回家的路上，雨一直下个不停，我一直在想她最后说的话。"
GOOD = (
    "第四章",
    "3月3日，我独自回到她的公寓，带着2个空箱子。",
    "“她的笔记你留了吗？”她母亲在门口问我。",
    "我告诉自己，只读前10页，绝不多读！",
    "到中午我已经读完了全部300页，双手抖个不停。",
    "除了我，没有人知道这本书的存在。",
)


def precision_agent(handler):
    config = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {"translation_mode": "best_of_three", "align_retry_limit": 1},
        }
    )
    client = FakeClient(handler=handler)
    events = []
    client.set_event_sink(lambda event, **data: events.append((event, data)))
    return PrecisionAgent(client, config), client, events


def book_inputs():
    return PrecisionInputs(BOOK, (), CONTEXT, "style", "synopsis", "digest", [[]] * 6, "", False)


def segment_ids_of(call):
    return segment_ids(json.loads(call["messages"][-1]["content"].split("Task (JSON):\n")[1]))


def respond_with(bad):
    """Answer whole-batch requests with ``bad`` (id -> text) and smaller ones correctly."""

    def handler(messages, *_):
        task = json.loads(messages[-1]["content"].split("Task (JSON):\n")[1])
        ids = segment_ids(task)
        if len(ids) == len(BOOK):
            return json.dumps(ids_reply(bad))
        return json.dumps(ids_reply({i: GOOD[i] for i in ids}))

    return handler


def test_aligned_reply_is_accepted_in_one_request():
    worker, client, events = precision_agent(respond_with(dict(enumerate(GOOD))))
    assert worker.translate(book_inputs()) == list(GOOD)
    assert len(client.calls) == 1 and events == []


def test_context_taken_as_ids_shifts_everything_back_and_is_rejected_whole():
    # The model numbered the last context paragraph as id 0 and pushed the batch back by one.
    shifted = dict(enumerate([CONTEXT.splitlines()[-1], *GOOD[:-1]]))
    worker, client, events = precision_agent(respond_with(shifted))
    assert worker.translate(book_inputs()) == list(GOOD)
    assert [(event, data["reason"]) for event, data in events] == [
        ("precision_reply_misaligned", "context_echo")
    ]
    # Nothing of the shifted reply is used; the request is halved and answered correctly.
    assert [segment_ids_of(call) for call in client.calls] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2],
        [3, 4, 5],
    ]


def test_skipped_title_shifts_entries_forward_and_the_shifted_stretch_is_redone():
    # The title was skipped: id 0 got the translation of segment 1, and so on.
    shifted = dict(enumerate([*GOOD[1:], "（多余）"]))
    assert accepted_segments(BOOK, list(range(6)), ids_reply(shifted), target_scripts=ZH) == {}
    worker, client, events = precision_agent(respond_with(shifted))
    assert worker.translate(book_inputs()) == list(GOOD)
    assert [segment_ids_of(call) for call in client.calls][1:] == [[0, 1, 2], [3, 4, 5]]


def test_copied_neighbor_source_is_pending_and_retranslated_alone():
    copied = {**dict(enumerate(GOOD)), 3: BOOK[4]}
    worker, client, events = precision_agent(respond_with(copied))
    assert worker.translate(book_inputs()) == list(GOOD)
    assert [segment_ids_of(call) for call in client.calls] == [list(range(6)), [3]]
    assert events == []
