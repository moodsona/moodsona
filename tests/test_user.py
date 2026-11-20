"""Tests for SimulatedUser, a persona with the same ``reply(history)`` method as a companion.

No test calls a model: the user model is a FakeProvider and the companion a
test double or a list of fixed replies.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import moodsona
from moodsona.companion import Companion, EchoCompanion, ScriptedCompanion
from moodsona.llm.base import ChatMessage, LLMError, LLMProvider
from moodsona.llm.fake import FakeCall, FakeProvider, Responder, plumbing_reply
from moodsona.metrics import compute_metrics
from moodsona.personas import load_bundled_persona
from moodsona.prompts import NOTES_MARKER
from moodsona.runner import Transcript, TurnRecord, run
from moodsona.simulator import Simulator, SimulatorConfig
from moodsona.state import PersonaSeed
from moodsona.user import SimulatedUser
from support import block, make_seed

Message = dict[str, Any]

#: Lines different enough that the own-repeat rail never fires within a day.
TOPICS = (
    "the tomatoes split overnight",
    "rain flattened the beans",
    "a neighbour borrowed my spade",
    "found slugs under the boards",
    "the shed door is stuck again",
    "picked the first courgette",
)
INTENTS = ("reply", "end_day", "skip_tomorrow")


# -- helpers ----------------------------------------------------------------


def patterned(
    intents: Mapping[tuple[int, int], str] | None = None,
    *,
    junk: Collection[tuple[int, int]] = (),
    bad_notes: Collection[int] = (),
) -> Responder:
    """Valid turns with the given intent per (day, turn), "reply" by default.

    A turn in ``junk`` fails its first attempt (the repair round passes). Day
    notes come from plumbing_reply, except for days in ``bad_notes``, whose
    notes never parse.
    """
    table = dict(intents or {})

    def respond(messages: Sequence[ChatMessage]) -> str:
        payload = json.loads(messages[1].content)
        day = payload["day"]
        if messages[0].content.startswith(NOTES_MARKER):
            return "no notes today" if day in bad_notes else plumbing_reply(messages)
        turn = payload["turn_in_day"]
        if (day, turn) in junk and len(messages) == 2:
            return "junk"
        state = payload["state"]
        return block(
            TOPICS[(turn - 1) % len(TOPICS)],
            intent=table.get((day, turn), "reply"),
            thought=f"secret thought {day}-{turn}",
            emotion=state["emotion"] + 1,
            energy=state["energy"] - 1,
        )

    return respond


def is_notes(call: FakeCall) -> bool:
    return call.messages[0].content.startswith(NOTES_MARKER)


def turn_payloads(provider: FakeProvider) -> list[dict[str, Any]]:
    return [json.loads(c.messages[1].content) for c in provider.calls if not is_notes(c)]


def notes_payloads(provider: FakeProvider) -> list[dict[str, Any]]:
    return [json.loads(c.messages[1].content) for c in provider.calls if is_notes(c)]


def say(text: str) -> Message:
    """A companion message, OpenAI style."""
    return {"role": "assistant", "content": text}


def said(text: str) -> Message:
    """The simulated user's message, as a harness adds it to the history."""
    return {"role": "user", "content": text}


def talk(
    user: SimulatedUser, replies: Sequence[str], history: Sequence[Message] = ()
) -> list[Message]:
    """Per reply: the simulated user writes, then the companion answers with that reply."""
    messages = list(history)
    for reply in replies:
        messages += [said(user.reply(messages)), say(reply)]
    return messages


def openai_loop(user: SimulatedUser, companion: Companion) -> Transcript:
    """Drive ``user`` the way an eval harness does, with OpenAI-style messages.

    Each message also carries a ``day`` key, which the simulated user ignores
    and the companion gets as :attr:`ChatMessage.day`, as from run().
    """
    history: list[Message] = []
    while not user.done:
        history.append({**said(user.reply(history)), "day": user.day})
        bot = companion.reply(
            tuple(ChatMessage(m["role"], m["content"], m["day"]) for m in history)
        )
        history.append({**say(bot), "day": user.day})
    return user.finish(history)


def reference_run(
    seed: PersonaSeed,
    provider: LLMProvider,
    companion: Companion,
    days: int,
    turns_per_day: int,
    rng_seed: int = 0,
) -> Transcript:
    """run() before it was built on SimulatedUser, plus day tags: the oracle for run()."""
    simulator = Simulator(seed, provider, rng=random.Random(rng_seed))
    transcript = Transcript(seed.id, rng_seed, days=days, turns_per_day=turns_per_day)
    history: list[ChatMessage] = []
    for day in range(1, days + 1):
        if simulator.is_skipped(day):
            transcript.skipped_days.append(day)
            continue
        simulator.start_day(day)
        today: list[ChatMessage] = []
        for turn_in_day in range(1, turns_per_day + 1):
            turn = simulator.next_turn(today, day=day, turn_in_day=turn_in_day)
            user = ChatMessage("user", turn.text, day=day)
            history.append(user)
            reply = companion.reply(tuple(history))
            answer = ChatMessage("assistant", reply, day=day)
            history.append(answer)
            today += [user, answer]
            transcript.turns.append(
                TurnRecord(
                    day=day,
                    turn=turn_in_day,
                    hidden=turn.hidden.as_dict() if turn.hidden else None,
                    visible=turn.text,
                    companion_reply=reply,
                    mood=simulator.state.mood,
                    polluted=turn.polluted,
                    problems=turn.problems,
                    forced=turn.forced,
                    rules_applied=turn.rules_applied,
                )
            )
            if turn.ends_day:
                break
        note = simulator.end_day(today, day=day)
        if note is not None:
            transcript.notes.append(note)
    return transcript


# -- run() is a SimulatedUser loop ------------------------------------------

Simulate = Callable[[FakeProvider, Companion], Transcript]


@pytest.mark.parametrize(
    ("seed", "days", "turns_per_day", "rng_seed", "responder", "skipped"),
    [
        pytest.param(
            make_seed(),
            5,
            3,
            1,
            patterned({(2, 2): "end_day", (3, 1): "skip_tomorrow"}),
            [4],
            id="end-day-and-skip",
        ),
        pytest.param(
            make_seed(memory_days_visible=0),
            4,
            2,
            3,
            patterned({(1, 1): "skip_tomorrow"}),
            [2],
            id="no-day-memory",
        ),
        pytest.param(
            make_seed(overnight_energy_jitter=15),
            3,
            4,
            7,
            patterned({(1, 3): "end_day"}, junk={(2, 2)}, bad_notes={1}),
            [],
            id="jitter-repairs-and-failed-notes",
        ),
        pytest.param(
            load_bundled_persona("anxious-newcomer"),
            4,
            2,
            11,
            patterned({(3, 2): "skip_tomorrow"}),
            [4],
            id="bundled-skips-the-last-day",
        ),
        pytest.param(
            load_bundled_persona("boundary-tester"),
            2,
            6,
            0,
            None,
            [],
            id="bundled-plumbing",
        ),
        pytest.param(make_seed(), 1, 1, 0, patterned(), [], id="one-turn"),
    ],
)
def test_run_and_a_hand_written_loop_match_the_old_run(
    tmp_path: Path,
    seed: PersonaSeed,
    days: int,
    turns_per_day: int,
    rng_seed: int,
    responder: Responder | None,
    skipped: list[int],
) -> None:
    def once(name: str, simulate: Simulate) -> tuple[str, list[FakeCall], list[Any]]:
        provider = FakeProvider(responder)
        companion = ScriptedCompanion(["Mm, tell me more.", "And then?", "Oh no."], cycle=True)
        transcript = simulate(provider, companion)
        path = tmp_path / f"{name}.jsonl"
        transcript.to_jsonl(path)
        assert Transcript.from_jsonl(path).skipped_days == skipped
        return path.read_text(encoding="utf-8"), provider.calls, companion.histories

    def by_hand(provider: FakeProvider, companion: Companion) -> Transcript:
        user = SimulatedUser(
            seed, provider, turns_per_day=turns_per_day, days=days, rng_seed=rng_seed
        )
        return openai_loop(user, companion)

    def with_run(provider: FakeProvider, companion: Companion) -> Transcript:
        return run(
            seed,
            companion,
            user_model=provider,
            days=days,
            turns_per_day=turns_per_day,
            rng_seed=rng_seed,
        )

    old = once("old", lambda p, c: reference_run(seed, p, c, days, turns_per_day, rng_seed))
    assert once("run", with_run) == old
    assert once("loop", by_hand) == old


@settings(max_examples=60, deadline=None)
@given(
    days=st.integers(1, 5),
    turns_per_day=st.integers(1, 4),
    intents=st.dictionaries(
        st.tuples(st.integers(1, 5), st.integers(1, 4)), st.sampled_from(INTENTS)
    ),
    memory=st.sampled_from([0, 3]),
    rng_seed=st.integers(0, 2**16),
)
def test_days_and_turns_stay_within_bounds_for_any_intents(
    days: int,
    turns_per_day: int,
    intents: dict[tuple[int, int], str],
    memory: int,
    rng_seed: int,
) -> None:
    seed = make_seed(memory_days_visible=memory, overnight_energy_jitter=10)
    provider = FakeProvider(patterned(intents))
    user = SimulatedUser(seed, provider, turns_per_day=turns_per_day, days=days, rng_seed=rng_seed)
    transcript = openai_loop(user, EchoCompanion())

    per_day = Counter(t.day for t in transcript.turns)
    assert all(count <= turns_per_day for count in per_day.values())
    for day, count in per_day.items():
        assert [t.turn for t in transcript.turns if t.day == day] == list(range(1, count + 1))
    written = [*per_day, *transcript.skipped_days, *(n.day for n in transcript.notes)]
    assert all(1 <= day <= days for day in written)
    assert not set(per_day) & set(transcript.skipped_days)

    # And it is exactly what run() did before it was built on SimulatedUser.
    old = FakeProvider(patterned(intents))
    assert transcript == reference_run(seed, old, EchoCompanion(), days, turns_per_day, rng_seed)
    assert provider.calls == old.calls


@settings(max_examples=40, deadline=None)
@given(
    turns_per_day=st.integers(1, 4),
    replies=st.integers(1, 14),
    intents=st.dictionaries(
        st.tuples(st.integers(1, 28), st.integers(1, 4)), st.sampled_from(INTENTS)
    ),
)
def test_an_unbounded_user_rolls_days_over_and_is_never_done(
    turns_per_day: int, replies: int, intents: dict[tuple[int, int], str]
) -> None:
    user = SimulatedUser(make_seed(), FakeProvider(patterned(intents)), turns_per_day=turns_per_day)
    history: list[Message] = []
    for _ in range(replies):
        assert not user.done
        history = talk(user, ["I see."], history)
    assert not user.done

    transcript = user.transcript(history)
    assert len(transcript.turns) == replies
    per_day = Counter(t.day for t in transcript.turns)
    assert all(count <= turns_per_day for count in per_day.values())
    assert list(per_day) == sorted(per_day)
    assert not set(per_day) & set(transcript.skipped_days)
    assert user.day == transcript.turns[-1].day == transcript.days
    assert [n.day for n in transcript.notes] == sorted(per_day)[:-1]


# -- history ------------------------------------------------------------------


def test_openai_style_history_with_parts_tools_and_system_messages() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=3)
    history: list[Message] = [{"role": "system", "content": "You are a kind companion."}]
    history.append(said(user.reply(history)))
    history += [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Split already?"},
                {"type": "image_url", "image_url": {"url": "data:,"}},
            ],
        },
        {"role": "assistant", "content": "  "},
    ]
    history.append(said(user.reply(history)))
    history.append(say("Do you grow anything else?"))

    first, second = turn_payloads(provider)
    assert first["visible_conversation"] == []
    assert second["visible_conversation"] == [
        {"from": "character", "text": TOPICS[0]},
        {"from": "companion", "text": "Split already?"},
    ]
    transcript = user.transcript(history)
    assert [t.companion_reply for t in transcript.turns] == [
        "Split already?",
        "Do you grow anything else?",
    ]


def test_chat_messages_are_shown_as_they_are_like_run_shows_them() -> None:
    # An empty companion reply stays in the day's conversation, as run() has
    # always shown it; only an OpenAI-style message without text is skipped.
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider)
    history = [ChatMessage("system", "be kind")]
    history.append(ChatMessage("user", user.reply(history)))
    history.append(ChatMessage("assistant", ""))
    user.reply(history)
    assert turn_payloads(provider)[1]["visible_conversation"] == [
        {"from": "character", "text": TOPICS[0]},
        {"from": "companion", "text": ""},
    ]


def test_the_bot_may_speak_first() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider)
    history: list[Message] = [say("Hi! How was your week?")]
    assert user.reply(history) == TOPICS[0]
    assert turn_payloads(provider)[0]["visible_conversation"] == [
        {"from": "companion", "text": "Hi! How was your week?"}
    ]
    history += [said(TOPICS[0]), say("Oh no, all of them?")]
    assert user.transcript(history).turns[0].companion_reply == "Oh no, all of them?"


def test_a_user_message_from_elsewhere_ends_the_companion_reply() -> None:
    user = SimulatedUser(make_seed(), FakeProvider(patterned()))
    history = [
        said(user.reply([])),
        say("Split already?"),
        said("a line from the harness's own script"),
        say("Tell me more."),
    ]
    user.reply(history)
    assert user.transcript().turns[0].companion_reply == "Split already?"


def test_a_blank_persona_message_keeps_the_conversation_in_step() -> None:
    # Every attempt fails to parse, so each turn falls back to an empty line.
    config = SimulatorConfig(fallback_message="")
    user = SimulatedUser(make_seed(), FakeProvider(lambda _: "not a turn"), config=config)
    replies = ["Hello?", "Are you there?", "Take your time."]
    turns = user.transcript(talk(user, replies)).turns
    assert [t.companion_reply for t in turns] == replies
    assert all(t.polluted and t.visible == "" for t in turns)


def test_a_shorter_history_is_another_conversation() -> None:
    user = SimulatedUser(make_seed(), FakeProvider(patterned()))
    history = talk(user, ["Hm."])
    user.reply(history)  # the user has now seen these two messages and its own reply
    for shorter in ([], history[:1], history):
        with pytest.raises(ValueError, match="one SimulatedUser per conversation"):
            user.reply(shorter)
    with pytest.raises(ValueError, match="one SimulatedUser per conversation"):
        user.transcript(history)
    with pytest.raises(ValueError, match="one SimulatedUser per conversation"):
        user.finish(history)


def test_a_history_item_must_be_a_message() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider)
    bad: Any = ["hello"]
    with pytest.raises(TypeError, match="ChatMessage or a mapping"):
        user.reply(bad)
    assert (user.day, provider.calls) == (0, [])


# -- days ---------------------------------------------------------------------


def test_the_next_reply_writes_the_day_notes_and_starts_the_next_day() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=2)
    history = talk(user, ["Mm.", "Long day?"])  # day 1 is full
    assert user.day == 1
    assert notes_payloads(provider) == []

    history = talk(user, ["Morning!"], history)  # the first message of day 2
    assert user.day == 2
    assert [is_notes(c) for c in provider.calls] == [False, False, True, False]
    # The notes read the whole day, the companion's last reply included.
    (notes,) = notes_payloads(provider)
    assert notes["day"] == 1
    assert [m["text"] for m in notes["visible_conversation"]] == [
        TOPICS[0],
        "Mm.",
        TOPICS[1],
        "Long day?",
    ]
    # Day 2 starts with an empty conversation of its own and reads day 1's notes.
    day2 = turn_payloads(provider)[2]
    assert (day2["day"], day2["turn_in_day"], day2["visible_conversation"]) == (2, 1, [])
    assert [entry["day"] for entry in day2["memory_log"]] == [1]

    transcript = user.transcript(history)
    assert [n.day for n in transcript.notes] == [1]
    assert [(t.day, t.turn, t.companion_reply) for t in transcript.turns] == [
        (1, 1, "Mm."),
        (1, 2, "Long day?"),
        (2, 1, "Morning!"),
    ]


def test_a_skipped_day_has_no_messages() -> None:
    provider = FakeProvider(patterned({(1, 2): "skip_tomorrow"}))
    user = SimulatedUser(make_seed(), provider, turns_per_day=3)
    history = talk(user, ["Hm.", "See you.", "Back already?"])
    transcript = user.transcript(history)
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (1, 2), (3, 1)]
    assert transcript.skipped_days == [2]
    assert user.day == 3
    assert turn_payloads(provider)[-1]["days_since_last_talk"] == 2


def test_done_after_the_last_day_and_no_reply_after_it() -> None:
    user = SimulatedUser(make_seed(), FakeProvider(patterned()), turns_per_day=2, days=2)
    history: list[Message] = []
    for _ in range(3):
        history = talk(user, ["Hm."], history)
        assert not user.done
    history = talk(user, ["Hm."], history)
    assert user.done
    with pytest.raises(RuntimeError, match=r"all 2 days.*finish\(history\)"):
        user.reply(history)
    transcript = user.finish(history)
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [n.day for n in transcript.notes] == [1, 2]
    with pytest.raises(RuntimeError, match=r"finish\(\) was called"):
        user.reply(history)


def test_done_when_the_days_left_are_skipped() -> None:
    provider = FakeProvider(patterned({(2, 1): "skip_tomorrow"}))
    user = SimulatedUser(make_seed(), provider, turns_per_day=2, days=3)
    history = talk(user, ["Hm.", "Go on."])
    assert (user.day, user.done) == (1, False)
    history = talk(user, ["Bye then."], history)
    assert user.done  # day 3 is skipped, so day 2's only turn was the last one
    transcript = user.finish(history)
    assert (transcript.skipped_days, transcript.days) == ([3], 3)
    assert [n.day for n in transcript.notes] == [1, 2]


# -- finish, transcript, metrics ----------------------------------------------


def test_finish_is_idempotent() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=2, days=1, rng_seed=5)
    history = talk(user, ["Hm.", "Go on."])
    first = user.finish(history)
    calls = len(provider.calls)
    assert [n.day for n in first.notes] == [1]
    assert (first.persona_id, first.rng_seed, first.days, first.turns_per_day) == (
        "test-gardener",
        5,
        1,
        2,
    )
    assert [t.companion_reply for t in first.turns] == ["Hm.", "Go on."]

    assert user.finish(history) == first
    assert user.finish() == first
    assert user.transcript() == first
    assert len(provider.calls) == calls
    first.turns.clear()  # the caller's copy
    assert len(user.finish().turns) == 2


def test_finish_before_the_end_writes_the_open_day() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=3)
    history = talk(user, ["Hm.", "Go on."])
    transcript = user.finish(history)
    assert user.done
    assert [n.day for n in transcript.notes] == [1]
    assert (transcript.days, transcript.turns[-1].companion_reply) == (1, "Go on.")
    assert [m["text"] for m in notes_payloads(provider)[0]["visible_conversation"]][-1] == "Go on."


def test_finish_without_history_reads_the_day_as_last_seen() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=3)
    talk(user, ["Hm.", "Go on."])
    transcript = user.finish()
    assert transcript.turns[-1].companion_reply == ""
    assert [m["text"] for m in notes_payloads(provider)[0]["visible_conversation"]] == [
        TOPICS[0],
        "Hm.",
        TOPICS[1],
    ]


def test_finish_before_the_first_reply_is_empty() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, days=3)
    assert (user.transcript().turns, user.transcript().days, user.day) == ([], 3, 0)
    transcript = user.finish()
    assert (transcript.turns, transcript.notes, provider.calls) == ([], [], [])
    assert SimulatedUser(make_seed(), provider).transcript().days is None


def test_transcript_and_metrics_make_no_model_calls() -> None:
    provider = FakeProvider(patterned())
    user = SimulatedUser(make_seed(), provider, turns_per_day=2)
    history = talk(user, ["Hm.", "Go on.", "Morning."])
    calls = list(provider.calls)

    snapshot = user.transcript(history)
    assert snapshot.turns[-1].companion_reply == "Morning."
    assert user.transcript().turns[-1].companion_reply == ""
    assert [n.day for n in snapshot.notes] == [1]  # day 2's notes are not written yet
    assert snapshot.days == 2
    metrics = user.metrics(history)
    assert metrics == compute_metrics(snapshot)
    assert (metrics.turns, metrics.active_days) == (3, 2)
    assert provider.calls == calls

    # Nothing changed: the conversation goes on where it was.
    talk(user, ["Then?"], history)
    assert [(t.day, t.turn) for t in user.transcript().turns][-1] == (2, 2)


def test_state_follows_the_latest_turn() -> None:
    user = SimulatedUser(make_seed(), FakeProvider(patterned()))
    assert user.state.mood == make_seed().initial_state
    history = talk(user, ["Hm.", "Go on."])
    assert user.state.mood == user.transcript(history).turns[-1].mood
    assert user.state.current_day == user.day == 1


# -- failures and arguments ---------------------------------------------------


def test_a_failed_notes_call_leaves_the_turn_open_and_a_retry_repeats_it() -> None:
    calls = 0
    script = patterned()

    def flaky(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls in (2, 4):  # day 1's notes, then day 2's first turn
            raise LLMError("timeout")
        return script(messages)

    seed = make_seed(overnight_energy_jitter=15)
    user = SimulatedUser(seed, FakeProvider(flaky), turns_per_day=1, rng_seed=3)
    history = talk(user, ["Hm."])
    with pytest.raises(LLMError):
        user.reply(history)  # the notes fail: the turn stays open
    assert user.day == 1
    assert user.transcript().notes == []
    assert user.transcript().turns[0].companion_reply == ""
    with pytest.raises(LLMError):
        user.reply(history)  # the notes pass, day 2's first turn fails
    assert user.day == 2
    assert user.transcript().turns[0].companion_reply == "Hm."
    history = talk(user, ["Ok."], history)

    transcript = user.transcript(history)
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (2, 1)]
    assert [n.day for n in transcript.notes] == [1]
    assert [t.companion_reply for t in transcript.turns] == ["Hm.", "Ok."]
    # The retries left no trace: the same run without failures.
    steady = SimulatedUser(seed, FakeProvider(script), turns_per_day=1, rng_seed=3)
    assert steady.transcript(talk(steady, ["Hm.", "Ok."])) == transcript


def rule_two_every_turn(intents: Mapping[tuple[int, int], str] | None = None) -> Responder:
    """Valid turns that all name rule 2, with the given intent per (day, turn)."""
    table = dict(intents or {})

    def respond(messages: Sequence[ChatMessage]) -> str:
        if messages[0].content.startswith(NOTES_MARKER):
            return plumbing_reply(messages)
        payload = json.loads(messages[1].content)
        day, turn = payload["day"], payload["turn_in_day"]
        return block(
            TOPICS[(turn - 1) % len(TOPICS)],
            intent=table.get((day, turn), "reply"),
            rules=[2],
        )

    return respond


def repeats(transcript: Transcript) -> list[tuple[int, int, list[int]]]:
    """(day, turn, repeat of each applied rule) for every turn."""
    return [(t.day, t.turn, [a.repeat for a in t.rules_applied or ()]) for t in transcript.turns]


def test_a_provider_error_and_its_retry_count_the_rule_once() -> None:
    calls = 0
    script = rule_two_every_turn()

    def flaky(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:  # day 1's second turn
            raise LLMError("timeout")
        return script(messages)

    user = SimulatedUser(make_seed(), FakeProvider(flaky), turns_per_day=3, days=1)
    history = talk(user, ["Hm."])
    with pytest.raises(LLMError):
        user.reply(history)
    assert user.state.rule_uses == {2: 1}
    history = talk(user, ["Go on.", "Mm."], history)
    assert repeats(user.finish(history)) == [(1, 1, [1]), (1, 2, [2]), (1, 3, [3])]


def test_the_rule_count_restarts_on_the_first_day_after_a_skip() -> None:
    provider = FakeProvider(rule_two_every_turn({(1, 2): "skip_tomorrow"}))
    transcript = run(make_seed(), EchoCompanion(), user_model=provider, days=3, turns_per_day=2)
    assert transcript.skipped_days == [2]
    assert repeats(transcript) == [(1, 1, [1]), (1, 2, [2]), (3, 1, [1]), (3, 2, [2])]


def test_a_failed_notes_call_in_finish_can_be_retried() -> None:
    failed = False
    script = patterned()

    def flaky(messages: Sequence[ChatMessage]) -> str:
        nonlocal failed
        if messages[0].content.startswith(NOTES_MARKER) and not failed:
            failed = True
            raise LLMError("timeout")
        return script(messages)

    user = SimulatedUser(make_seed(), FakeProvider(flaky), turns_per_day=2, days=1)
    history = talk(user, ["Hm.", "Go on."])
    with pytest.raises(LLMError):
        user.finish(history)
    transcript = user.finish(history)
    assert [n.day for n in transcript.notes] == [1]
    assert [t.companion_reply for t in transcript.turns] == ["Hm.", "Go on."]


@pytest.mark.parametrize("arguments", [{"turns_per_day": 0}, {"days": 0}, {"days": -2}])
def test_turns_per_day_and_days_must_be_positive(arguments: dict[str, Any]) -> None:
    provider = FakeProvider()
    with pytest.raises(ValueError, match="at least 1"):
        SimulatedUser(make_seed(), provider, **arguments)
    assert provider.calls == []


def test_an_unknown_language_fails_before_any_model_call() -> None:
    provider = FakeProvider()
    with pytest.raises(LookupError, match="No language profile for 'xx'"):
        SimulatedUser(make_seed(language="xx"), provider)
    assert provider.calls == []


def test_simulated_user_is_exported() -> None:
    assert moodsona.SimulatedUser is SimulatedUser
    assert "SimulatedUser" in moodsona.__all__


def test_the_model_parameter_is_called_user_model_like_in_run() -> None:
    user = SimulatedUser(make_seed(), user_model=FakeProvider(patterned()))
    assert user.reply([]) != ""
