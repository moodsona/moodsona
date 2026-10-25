"""Tests for the langwatch/scenario adapter.

They run against the real ``scenario`` package when it is installed (the
``moodsona[scenario]`` extra) and otherwise against a minimal stub of the three
names the adapter uses. No test calls a model: the user model is a FakeProvider
and the agent under test is a list of fixed replies.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle
import sys
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import moodsona.adapters
from moodsona.adapters.scenario import MoodsonaUser, user_simulator_agent
from moodsona.llm.base import ChatMessage, LLMError
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.metrics import metrics_from_jsonl
from moodsona.prompts import NOTES_MARKER
from moodsona.runner import Transcript
from moodsona.simulator import SimulatorConfig
from support import block, make_seed

# -- a minimal stand-in for the scenario package ---------------------------


class StubRole(Enum):
    USER = "User"
    AGENT = "Agent"
    JUDGE = "Judge"


class StubAgentAdapter(ABC):
    name: str | None = None
    role: ClassVar[StubRole] = StubRole.AGENT

    @abstractmethod
    async def call(self, input: Any) -> Any: ...


@dataclass
class StubAgentInput:
    thread_id: str
    messages: list[Any]
    new_messages: list[Any] = field(default_factory=list)
    scenario_state: Any = None


def make_stub() -> types.ModuleType:
    module = types.ModuleType("scenario")
    module.AgentAdapter = StubAgentAdapter  # type: ignore[attr-defined]
    module.AgentRole = StubRole  # type: ignore[attr-defined]
    module.AgentInput = StubAgentInput  # type: ignore[attr-defined]
    return module


@pytest.fixture
def scenario_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The real scenario package if it is installed, else the stub."""
    try:
        with warnings.catch_warnings():  # its import-time deprecation warnings are not ours
            warnings.simplefilter("ignore")
            real = importlib.import_module("scenario")
    except ModuleNotFoundError as exc:
        if exc.name != "scenario":
            raise  # scenario is installed but one of its own imports is missing
        stub = make_stub()
        monkeypatch.setitem(sys.modules, "scenario", stub)
        return stub
    except Exception as exc:  # an installed but broken package
        pytest.skip(f"scenario is installed but does not import: {exc}")
    if not all(hasattr(real, name) for name in ("AgentAdapter", "AgentRole", "AgentInput")):
        pytest.skip("an unrelated module named 'scenario' is installed")
    return real


# -- helpers ----------------------------------------------------------------


def scheduled(messages: Sequence[ChatMessage]) -> str:
    """Hidden blocks with a secret thought; emotion rises by one point each turn.

    Day 1 ends at turn 2 and day 2 skips day 3 at turn 1.
    """
    if messages[0].content.startswith(NOTES_MARKER):
        return plumbing_reply(messages)
    payload = json.loads(messages[1].content)
    day, turn = payload["day"], payload["turn_in_day"]
    state = payload["state"]
    intent = {(1, 2): "end_day", (2, 1): "skip_tomorrow"}.get((day, turn), "reply")
    return block(
        f"day {day} line {turn} about the weather",
        intent=intent,
        thought=f"secret thought {day}-{turn}",
        emotion=state["emotion"] + 1,
        energy=state["energy"] - 1,
    )


def turn_payloads(provider: FakeProvider) -> list[dict[str, Any]]:
    return [
        json.loads(call.messages[1].content)
        for call in provider.calls
        if not call.messages[0].content.startswith(NOTES_MARKER)
    ]


def converse(
    module: types.ModuleType, user: MoodsonaUser, replies: Sequence[Any], turns: int
) -> list[dict[str, Any]]:
    """Drive the user like scenario does: user message, then the agent's reply."""
    messages: list[dict[str, Any]] = []

    async def go() -> None:
        for i in range(turns):
            agent_input = module.AgentInput(
                thread_id="thread-1", messages=list(messages), new_messages=[], scenario_state=None
            )
            text = await user.call(agent_input)
            messages.append({"role": "user", "content": text})
            reply = replies[i % len(replies)]
            messages.extend(reply if isinstance(reply, list) else [reply])

    asyncio.run(go())
    return messages


def say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


# -- tests ------------------------------------------------------------------


def test_agent_plays_the_user_role(scenario_module: types.ModuleType) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider())
    assert isinstance(user, scenario_module.AgentAdapter)
    assert isinstance(user, MoodsonaUser)
    agent_class: Any = type(user)  # `role` comes from scenario's base class
    assert agent_class.role is scenario_module.AgentRole.USER


def test_only_the_visible_message_is_returned(scenario_module: types.ModuleType) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider(scheduled), turns_per_day=3)
    messages = converse(scenario_module, user, [say("Mm, tell me more.")], turns=4)
    sent = [m["content"] for m in messages if m["role"] == "user"]
    assert sent[0] == "day 1 line 1 about the weather"
    assert all(isinstance(text, str) for text in sent)
    joined = "\n".join(sent)
    assert "secret thought" not in joined
    assert "<state>" not in joined
    assert "<message>" not in joined


def test_hidden_state_and_day_memory_persist_across_calls(
    scenario_module: types.ModuleType,
) -> None:
    provider = FakeProvider(scheduled)
    seed = make_seed()
    user = user_simulator_agent(seed, provider, turns_per_day=3, rng_seed=1)
    converse(scenario_module, user, [say("Mm, tell me more.")], turns=5)

    # Day 1 ends early (end_day at turn 2), day 2 skips day 3 at turn 1.
    transcript = user.transcript()
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (1, 2), (2, 1), (4, 1), (4, 2)]
    assert transcript.skipped_days == [3]
    assert [n.day for n in transcript.notes] == [1, 2]
    assert user.day == 4

    # The emotion the model proposes builds on the state the adapter kept.
    emotions = [t.mood.emotion for t in transcript.turns]
    assert emotions == sorted(emotions)
    assert emotions[-1] > seed.initial_state.emotion
    assert user.simulator.state.mood == transcript.turns[-1].mood

    # Day 2 reads day 1's notes and starts with an empty conversation of its own.
    payloads = turn_payloads(provider)
    day2 = next(p for p in payloads if p["day"] == 2)
    assert [entry["day"] for entry in day2["memory_log"]] == [1]
    assert day2["visible_conversation"] == []
    day4 = next(p for p in payloads if p["day"] == 4)
    assert day4["days_since_last_talk"] == 2


@settings(max_examples=25, deadline=None)
@given(turns_per_day=st.integers(1, 4), calls=st.integers(1, 12))
def test_days_roll_over_after_turns_per_day(turns_per_day: int, calls: int) -> None:
    # Plumbing replies never end a day, so every day has exactly turns_per_day turns.
    user = MoodsonaUser(make_seed(), FakeProvider(), turns_per_day=turns_per_day)
    messages: list[dict[str, Any]] = []
    for _ in range(calls):
        messages.append({"role": "user", "content": user.respond(messages)})
        messages.append(say("I see."))
    turns = user.transcript(messages).turns
    assert [(t.day, t.turn) for t in turns] == [
        (i // turns_per_day + 1, i % turns_per_day + 1) for i in range(calls)
    ]
    assert len(user.transcript(messages).notes) == (calls - 1) // turns_per_day


def test_companion_messages_from_the_input_are_used(scenario_module: types.ModuleType) -> None:
    provider = FakeProvider(scheduled)
    user = user_simulator_agent(make_seed(), provider, turns_per_day=3)
    replies: list[Any] = [
        # Content as a list of parts, plus a tool call and its result around it.
        [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": [{"type": "text", "text": "Red already? Nice."}]},
        ],
        say("Do you grow anything else?"),
    ]
    messages = converse(scenario_module, user, replies, turns=2)

    second = turn_payloads(provider)[1]
    assert second["visible_conversation"] == [
        {"from": "character", "text": "day 1 line 1 about the weather"},
        {"from": "companion", "text": "Red already? Nice."},
    ]
    transcript = user.transcript(messages)
    assert [t.companion_reply for t in transcript.turns] == [
        "Red already? Nice.",
        "Do you grow anything else?",
    ]


def test_a_scripted_user_message_ends_the_companion_reply() -> None:
    user = MoodsonaUser(make_seed(), FakeProvider(scheduled), turns_per_day=3)
    first = user.respond([])
    later = [
        {"role": "user", "content": first},
        say("Red already?"),
        {"role": "user", "content": "a line from the scenario script"},
        say("Tell me more."),
    ]
    user.respond(later)
    assert user.transcript().turns[0].companion_reply == "Red already?"


def test_a_persona_without_day_memory_writes_no_notes() -> None:
    user = MoodsonaUser(make_seed(memory_days_visible=0), FakeProvider(), turns_per_day=1)
    messages: list[dict[str, Any]] = []
    for _ in range(3):
        messages += [{"role": "user", "content": user.respond(messages)}, say("I see.")]
    transcript = user.transcript(messages)
    assert [t.day for t in transcript.turns] == [1, 2, 3]
    assert transcript.notes == []


def test_a_failed_model_call_can_be_retried() -> None:
    calls = 0

    def flaky(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls in (2, 4):  # day 1's notes, then day 2's first turn
            raise LLMError("timeout")
        return plumbing_reply(messages)

    user = MoodsonaUser(make_seed(), FakeProvider(flaky), turns_per_day=1)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user.respond([])}, say("Hm.")]
    for _ in range(2):
        with pytest.raises(LLMError):
            user.respond(messages)
    messages += [{"role": "user", "content": user.respond(messages)}, say("Ok.")]
    transcript = user.transcript(messages)
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (2, 1)]
    assert [n.day for n in transcript.notes] == [1]
    assert [t.companion_reply for t in transcript.turns] == ["Hm.", "Ok."]


def test_the_agent_may_speak_first(scenario_module: types.ModuleType) -> None:
    provider = FakeProvider(scheduled)
    user = user_simulator_agent(make_seed(), provider)
    opener = [{"role": "system", "content": "be kind"}, say("Hi! How was your week?")]
    text = asyncio.run(
        user.call(
            scenario_module.AgentInput(
                thread_id="t", messages=opener, new_messages=[], scenario_state=None
            )
        )
    )
    assert text == "day 1 line 1 about the weather"
    assert turn_payloads(provider)[0]["visible_conversation"] == [
        {"from": "companion", "text": "Hi! How was your week?"}
    ]


def test_transcript_and_metrics_after_the_scenario(
    scenario_module: types.ModuleType, tmp_path: Path
) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider(), turns_per_day=2, rng_seed=7)
    messages = converse(scenario_module, user, [say("I see."), say("Go on.")], turns=5)

    # Without the final messages the last reply is unknown.
    assert user.transcript().turns[-1].companion_reply == ""
    transcript = user.transcript(messages)
    assert transcript.turns[-1].companion_reply == "I see."
    assert transcript.persona_id == "test-gardener"
    assert (transcript.rng_seed, transcript.turns_per_day, transcript.days) == (7, 2, 3)
    assert [m.content for m in transcript.visible_history()] == [m["content"] for m in messages]
    # Asking for the transcript does not change the user's state.
    assert len(user.transcript().turns) == 5

    path = tmp_path / "scenario.jsonl"
    transcript.to_jsonl(path)
    assert Transcript.from_jsonl(path) == transcript
    from_file = metrics_from_jsonl(path)
    metrics = user.metrics(messages)
    assert metrics == from_file
    assert (metrics.turns, metrics.active_days) == (5, 3)


def test_the_transcript_is_empty_before_the_first_call() -> None:
    user = MoodsonaUser(make_seed(), FakeProvider())
    transcript = user.transcript()
    assert (transcript.turns, transcript.days, user.day) == ([], None, 0)


def test_reusing_the_agent_for_another_conversation_is_an_error() -> None:
    user = MoodsonaUser(make_seed(), FakeProvider())
    first = user.respond([])
    user.respond([{"role": "user", "content": first}, say("Hm.")])
    with pytest.raises(ValueError, match="one agent per"):
        user.respond([])  # a new, shorter conversation


def test_the_agent_is_bound_to_one_scenario_thread(scenario_module: types.ModuleType) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider())

    def ask(thread_id: str, messages: list[dict[str, Any]]) -> str:
        agent_input = scenario_module.AgentInput(
            thread_id=thread_id, messages=messages, new_messages=[], scenario_state=None
        )
        return asyncio.run(user.call(agent_input))

    first = ask("run-1", [])
    so_far = [{"role": "user", "content": first}, say("Hm.")]
    ask("run-1", so_far)
    longer = [*so_far, {"role": "user", "content": "x"}, say("y")]
    with pytest.raises(ValueError, match="thread"):
        ask("run-2", longer)  # longer than before, but another conversation


def test_a_blank_persona_message_keeps_the_conversation_in_step() -> None:
    # Every attempt fails to parse, so each turn falls back to an empty line.
    config = SimulatorConfig(fallback_message="")
    user = MoodsonaUser(make_seed(), FakeProvider(lambda _: "not a turn"), config=config)
    messages: list[dict[str, Any]] = []
    replies = ("Hello?", "Are you there?", "Take your time.")
    for reply in replies:
        messages += [{"role": "user", "content": user.respond(messages)}, say(reply)]
    turns = user.transcript(messages).turns
    assert [t.companion_reply for t in turns] == list(replies)
    assert all(t.polluted and t.visible == "" for t in turns)


def test_call_hands_respond_a_snapshot_of_the_messages(
    scenario_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider())
    seen: list[Sequence[Any]] = []

    def fake_respond(messages: Sequence[Any], *, thread_id: str | None = None) -> str:
        seen.append(messages)
        return "hi"

    monkeypatch.setattr(user, "respond", fake_respond)
    agent_input = scenario_module.AgentInput(
        thread_id="t", messages=[say("Hello")], new_messages=[], scenario_state=None
    )
    asyncio.run(user.call(agent_input))
    assert seen == [agent_input.messages]
    assert seen[0] is not agent_input.messages


def test_days_covers_skipped_days() -> None:
    turns = 0

    def skip_then_fail(messages: Sequence[ChatMessage]) -> str:
        nonlocal turns
        if messages[0].content.startswith(NOTES_MARKER):
            return plumbing_reply(messages)
        turns += 1
        if turns > 1:  # the first turn after the skipped day
            raise LLMError("timeout")
        return block("see you in two days", intent="skip_tomorrow")

    user = MoodsonaUser(make_seed(), FakeProvider(skip_then_fail), turns_per_day=3)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user.respond([])}, say("Bye!")]
    with pytest.raises(LLMError):
        user.respond(messages)
    transcript = user.transcript(messages)
    assert (transcript.skipped_days, transcript.days) == ([2], 2)


def test_the_agent_class_attribute_needs_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scenario", None)  # blocks `import scenario`
    adapter = moodsona.adapters.scenario
    name = "MoodsonaUserAgent"
    assert not hasattr(adapter, name)
    with pytest.raises(AttributeError, match=r"moodsona\[scenario\]") as info:
        getattr(adapter, name)
    assert isinstance(info.value.__cause__, ImportError)
    missing = "NoSuchName"
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(adapter, missing)


def test_days_counts_only_days_with_turns() -> None:
    calls = 0

    def fails_on_day_2(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:  # day 2's first turn, after day 1's notes
            raise LLMError("timeout")
        return plumbing_reply(messages)

    user = MoodsonaUser(make_seed(), FakeProvider(fails_on_day_2), turns_per_day=1)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user.respond([])}, say("Hm.")]
    with pytest.raises(LLMError):
        user.respond(messages)
    assert user.transcript(messages).days == 1


def test_agents_share_one_class_and_pickle(scenario_module: types.ModuleType) -> None:
    first = user_simulator_agent(make_seed(), FakeProvider())
    second = user_simulator_agent(make_seed(), FakeProvider())
    assert type(first) is type(second)
    first.respond([])
    copy = pickle.loads(pickle.dumps(first))  # noqa: S301 - our own bytes, just written
    assert type(copy) is type(first)
    assert copy.simulator.state == first.simulator.state


def test_turns_per_day_must_be_positive(scenario_module: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="turns_per_day"):
        user_simulator_agent(make_seed(), FakeProvider(), turns_per_day=0)


def test_module_imports_without_scenario_and_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "scenario", None)  # blocks `import scenario`
    # Re-importing rebinds the package attribute too; have monkeypatch restore both.
    monkeypatch.setattr(moodsona.adapters, "scenario", sys.modules["moodsona.adapters.scenario"])
    monkeypatch.delitem(sys.modules, "moodsona.adapters.scenario")
    adapter = importlib.import_module("moodsona.adapters.scenario")
    with pytest.raises(ImportError, match=r"moodsona\[scenario\]"):
        adapter.user_simulator_agent(make_seed(), FakeProvider())


def test_an_unrelated_scenario_module_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scenario", types.ModuleType("scenario"))
    with pytest.raises(ImportError, match="langwatch-scenario"):
        user_simulator_agent(make_seed(), FakeProvider())
