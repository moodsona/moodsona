"""Tests for the langwatch/scenario adapter.

The adapter is a :class:`~moodsona.user.SimulatedUser` with scenario's
``call()``; how replies, days and transcripts behave is tested in
``test_user.py``. These tests cover the scenario side: the class, the call
path, thread binding, pickling and the import without scenario.

They run against the real ``scenario`` package when it is installed (the
``moodsona[scenario]`` extra) and otherwise against a minimal stub of the three
names they use. No test calls a model: the user model is a FakeProvider and the
agent under test is a list of fixed replies.
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

import moodsona.adapters
from moodsona.adapters.scenario import user_simulator_agent
from moodsona.llm.base import ChatMessage, LLMError
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.metrics import metrics_from_jsonl
from moodsona.prompts import NOTES_MARKER
from moodsona.runner import Transcript
from moodsona.user import SimulatedUser
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


def say(text: str) -> dict[str, Any]:
    """An agent (companion) message."""
    return {"role": "assistant", "content": text}


def said(text: str) -> dict[str, Any]:
    """A user message, as scenario adds the persona's line to the conversation."""
    return {"role": "user", "content": text}


def agent_input(
    module: types.ModuleType, messages: list[dict[str, Any]], thread_id: str = "t"
) -> Any:
    """The module's ``AgentInput`` for ``messages``."""
    return module.AgentInput(
        thread_id=thread_id, messages=messages, new_messages=[], scenario_state=None
    )


def ask(module: types.ModuleType, user: Any, messages: list[dict[str, Any]], thread: str) -> str:
    """One ``call``, as scenario makes it."""
    text: str = asyncio.run(user.call(agent_input(module, messages, thread)))
    return text


def converse(
    module: types.ModuleType, user: Any, replies: Sequence[str], turns: int
) -> list[dict[str, Any]]:
    """Drive the user like scenario does: ``call``, then the agent's reply."""
    messages: list[dict[str, Any]] = []

    async def go() -> None:
        for i in range(turns):
            messages.append(said(await user.call(agent_input(module, list(messages)))))
            messages.append(say(replies[i % len(replies)]))

    asyncio.run(go())
    return messages


# -- tests ------------------------------------------------------------------


def test_agent_is_a_simulated_user_in_the_user_role(scenario_module: types.ModuleType) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider())
    assert isinstance(user, scenario_module.AgentAdapter)
    assert isinstance(user, SimulatedUser)
    agent_class: Any = type(user)  # `role` comes from scenario's base class
    assert agent_class.role is scenario_module.AgentRole.USER


def test_calls_return_only_the_visible_message_and_keep_the_state(
    scenario_module: types.ModuleType,
) -> None:
    provider = FakeProvider(scheduled)
    seed = make_seed()
    user = user_simulator_agent(seed, provider, turns_per_day=3, rng_seed=1)
    messages = converse(scenario_module, user, ["Mm, tell me more."], turns=5)

    sent = [m["content"] for m in messages if m["role"] == "user"]
    assert sent[0] == "day 1 line 1 about the weather"
    joined = "\n".join(sent)
    for hidden in ("secret thought", "<state>", "<message>"):
        assert hidden not in joined

    # Day 1 ends early (end_day at turn 2), day 2 skips day 3 at turn 1.
    transcript = user.transcript(messages)
    assert [(t.day, t.turn) for t in transcript.turns] == [(1, 1), (1, 2), (2, 1), (4, 1), (4, 2)]
    assert transcript.skipped_days == [3]
    assert [n.day for n in transcript.notes] == [1, 2]
    assert user.day == 4
    assert user.state.mood == transcript.turns[-1].mood
    assert transcript.turns[-1].mood.emotion > seed.initial_state.emotion

    # Day 2 reads day 1's notes and starts with an empty conversation of its own.
    day2 = next(p for p in turn_payloads(provider) if p["day"] == 2)
    assert [entry["day"] for entry in day2["memory_log"]] == [1]
    assert day2["visible_conversation"] == []


def test_transcript_metrics_and_finish_after_the_scenario(
    scenario_module: types.ModuleType, tmp_path: Path
) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider(), turns_per_day=2, rng_seed=7)
    messages = converse(scenario_module, user, ["I see.", "Go on."], turns=5)

    # Without the final messages the last reply is unknown.
    assert user.transcript().turns[-1].companion_reply == ""
    transcript = user.transcript(messages)
    assert transcript.turns[-1].companion_reply == "I see."
    assert transcript.persona_id == "test-gardener"
    assert (transcript.rng_seed, transcript.turns_per_day, transcript.days) == (7, 2, 3)
    assert [m.content for m in transcript.visible_history()] == [m["content"] for m in messages]

    path = tmp_path / "scenario.jsonl"
    transcript.to_jsonl(path)
    assert Transcript.from_jsonl(path) == transcript
    metrics = user.metrics(messages)
    assert metrics == metrics_from_jsonl(path)
    assert (metrics.turns, metrics.active_days) == (5, 3)

    # finish() also writes the last day's notes.
    finished = user.finish(messages)
    assert finished.turns == transcript.turns
    assert [n.day for n in finished.notes] == [1, 2, 3]
    assert user.done


def test_the_agent_is_bound_to_one_scenario_thread(scenario_module: types.ModuleType) -> None:
    user = user_simulator_agent(make_seed(), FakeProvider())
    so_far = [said(ask(scenario_module, user, [], "run-1")), say("Hm.")]
    ask(scenario_module, user, so_far, "run-1")
    longer = [*so_far, said("x"), say("y")]
    with pytest.raises(ValueError, match="thread"):
        ask(scenario_module, user, longer, "run-2")  # longer than before, but another run


def test_a_failed_first_call_does_not_bind_the_thread(scenario_module: types.ModuleType) -> None:
    calls = 0

    def fails_once(messages: Sequence[ChatMessage]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMError("timeout")
        return plumbing_reply(messages)

    user = user_simulator_agent(make_seed(), FakeProvider(fails_once))
    with pytest.raises(LLMError):
        ask(scenario_module, user, [], "run-1")
    # The failed run left no turn behind, so another run may use the agent.
    so_far = [said(ask(scenario_module, user, [], "run-2")), say("Hm.")]
    ask(scenario_module, user, so_far, "run-2")
    with pytest.raises(ValueError, match="'run-2'"):
        ask(scenario_module, user, [*so_far, said("x"), say("y")], "run-1")


def test_call_hands_reply_a_snapshot_of_the_messages(
    scenario_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    user: Any = user_simulator_agent(make_seed(), FakeProvider())
    seen: list[Sequence[Any]] = []

    def fake_reply(history: Sequence[Any]) -> str:
        seen.append(history)
        return "hi"

    monkeypatch.setattr(user, "reply", fake_reply)
    live = agent_input(scenario_module, [say("Hello")])
    assert asyncio.run(user.call(live)) == "hi"
    assert seen == [live.messages]
    assert seen[0] is not live.messages


def test_agents_share_one_class_and_pickle(scenario_module: types.ModuleType) -> None:
    first = user_simulator_agent(make_seed(), FakeProvider())
    second = user_simulator_agent(make_seed(), FakeProvider())
    assert type(first) is type(second)
    first.reply([])
    copy = pickle.loads(pickle.dumps(first))  # noqa: S301 - our own bytes, just written
    assert type(copy) is type(first)
    assert copy.state == first.state


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
