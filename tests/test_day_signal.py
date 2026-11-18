"""Every message the companion sees carries its simulated day.

No test calls a model: the user model is a FakeProvider, the companion a test
double and the provider SDKs stand-in clients.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from moodsona.companion import Companion, ScriptedCompanion
from moodsona.llm.anthropic import AnthropicProvider
from moodsona.llm.base import ChatMessage, LLMProvider
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.llm.openai import OpenAIProvider
from moodsona.prompts import NOTES_MARKER
from moodsona.runner import Transcript, run
from moodsona.user import SimulatedUser
from support import block, make_seed

#: Lines different enough that the own-repeat rail never fires within a day.
TOPICS = (
    "the tomatoes split overnight",
    "rain flattened the beans",
    "a neighbour borrowed my spade",
    "found slugs under the boards",
)
INTENTS = ("reply", "end_day", "skip_tomorrow")
#: Day 2 ends at turn 2; day 3 ends at turn 1 and skips day 4.
SCHEDULE = {(2, 2): "end_day", (3, 1): "skip_tomorrow"}


def scheduled(intents: Mapping[tuple[int, int], str]) -> Any:
    """Valid turns with the given intent per (day, turn), "reply" by default."""

    def respond(messages: Sequence[ChatMessage]) -> str:
        if messages[0].content.startswith(NOTES_MARKER):
            return plumbing_reply(messages)
        payload = json.loads(messages[1].content)
        day, turn = payload["day"], payload["turn_in_day"]
        return block(
            TOPICS[(turn - 1) % len(TOPICS)],
            intent=intents.get((day, turn), "reply"),
            emotion=payload["state"]["emotion"],
            energy=payload["state"]["energy"],
        )

    return respond


def simulate(provider: LLMProvider, companion: Companion, days: int, turns: int) -> Transcript:
    """The one call of run() in this file, so a change of its signature is one line."""
    return run(make_seed(), provider, companion, days=days, turns_per_day=turns)


def assert_days_are_consistent(history: Sequence[ChatMessage], transcript: Transcript) -> None:
    """``history`` alternates user and assistant messages, each pair tagged with its turn's day."""
    days = [m.day for m in history]
    assert all(isinstance(day, int) and day >= 1 for day in days)
    assert days == sorted(days)  # type: ignore[type-var]
    assert not set(days) & set(transcript.skipped_days)
    pairs = [(m.role, m.day) for m in history]
    expected = [(role, t.day) for t in transcript.turns for role in ("user", "assistant")]
    assert pairs == expected[: len(pairs)]
    # A new day starts exactly where the transcript starts a new day.
    starts = [i for i in range(1, len(history)) if history[i].day != history[i - 1].day]
    assert all(history[i].role == "user" for i in starts)
    assert all(history[i].day > history[i - 1].day for i in starts)  # type: ignore[operator]


# -- ChatMessage --------------------------------------------------------------


def test_day_defaults_to_none_and_takes_part_in_equality() -> None:
    message = ChatMessage("user", "x")
    assert message.day is None
    assert ChatMessage("user", "x", day=2).day == 2
    assert message != ChatMessage("user", "x", day=1)
    assert message == ChatMessage("user", "x", None)


# -- providers never send the tag -----------------------------------------------


class _Client:
    """Stands in for both SDK clients and records each request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = self
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        text = SimpleNamespace(type="text", text="ok")
        return SimpleNamespace(
            content=[text], choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


@pytest.mark.parametrize("adapter", [AnthropicProvider, OpenAIProvider])
def test_providers_send_only_role_and_content(adapter: Any) -> None:
    client = _Client()
    provider = adapter("test-model", client=client)
    provider.complete(
        [
            ChatMessage("system", "be brief", day=1),
            ChatMessage("user", "hi", day=1),
            ChatMessage("assistant", "hello", day=1),
            ChatMessage("user", "back again", day=2),
        ]
    )
    (request,) = client.requests
    assert "day" not in json.dumps(request)
    assert all(set(m) == {"role", "content"} for m in request["messages"])


# -- run() ----------------------------------------------------------------------


def test_run_tags_every_message_the_companion_sees_with_its_day() -> None:
    companion = ScriptedCompanion(["Mm, tell me more."], cycle=True)
    transcript = simulate(FakeProvider(scheduled(SCHEDULE)), companion, days=5, turns=3)
    assert transcript.skipped_days == [4]
    assert sorted({t.day for t in transcript.turns}) == [1, 2, 3, 5]

    for history in companion.histories:
        assert history[-1].role == "user"
        assert_days_are_consistent(history, transcript)
    # The companion learns of day 5 from the first message of day 5.
    final = companion.histories[-1]
    assert [m.day for m in final if m.role == "user"] == [t.day for t in transcript.turns]


def test_visible_history_carries_the_same_days_as_the_live_run() -> None:
    companion = ScriptedCompanion(["Mm, tell me more."], cycle=True)
    transcript = simulate(FakeProvider(scheduled(SCHEDULE)), companion, days=5, turns=3)

    replayed = transcript.visible_history()
    assert [m.day for m in replayed[::2]] == [t.day for t in transcript.turns]
    assert [m.day for m in replayed[1::2]] == [t.day for t in transcript.turns]
    assert_days_are_consistent(replayed, transcript)
    # Exactly what the companion saw, plus its own last reply.
    assert replayed[:-1] == list(companion.histories[-1])
    assert replayed[-1] == ChatMessage("assistant", "Mm, tell me more.", day=5)


# -- SimulatedUser ----------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    days=st.integers(1, 5),
    turns_per_day=st.integers(1, 3),
    intents=st.dictionaries(
        st.tuples(st.integers(1, 5), st.integers(1, 3)), st.sampled_from(INTENTS)
    ),
)
def test_user_day_after_reply_tags_the_turn_it_wrote(
    days: int, turns_per_day: int, intents: dict[tuple[int, int], str]
) -> None:
    provider = FakeProvider(scheduled(intents))
    user = SimulatedUser(make_seed(), provider, turns_per_day=turns_per_day, days=days)
    history: list[ChatMessage] = []
    while not user.done:
        history.append(ChatMessage("user", user.reply(history), day=user.day))
        history.append(ChatMessage("assistant", "I see.", day=user.day))
    transcript = user.finish(history)

    assert_days_are_consistent(history, transcript)
    assert len(history) == 2 * len(transcript.turns)


def loop(user: SimulatedUser, tag: str) -> Transcript:
    """The harness loop, with day-tagged, untagged or dict messages carrying a ``day`` key."""
    history: list[Any] = []
    while not user.done:
        text = user.reply(history)
        day = user.day
        for role, content in (("user", text), ("assistant", f"Heard: {text}")):
            if tag == "tagged":
                history.append(ChatMessage(role, content, day=day))  # type: ignore[arg-type]
            elif tag == "untagged":
                history.append(ChatMessage(role, content))  # type: ignore[arg-type]
            else:
                history.append({"role": role, "content": content, "day": day})
    return user.finish(history)


def test_simulated_user_reads_tagged_and_untagged_histories_alike() -> None:
    results = []
    for tag in ("untagged", "tagged", "dict"):
        provider = FakeProvider(scheduled(SCHEDULE))
        user = SimulatedUser(make_seed(), provider, turns_per_day=3, days=5, rng_seed=4)
        results.append((loop(user, tag), provider.calls))
    assert results[0] == results[1] == results[2]
    assert results[0][0].skipped_days == [4]
