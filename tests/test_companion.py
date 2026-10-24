from __future__ import annotations

import pytest

from moodsona.companion import Companion, EchoCompanion, ScriptedCompanion, ScriptExhaustedError
from moodsona.llm.base import ChatMessage

HISTORY = [
    ChatMessage("user", "hi"),
    ChatMessage("assistant", "hello"),
    ChatMessage("user", "rainy day"),
]


def test_echo_companion() -> None:
    assert EchoCompanion().reply(HISTORY) == "You said: rainy day"
    assert EchoCompanion(prefix="").reply(HISTORY) == "rainy day"
    assert EchoCompanion().reply([]) == "You said: "
    assert isinstance(EchoCompanion(), Companion)


def test_scripted_companion_records_histories() -> None:
    companion = ScriptedCompanion(["a", "b"])
    assert companion.reply(HISTORY[:1]) == "a"
    assert companion.reply(HISTORY) == "b"
    assert companion.histories == [tuple(HISTORY[:1]), tuple(HISTORY)]
    with pytest.raises(ScriptExhaustedError, match="call 3"):
        companion.reply(HISTORY)


def test_scripted_companion_can_cycle() -> None:
    companion = ScriptedCompanion(["a", "b"], cycle=True)
    assert [companion.reply([]) for _ in range(5)] == ["a", "b", "a", "b", "a"]
    with pytest.raises(ValueError, match="at least one"):
        ScriptedCompanion([])


def test_public_api() -> None:
    import moodsona

    assert moodsona.__version__
    assert set(moodsona.__all__) <= set(dir(moodsona))
