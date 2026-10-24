"""moodsona: an emotional user simulator for testing AI companion chatbots.

Quick start (offline)::

    from moodsona import EchoCompanion, FakeProvider, load_bundled_persona, run

    persona = load_bundled_persona("anxious-newcomer")
    transcript = run(persona, FakeProvider(), EchoCompanion(), days=2, turns_per_day=3)
    for point in transcript.trajectory():
        print(point)

Swap ``FakeProvider()`` for ``AnthropicProvider()`` or ``OpenAIProvider()``
(from ``moodsona.llm.anthropic`` / ``moodsona.llm.openai``) and ``EchoCompanion``
for your own chatbot wrapped in the :class:`Companion` protocol.
"""

from importlib.metadata import PackageNotFoundError, version

from moodsona.companion import Companion, EchoCompanion, ScriptedCompanion, ScriptExhaustedError
from moodsona.llm import ChatMessage, FakeProvider, LLMError, LLMProvider, LLMResponse
from moodsona.parse import HiddenBlock, ParseError
from moodsona.personas import (
    PersonaError,
    list_bundled_personas,
    load_bundled_persona,
    load_persona,
    persona_from_mapping,
)
from moodsona.runner import TrajectoryPoint, Transcript, TurnRecord, run
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig, SimulatorTurn
from moodsona.state import (
    DayNote,
    Intent,
    Mood,
    PersonaSeed,
    PersonaState,
    Pronouns,
    ReactionRule,
    Stance,
)

try:
    __version__ = version("moodsona")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "ChatMessage",
    "Companion",
    "DayNote",
    "EchoCompanion",
    "FakeProvider",
    "HiddenBlock",
    "Intent",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Mood",
    "ParseError",
    "PersonaError",
    "PersonaSeed",
    "PersonaState",
    "Pronouns",
    "ReactionRule",
    "SafetyHook",
    "ScriptExhaustedError",
    "ScriptedCompanion",
    "Simulator",
    "SimulatorConfig",
    "SimulatorTurn",
    "Stance",
    "TrajectoryPoint",
    "Transcript",
    "TurnRecord",
    "__version__",
    "list_bundled_personas",
    "load_bundled_persona",
    "load_persona",
    "persona_from_mapping",
    "run",
]
