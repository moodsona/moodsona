"""A moodsona persona as the user simulator of a langwatch/scenario test (experimental).

Install with ``pip install 'moodsona[scenario]'``. Build the agent with
:func:`user_simulator_agent` and put it in ``scenario.run(agents=[...])`` in
place of scenario's own user simulator::

    user = user_simulator_agent(load_bundled_persona("anxious-newcomer"), provider)
    result = await scenario.run(
        name="first week", description="...", agents=[MyCompanion(), user, judge]
    )
    transcript = user.finish(result.messages)

The agent is a :class:`~moodsona.user.SimulatedUser` with scenario's
``call()``: each call returns ``user.reply(input.messages)``, so days, hidden
state, ``transcript()``, ``metrics()`` and ``finish()`` work as described in
``docs/simulated-user.md``; ``docs/integrations/scenario.md`` covers the
scenario side. The ``scenario`` package is imported only by
:func:`user_simulator_agent`; importing this module does not need it.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import types
from typing import TYPE_CHECKING, Any, cast

from moodsona.llm.base import LLMProvider
from moodsona.simulator import SafetyHook, SimulatorConfig
from moodsona.state import PersonaSeed
from moodsona.user import SimulatedUser

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

__all__ = ["DEFAULT_TURNS_PER_DAY", "user_simulator_agent"]

#: Persona messages per moodsona day unless ``turns_per_day`` says otherwise.
DEFAULT_TURNS_PER_DAY = 4

_INSTALL = "install it with: pip install 'moodsona[scenario]'"

#: Name of the class :func:`user_simulator_agent` builds (an attribute of this
#: module once scenario is installed).
_AGENT_CLASS = "MoodsonaUserAgent"


class _ScenarioUser(SimulatedUser):
    """A :class:`SimulatedUser` with scenario's ``call()``; needs no scenario installed."""

    #: The scenario thread this persona talks in, from its first successful reply.
    _thread_id: str | None = None

    async def call(self, input: Any) -> str:
        """Return the persona's next visible message (scenario's entry point).

        :meth:`~moodsona.user.SimulatedUser.reply` makes blocking model calls;
        it runs in a worker thread, on a copy of the messages, so the event
        loop stays free. Cancelling the call from outside (a test timeout, for
        example) does not stop a model call that has already started; the
        thread finishes it in the background.

        The first successful reply binds the agent to ``input.thread_id``; a
        failed first call binds nothing, so the agent stays usable.

        Args:
            input: scenario's ``AgentInput``. Only ``input.messages``, the
                conversation so far as OpenAI-style messages, and
                ``input.thread_id`` are read.

        Returns:
            The message text; scenario adds it to the conversation as a user message.

        Raises:
            ValueError: The agent is already in another scenario run.
            moodsona.llm.LLMError: The model failed.
        """
        thread_id = input.thread_id
        if self._thread_id is not None and thread_id != self._thread_id:
            raise ValueError(
                f"this persona is in scenario thread {self._thread_id!r}, not {thread_id!r}; "
                "use one agent per scenario run"
            )
        text = await asyncio.to_thread(self.reply, list(input.messages))
        self._thread_id = thread_id
        return text


def user_simulator_agent(
    persona: PersonaSeed,
    model: LLMProvider,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    rng_seed: int = 0,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
    config: SimulatorConfig | None = None,
) -> SimulatedUser:
    """A scenario ``AgentAdapter`` with the user role that plays ``persona``.

    Pass it to ``scenario.run(agents=[...])`` instead of scenario's
    ``UserSimulatorAgent``. Use one agent per scenario run: it keeps the
    persona's state from call to call. The arguments are those of
    :class:`~moodsona.user.SimulatedUser`; there is no ``days`` limit, because
    scenario decides when the run stops.

    Args:
        persona: The persona.
        model: The model that writes the persona's lines.
        turns_per_day: Maximum persona messages per moodsona day.
        rng_seed: Seed for overnight energy and typing noise.
        language: Language profile; None (the default) uses the persona's
            ``language`` (see :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector (see :class:`~moodsona.simulator.Simulator`).
        config: Optional simulator settings.

    Returns:
        An instance of a subclass of both :class:`~moodsona.user.SimulatedUser`
        and ``scenario.AgentAdapter``.

    Raises:
        ImportError: langwatch-scenario is not installed.
        ValueError: ``turns_per_day`` is below 1.
        LookupError: No language profile covers the persona's language.
    """
    return _agent_class()(
        persona,
        model,
        turns_per_day=turns_per_day,
        rng_seed=rng_seed,
        language=language,
        safety_hook=safety_hook,
        config=config,
    )


def _agent_class() -> type[_ScenarioUser]:
    """The scenario-facing class, built once per scenario base class; imports scenario."""
    try:
        # Attribute access on the imported module rather than `from scenario import
        # ...`: mypy then needs neither scenario's package nor an ignore.
        scenario = importlib.import_module("scenario")
        base, role = scenario.AgentAdapter, scenario.AgentRole.USER
    except (ImportError, AttributeError) as exc:  # AttributeError: an unrelated `scenario`
        raise ImportError(
            f"moodsona's scenario adapter needs langwatch-scenario; {_INSTALL}"
        ) from exc
    return _build_agent_class(base, role)


@functools.cache
def _build_agent_class(base: type, role: Any) -> type[_ScenarioUser]:
    def body(namespace: dict[str, Any]) -> None:
        namespace["__module__"] = __name__
        namespace["__doc__"] = "A :class:`SimulatedUser` that scenario runs as its user simulator."
        namespace["role"] = role

    # new_class rather than a class statement: mypy cannot check a base class
    # that is only known at run time, and this way needs no type: ignore.
    agent_class = types.new_class(_AGENT_CLASS, (_ScenarioUser, base), exec_body=body)
    return cast("type[_ScenarioUser]", agent_class)


if not TYPE_CHECKING:  # hidden from mypy, so a misspelt import still fails type checking

    def __getattr__(name: str) -> Any:
        """Resolve ``MoodsonaUserAgent`` on demand, so pickle can find the class."""
        if name == _AGENT_CLASS:
            try:
                return _agent_class()
            except ImportError as exc:
                message = f"{__name__}.{name} needs langwatch-scenario; {_INSTALL}"
                raise AttributeError(message) from exc
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
