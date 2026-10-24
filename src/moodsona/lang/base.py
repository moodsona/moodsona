"""The language profile protocol.

A language profile holds everything in moodsona that depends on the language a
simulated user writes in: spotting a reply in the wrong language, spotting a
bare agreement or a reciprocal question, and adding the typing noise of a
person on a phone keyboard. The simulator core talks only to this protocol, so
adding a language means adding one profile and registering it in
:mod:`moodsona.lang`.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

__all__ = ["LanguageProfile"]


@runtime_checkable
class LanguageProfile(Protocol):
    """Language-specific heuristics used by the simulator.

    Attributes:
        code: BCP-47 language tag the profile covers, for example ``"en"``.
    """

    code: str

    def is_wrong_language(self, text: str) -> bool:
        """Tell whether ``text`` is written in a different language.

        Args:
            text: A message written by the simulated user.

        Returns:
            True when the message is clearly not in this profile's language.
            Text without letters (emoji, numbers, punctuation, empty) is never
            the wrong language.
        """
        ...

    def is_agreement(self, text: str) -> bool:
        """Tell whether ``text`` is only a short affirmative reply.

        Args:
            text: A message written by the simulated user.

        Returns:
            True for replies such as "yeah" or "makes sense" that carry nothing
            but agreement; False for negations and for longer messages.
        """
        ...

    def is_ask_back(self, text: str) -> bool:
        """Tell whether ``text`` ends with a reciprocal question.

        Args:
            text: A message written by the simulated user.

        Returns:
            True when the message ends by turning the question back to the
            other side ("and you?", "what about you?").
        """
        ...

    def add_typing_noise(self, text: str, rng: random.Random, level: float) -> str:
        """Return ``text`` with messenger-style typing noise.

        Args:
            text: A clean message.
            rng: Source of randomness; the same seed gives the same output.
            level: Noise intensity in ``[0, 1]``; ``0`` returns ``text`` as is.

        Returns:
            The noisy message.
        """
        ...
