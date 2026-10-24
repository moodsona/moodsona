"""Language profiles: everything in moodsona that depends on the language.

Use :func:`get_profile` to look a profile up by its BCP-47 tag::

    from moodsona.lang import get_profile

    english = get_profile("en")
    english.is_ask_back("fine, you?")  # True

v0.1 ships English only. A new language is one module with a class that
satisfies :class:`LanguageProfile`, plus one entry in ``_REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Final

from moodsona.lang.base import LanguageProfile
from moodsona.lang.english import EnglishProfile, NoiseRates

__all__ = [
    "EnglishProfile",
    "LanguageProfile",
    "NoiseRates",
    "available_languages",
    "get_profile",
]

_REGISTRY: Final[MappingProxyType[str, Callable[[], LanguageProfile]]] = MappingProxyType(
    {"en": EnglishProfile}
)


def available_languages() -> tuple[str, ...]:
    """Return the language tags that have a profile, sorted.

    Returns:
        The registered BCP-47 primary language tags, for example ``("en",)``.
    """
    return tuple(sorted(_REGISTRY))


def get_profile(code: str) -> LanguageProfile:
    """Return a fresh language profile with default settings.

    The tag is matched case-insensitively and ``_`` is accepted for ``-``. A
    regional tag falls back to its language: ``"en-US"`` and ``"en_GB"`` give
    the English profile.

    Args:
        code: BCP-47 language tag, for example ``"en"`` or ``"en-US"``.

    Returns:
        A new profile instance for that language.

    Raises:
        LookupError: If no profile covers ``code``; the message lists the
            available tags.
    """
    normalised = code.strip().replace("_", "-").lower()
    factory = _REGISTRY.get(normalised) or _REGISTRY.get(normalised.split("-", 1)[0])
    if factory is None:
        available = ", ".join(available_languages())
        raise LookupError(f"No language profile for {code!r}. Available: {available}.")
    return factory()
