"""Checks on a simulated user's message before it reaches the companion.

A simulation is only evidence while the simulator keeps writing as the
fictional user. Models sometimes break role instead: they state their own
identity, refuse, or talk about the task. Those lines must never enter the
transcript as if a person had sent them, so the simulator retries them.

The patterns are deliberately narrow. A simulated user may well say "I can't
sleep" or ask "are you ChatGPT?"; only the simulator claiming an assistant
identity for itself, refusing the task, or leaking the scaffold counts.
Language-specific checks (wrong language, reflexive agreement, asking back)
live in the language profile, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Straight or typographic apostrophe.
_APOS = "['\u2019]"
_VENDORS = (
    r"claude|anthropic|openai|chatgpt|gpt-?\d[\w.-]*|gemini|grok|xai|llama|deepseek|qwen|mistral"
)

_IDENTITY = re.compile(
    r"\bas\s+an?\s+(?:ai|artificial\s+intelligence|assistant|language\s+model|chatbot)\b"
    rf"|\bi\s*(?:am|{_APOS}m)\s+(?:just\s+|only\s+)?(?:an?\s+)?"
    r"(?:ai\b|ai\s+(?:model|assistant)|artificial\s+intelligence|(?:large\s+)?language\s+model"
    r"|virtual\s+assistant|chatbot)"
    # "I'm Claude." / "I am Gemini, an AI ..." but not "I'm Gemini so I overthink"
    # (astrology) or "I'm Grok-level tired": the name must end the clause or be
    # followed by a word that makes it an assistant.
    rf"|\bi\s*(?:am|{_APOS}m)\s+(?:{_VENDORS})"
    r"(?=\s*(?:[.!?,;:]|$)|\s+(?:ai|assistant|(?:language\s+)?model|chatbot|bot)\b)"
    rf"|\b(?:made|created|trained|developed|built)\s+by\s+(?:{_VENDORS})\b",
    re.IGNORECASE | re.MULTILINE,
)

_REFUSAL = re.compile(
    rf"\bi\s+(?:can(?:not|{_APOS}t| not)|won{_APOS}t|will\s+not|am\s+not\s+able\s+to)\s+"
    r"(?:help\s+with\s+(?:that|this)|assist\s+with|continue\s+(?:this|with\s+this)"
    r"|comply|role-?play|pretend\s+to\s+be|play\s+(?:this|that)\s+(?:role|character)"
    r"|write\s+(?:this|that)\s+(?:message|line))"
    r"|\bi\s+must\s+decline\b"
    rf"|\bi(?:{_APOS}m|\s+am)\s+sorry,?\s+but\s+i\s+(?:can|won|will|am)",
    re.IGNORECASE,
)

# Scaffold vocabulary: payload keys and talk about the prompt or the fiction.
_META = re.compile(
    r"\b(?:inner_thought|hidden_goal|reaction_rules|visible_conversation|memory_log"
    r"|blocked_openers|rules_triggered|turn_in_day)\b"
    r"|<\s*/?\s*(?:state|message)\s*>"
    r"|\bsystem\s+prompt\b|\bthe\s+(?:prompt|persona|hidden\s+state)\b"
    r"|\bthis\s+(?:role-?play|simulation|fiction)\b|\b(?:fictional|simulated)\s+(?:character|user)\b"
    r"|\bin\s+character\b",
    re.IGNORECASE,
)

ROLE_BREAK_KINDS = ("empty_output", "identity_claim", "task_refusal", "meta_content")


def find_role_break(text: str) -> str | None:
    """Return the kind of the first role break in ``text``, or None.

    Args:
        text: A simulated user's visible message (or notes).

    Returns:
        One of ``ROLE_BREAK_KINDS`` or None when the text is fine.
    """
    if not text.strip():
        return "empty_output"
    if _IDENTITY.search(text):
        return "identity_claim"
    if _REFUSAL.search(text):
        return "task_refusal"
    if _META.search(text):
        return "meta_content"
    return None


_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> set[str]:
    """The set of lower-case words in ``text``."""
    return {w.lower() for w in _WORD.findall(text)}


def repeats_earlier(
    text: str, earlier: Iterable[str], *, threshold: float = 0.7, min_words: int = 3
) -> bool:
    """Tell whether ``text`` nearly repeats one of the ``earlier`` lines.

    Similarity is the Jaccard index of the two word sets; lines shorter than
    ``min_words`` words are never counted as repeats ("ok", "yeah sure").
    """
    mine = words(text)
    if len(mine) < min_words:
        return False
    for line in earlier:
        other = words(line)
        if other and len(mine & other) / len(mine | other) >= threshold:
            return True
    return False


def opener(text: str, *, max_words: int = 3) -> str | None:
    """The first few words of a message, lower-cased, for the anti-mirroring list."""
    found = _WORD.findall(text.lower())[:max_words]
    return " ".join(found) if found else None
