"""English language profile.

Everything here is a heuristic tuned for chat messages a few words to a few
sentences long. The limits of each heuristic are written down next to it; the
tests pin the cases that are known to work.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, fields
from typing import Final

from moodsona.lang.noise_tables import APOSTROPHE_DROPS, QWERTY_NEIGHBOURS

__all__ = ["EnglishProfile", "NoiseRates"]

# Apostrophe look-alikes that phone keyboards and models produce.
_APOSTROPHES: Final = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})


def _straight_apostrophes(text: str) -> str:
    return text.translate(_APOSTROPHES)


# ------------------------------------------------------------ wrong language


def _script_counts(text: str) -> tuple[int, int]:
    """Count Latin and non-Latin letters in ``text``.

    The text is NFKC-normalised first, so styled letters (fullwidth, "math
    bold", the information sign) count as the plain Latin letters they look
    like. Letters without a Unicode name and modifier letters are ignored.
    """
    latin = other = 0
    for char in unicodedata.normalize("NFKC", text):
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if not name or name.startswith("MODIFIER"):
            continue
        if name.startswith("LATIN"):
            latin += 1
        else:
            other += 1
    return latin, other


# ----------------------------------------------------------------- agreement

# Patterns run on text normalised by `_agreement_text`: lowercase, apostrophes
# removed ("you're" -> "youre"), punctuation and emoji replaced by spaces.
_AGREEMENT_UNITS: Final = (
    r"ye+a+h*",
    r"ya+h?",
    r"ye+s+",
    r"ye+p+",
    r"yu+p+",
    r"o+k+",
    r"oka+y+",
    r"okey",
    r"k+",
    r"alright",
    r"all right",
    r"aight",
    r"sure",
    r"for sure",
    r"sure thing",
    r"of course",
    r"ofc",
    r"(?:so |very |too |thats (?:so )?|its )?true",
    r"(?:thats )?fair(?: enough| point)?",
    r"good point",
    r"valid",
    r"(?:that |it )?(?:totally |kinda |kind of )?makes? (?:total |a lot of |so much )?sense",
    r"(?:thats )?right",
    r"(?:youre|you are|you re|ur|u r|u are)"
    r"(?: (?:so|totally|absolutely|completely|probably|really|actually))? right",
    r"exactly",
    r"totally",
    r"definitely",
    r"absolutely",
    r"indeed",
    r"(?:thats )?correct",
    r"agreed",
    r"(?:i )?(?:totally |completely )?agree",
    r"same(?: here)?",
    r"me too",
    r"i (?:guess|suppose)(?: so)?",
    r"i think so",
    r"(?:guess|suppose) so",
    r"ig",
    r"sounds (?:good|right|fair)",
    r"got it",
    r"gotcha",
    r"understood",
    r"m+h+m+",
    r"uh huh",
    r"no doubt",
    r"(?:i know|ikr)(?: right)?",
    r"fr(?:fr)?",
    r"cool",
)
# Words that may surround an agreement without changing it ("oh yeah lol").
_FILLER_UNITS: Final = (
    r"lol",
    r"lmao",
    r"ha(?:ha)+h?",
    r"hah?",
    r"he(?:he)+",
    r"o+h+",
    r"a+h+",
    r"well",
    r"h+m+",
    r"u+m+",
    r"u+h+",
    r"i mean",
    r"tbh",
    r"honestly",
    r"actually",
    r"really",
    r"then",
    r"so",
    r"just",
    r"and",
    r"man",
    r"dude",
)
_AGREEMENT: Final = "|".join(_AGREEMENT_UNITS)
_UNIT: Final = f"(?:{_AGREEMENT}|{'|'.join(_FILLER_UNITS)})"
_ONLY_AGREEMENT: Final = re.compile(rf"{_UNIT}(?: {_UNIT})*")
_HAS_AGREEMENT: Final = re.compile(rf"(?:^| )(?:{_AGREEMENT})(?= |$)")
# Longer than this after normalisation is a message, not a reply.
_MAX_AGREEMENT_CHARS: Final = 60
_NOT_WORD: Final = re.compile(r"[^\w\s]|_")


def _agreement_text(text: str) -> str:
    lowered = _straight_apostrophes(text).casefold().replace("'", "")
    return " ".join(_NOT_WORD.sub(" ", lowered).split())


# ------------------------------------------------------------------ ask-back

_SKIPPED_TRAILING_CATEGORIES: Final = frozenset({"So", "Sk", "Cf", "Mn", "Me", "Co", "Cs"})
_EMOTICON_END: Final = re.compile(r"(?:[:;=][-']?[()\[\]dpo3*|/\\]+|<3+|\)+)$")
_ASK_LEAD: Final = r"(?:^|(?P<boundary>[.,!?;:\u2026\u2013\u2014-])\s*|\s)"
_ASK_TAIL: Final = (
    r"(?:\s+(?:then|tho|though|lol|lmao|haha+|today|tonight))?\s*(?P<mark>[?!.\u2026]*)$"
)
_TIME_WORD: Final = r"(?:day|week|night|weekend|evening|morning)"
# Unmistakable on their own, wherever they stand.
_ASK_STRONG: Final = re.compile(
    _ASK_LEAD
    + r"(?P<phrase>hbu|wbu|hby|wby|hru"
    + r"|(?:what|how) about (?:you|u|ya|yourself|urself|yours|(?:your|ur) "
    + _TIME_WORD
    + r")"
    + r"|how(?:re| are| r| have|ve)? (?:you|u|ya)(?: (?:doing|feeling|holding up|been))?"
    + r"|how(?:s| is| was) (?:your|ur) "
    + _TIME_WORD
    + r"(?: going| been)?"
    + r"|how was yours)"
    + _ASK_TAIL
)
# "and you": needs a question mark or a clause boundary before it.
_ASK_AND: Final = re.compile(
    _ASK_LEAD + r"(?P<phrase>(?:and|n) (?:you|u|ya|yourself|urself|yours))" + _ASK_TAIL
)
# A bare "you?": needs both, so "thank you?" does not count.
_ASK_BARE: Final = re.compile(_ASK_LEAD + r"(?P<phrase>you|u|ya|yourself|urself|yours)" + _ASK_TAIL)


def _strip_trailing_decoration(text: str) -> str:
    """Remove trailing whitespace, emoji and emoticons such as ``:)``."""
    while True:
        end = len(text)
        while end and (
            text[end - 1].isspace()
            or unicodedata.category(text[end - 1]) in _SKIPPED_TRAILING_CATEGORIES
        ):
            end -= 1
        stripped = _EMOTICON_END.sub("", text[:end])
        if stripped == text:
            return text
        text = stripped


def _at_clause_start(match: re.Match[str]) -> bool:
    return match["boundary"] is not None or match.start("phrase") == 0


# -------------------------------------------------------------------- noise


@dataclass(frozen=True, slots=True)
class NoiseRates:
    """Probabilities of each kind of typing noise at ``level=1``.

    ``add_typing_noise`` multiplies every rate by ``level``. Message-level
    rates decide once per message (a person either capitalises or does not);
    ``word_typo`` is drawn for every eligible word.

    Attributes:
        lowercase_starts: Lowercase the first letter of every sentence.
        lowercase_i: Write the pronoun "I" (and "I'm", "I've") as "i".
        drop_apostrophes: Type contractions without the apostrophe ("dont").
        word_typo: One keyboard slip in a word of three or more letters.
        drop_final_period: Leave out a final "." (a "?" or "!" always stays).
    """

    lowercase_starts: float = 0.8
    lowercase_i: float = 0.5
    drop_apostrophes: float = 0.7
    word_typo: float = 0.15
    drop_final_period: float = 0.7

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NoiseRates.{field.name} must be in [0, 1], got {value!r}.")


# Relative weights of the four keyboard slips.
_TYPO_KINDS: Final = ("adjacent_key", "swap", "drop", "double")
_TYPO_WEIGHTS: Final = (0.35, 0.25, 0.25, 0.15)
# The output length stays within 1/5 = 20 % of the input length.
_MAX_LENGTH_CHANGE_DIVISOR: Final = 5
_MIN_TYPO_WORD: Final = 3

# Trailing sentence punctuation is not part of a protected token.
_PROTECTED_END: Final = r"(?=[.,!?;:)\]'\"\u2026]*(?:\s|$))"
_URL: Final = r"(?:https?://|www\.)[^\s<>\"']+?" + _PROTECTED_END
_EMAIL: Final = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
_DOMAIN: Final = r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?:/\S*?)?" + _PROTECTED_END
_HANDLE: Final = r"[@#]\w+"
_HAS_DIGIT: Final = r"[^\s\d]*\d\S*?" + _PROTECTED_END
_TOKEN: Final = re.compile(
    rf"(?P<protected>{_URL}|{_EMAIL}|{_DOMAIN}|{_HANDLE}|{_HAS_DIGIT})"
    r"|(?P<word>[A-Za-z]+(?:['\u2019][A-Za-z]+)*)"
    r"|(?P<other>.)",
    re.DOTALL,
)
_SENTENCE_END: Final = frozenset(".!?\n\u2026")
_CLAUSE_MARK: Final = frozenset(",;:")


def _keyboard_slip(word: str, kind: str, rng: random.Random) -> str:
    """Apply one slip of ``kind`` to an ASCII word of three or more letters."""
    if kind == "adjacent_key":
        index = rng.randrange(len(word))
        neighbour = rng.choice(QWERTY_NEIGHBOURS[word[index].lower()])
        if word[index].isupper():
            neighbour = neighbour.upper()
        return word[:index] + neighbour + word[index + 1 :]
    if kind == "swap":
        # Never the first letter: "oTday" is not how thumbs slip.
        positions = [i for i in range(1, len(word) - 1) if word[i] != word[i + 1]]
        if not positions:
            return word
        index = rng.choice(positions)
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    if kind == "drop":
        # Only an inner letter, and only from words that stay three letters long.
        if len(word) <= _MIN_TYPO_WORD:
            return word
        index = rng.randrange(1, len(word) - 1)
        return word[:index] + word[index + 1 :]
    index = rng.randrange(len(word))
    return word[: index + 1] + word[index] + word[index + 1 :]


def _apply_case(word: str, *, sentence_start: bool, lower_starts: bool, lower_i: bool) -> str:
    if lower_i and word[0] == "I" and (len(word) == 1 or word[1] in "'\u2019"):
        return word[0].lower() + word[1:]
    is_acronym = len(word) > 1 and word.isupper()
    if sentence_start and lower_starts and not is_acronym:
        return word[0].lower() + word[1:]
    return word


def _last_visible(pieces: list[str]) -> int | None:
    for index in range(len(pieces) - 1, -1, -1):
        if pieces[index] and not pieces[index].isspace():
            return index
    return None


# ------------------------------------------------------------------ profile


class EnglishProfile:
    """Language profile for English.

    Args:
        min_latin_share: A message with at least ``min_letters`` letters is the
            wrong language when fewer than this share of its letters are Latin.
        min_letters: Below this many letters the share rule is skipped (only
            the majority rule applies), so "ok да" is not flagged.
        noise_rates: Probabilities used by :meth:`add_typing_noise`.

    Raises:
        ValueError: If ``min_latin_share`` is outside ``[0, 1]`` or
            ``min_letters`` is negative.
    """

    code: str = "en"

    def __init__(
        self,
        *,
        min_latin_share: float = 0.6,
        min_letters: int = 10,
        noise_rates: NoiseRates | None = None,
    ) -> None:
        if not 0.0 <= min_latin_share <= 1.0:
            raise ValueError(f"min_latin_share must be in [0, 1], got {min_latin_share!r}.")
        if min_letters < 0:
            raise ValueError(f"min_letters must be >= 0, got {min_letters!r}.")
        self.min_latin_share = min_latin_share
        self.min_letters = min_letters
        self.noise_rates = noise_rates if noise_rates is not None else NoiseRates()

    def __repr__(self) -> str:
        return (
            f"EnglishProfile(min_latin_share={self.min_latin_share!r}, "
            f"min_letters={self.min_letters!r}, noise_rates={self.noise_rates!r})"
        )

    def is_wrong_language(self, text: str) -> bool:
        """Tell whether ``text`` is written in a non-Latin script.

        Two rules, on letters only (digits, punctuation, emoji and spaces are
        ignored, so a message without letters is never flagged):

        1. Majority: more non-Latin letters than Latin ones. Catches short
           replies such as "Привет" or "你好".
        2. Share: at least ``min_letters`` letters and a Latin share below
           ``min_latin_share``. Catches mixed messages that mostly left English.

        Limits: this is a script check, not language identification. French,
        Spanish or German written in Latin letters pass as English; English
        transliterated into another script is flagged; a lone Greek "µ" is
        flagged.

        Args:
            text: A message written by the simulated user.

        Returns:
            True when the message is not written in Latin letters.
        """
        latin, other = _script_counts(text)
        letters = latin + other
        if letters == 0:
            return False
        if other > latin:
            return True
        return letters >= self.min_letters and latin / letters < self.min_latin_share

    def is_agreement(self, text: str) -> bool:
        """Tell whether ``text`` is nothing but a short agreement.

        The text is lowercased, apostrophes are removed ("you're" and "youre"
        match alike) and punctuation and emoji are dropped. It counts as an
        agreement when every remaining word belongs to an agreement phrase
        ("yeah", "makes sense", "you're so right") or to a filler ("lol",
        "honestly"), and at least one agreement phrase is present.

        Because every word must be covered, negations and contrasts fail on
        their own: "not sure", "okay but no", "I guess not", "yeah no". A
        reply that agrees and then says more ("true, I should sleep") is not a
        bare agreement and returns False. Sarcasm is not detected.

        Args:
            text: A message written by the simulated user.

        Returns:
            True for a bare affirmative reply.
        """
        normalised = _agreement_text(text)
        if not normalised or len(normalised) > _MAX_AGREEMENT_CHARS:
            return False
        return bool(_ONLY_AGREEMENT.fullmatch(normalised) and _HAS_AGREEMENT.search(normalised))

    def is_ask_back(self, text: str) -> bool:
        """Tell whether ``text`` ends by asking the same question back.

        Trailing emoji, emoticons and a closing "lol"/"then" are skipped. Then
        the end of the message must be one of:

        - an abbreviation or explicit phrase, always: "hbu", "wbu",
          "what about you", "how about yourself", "how are you doing",
          "how was your day";
        - "and you" / "and yourself": with a question mark, or starting a
          clause ("Fine, and you");
        - a bare "you?" / "yourself?": with a question mark *and* starting a
          clause, so "Good, you?" counts but "thank you?" does not.

        Limits: a question about a third person ("and you guys?") or a
        question in the middle of the message is not detected.

        Args:
            text: A message written by the simulated user.

        Returns:
            True when the message ends with a reciprocal question.
        """
        tail = _strip_trailing_decoration(_straight_apostrophes(text).casefold())
        tail = tail.replace("'", "")
        if not tail:
            return False
        if _ASK_STRONG.search(tail):
            return True
        match = _ASK_AND.search(tail)
        if match and ("?" in match["mark"] or _at_clause_start(match)):
            return True
        match = _ASK_BARE.search(tail)
        return bool(match and "?" in match["mark"] and _at_clause_start(match))

    def add_typing_noise(self, text: str, rng: random.Random, level: float) -> str:
        """Return ``text`` as a hurried person would type it on a phone.

        Possible changes, each with a probability of ``rate * level`` (see
        :class:`NoiseRates`): lowercase sentence starts, a lowercase "i",
        contractions without the apostrophe ("dont", "im"), a dropped final
        period, and per word one keyboard slip: a QWERTY neighbour key, two
        swapped letters, a dropped inner letter or a doubled letter.

        Guarantees:

        - ``level == 0`` or an empty ``text`` returns ``text`` unchanged;
        - the same ``rng`` state gives the same output;
        - URLs, e-mail addresses, bare domains, @mentions, #hashtags and any
          token containing a digit are never changed;
        - no word is removed and only ASCII words are touched; at most one
          letter per word changes, so the meaning survives;
        - the length changes by at most 20 % (slips that would exceed this
          are skipped), so a very short text only gets length-preserving noise.

        Args:
            text: A clean message.
            rng: Source of randomness. Only this object is used, never the
                global ``random`` module.
            level: Noise intensity in ``[0, 1]``.

        Returns:
            The noisy message.

        Raises:
            ValueError: If ``level`` is outside ``[0, 1]`` or NaN.
        """
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level!r}.")
        if not text or level == 0.0:
            return text
        rates = self.noise_rates
        lower_starts = rng.random() < rates.lowercase_starts * level
        lower_i = rng.random() < rates.lowercase_i * level
        drop_apostrophes = rng.random() < rates.drop_apostrophes * level
        max_change = len(text) // _MAX_LENGTH_CHANGE_DIVISOR
        change = 0

        pieces: list[str] = []
        kinds: list[str] = []
        sentence_start = True
        for match in _TOKEN.finditer(text):
            kind = match.lastgroup or "other"
            token = match.group()
            if kind == "word":
                candidate = self._noisy_word(token, rng, level, drop_apostrophes=drop_apostrophes)
                if abs(change + len(candidate) - len(token)) <= max_change:
                    change += len(candidate) - len(token)
                    token = candidate
                token = _apply_case(
                    token,
                    sentence_start=sentence_start,
                    lower_starts=lower_starts,
                    lower_i=lower_i,
                )
                sentence_start = False
            elif kind == "protected" or token in _CLAUSE_MARK:
                sentence_start = False
            elif token in _SENTENCE_END:
                sentence_start = True
            pieces.append(token)
            kinds.append(kind)

        last = _last_visible(pieces)
        if (
            last is not None
            and kinds[last] == "other"
            and pieces[last] == "."
            and (last == 0 or pieces[last - 1] != ".")
            and rng.random() < rates.drop_final_period * level
            and abs(change - 1) <= max_change
        ):
            pieces[last] = ""
        return "".join(pieces)

    def _noisy_word(
        self, word: str, rng: random.Random, level: float, *, drop_apostrophes: bool
    ) -> str:
        straight = _straight_apostrophes(word).lower()
        if "'" in straight:
            if drop_apostrophes and straight in APOSTROPHE_DROPS:
                typed = APOSTROPHE_DROPS[straight]
                # "I'm" is typed "im"; other contractions keep their capital.
                if word[0].isupper() and not straight.startswith("i'"):
                    typed = typed[0].upper() + typed[1:]
                return typed
            return word
        if len(word) < _MIN_TYPO_WORD or word.isupper():
            return word
        if rng.random() >= self.noise_rates.word_typo * level:
            return word
        kind = rng.choices(_TYPO_KINDS, weights=_TYPO_WEIGHTS)[0]
        return _keyboard_slip(word, kind, rng)
