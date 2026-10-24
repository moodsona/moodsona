"""Data tables for English typing noise.

Kept apart from the logic so that the tables can be reviewed (and corrected)
without reading any code.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

__all__ = ["APOSTROPHE_DROPS", "QWERTY_NEIGHBOURS"]

#: Letters that touch each letter on a US QWERTY keyboard: the keys to the left
#: and right on the same row, plus the touching keys on the rows above and below
#: (rows are staggered, so a key touches two keys on each neighbouring row).
#: The relation is symmetric; a test checks that.
QWERTY_NEIGHBOURS: Final = MappingProxyType(
    {
        "q": "wa",
        "w": "qeas",
        "e": "wrsd",
        "r": "etdf",
        "t": "ryfg",
        "y": "tugh",
        "u": "yihj",
        "i": "uojk",
        "o": "ipkl",
        "p": "ol",
        "a": "qwsz",
        "s": "weadzx",
        "d": "ersfxc",
        "f": "rtdgcv",
        "g": "tyfhvb",
        "h": "yugjbn",
        "j": "uihknm",
        "k": "iojlm",
        "l": "opk",
        "z": "asx",
        "x": "sdzc",
        "c": "dfxv",
        "v": "fgcb",
        "b": "ghvn",
        "n": "hjbm",
        "m": "jkn",
    }
)

#: Contractions that people commonly type without the apostrophe, mapped to the
#: lowercase form they type. Keys are lowercase with a straight apostrophe.
#: Contractions whose apostrophe-less form reads as a different word in context
#: ("we'll" -> "well", "I'll" -> "ill", "we're" -> "were", "she'll" -> "shell")
#: are left out on purpose. "it's" -> "its" and "let's" -> "lets" stay in: the
#: slip is very common and a reader still gets the meaning.
APOSTROPHE_DROPS: Final = MappingProxyType(
    {
        "aren't": "arent",
        "can't": "cant",
        "couldn't": "couldnt",
        "didn't": "didnt",
        "doesn't": "doesnt",
        "don't": "dont",
        "hadn't": "hadnt",
        "hasn't": "hasnt",
        "haven't": "havent",
        "i'd": "id",
        "i'm": "im",
        "i've": "ive",
        "isn't": "isnt",
        "it's": "its",
        "let's": "lets",
        "shouldn't": "shouldnt",
        "that's": "thats",
        "there's": "theres",
        "they're": "theyre",
        "they've": "theyve",
        "wasn't": "wasnt",
        "weren't": "werent",
        "what's": "whats",
        "won't": "wont",
        "wouldn't": "wouldnt",
        "you'd": "youd",
        "you're": "youre",
        "you've": "youve",
    }
)
