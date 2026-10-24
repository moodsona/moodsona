from __future__ import annotations

import random
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona.lang import EnglishProfile, LanguageProfile, NoiseRates
from moodsona.lang.english import _keyboard_slip
from moodsona.lang.noise_tables import APOSTROPHE_DROPS, QWERTY_NEIGHBOURS

EN = EnglishProfile()
CURLY = chr(0x2019)  # right single quotation mark, the phone-keyboard apostrophe
# Styled Latin letters, built from code points so the source stays plain ASCII.
FULLWIDTH_HELLO = "".join(chr(ord(c) + 0xFEE0) for c in "hello")
MATH_BOLD_HELLO = "".join(chr(0x1D41A + ord(c) - ord("a")) for c in "hello")
ALL_NOISE = NoiseRates(
    lowercase_starts=1.0,
    lowercase_i=1.0,
    drop_apostrophes=1.0,
    word_typo=1.0,
    drop_final_period=1.0,
)
NO_NOISE = NoiseRates(
    lowercase_starts=0.0,
    lowercase_i=0.0,
    drop_apostrophes=0.0,
    word_typo=0.0,
    drop_final_period=0.0,
)


def _only(**rates: float) -> EnglishProfile:
    values = dict.fromkeys(NoiseRates.__dataclass_fields__, 0.0)
    values.update(rates)
    return EnglishProfile(noise_rates=NoiseRates(**values))


def test_satisfies_protocol() -> None:
    profile: LanguageProfile = EnglishProfile()
    assert isinstance(profile, LanguageProfile)
    assert profile.code == "en"


@pytest.mark.parametrize(
    "kwargs",
    [{"min_latin_share": -0.1}, {"min_latin_share": 1.5}, {"min_letters": -1}],
)
def test_constructor_validates(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        EnglishProfile(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_noise_rates_validate(value: float) -> None:
    with pytest.raises(ValueError, match="word_typo"):
        NoiseRates(word_typo=value)


# ------------------------------------------------------------ wrong language


@pytest.mark.parametrize(
    "text",
    [
        "Привет, как дела?",
        "Привет",
        "สวัสดีครับ",
        "你好",
        "今天很累",
        "こんにちは、元気ですか",
        "안녕하세요",
        "مرحبا كيف حالك",
        "שלום מה שלומך",
        "Ευχαριστώ πολύ",
        "ok, 今天真的很累",
    ],
)
def test_flags_non_latin_text(text: str) -> None:
    assert EN.is_wrong_language(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "123",
        "12:30!!",
        "?!...",
        "😊😊😊",
        "👍🏽",
        "hello",
        "I'm fine, thanks 😊",
        "café au lait, naïve résumé",
        "Ich bin müde",  # German in Latin letters: a script check cannot tell
        "ok да",  # too short for the share rule, and not a non-Latin majority
        "I had gyros and σουβλάκι today",  # a borrowed word or two is fine
        FULLWIDTH_HELLO,
        MATH_BOLD_HELLO,
    ],
)
def test_does_not_flag_english_or_letterless_text(text: str) -> None:
    assert not EN.is_wrong_language(text)


def test_share_threshold_is_configurable() -> None:
    text = "hello there my friend привет"  # 18 Latin letters out of 24
    assert not EN.is_wrong_language(text)
    assert EnglishProfile(min_latin_share=0.9).is_wrong_language(text)


def test_min_letters_gates_the_share_rule() -> None:
    text = "ok да"  # Latin share 0.5, but only four letters
    assert not EN.is_wrong_language(text)
    assert EnglishProfile(min_letters=4).is_wrong_language(text)


_EMOJI = st.characters(min_codepoint=0x1F300, max_codepoint=0x1F64F)
_ASCII = st.sampled_from(string.ascii_letters + string.digits + string.punctuation + " ")


@given(st.text(alphabet=st.one_of(_ASCII, _EMOJI)))
def test_ascii_and_emoji_are_never_wrong_language(text: str) -> None:
    assert not EN.is_wrong_language(text)


# ----------------------------------------------------------------- agreement


@pytest.mark.parametrize(
    "text",
    [
        "yeah",
        "Yeah.",
        "yeahhh",
        "yep",
        "ok",
        "Ok!",
        "okay",
        "k",
        "sure",
        "Sure thing",
        "I guess",
        "i guess so",
        "true",
        "so true",
        "That's true",
        "fair",
        "Fair enough.",
        "makes sense",
        "That makes sense",
        "you're right",
        f"You{CURLY}re right",
        "youre so right lol",
        "you are right",
        "exactly",
        "totally",
        "agreed",
        "I agree",
        "same",
        "me too",
        "sounds good",
        "got it",
        "mhm",
        "oh yeah haha",
        "yeah, makes sense",
        "honestly fair",
        "ok 👍",
        "i know right",
    ],
)
def test_agreement_positive(text: str) -> None:
    assert EN.is_agreement(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "...",
        "😊",
        "not sure",
        "Not really",
        "okay but no",
        "I guess not",
        "yeah no",
        "no",
        "nope",
        "sure, but I don't think so",
        "you're wrong",
        "true, I should sleep more",
        "ok so what happened was my boss called",
        "hmm",
        "lol",
        "what?",
        "yeah?? are you serious",
        "ok привет",
        "I don't agree",
        "that doesn't make sense",
    ],
)
def test_agreement_negative(text: str) -> None:
    assert not EN.is_agreement(text)


def test_long_message_is_never_a_bare_agreement() -> None:
    assert not EN.is_agreement(" ".join(["yeah"] * 20))


# ------------------------------------------------------------------ ask-back


@pytest.mark.parametrize(
    "text",
    [
        "and you?",
        "And you",
        "I'm ok. And you?",
        "Fine, and you",
        "what about you?",
        "I don't know. What about you?",
        "how about yourself?",
        "how about u",
        "hbu",
        "ok hbu?",
        "tired lol wbu",
        "you?",
        "Good, you?",
        "same old. you?",
        "how are you?",
        "how are you doing today? :)",
        "what about you then?",
        "hbu 😊",
        "how was your day?",
        "how's your week going?",
        f"how{CURLY}s your day",
        "and yourself?",
        "not bad, u?",
    ],
)
def test_ask_back_positive(text: str) -> None:
    assert EN.is_ask_back(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "😊",
        "thank you?",
        "Thank you",
        "I love you",
        "me and you",
        "fine and you",
        "what about you guys?",
        "and you guys?",
        "what about you? I had a long day",  # not at the end
        "are you sure?",
        "what do you think about it?",
        "hbu1",
        "you're right",
    ],
)
def test_ask_back_negative(text: str) -> None:
    assert not EN.is_ask_back(text)


# --------------------------------------------------------------------- noise

SAMPLE = "I don't know, I'm just tired today. What about you?"


def test_noise_tables_are_consistent() -> None:
    assert set(QWERTY_NEIGHBOURS) == set(string.ascii_lowercase)
    for key, neighbours in QWERTY_NEIGHBOURS.items():
        assert key not in neighbours
        for neighbour in neighbours:
            assert key in QWERTY_NEIGHBOURS[neighbour], (key, neighbour)
    for contraction, typed in APOSTROPHE_DROPS.items():
        assert contraction == contraction.lower()
        assert "'" in contraction
        assert typed == contraction.replace("'", "")


def test_level_zero_and_empty_text_are_unchanged() -> None:
    assert EN.add_typing_noise(SAMPLE, random.Random(1), 0.0) == SAMPLE
    assert EN.add_typing_noise("", random.Random(1), 1.0) == ""


@pytest.mark.parametrize("level", [-0.1, 1.1, float("nan"), float("inf")])
def test_level_out_of_range_raises(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        EN.add_typing_noise(SAMPLE, random.Random(1), level)


def test_noise_uses_only_the_given_rng() -> None:
    random.seed(0)
    first = EN.add_typing_noise(SAMPLE * 3, random.Random(7), 1.0)
    random.seed(99)
    second = EN.add_typing_noise(SAMPLE * 3, random.Random(7), 1.0)
    assert first == second


def test_full_noise_changes_text_for_most_seeds() -> None:
    changed = sum(
        EN.add_typing_noise(SAMPLE, random.Random(seed), 1.0) != SAMPLE for seed in range(50)
    )
    assert changed >= 45


def test_lowercase_sentence_starts_keeps_acronyms() -> None:
    profile = _only(lowercase_starts=1.0)
    text = "Hello there. How are you? NASA called! Fine"
    assert profile.add_typing_noise(text, random.Random(0), 1.0) == (
        "hello there. how are you? NASA called! fine"
    )


def test_lowercase_i() -> None:
    profile = _only(lowercase_i=1.0)
    text = "Yes I think I'm fine and I've eaten"
    assert profile.add_typing_noise(text, random.Random(0), 1.0) == (
        "Yes i think i'm fine and i've eaten"
    )


def test_drop_apostrophes() -> None:
    profile = _only(drop_apostrophes=1.0)
    text = f"Don't worry, I'm fine and it{CURLY}s ok. We'll see, I'll call."
    assert profile.add_typing_noise(text, random.Random(0), 1.0) == (
        "Dont worry, im fine and its ok. We'll see, I'll call."
    )


def test_drop_final_period_only() -> None:
    profile = _only(drop_final_period=1.0)
    rng = random.Random(0)
    assert profile.add_typing_noise("See you soon.", rng, 1.0) == "See you soon"
    assert profile.add_typing_noise("See you soon. ", rng, 1.0) == "See you soon "
    assert profile.add_typing_noise("Really?", rng, 1.0) == "Really?"
    assert profile.add_typing_noise("Wow!", rng, 1.0) == "Wow!"
    assert profile.add_typing_noise("Well...", rng, 1.0) == "Well..."
    assert profile.add_typing_noise("ok.", rng, 1.0) == "ok."  # too short to shrink by 20 %


def _is_one_slip(original: str, typed: str) -> bool:
    if original == typed:
        return True
    if len(typed) == len(original):
        diff = [i for i, (a, b) in enumerate(zip(original, typed, strict=True)) if a != b]
        if len(diff) == 1:
            a, b = original[diff[0]].lower(), typed[diff[0]].lower()
            return b in QWERTY_NEIGHBOURS.get(a, "")
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and original[diff[0]] == typed[diff[1]]
            and original[diff[1]] == typed[diff[0]]
        )
    longer, shorter = (original, typed) if len(original) > len(typed) else (typed, original)
    if len(longer) - len(shorter) != 1:
        return False
    return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))


def test_word_typos_change_at_most_one_letter_per_word() -> None:
    profile = _only(word_typo=1.0)
    text = "the quick brown fox jumps over the lazy sleeping dog"
    for seed in range(30):
        noisy = profile.add_typing_noise(text, random.Random(seed), 1.0)
        assert noisy != text
        pairs = zip(text.split(), noisy.split(), strict=True)
        assert all(_is_one_slip(a, b) for a, b in pairs), noisy


@pytest.mark.parametrize("kind", ["adjacent_key", "swap", "drop", "double"])
def test_each_slip_kind(kind: str) -> None:
    for seed in range(20):
        word = "Thinking"
        typed = _keyboard_slip(word, kind, random.Random(seed))
        assert typed != word
        assert _is_one_slip(word, typed)
        assert typed[0] in "Tt" or kind == "adjacent_key"
        if kind == "drop":
            assert typed[0] == word[0]
            assert typed[-1] == word[-1]


def test_slips_that_cannot_apply_return_the_word() -> None:
    assert _keyboard_slip("aaa", "swap", random.Random(0)) == "aaa"
    assert _keyboard_slip("cat", "drop", random.Random(0)) == "cat"


def test_short_words_and_acronyms_get_no_slips() -> None:
    profile = _only(word_typo=1.0)
    text = "ok so I am at NASA HQ by my TV"
    assert profile.add_typing_noise(text, random.Random(3), 1.0) == text


def test_protected_tokens_are_untouched() -> None:
    profile = EnglishProfile(noise_rates=ALL_NOISE)
    text = (
        "Look at https://example.com/some/Path?q=Hello. Ask @SomeoneThere or "
        "write to Someone.Else@example.org, see #TodayTopic and example.net/About "
        "at 7pm on 2026-10-01 with 3.14159 cups."
    )
    protected = [
        "https://example.com/some/Path?q=Hello",
        "@SomeoneThere",
        "Someone.Else@example.org",
        "#TodayTopic",
        "example.net/About",
        "7pm",
        "2026-10-01",
        "3.14159",
    ]
    for seed in range(30):
        noisy = profile.add_typing_noise(text, random.Random(seed), 1.0)
        for token in protected:
            assert token in noisy, (token, noisy)


def test_non_ascii_words_are_untouched() -> None:
    profile = EnglishProfile(noise_rates=ALL_NOISE)
    text = "Привет, как дела?"
    assert profile.add_typing_noise(text, random.Random(0), 1.0) == text


def test_lowercase_start_after_protected_token_sentence() -> None:
    profile = _only(lowercase_starts=1.0)
    text = "It ends at 5pm. Then we go."
    assert profile.add_typing_noise(text, random.Random(0), 1.0) == "it ends at 5pm. then we go."


# --------------------------------------------------------- noise properties

_LEVELS = st.floats(min_value=0.0, max_value=1.0)
_SEEDS = st.integers(min_value=0, max_value=2**32)


@given(st.text(max_size=300), _SEEDS, _LEVELS)
def test_noise_is_deterministic_for_a_seed(text: str, seed: int, level: float) -> None:
    first = EN.add_typing_noise(text, random.Random(seed), level)
    second = EN.add_typing_noise(text, random.Random(seed), level)
    assert first == second


@given(st.text(max_size=300), _SEEDS)
def test_level_zero_is_identity(text: str, seed: int) -> None:
    assert EN.add_typing_noise(text, random.Random(seed), 0.0) == text


@given(st.text(max_size=300), _SEEDS, _LEVELS)
def test_length_stays_within_twenty_percent(text: str, seed: int, level: float) -> None:
    noisy = EnglishProfile(noise_rates=ALL_NOISE).add_typing_noise(text, random.Random(seed), level)
    assert abs(len(noisy) - len(text)) <= 0.2 * len(text)
    assert bool(noisy) == bool(text)


_WORD = st.text(alphabet=string.ascii_letters, min_size=1, max_size=12)
_WORDS = st.lists(_WORD, min_size=1, max_size=25)
_URL = st.builds(
    lambda host, tld, path: f"https://{host}.{tld}/{path}",
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10),
    st.sampled_from(["com", "org", "io", "net"]),
    st.text(alphabet=string.ascii_letters + string.digits + "/_-?=&", max_size=20),
)


@given(_WORDS, _URL, st.integers(min_value=0, max_value=25), _SEEDS, _LEVELS)
def test_urls_are_untouched(
    words: list[str], url: str, position: int, seed: int, level: float
) -> None:
    words.insert(min(position, len(words)), url)
    text = " ".join(words) + "."
    noisy = EnglishProfile(noise_rates=ALL_NOISE).add_typing_noise(text, random.Random(seed), level)
    assert url in noisy


@given(_WORDS, st.sampled_from(["", ".", "!", "?"]), _SEEDS, _LEVELS)
def test_no_word_is_removed(words: list[str], end: str, seed: int, level: float) -> None:
    text = " ".join(words) + end
    noisy = EN.add_typing_noise(text, random.Random(seed), level)
    assert len(noisy.split()) == len(text.split())
    if end in "?!":
        assert noisy.endswith(end)
