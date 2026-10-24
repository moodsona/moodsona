from __future__ import annotations

import pytest

from moodsona.lang import (
    EnglishProfile,
    LanguageProfile,
    available_languages,
    get_profile,
)


def test_english_is_registered() -> None:
    profile: LanguageProfile = get_profile("en")
    assert isinstance(profile, EnglishProfile)
    assert isinstance(profile, LanguageProfile)
    assert profile.code == "en"


@pytest.mark.parametrize("code", ["en", "EN", " en ", "en-US", "en_GB", "En-au"])
def test_tag_variants_resolve_to_english(code: str) -> None:
    assert get_profile(code).code == "en"


@pytest.mark.parametrize("code", ["ru", "de", "", "eng", "english", "-en"])
def test_unknown_code_raises_lookup_error_listing_available(code: str) -> None:
    with pytest.raises(LookupError, match=r"Available: en\."):
        get_profile(code)


def test_error_message_names_the_requested_code() -> None:
    with pytest.raises(LookupError, match="'xx-YY'"):
        get_profile("xx-YY")


def test_available_languages() -> None:
    assert available_languages() == ("en",)


def test_every_registered_profile_matches_its_tag() -> None:
    for code in available_languages():
        profile = get_profile(code)
        assert isinstance(profile, LanguageProfile)
        assert profile.code == code


def test_each_call_returns_a_fresh_instance() -> None:
    assert get_profile("en") is not get_profile("en")
