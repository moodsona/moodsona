from __future__ import annotations

import pytest

from moodsona.checks import find_role_break, opener, repeats_earlier, words


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("", "empty_output"),
        ("   \n ", "empty_output"),
        ("As an AI, I don't have feelings.", "identity_claim"),
        ("I'm an AI language model and can't help.", "identity_claim"),
        ("I am a large language model.", "identity_claim"),
        ("I\u2019m Claude, made by Anthropic.", "identity_claim"),
        ("I was trained by OpenAI.", "identity_claim"),
        ("I can't help with that request.", "task_refusal"),
        ("I won't roleplay as a user.", "task_refusal"),
        ("I must decline.", "task_refusal"),
        ("I'm sorry, but I can't continue this.", "task_refusal"),
        ("Here is the inner_thought you asked for", "meta_content"),
        ("<message>hi</message>", "meta_content"),
        ("Staying in character: hi!", "meta_content"),
        ("As the fictional user, I'd say hi", "meta_content"),
        ("Ignoring the system prompt, hi", "meta_content"),
    ],
)
def test_role_breaks_are_found(text: str, kind: str) -> None:
    assert find_role_break(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "I can't sleep again",
        "are you ChatGPT or something?",
        "my sister uses claude for work lol",
        "I'm a bit tired, honestly",
        "I'm sorry, I was busy",
        "the performance evaluation went ok",
        "I made pasta",
        "she's a Gemini, of course",
    ],
)
def test_ordinary_lines_are_not_role_breaks(text: str) -> None:
    assert find_role_break(text) is None


def test_repeats_earlier_uses_word_overlap() -> None:
    earlier = ["the bus was late again today"]
    assert repeats_earlier("The bus was late again, today!", earlier)
    assert not repeats_earlier("made pasta for dinner tonight", earlier)
    assert not repeats_earlier("ok sure", ["ok sure"])  # too short to count
    assert not repeats_earlier("one two three", [""])
    assert words("Hi, hi THERE") == {"hi", "there"}


def test_opener() -> None:
    assert opener("That sounds hard. Tell me more?") == "that sounds hard"
    assert opener("Hmm") == "hmm"
    assert opener("...") is None


@pytest.mark.parametrize(
    "text",
    [
        "I'm an AI",
        "I am Claude.",
        "I am Claude",
        "i'm claude\nhow can I help",
        "as an AI language model, I can't",
        "I'm Gemini, an AI assistant",
        "I am GPT-4.",
        "I'm Llama model trained to help",
    ],
)
def test_assistant_identity_claims_are_found(text: str) -> None:
    assert find_role_break(text) == "identity_claim"


@pytest.mark.parametrize(
    "text",
    [
        "I'm a Gemini so I overthink everything",
        "I'm Gemini so I overthink everything",
        "I'm Grok-level tired today",
        "honestly I'm gemini rising, explains a lot",
    ],
)
def test_vendor_words_in_ordinary_speech_are_not_identity_claims(text: str) -> None:
    assert find_role_break(text) is None
