from __future__ import annotations

import json

import pytest

from moodsona.parse import HiddenBlock, ParseError, clean_message, parse_day_notes, parse_turn
from moodsona.state import Intent, Stance
from support import block, notes


def test_hidden_block_is_parsed_and_clamped() -> None:
    hidden, message = parse_turn(
        '<state>{"inner_thought": "Tired,  but it was nice.", "emotion": 130, '
        '"energy": "-5", "intent": "Short_Reply "}</state><message>"yep"</message>'
    )
    assert hidden == HiddenBlock("Tired, but it was nice.", 100, 0, Intent.SHORT_REPLY, ())
    assert message == "yep"


def test_rules_triggered_keeps_plausible_numbers_only() -> None:
    hidden, _ = parse_turn(block(rules=[2, "3", 2, 0, -1, True, "x", 1.5]))
    assert hidden.rules_triggered == (2, 3)
    hidden, _ = parse_turn(block().replace('"rules_triggered": []', '"rules_triggered": 4'))
    assert hidden.rules_triggered == ()
    raw = '<state>{"inner_thought": "ok", "emotion": 1, "energy": 2, "intent": "reply"}</state>'
    assert parse_turn(raw + "<message>hi</message>")[0].rules_triggered == ()


def test_tolerated_forms() -> None:
    fenced = (
        '<STATE>```json\n{"inner_thought": "ok", "emotion": 40.6, "energy": "70", '
        '"intent": "reply"}\n```</STATE>\n<message>  first bubble \n\n second   bubble'
    )
    hidden, message = parse_turn(fenced)
    assert (hidden.emotion, hidden.energy) == (41, 70)
    assert message == "first bubble\nsecond bubble"  # missing </message> is fine
    assert parse_turn("Sure! " + block("hi there") + " trailing")[1] == "hi there"


def test_empty_message_is_returned_for_the_rails_to_judge() -> None:
    assert parse_turn(block(""))[1] == ""


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("just a message with no block", "missing <state>"),
        ("<message>hi</message><state>{}</state>", "missing <message>"),
        ("<state>not json</state><message>hi</message>", "no JSON object"),
        ("<state>{oops}</state><message>hi</message>", "not valid JSON"),
        ("<state>[1, 2]</state><message>hi</message>", "no JSON object"),
        (block(intent="leave"), "unknown intent"),
        (block(thought="   "), "inner_thought"),
        (block().replace('"inner_thought": "Fine, I will answer.", ', ""), "inner_thought"),
        (block(emotion=True), "boolean"),
        (block(emotion="lots"), "not a number"),
        (block(energy=None), "not a number"),
        (block(energy=[5]), "not a number"),
        (block(emotion="nan"), "finite"),
        (block(emotion="inf"), "finite"),
    ],
)
def test_malformed_turn_output_is_rejected(raw: str, reason: str) -> None:
    with pytest.raises(ParseError, match=reason):
        parse_turn(raw)


def test_two_json_objects_in_the_state_are_rejected() -> None:
    with pytest.raises(ParseError, match="not valid JSON"):
        parse_turn('<state>{"a": 1}, {"b": 2}</state><message>hi</message>')


def test_hidden_block_as_dict_is_json_ready() -> None:
    hidden, _ = parse_turn(block(rules=[1]))
    assert json.loads(json.dumps(hidden.as_dict()))["intent"] == "reply"


def test_day_notes_are_parsed() -> None:
    raw = "Here you go:\n```json\n" + notes([" a  b ", "", "c", 5, "d", "e", "f", "g"]) + "\n```"
    lines, stance = parse_day_notes(raw)
    assert lines == ("a b", "c", "d", "e", "f")
    assert stance == Stance.CURIOUS


@pytest.mark.parametrize("stance", [None, "furious", 3])
def test_day_notes_with_unknown_stance_keep_the_lines(stance: object) -> None:
    raw = json.dumps({"memory": ["went well"], "stance": stance})
    assert parse_day_notes(raw) == (("went well",), None)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("no json here", "no JSON object"),
        ("{broken", "no JSON object"),
        ('{"memory": [1, }', "not valid JSON"),
        ('{"stance": "open"}', "memory is missing"),
        ('{"memory": "one line"}', "memory is missing"),
        ('{"memory": ["  ", 3]}', "memory is empty"),
    ],
)
def test_malformed_day_notes_are_rejected(raw: str, reason: str) -> None:
    with pytest.raises(ParseError, match=reason):
        parse_day_notes(raw)


def test_clean_message_strips_wrapping_quotes() -> None:
    assert clean_message(" \u201cwell, ok\u201d ") == "well, ok"
    assert clean_message("'yeah'") == "yeah"


def test_non_ascii_digit_rule_numbers_are_ignored() -> None:
    hidden, _ = parse_turn(block(rules=["²", "٣", "2"]))
    assert hidden.rules_triggered == (2,)


@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["huge", "huge-negative"])
def test_scores_too_large_for_a_float_are_a_parse_error(value: int) -> None:
    with pytest.raises(ParseError, match="emotion"):
        parse_turn(block(emotion=value))


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ('"hello"', "hello"),
        ("«hello»", "hello"),
        ("he said 'no'", "he said 'no'"),
        ("'cause im tired", "'cause im tired"),
        ('"quoted" then not', '"quoted" then not'),
        ("'", "'"),
    ],
)
def test_clean_message_strips_one_matched_wrapping_pair_only(raw: str, clean: str) -> None:
    assert clean_message(raw) == clean
