"""The logic line: parsing, its refusals, editing it, and what it computes; plus the At a glance text."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import compile as C
from openstrategy_studio.expressions import (append_letter, condition_name_problem, condition_text, logic_names,
                                             logic_text, next_letter, parse_logic, plan, remove_letter,
                                             rename_in_logic)


def test_parse_shapes():
    assert parse_logic("A") == "A"
    assert parse_logic("A and B and C") == {"op": "and", "args": ["A", "B", "C"]}
    assert parse_logic("not (A or B)") == {"not": {"op": "or", "args": ["A", "B"]}}
    assert parse_logic("(A and (B or C)) and not D") == {
        "op": "and", "args": [{"op": "and", "args": ["A", {"op": "or", "args": ["B", "C"]}]}, {"not": "D"}]}
    assert parse_logic("a XOR b".replace("a", "A").replace("b", "B")) == {"op": "xor", "args": ["A", "B"]}
    assert logic_names(parse_logic("(A nand B) nor not C")) == {"A", "B", "C"}


def test_conditions_can_have_names():
    """Renamed conditions read in the logic line; the operator words still read as operators."""
    assert parse_logic("trend and not (rsi_low or Gap2)") == {
        "op": "and", "args": ["trend", {"not": {"op": "or", "args": ["rsi_low", "Gap2"]}}]}
    assert parse_logic("trend AND Not x") == {"op": "and", "args": ["trend", {"not": "x"}]}
    assert logic_names(parse_logic("(trend xor A) nor rsi_low")) == {"trend", "A", "rsi_low"}
    assert logic_text(parse_logic("(trend or B) and not rsi_low")) == "(trend or B) and not rsi_low"


def test_condition_names_are_checked():
    assert condition_name_problem("trend", {"A"}) is None
    assert condition_name_problem("rsi_low2", set()) is None
    for bad in ("", "2x", "_x", "rsi low", "a-b", "é"):
        assert "letters, digits and underscores" in condition_name_problem(bad, set()), bad
    for word in ("and", "NOT", "Xnor"):
        assert "operator" in condition_name_problem(word, set()), word
    assert "already has a condition called B" in condition_name_problem("B", {"A", "B"})


def test_renaming_rewrites_only_that_name():
    assert rename_in_logic("(A and AA) or not A", "A", "trend") == "(trend and AA) or not trend"
    # The rest stays as written, brackets and spacing included; a half-typed line is renamed too.
    assert rename_in_logic("((B))  and   A", "B", "gap") == "((gap))  and   A"
    assert rename_in_logic("A and (B", "B", "gap") == "A and (gap"
    assert rename_in_logic("BA and B", "B", "x") == "BA and x"


@pytest.mark.parametrize("text, message", [
    ("", "write which conditions"), ("A and B or C", "meet without brackets"), ("(A and B", "never closed"),
    ("A and B)", "closed that was never opened"), ("A and", "ends too soon"), ("and A", "needs a condition"),
    ("A B", "follows without an operator"), ("A & B", "not a condition name"), ("A and 3", "not a condition name"),
    ("A and _x", "not a condition name"),
    # Never evaluated: a call reads as a name and a bracket with no operator between, and is refused.
    ("eval(A)", "follows without an operator"), ("__import__('os')", "not a condition name"),
])
def test_parse_refuses_with_a_message(text, message):
    with pytest.raises(ValueError, match=message):
        parse_logic(text)


def test_text_round_trip():
    for text in ("A", "A and B", "not A", "(A or B) and not C", "not (A xor B)", "A nand (B nor C)"):
        assert logic_text(parse_logic(text)) == text
        assert parse_logic(logic_text(parse_logic(text))) == parse_logic(text)


def test_editing_the_line():
    assert next_letter({"A", "B", "D"}) == "C"
    assert next_letter({chr(ord("A") + i) for i in range(26)}) == "AA"
    assert append_letter("", "A") == "A"
    assert append_letter("A and B", "C") == "A and B and C"
    assert append_letter("A or B", "C") == "(A or B) and C"
    assert append_letter("A or B", "C", "or") == "A or B or C"
    assert remove_letter("A and B and C", "B") == "A and C"
    assert remove_letter("(A or B) and not C", "C") == "A or B"
    assert remove_letter("A", "A") == ""


def test_every_operator_computes_what_it_says():
    index = pd.RangeIndex(8)
    bits = {letter: pd.DataFrame({"X": [(i >> shift) & 1 == 1 for i in range(8)]}, index=index)
            for letter, shift in (("A", 0), ("B", 1), ("C", 2))}
    a, b, c = (bits[letter]["X"].to_numpy() for letter in "ABC")
    expected = {"A and B and C": a & b & c, "A or B or C": a | b | c, "A xor B xor C": a ^ b ^ c,
                "A nand B nand C": ~(a & b & c), "A nor B nor C": ~(a | b | c), "A xnor B": ~(a ^ b),
                "not (A and B) or C": ~(a & b) | c}
    for text, values in expected.items():
        got = C.evaluate_logic(parse_logic(text), bits)["X"].to_numpy()
        np.testing.assert_array_equal(got, values, err_msg=text)


def test_conditions_in_words():
    assert condition_text({"left": {"feature": "fast"}, "op": "crosses_above", "right": {"feature": "slow"}}) == \
        "fast crosses above slow"
    assert condition_text({"left": {"price": "close"}, "op": ">", "right": {"feature": "upper", "shift": 1}}) == \
        "close is above upper 1 bar ago"
    assert condition_text({"pattern": "hammer", "side": "bullish"}) == "hammer appears (bullish)"
    assert condition_text({"filter": "weekday", "days": [0, 4]}) == "on Mon, Fri"
    assert "on 1w bars, close is above open" == condition_text(
        {"filter": "higher_timeframe", "timeframe": "1w", "left": {"price": "close"}, "op": ">",
         "right": {"price": "open"}})


def test_the_strategy_reads_as_it_would_be_told(ma_spec):
    """The At a glance panel's strategy on a page (the owner, 25 November 2026): the market, each rule, each
    trade's size and stops, the features and what is optimised, in sentences."""
    sentences = plan(ma_spec)
    by_lead = {sentence["lead"]: sentence for sentence in sentences[1:]}
    assert sentences[0]["lead"] == "Spread bets on EURUSD=X, long only"
    assert sentences[0]["text"].startswith(", on daily bars from 2019-01-01 to 2024-12-31, starting with 10,000")
    assert by_lead["Buy"]["text"] == " when fast crosses above slow."
    assert by_lead["Sell"]["text"] == " when fast crosses below slow, or at the stop or the target."
    assert by_lead["Each trade"]["text"] == (" risks 100, with a stop loss 60 points from the entry and a take profit "
                                             "120 points from the entry.")
    assert "fast is SMA(close, 50)" in by_lead["Where"]["items"]
    assert by_lead["Optimising"]["text"].startswith(" 2 settings") and len(by_lead["Optimising"]["items"]) == 2
    assert [sentence["tone"] for sentence in sentences[:4]] == ["setup", "entry", "exit", "risk"]
    # A share of the balance in percent, and a stop measured as one.
    ma_spec["risk"]["sizing"] = {"risk_pct": 0.2, "risk_of": "balance"}
    ma_spec["risk"]["stop"].update(unit="pct", pct=0.13138)
    assert plan(ma_spec)[3]["text"].startswith(" risks 20% of the balance, with a stop loss 13.14% from the entry")


def test_a_long_rule_reads_as_a_list():
    """Three conditions joined one way are a list; mixed operators keep the logic line, each condition named;
    an exit that is a list says after it that the stop closes it too."""
    from openstrategy_studio.spec import load_template

    spec = load_template("ma_cross")
    cross = {"left": {"feature": "fast"}, "op": "crosses_above", "right": {"feature": "slow"}}
    low = {"left": {"price": "close"}, "op": "<", "right": {"const": 1.2}}
    friday = {"filter": "weekday", "days": [4]}
    spec["rules"]["long_entry"] = {"conditions": {"A": cross, "B": low, "C": friday}, "logic": "A and B and C"}
    spec["rules"]["long_exit"] = {"conditions": {"A": cross, "B": low, "C": friday}, "logic": "(A and B) or C"}
    by_lead = {sentence["lead"]: sentence for sentence in plan(spec)}
    assert by_lead["Buy"]["text"] == " when all of these are true:"
    assert by_lead["Buy"]["items"] == ["fast crosses above slow", "close is below 1.2", "on Fri"]
    assert by_lead["Sell"]["text"] == " when (A and B) or C, where:"
    assert by_lead["Sell"]["items"][0] == "A: fast crosses above slow"
    assert by_lead["Sell"]["after"] == "Or at the stop or the target."
    # Two conditions read as one sentence, brackets and all.
    spec["rules"]["long_entry"] = {"conditions": {"A": cross, "B": low}, "logic": "A and not B"}
    assert {s["lead"]: s for s in plan(spec)}["Buy"]["text"] == " when fast crosses above slow and not close is below 1.2."


def test_a_conditions_timing_in_words():
    cross = {"left": {"feature": "fast"}, "op": "crosses_above", "right": {"feature": "slow"}}
    assert condition_text({**cross, "within_bars": 5}) == "fast crosses above slow, on any of the last 5 bars"
    assert condition_text({**cross, "for_bars": 3}) == "fast crosses above slow, on each of the last 3 bars"
    assert condition_text({**cross, "bars_ago": 1}) == "fast crosses above slow, 1 bar ago"
    assert condition_text({"pattern": "hammer", "side": "bullish", "bars_ago": 2}) == "hammer appears (bullish), 2 bars ago"
    weekly = {"filter": "higher_timeframe", "timeframe": "1w", "left": {"price": "close"}, "op": ">",
              "right": {"price": "open"}, "for_bars": 2}
    assert condition_text(weekly) == "on 1w bars, close is above open, on each of the last 2 bars"
