"""The logic line: parsing, its refusals, editing it, and what it computes; plus the At a glance text."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import compile as C
from openstrategy_studio.expressions import (append_letter, condition_name_problem, condition_text, logic_names,
                                             logic_text, next_letter, parse_logic, remove_letter, rename_in_logic,
                                             summary)


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


def test_summary_covers_every_step(ma_spec):
    rows = dict(summary(ma_spec))
    assert rows["Market"].startswith("EURUSD=X · 1d")
    assert rows["Trades"] == "on signals, spread bets"
    assert rows["Long entry"] == "A" and "stop 60 pts" in rows["Risk"] and "rates: static" in rows["Account"]
    assert rows["Optimise"] == "2 variables"
    assert ("Feature", "fast = SMA(close, 50)") in summary(ma_spec)
