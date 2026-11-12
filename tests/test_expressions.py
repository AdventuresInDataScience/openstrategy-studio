"""The expression editor converts to and from the rule tree, and never evaluates text."""

import pytest

from openstrategy_studio import expressions as E
from openstrategy_studio.spec import validate_spec


def test_parse_builds_rules_and_features():
    rule, new = E.parse("rsi(14) < 30 and close > ma(200)", {})
    assert rule == {"all": [{"left": {"feature": "rsi_14"}, "op": "<", "right": {"const": 30}},
                            {"left": {"price": "close"}, "op": ">", "right": {"feature": "ma_200"}}]}
    assert new["rsi_14"]["block"] == "rsi" and new["rsi_14"]["window"] == 14
    assert new["ma_200"]["window"] == 200


def test_round_trip_through_text(ma_spec):
    features = ma_spec["features"]
    for slot in ("long_entry", "long_exit"):
        text = E.to_text(ma_spec["rules"][slot], features)
        rule, new = E.parse(text, features)
        assert new == {}
        assert rule == ma_spec["rules"][slot]["all"][0]


def test_calls_outputs_shifts_and_functions():
    text = "crosses_above(close, bbands(20).upper[1]) or (not rising(close, 3) and between(rsi(14), 40, 60))"
    rule, new = E.parse(text, {})
    assert rule["any"][0] == {"left": {"price": "close"}, "op": "crosses_above",
                              "right": {"feature": "bbands_20_2_0", "output": "upper", "shift": 1}}
    again, _ = E.parse(E.to_text(rule, new), new)
    assert again == rule


def test_patterns_and_filters():
    rule, _ = E.parse("pattern('hammer', side='bullish', shadow_factor=1.5) and session('07:00', '16:00')", {})
    assert rule["all"][0] == {"pattern": "hammer", "module": "candle", "side": "bullish",
                              "params": {"shadow_factor": 1.5}}
    assert rule["all"][1] == {"filter": "session", "from": "07:00", "to": "16:00"}


def test_rejects_anything_outside_the_whitelist():
    for text in ("__import__('os').system('echo hi')", "close.__class__ > 1", "[x for x in close] > 1",
                 "lambda: 1", "close + 1 > 2", "open('file') > 1", "nope > 1"):
        with pytest.raises(ValueError):
            E.parse(text, {})


def test_parsed_rules_validate(ma_spec):
    rule, new = E.parse("ma(20) > ma(50) and weekday(0, 1, 2)", ma_spec["features"])
    ma_spec["features"].update(new)
    ma_spec["rules"]["long_entry"] = rule
    assert validate_spec(ma_spec) == []
