"""Tests 5 and 6: spec validation and round trip (spec version 2)."""

import copy

import pytest

from openstrategy_studio import spec as S


def problems_for(spec, **changes):
    spec = copy.deepcopy(spec)
    for path, value in changes.items():
        S.set_path(spec, path.replace("__", "."), value)
    return S.validate_spec(spec)


def says(problems, text):
    assert any(text in p for p in problems), (text, problems)


def test_templates_are_valid():
    for name in S.list_templates():
        assert S.validate_spec(S.load_template(name)) == [], name


def test_an_old_version_is_refused(ma_spec):
    says(problems_for(ma_spec, version=1), "the Studio reads version 2")


def test_setup_names_what_and_how(ma_spec):
    says(problems_for(ma_spec, setup__kind="hold"), "what you are building")
    says(problems_for(ma_spec, setup__trading="futures"), "how you are trading it")
    spec = copy.deepcopy(ma_spec)
    spec["instruments"]["EURUSD=X"]["mode"] = "cfd"
    says(S.validate_spec(spec), "set once, on the Setup step")


def test_two_sizing_arguments(ma_spec):
    says(problems_for(ma_spec, risk__sizing={"risk_pct": 0.01, "stake": 2}), "Choose one way to size positions")


def test_fixed_plus_trailing_stop(ma_spec):
    says(problems_for(ma_spec, risk__stop={"kind": "fixed", "points": 60, "trailing_points": 30}),
         "cannot be used together")


def test_risk_pct_needs_fixed_stop(ma_spec):
    says(problems_for(ma_spec, risk__stop={"kind": "none"}), "needs a fixed stop")


def test_dangling_feature_reference(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_entry"]["conditions"]["A"]["right"] = {"feature": "missing"}
    says(S.validate_spec(spec), "'missing' that does not exist")


def test_feature_settings_are_checked(ma_spec):
    says(problems_for(ma_spec, features__fast__indicator="talib:NOPE"), "unknown indicator")
    says(problems_for(ma_spec, features__fast__params__window=5), "not a setting of")
    says(problems_for(ma_spec, features__fast__params__timeperiod=5.5), "whole number")
    says(problems_for(ma_spec, features__fast__inputs__nope="close"), "not an input of")
    says(problems_for(ma_spec, features__fast__inputs__close="vwap"), "a price field or another feature")
    bands = copy.deepcopy(ma_spec)
    bands["features"]["bb"] = {"indicator": "talib:BBANDS", "inputs": {}, "params": {"matype": "LINEAR"}}
    says(S.validate_spec(bands), "must be one of SMA")


def test_feature_cycle(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["features"]["fast"]["inputs"]["close"] = {"feature": "slow"}
    spec["features"]["slow"]["inputs"]["close"] = {"feature": "fast"}
    says(S.validate_spec(spec), "loop")


def test_rules_and_portfolio_settings_do_not_mix(ma_spec, momentum_spec):
    spec = copy.deepcopy(ma_spec)
    spec["weighting"] = {"scheme": "equal"}
    says(S.validate_spec(spec), "has no portfolio settings")
    spec = copy.deepcopy(momentum_spec)
    spec["rules"] = {"long_entry": copy.deepcopy(ma_spec["rules"]["long_entry"])}
    says(S.validate_spec(spec), "has no entry or exit rules")


def test_filter_in_exit_is_rejected(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_exit"]["conditions"]["B"] = {"filter": "weekday", "days": [0]}
    spec["rules"]["long_exit"]["logic"] = "A and B"
    says(S.validate_spec(spec), "only be used in entry rules")


def test_logic_line_problems(ma_spec):
    spec = copy.deepcopy(ma_spec)
    rule = spec["rules"]["long_entry"]
    rule["conditions"]["B"] = {"left": {"price": "close"}, "op": ">", "right": {"const": 1.0}}
    rule["conditions"]["C"] = {"left": {"price": "close"}, "op": "<", "right": {"const": 2.0}}
    for logic, text in (("A and B or C", "meet without brackets"), ("A and (B", "never closed"),
                        ("A and D and B and C", "D, which is not a condition"), ("A and B", "C is not in the logic"),
                        ("A B C", "follows without an operator"), ("A and 3", "not a condition letter")):
        rule["logic"] = logic
        says(S.validate_spec(spec), text)
    rule["logic"] = "(A and B) or not C"
    assert S.validate_spec(spec) == []


def test_pattern_condition_needs_a_pattern_feature(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_entry"]["conditions"]["A"] = {"pattern": "fast", "side": "bullish"}
    says(S.validate_spec(spec), "'fast' is not a pattern")


def test_optimise_variables_are_checked(ma_spec):
    says(problems_for(ma_spec, optimise__variables={"features.gone.params.timeperiod": {"start": 1, "stop": 2}}),
         "no longer exists")
    variables = copy.deepcopy(ma_spec["optimise"]["variables"])
    variables["features.fast.params.timeperiod"] = {"start": 100, "stop": 10}
    says(problems_for(ma_spec, optimise__variables=variables), "start must be below the stop")
    variables["features.fast.params.timeperiod"] = {"values": []}
    says(problems_for(ma_spec, optimise__variables=variables), "at least one value")
    variables["features.fast.params.timeperiod"] = {"values": [10, 20.5]}
    says(problems_for(ma_spec, optimise__variables=variables), "whole number")


def test_threshold_needs_calendar(momentum_spec):
    says(problems_for(momentum_spec, schedule__rebalance="every_bar", schedule__calendar=None,
                      schedule__threshold=0.05), "needs a calendar")


def test_rebalance_must_be_named(momentum_spec):
    says(problems_for(momentum_spec, schedule__rebalance="sometimes"), "Rebalance never")


def test_order_timing_needs_intraday_bars(ma_spec):
    says(problems_for(ma_spec, risk__orders={"good_for_day": True}), "need intraday bars")
    spec = copy.deepcopy(ma_spec)
    spec["data"]["timeframe"] = "1h"
    spec["risk"]["orders"] = {"good_for_day": True, "flat_by_day_end": True}
    assert not any("intraday" in p for p in S.validate_spec(spec))


def test_fx_instrument_needs_both_currencies(ma_spec):
    """docs/UPSTREAM.md issue 3: with a leg missing, vectorbtX skips the carry with only a warning."""
    for missing in ("base_currency", "quote_currency"):
        spec = copy.deepcopy(ma_spec)
        del spec["instruments"]["EURUSD=X"][missing]
        says(S.validate_spec(spec), "name both of its currencies")
    # Not FX: an instrument without a pair is fine.
    spec = copy.deepcopy(ma_spec)
    spec["instruments"]["EURUSD=X"]["asset_class"] = "stocks"
    del spec["instruments"]["EURUSD=X"]["base_currency"]
    assert not any("currencies" in p for p in S.validate_spec(spec))


def test_borrow_rate_is_for_buying_outright_and_never_negative(shared_spec, ma_spec):
    """vectorbtX 9fa427e: the borrow fee follows how a thing is traded. A direct (equity)
    short borrows what it sells; a spread bet or CFD short pays its funding instead."""
    equity = copy.deepcopy(shared_spec)
    symbol = equity["data"]["symbols"][0]
    for rate in (0.0, 0.005):
        equity["instruments"][symbol]["borrow_rate"] = rate
        assert not any("borrow_rate" in p for p in S.validate_spec(equity)), rate
    equity["instruments"][symbol]["borrow_rate"] = -0.01
    says(S.validate_spec(equity), "0 or more")
    spread_bet = copy.deepcopy(ma_spec)
    spread_bet["instruments"]["EURUSD=X"]["borrow_rate"] = 0.005
    says(S.validate_spec(spread_bet), "only when buying outright")


def test_cash_model(shared_spec, ma_spec, momentum_spec):
    assert S.cash_model(shared_spec) == "shared"
    separate = copy.deepcopy(shared_spec)
    separate["account"] = {"init_cash": 10_000, "freq": "D"}
    assert S.cash_model(separate) == "per_instrument"          # vectorbt's own default, named
    assert S.cash_model(ma_spec) is None                       # one symbol is one account either way
    assert S.cash_model(momentum_spec) is None                 # weights are shares of one account


def test_cash_settings_are_checked(shared_spec, momentum_spec):
    says(problems_for(shared_spec, account__cash="joint"), "Cash must be per_instrument")
    says(problems_for(shared_spec, account__when_cash_short="halve"), "When cash runs short")
    says(problems_for(shared_spec, account__cash="per_instrument"), "only apply to a shared account")
    says(problems_for(shared_spec, account__entry_priority={"by": "volume"}), "must be by")
    says(problems_for(shared_spec, account__entry_priority={"by": "symbols", "order": ["AAPL", "TSLA"]}),
         "not in the list: TSLA")
    says(problems_for(shared_spec, account__entry_priority={"by": "symbols", "order": ["AAPL", "AAPL"]}), "twice")
    says(problems_for(shared_spec, account__entry_priority={"by": "random"}), "needs a random seed")
    assert problems_for(shared_spec, account__entry_priority={"by": "random"}, engine_extra__seed=1) == []
    says(problems_for(shared_spec, account__entry_priority__score={"const": 1}), "a number ranks them all the same")
    says(problems_for(shared_spec, account__entry_priority__score={"feature": "missing"}), "does not exist")
    says(problems_for(shared_spec, engine_extra__cash_sharing=True), "not as engine settings")
    says(problems_for(momentum_spec, account__cash="per_instrument"), "shares of one account")
    says(problems_for(momentum_spec, account__entry_priority={"by": "random"}), "only entry and exit rule")


def test_good_spec_has_no_problems(ma_spec):
    assert S.validate_spec(ma_spec) == []


def test_round_trip(ma_spec, tmp_path):
    path = tmp_path / "strategy.json"
    S.save_spec(ma_spec, path)
    loaded = S.load_spec(path)
    S.save_spec(loaded, path)
    assert S.load_spec(path) == ma_spec
    assert S.spec_hash(loaded) == S.spec_hash(ma_spec)
    assert len(S.spec_hash(ma_spec)) == 16


def test_apply_params_writes_values_and_drops_the_ranges(ma_spec):
    applied = S.apply_params(ma_spec, {"features.fast.params.timeperiod": 20})
    assert applied["features"]["fast"]["params"]["timeperiod"] == 20
    assert applied["features"]["slow"]["params"]["timeperiod"] == 200
    assert "optimise" not in applied
    # The original is untouched.
    assert ma_spec["features"]["fast"]["params"]["timeperiod"] == 50


def test_apply_params_rejects_unknown_paths(ma_spec):
    with pytest.raises(ValueError, match="cannot be varied"):
        S.apply_params(ma_spec, {"features.nope.params.timeperiod": 3})


def test_tunable_inputs_include_the_risk_dials(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["risk"]["exits"] = {"time_bars": 20}
    paths = {row["path"]: row for row in S.tunable_inputs(spec)}
    for path in ("features.fast.params.timeperiod", "risk.stop.points", "risk.target.points", "risk.sizing.risk_pct",
                 "risk.exits.time_bars"):
        assert path in paths, path
    assert paths["risk.exits.time_bars"]["type"] == "int" and paths["risk.stop.points"]["value"] == 60
    assert S.apply_params(spec, {"risk.stop.points": 80.0})["risk"]["stop"]["points"] == 80.0


def test_variable_values():
    assert S.variable_values({"start": 10, "stop": 50, "step": 10}, "int") == [10, 20, 30, 40, 50]
    assert S.variable_values({"values": [1, 3]}, "int") == [1, 3]
    assert S.variable_values({"start": 0.5, "stop": 1.0, "step": 0.25}, "float") == [0.5, 0.75, 1.0]


def test_feature_order_respects_dependencies():
    features = {"b": {"indicator": "talib:RSI", "inputs": {"close": {"feature": "a"}}},
                "a": {"indicator": "talib:SMA", "inputs": {"close": "close"}}}
    assert S.feature_order(features) == ["a", "b"]
