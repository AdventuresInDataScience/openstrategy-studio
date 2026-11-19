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


def test_an_exit_rule_takes_every_condition_kind(ma_spec):
    """The owner's call (17 November 2026): a weekday or higher-timeframe condition in an exit is a
    condition like any other. "A or Friday" is a time exit."""
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_exit"]["conditions"]["B"] = {"filter": "weekday", "days": [4]}
    spec["rules"]["long_exit"]["conditions"]["C"] = {"filter": "higher_timeframe", "timeframe": "1w",
                                                     "left": {"price": "close"}, "op": "<", "right": {"price": "open"}}
    spec["rules"]["long_exit"]["logic"] = "A or B or C"
    assert S.validate_spec(spec) == []


def test_a_condition_takes_one_timing_of_whole_bars(ma_spec):
    spec = copy.deepcopy(ma_spec)
    condition = spec["rules"]["long_entry"]["conditions"]["A"]
    for key in S.TIMING_KEYS:
        condition.update({key: 3})
        assert S.validate_spec(spec) == [], key
        del condition[key]
    condition.update(for_bars=3, within_bars=2)
    says(S.validate_spec(spec), "give one timing, not 2")
    condition.pop("within_bars")
    condition["for_bars"] = 0
    says(S.validate_spec(spec), "for bars must be at least 1")
    condition["for_bars"] = 2.5
    says(S.validate_spec(spec), "for bars must be a whole number")
    condition.pop("for_bars")
    spec["rules"]["long_entry"]["conditions"]["B"] = {"filter": "month", "months": [1], "bars_ago": 1}
    spec["rules"]["long_entry"]["logic"] = "A and B"
    says(S.validate_spec(spec), "takes no timing")
    # On a higher timeframe it counts the higher bars, and is checked on the condition itself.
    spec["rules"]["long_entry"]["conditions"]["B"] = {"filter": "higher_timeframe", "timeframe": "1w", "for_bars": 2,
                                                      "left": {"price": "close"}, "op": ">", "right": {"price": "open"}}
    assert S.validate_spec(spec) == []
    paths = {row["path"] for row in S.tunable_inputs(spec)}
    assert "rules.long_entry.conditions.B.for_bars" in paths


def test_what_a_trade_too_big_to_fund_does(ma_spec, shared_spec):
    """Any rule strategy chooses skip or partial; the default follows the account."""
    assert S.when_cash_short(ma_spec) == "partial"
    single = copy.deepcopy(ma_spec)
    single["account"]["when_cash_short"] = "skip"
    assert S.validate_spec(single) == [] and S.when_cash_short(single) == "skip"
    shared = copy.deepcopy(shared_spec)
    shared["account"].pop("when_cash_short", None)
    assert S.when_cash_short(shared) == "skip"
    # The priority still needs a shared account.
    single["account"]["entry_priority"] = {"by": "symbols"}
    says(S.validate_spec(single), "only applies to a shared account")


def test_logic_line_problems(ma_spec):
    spec = copy.deepcopy(ma_spec)
    rule = spec["rules"]["long_entry"]
    rule["conditions"]["B"] = {"left": {"price": "close"}, "op": ">", "right": {"const": 1.0}}
    rule["conditions"]["C"] = {"left": {"price": "close"}, "op": "<", "right": {"const": 2.0}}
    for logic, text in (("A and B or C", "meet without brackets"), ("A and (B", "never closed"),
                        ("A and D and B and C", "D, which is not a condition"), ("A and B", "C is not in the logic"),
                        ("A B C", "follows without an operator"), ("A and 3", "not a condition name"),
                        ("a and B and C", "capitals included: a is not A")):
        rule["logic"] = logic
        says(S.validate_spec(spec), text)
    rule["logic"] = "(A and B) or not C"
    assert S.validate_spec(spec) == []
    # A condition's name follows the same rules as the Build page's Name box.
    rule["conditions"]["not"] = rule["conditions"].pop("C")
    rule["logic"] = "(A and B) or not not"
    says(S.validate_spec(spec), "would read as an operator")


def test_renaming_a_condition_carries_the_logic_and_the_variables(ma_spec, ohlcv):
    """Renamed in place: the same position, the logic line and the optimise variables follow, and
    the strategy computes exactly what it did."""
    pytest.importorskip("vectorbt")
    import pandas as pd

    from openstrategy_studio import compile as C

    spec = copy.deepcopy(ma_spec)
    rule = spec["rules"]["long_entry"]
    rule["conditions"]["B"] = {"left": {"price": "close"}, "op": ">", "right": {"const": 1.0}}
    rule["logic"] = "(A)  and B"
    spec["optimise"]["variables"]["rules.long_entry.conditions.A.op"] = {"values": ["crosses_above", "crosses_below"]}
    spec["optimise"]["variables"]["rules.long_entry.conditions.B.right.const"] = {"start": 0.5, "stop": 1.5}
    before = copy.deepcopy(spec)
    S.rename_condition(spec, "rules.long_entry", "A", "trend")
    rule = spec["rules"]["long_entry"]
    assert list(rule["conditions"]) == ["trend", "B"] and rule["logic"] == "(trend)  and B"
    assert rule["conditions"]["trend"] == before["rules"]["long_entry"]["conditions"]["A"]
    assert list(spec["optimise"]["variables"]) == [
        "features.fast.params.timeperiod", "features.slow.params.timeperiod",
        "rules.long_entry.conditions.trend.op", "rules.long_entry.conditions.B.right.const"]
    # Only that rule: the exit's own condition A is untouched.
    assert spec["rules"]["long_exit"] == before["rules"]["long_exit"]
    assert S.validate_spec(spec) == []
    params = {"rules.long_entry.conditions.trend.op": "crosses_below"}
    renamed = C.compile_signals(S.apply_params(spec, params), ohlcv)
    original = C.compile_signals(S.apply_params(before, {"rules.long_entry.conditions.A.op": "crosses_below"}), ohlcv)
    pd.testing.assert_frame_equal(renamed["entries"], original["entries"])
    # Inside a version of a rule too.
    search = S.load_template("structure_search")
    S.rename_condition(search, "rules.long_entry.options.reversion", "A", "oversold")
    assert S.validate_spec(search) == []
    assert "rules.long_entry.options.reversion.conditions.oversold.right.const" in search["optimise"]["variables"]


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
    says(problems_for(shared_spec, account__when_cash_short="halve"), "When a trade is bigger than the account can fund")
    says(problems_for(shared_spec, account__cash="per_instrument"), "The entry priority only applies to a shared account")
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
    # Risk sizing and the extra exits need a fixed stop, so its kind is not offered; the target's is.
    assert "risk.stop.kind" not in paths
    assert paths["risk.target.kind"]["options"] == ["none", "fixed"]


def test_every_value_in_a_condition_can_be_optimised(ma_spec, ohlcv):
    """Bars ago on a line at 0 included (a trial may try 0 against 1 and 2), a pattern's side, and a
    higher-timeframe condition's timeframe; a number has no bars ago."""
    from openstrategy_studio import indicators
    from openstrategy_studio.compile import compile_signals

    spec = copy.deepcopy(ma_spec)
    pattern = next(item for item in indicators.catalogue() if item.startswith("pattern:"))
    spec["features"]["engulf"] = indicators.new_feature(pattern)
    entry = spec["rules"]["long_entry"]
    entry["conditions"]["B"] = {"pattern": "engulf", "side": "bullish"}
    entry["conditions"]["C"] = {"filter": "higher_timeframe", "timeframe": "1w", "left": {"price": "close"},
                                "op": ">", "right": {"const": 0.0}}
    entry["logic"] = "A and B and C"
    paths = {row["path"]: row for row in S.tunable_inputs(spec)}
    base = "rules.long_entry.conditions"
    assert paths[f"{base}.A.left.shift"]["value"] == 0 and paths[f"{base}.A.right.shift"]["type"] == "int"
    assert paths[f"{base}.B.side"]["options"] == ["bullish", "bearish", "either"]
    assert paths[f"{base}.C.timeframe"]["options"] == ["1w", "1M"]           # above the data's 1d only
    assert f"{base}.C.left.shift" in paths and f"{base}.C.right.shift" not in paths
    assert S.default_variable(paths[f"{base}.A.left.shift"]) == {"start": 0, "stop": 2, "step": 1}
    # A trial's 0 is the current bar: the same signals as no shift at all.
    tried = S.apply_params(ma_spec, {f"{base}.A.left.shift": 0})
    assert tried["rules"]["long_entry"]["conditions"]["A"]["left"]["shift"] == 0 and not S.validate_spec(tried)
    assert compile_signals(tried, ohlcv)["entries"].equals(compile_signals(S.apply_params(ma_spec), ohlcv)["entries"])


def test_the_stop_and_target_can_be_switched_on_and_off_by_the_optimiser():
    # A strategy with neither, sized without risk: both kinds are choices, and their distances stay.
    spec = S.load_template("rsi_reversion")
    paths = {row["path"]: row for row in S.tunable_inputs(spec)}
    assert paths["risk.stop.kind"]["options"] == ["none", "fixed", "trailing"]
    assert paths["risk.stop.kind"]["value"] == "none" and paths["risk.stop.points"]["value"] == 50
    assert "not used now" in paths["risk.stop.points"]["label"]
    trial = S.apply_params(spec, {"risk.stop.kind": "trailing", "risk.stop.points": 80.0, "risk.target.kind": "fixed"})
    assert trial["risk"]["stop"] == {"kind": "trailing", "unit": "points", "points": 80.0, "pct": 0.02}
    assert trial["risk"]["target"] == {"kind": "fixed", "unit": "points", "points": 100, "pct": 0.04}
    assert S.validate_spec(trial) == []
    # Every kind and unit a variable may pick is valid with the rest of the spec.
    spec["optimise"] = {"variables": {"risk.stop.kind": {"values": ["none", "fixed", "trailing"]},
                                      "risk.stop.unit": {"values": ["points", "pct"]},
                                      "risk.target.kind": {"values": ["none", "fixed"]},
                                      "risk.target.unit": {"values": ["points", "pct"]}}}
    assert S.validate_spec(spec) == []


def test_a_stop_in_percent_is_a_fraction_of_the_price_and_cannot_size_a_risk_yet(ma_spec):
    spec = copy.deepcopy(ma_spec)
    paths = {row["path"]: row for row in S.tunable_inputs(spec)}
    # Risk-sized, so the stop stays fixed and in points; the target may switch either way.
    assert "risk.stop.unit" not in paths and paths["risk.target.unit"]["options"] == ["points", "pct"]
    assert "not used now" in paths["risk.stop.pct"]["label"]
    spec["risk"]["stop"]["unit"] = "pct"
    assert any("cannot size a trade from a stop in %" in problem or "cannot size from a stop in %" in problem
               for problem in S.validate_spec(spec))
    spec["risk"]["sizing"] = {"stake": 1.0}
    assert S.validate_spec(spec) == []
    spec["risk"]["stop"]["pct"] = 2                         # meant 2%, typed as 200%
    assert any("0.02 is 2%" in problem for problem in S.validate_spec(spec))
    spec["risk"]["stop"]["pct"] = 0.02
    spec["risk"]["exits"] = {"time_bars": 10}
    assert any("measure the stop in points to use them" in problem for problem in S.validate_spec(spec))


def test_a_distance_measured_the_other_way_is_named(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["optimise"] = {"variables": {"risk.target.pct": {"start": 0.01, "stop": 0.05, "step": 0.01}}}
    assert any("the take profit is measured in points, so this distance changes nothing" in problem
               for problem in S.validate_spec(spec))
    spec["optimise"]["variables"]["risk.target.unit"] = {"values": ["points", "pct"]}
    assert S.validate_spec(spec) == []


def test_a_distance_searched_with_its_stop_off_is_named():
    # Every trial would be the same backtest, so validate_spec says what to do instead.
    spec = S.load_template("rsi_reversion")
    spec["optimise"] = {"variables": {"risk.stop.points": {"start": 20.0, "stop": 80.0, "step": 20.0}}}
    assert any("there is no stop, so its distance changes nothing" in problem for problem in S.validate_spec(spec))
    spec["optimise"]["variables"]["risk.stop.kind"] = {"values": ["none", "fixed"]}
    assert S.validate_spec(spec) == []


def test_a_stop_or_target_without_its_kind_is_rejected(ma_spec):
    # ⚠️ Read as none, it would trade unprotected (or past its target) with nothing to say so.
    spec = copy.deepcopy(ma_spec)
    spec["risk"]["target"] = {"points": 120}
    assert any("take profit must say its kind" in problem for problem in S.validate_spec(spec))
    spec = copy.deepcopy(ma_spec)
    spec["risk"]["stop"] = {"points": 60}
    assert any("stop must say its kind" in problem for problem in S.validate_spec(spec))


def test_variable_values():
    assert S.variable_values({"start": 10, "stop": 50, "step": 10}, "int") == [10, 20, 30, 40, 50]
    assert S.variable_values({"values": [1, 3]}, "int") == [1, 3]
    assert S.variable_values({"start": 0.5, "stop": 1.0, "step": 0.25}, "float") == [0.5, 0.75, 1.0]


def test_feature_order_respects_dependencies():
    features = {"b": {"indicator": "talib:RSI", "inputs": {"close": {"feature": "a"}}},
                "a": {"indicator": "talib:SMA", "inputs": {"close": "close"}}}
    assert S.feature_order(features) == ["a", "b"]


def test_feature_names_are_checked():
    assert S.feature_name_problem("fast", {"slow"}) is None and S.feature_name_problem("ema_2", set()) is None
    for bad in ("", "2x", "_x", "my ema", "a-b", "é", "close", "volume"):
        assert "cannot name a feature" in S.feature_name_problem(bad, set()), bad
    assert "already a feature called slow" in S.feature_name_problem("slow", {"slow"})
    spec = S.load_template("ma_cross")
    spec["features"]["_fast"] = spec["features"].pop("fast")
    says(S.validate_spec(spec), "cannot name a feature")


def _compiled(spec):
    """What a spec computes, on synthetic data for its own symbols: its signals, or its weights."""
    from openstrategy_studio import compile as C
    from conftest import make_ohlcv

    data = make_ohlcv(300, tuple(spec["data"]["symbols"]), seed=4, base=100.0)
    applied = S.apply_params(spec)
    if S.strategy_kind(applied) == "signals":
        return C.compile_signals(applied, data)
    return C.compile_weights(applied, data)


def test_renaming_a_feature_carries_every_reference():
    """Every feature of every template, renamed: nothing refers to the old name, the spec is valid,
    the feature keeps its place, and the strategy computes exactly what it did. Between them the
    templates hold pattern conditions, a selection score, an entry priority, versions of a rule
    and shifted references; a chained input and a higher-timeframe filter are added here."""
    pytest.importorskip("vectorbt")
    import json

    import pandas as pd

    for template in S.list_templates():
        base = S.load_template(template)
        if base.get("features") and S.strategy_kind(base) == "signals":
            first = next(iter(base["features"]))
            # A feature that reads another, and a weekly filter on that feature (entries only).
            base["features"]["chained"] = {"indicator": "talib:EMA", "inputs": {"close": {"feature": first}},
                                           "params": {"timeperiod": 5}}
            entry = base["rules"]["long_entry"]
            rule = entry["options"][entry["choose"]] if S.is_choice(entry) else entry
            rule["conditions"]["Z"] = {"filter": "higher_timeframe", "timeframe": "1w",
                                       "left": {"feature": first}, "op": ">", "right": {"feature": "chained"}}
            rule["logic"] = f"({rule['logic']}) or Z"
        assert S.validate_spec(base) == [], template
        before = _compiled(base)
        for old in list(base.get("features") or {}):
            spec = json.loads(json.dumps(base))
            new = f"{old}_renamed"
            S.rename_feature(spec, old, new)
            order = [new if name == old else name for name in base["features"]]
            assert list(spec["features"]) == order, (template, old)
            text = json.dumps({key: value for key, value in spec.items() if key != "optimise"})
            assert f'"feature": "{old}"' not in text and f'"pattern": "{old}"' not in text, (template, old)
            for path in base["optimise"]["variables"]:
                moved = path.replace(f"features.{old}.", f"features.{new}.", 1) if path.startswith(f"features.{old}.") \
                    else path
                assert moved in spec["optimise"]["variables"], (template, old, path)
            assert S.validate_spec(spec) == [], (template, old, S.validate_spec(spec))
            after = _compiled(spec)
            for name, frame in before.items():
                if isinstance(frame, pd.DataFrame):
                    pd.testing.assert_frame_equal(after[name], frame, obj=f"{template} {old} {name}")
