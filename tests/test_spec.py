"""Tests 5 and 6: spec validation and round trip."""

import copy

from openstrategy_studio import spec as S


def problems_for(spec, **changes):
    spec = copy.deepcopy(spec)
    for path, value in changes.items():
        S.set_path(spec, path.replace("__", "."), value)
    return S.validate_spec(spec)


def test_templates_are_valid():
    for name in S.list_templates():
        assert S.validate_spec(S.load_template(name)) == [], name


def test_two_sizing_arguments(ma_spec):
    problems = problems_for(ma_spec, risk__sizing={"risk_pct": 0.01, "stake": 2})
    assert any("Choose one way to size positions" in p for p in problems)


def test_fixed_plus_trailing_stop(ma_spec):
    problems = problems_for(ma_spec, risk__stop={"kind": "fixed", "points": 60, "trailing_points": 30})
    assert any("cannot be used together" in p for p in problems)


def test_risk_pct_needs_fixed_stop(ma_spec):
    problems = problems_for(ma_spec, risk__stop={"kind": "none"})
    assert any("needs a fixed stop" in p for p in problems)


def test_dangling_feature_reference(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_entry"]["all"][0]["right"] = {"feature": "missing"}
    assert any("'missing' that does not exist" in p for p in S.validate_spec(spec))


def test_feature_cycle(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["features"]["fast"]["source"] = {"feature": "slow"}
    spec["features"]["slow"]["source"] = {"feature": "fast"}
    assert any("loop" in p for p in S.validate_spec(spec))


def test_rules_plus_weights(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["weighting"] = {"scheme": "equal"}
    spec["schedule"] = {"calendar": {"freq": "M", "on": "first"}}
    assert S.strategy_kind(spec) == "conflict"
    assert any("not both" in p for p in S.validate_spec(spec))


def test_filter_in_exit_is_rejected(ma_spec):
    spec = copy.deepcopy(ma_spec)
    spec["rules"]["long_exit"]["all"].append({"filter": "weekday", "days": [0]})
    assert any("only be used in entry rules" in p for p in S.validate_spec(spec))


def test_bad_opt_bounds(ma_spec):
    problems = problems_for(ma_spec, features__fast__window={"value": 50, "opt": {"min": 100, "max": 10}})
    assert any("minimum must be below the maximum" in p for p in problems)


def test_threshold_needs_calendar(momentum_spec):
    problems = problems_for(momentum_spec, schedule__calendar=None, schedule__threshold=0.05)
    assert any("needs a calendar" in p for p in problems)


def test_fx_instrument_needs_both_currencies(ma_spec):
    """docs/UPSTREAM.md issue 3: with a leg missing, vectorbtX skips the carry with only a warning."""
    for missing in ("base_currency", "quote_currency"):
        spec = copy.deepcopy(ma_spec)
        del spec["instruments"]["EURUSD=X"][missing]
        assert any("name both of its currencies" in p for p in S.validate_spec(spec)), missing
    # Not FX: an instrument without a pair is fine.
    spec = copy.deepcopy(ma_spec)
    spec["instruments"]["EURUSD=X"]["asset_class"] = "stocks"
    del spec["instruments"]["EURUSD=X"]["base_currency"]
    assert not any("currencies" in p for p in S.validate_spec(spec))


def test_cash_model(shared_spec, ma_spec, momentum_spec):
    assert S.cash_model(shared_spec) == "shared"
    separate = copy.deepcopy(shared_spec)
    separate["account"] = {"init_cash": 10_000, "freq": "D"}
    assert S.cash_model(separate) == "per_instrument"          # vectorbt's own default, named
    assert S.cash_model(ma_spec) is None                       # one symbol is one account either way
    assert S.cash_model(momentum_spec) is None                 # weights are shares of one account


def test_cash_settings_are_checked(shared_spec, momentum_spec):
    def says(problems, text):
        assert any(text in p for p in problems), (text, problems)

    says(problems_for(shared_spec, account__cash="joint"), "Cash must be per_instrument")
    says(problems_for(shared_spec, account__when_cash_short="halve"), "When cash runs short")
    says(problems_for(shared_spec, account__cash="per_instrument"), "only apply to a shared account")
    says(problems_for(shared_spec, account__entry_priority={"by": "volume"}), "must be by")
    says(problems_for(shared_spec, account__entry_priority={"by": "symbols", "order": ["AAPL", "TSLA"]}),
         "not in the list: TSLA")
    says(problems_for(shared_spec, account__entry_priority={"by": "symbols", "order": ["AAPL", "AAPL"]}), "twice")
    says(problems_for(shared_spec, account__entry_priority={"by": "random"}), "needs a random seed")
    assert problems_for(shared_spec, account__entry_priority={"by": "random"}, engine_extra__seed=1) == []
    says(problems_for(shared_spec, account__entry_priority__score={"const": 1}), "a constant ranks them all the same")
    says(problems_for(shared_spec, account__entry_priority__score={"feature": "missing"}), "does not exist")
    says(problems_for(shared_spec, engine_extra__cash_sharing=True), "not as engine settings")
    says(problems_for(momentum_spec, account__cash="per_instrument"), "shares of one account")
    says(problems_for(momentum_spec, account__entry_priority={"by": "random"}), "only entry and exit rule")
    hold = copy.deepcopy(shared_spec)
    hold["rules"] = {}
    says(S.validate_spec(hold), "Buy and hold in one shared account")


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


def test_apply_params_leaves_no_opt(ma_spec):
    applied = S.apply_params(ma_spec, {"features.fast.window": 20})
    assert applied["features"]["fast"]["window"] == 20
    assert applied["features"]["slow"]["window"] == 200
    assert S.collect_opts(applied) == {}
    assert "opt" not in S.canonical_json(applied)
    # The original is untouched.
    assert ma_spec["features"]["fast"]["window"]["value"] == 50


def test_apply_params_rejects_unknown_paths(ma_spec):
    import pytest
    with pytest.raises(ValueError, match="not optimisable"):
        S.apply_params(ma_spec, {"features.nope.window": 3})


def test_collect_opts_paths_are_dotted(ma_spec):
    assert list(S.collect_opts(ma_spec)) == ["features.fast.window", "features.slow.window"]


def test_feature_order_respects_dependencies():
    features = {"b": {"block": "rsi", "source": {"feature": "a"}}, "a": {"block": "ma", "source": "close"}}
    assert S.feature_order(features) == ["a", "b"]
