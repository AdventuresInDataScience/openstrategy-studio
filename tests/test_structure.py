"""Structure search (phase 3): versions of a rule, conditional spaces, and their backends."""

import copy

import pandas as pd
import pytest

from openstrategy_studio import compile as C, optimise as O
from openstrategy_studio.spec import apply_params, collect_choices, load_template, resolve_choices, validate_spec

INSIDE = "rules.long_entry.options.reversion.conditions.A.right.const"


@pytest.fixture
def search_spec():
    return copy.deepcopy(load_template("structure_search"))


def test_template_is_valid_in_every_version(search_spec):
    assert validate_spec(search_spec) == []
    assert list(collect_choices(search_spec)) == ["rules.long_entry"]
    assert INSIDE in search_spec["optimise"]["variables"]


def test_apply_params_resolves_the_chosen_version(search_spec):
    applied = apply_params(search_spec, {"rules.long_entry.choose": "breakout"})
    assert applied["rules"]["long_entry"] == search_spec["rules"]["long_entry"]["options"]["breakout"]
    assert collect_choices(applied) == {} and "optimise" not in applied
    default = apply_params(search_spec)
    assert default["rules"]["long_entry"]["conditions"]["A"]["op"] == "crosses_above"
    # A value inside the unchosen version is accepted and simply has no effect.
    other = apply_params(search_spec, {INSIDE: 20.0, "rules.long_entry.choose": "breakout"})
    assert other == applied


def test_apply_params_refuses_a_value_the_strategy_cannot_vary(search_spec):
    with pytest.raises(ValueError, match="cannot be varied"):
        apply_params(search_spec, {"features.rsi.params.nope": 3})


def test_nested_choices_resolve():
    node = {"choose": "a", "options": {"a": {"choose": "y", "options": {"x": 1, "y": 2}}, "b": 3}}
    assert resolve_choices({"k": node}) == {"k": 2}


def test_an_unchosen_version_is_still_validated(search_spec):
    search_spec["rules"]["long_entry"]["options"]["breakout"]["conditions"]["A"]["right"] = {"feature": "missing"}
    problems = validate_spec(search_spec)
    assert any(p.startswith("With version 'breakout' at rules.long_entry") and "missing" in p for p in problems)


def test_a_variable_may_only_pick_existing_versions(search_spec):
    search_spec["optimise"]["variables"]["rules.long_entry.choose"]["values"].append("nope")
    assert any("every value must be one of" in p for p in validate_spec(search_spec))


def test_flat_space_fixes_the_structure(search_spec):
    space = O.build_space(search_spec)
    assert "rules.long_entry.choose" not in space
    assert INSIDE in space                                        # inside the version in use
    assert "features.upper.params.timeperiod" in space and "features.rsi.params.timeperiod" in space
    assert space["features.rsi.params.timeperiod"]["int"] is True
    search_spec["rules"]["long_entry"]["choose"] = "breakout"
    assert INSIDE not in O.build_space(search_spec)


def test_conditional_space_nests_each_version(search_spec):
    space = O.build_space(search_spec, conditional=True)
    selector = space["rules.long_entry.choose"]
    assert selector["values"] == ["reversion", "breakout"]
    assert set(selector["next_level"]["reversion"]) == {INSIDE}
    assert "breakout" not in selector["next_level"]          # that version has no variables of its own
    assert O.is_conditional(space)
    path = {"rules.long_entry.choose": "reversion", INSIDE: 25.0, "features.rsi.params.timeperiod": 10}
    assert set(O.active_space(space, path)) == set(path)
    assert "next_level" not in O.butchc_space(space)["features.rsi.params.timeperiod"]
    assert set(O.butchc_space(space)["rules.long_entry.choose"]["next_level"]["reversion"]) == {INSIDE}


def test_flat_backends_refuse_conditional_spaces(search_spec):
    space = O.build_space(search_spec, conditional=True)
    for backend in ("grid", "random", "scipy"):
        with pytest.raises(ValueError, match="flat spaces only"):
            O.BACKENDS[backend]["fn"](space, lambda batch: [0.0] * len(batch), 4, "maximize", 0, {})


def test_compiled_version_matches_the_written_out_rule(search_spec, ohlcv):
    pytest.importorskip("vectorbt")
    chosen = apply_params(search_spec, {"rules.long_entry.choose": "breakout"})
    direct = copy.deepcopy(chosen)
    direct["rules"]["long_entry"] = {"conditions": {"Z": {"left": {"price": "close"}, "op": ">",
                                                          "right": {"feature": "upper", "shift": 1}}}, "logic": "Z"}
    pd.testing.assert_frame_equal(C.compile_signals(chosen, ohlcv)["entries"], C.compile_signals(direct, ohlcv)["entries"])


def conditional_objective(params):
    # "b" with x near 7 scores 2; "a" with y near 3 scores at most 1.
    if params["family"] == "b":
        return 2.0 - abs(params["x"] - 7.0) / 10
    return 1.0 - abs(params["y"] - 3) / 10


CONDITIONAL = {"family": {"values": ["a", "b"], "next_level": {"a": {"y": {"min": 0, "max": 10, "int": True}},
                                                               "b": {"x": {"min": 0.0, "max": 10.0}}}}}


def test_optuna_searches_a_conditional_space():
    pytest.importorskip("optuna")
    seen = []

    def evaluate_batch(batch):
        seen.extend(batch)
        return [conditional_objective(params) for params in batch]

    O.run_optuna(CONDITIONAL, evaluate_batch, 60, "maximize", 0, {"batch": 4})
    assert all(("x" in p) != ("y" in p) for p in seen), "only the chosen branch's parameters are sampled"
    best = max(seen, key=conditional_objective)
    assert best["family"] == "b"


def test_butchc_searches_a_conditional_space():
    pytest.importorskip("butchc")
    seen = []

    def evaluate_batch(batch):
        seen.extend(batch)
        return [conditional_objective(params) for params in batch]

    O.run_butchc(CONDITIONAL, evaluate_batch, 80, "maximize", 0, {"batch": 4})
    assert max(seen, key=conditional_objective)["family"] == "b"
