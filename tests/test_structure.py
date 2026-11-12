"""Structure search (phase 3): choice nodes, conditional spaces, and their backends."""

import copy

import pandas as pd
import pytest

from openstrategy_studio import compile as C, optimise as O
from openstrategy_studio.spec import (apply_params, collect_choices, collect_opts, load_template, resolve_choices,
                                      validate_spec)


@pytest.fixture
def search_spec():
    return copy.deepcopy(load_template("structure_search"))


def test_template_is_valid_in_every_alternative(search_spec):
    assert validate_spec(search_spec) == []
    assert list(collect_choices(search_spec)) == ["rules.long_entry"]


def test_apply_params_resolves_the_chosen_option(search_spec):
    applied = apply_params(search_spec, {"rules.long_entry.choose": "breakout"})
    assert applied["rules"]["long_entry"] == search_spec["rules"]["long_entry"]["options"]["breakout"]
    assert collect_choices(applied) == {} and collect_opts(applied) == {}
    default = apply_params(search_spec)
    assert default["rules"]["long_entry"]["all"][0]["op"] == "crosses_above"
    # A param inside the unchosen option is accepted and simply has no effect.
    other = apply_params(search_spec, {"rules.long_entry.options.reversion.all.0.right.const": 20.0,
                                       "rules.long_entry.choose": "breakout"})
    assert other == applied


def test_nested_choices_resolve():
    node = {"choose": "a", "options": {"a": {"choose": "y", "options": {"x": 1, "y": 2}}, "b": 3}}
    assert resolve_choices({"k": node}) == {"k": 2}


def test_an_unchosen_alternative_is_still_validated(search_spec):
    search_spec["rules"]["long_entry"]["options"]["breakout"]["all"][0]["right"] = {"feature": "missing"}
    problems = validate_spec(search_spec)
    assert any(p.startswith("With 'breakout' chosen at rules.long_entry") and "missing" in p for p in problems)


def test_bad_choice_structure(search_spec):
    search_spec["rules"]["long_entry"]["choose"]["opt"]["values"].append("nope")
    assert any("may only pick existing options" in p for p in validate_spec(search_spec))


def test_flat_space_fixes_the_structure(search_spec):
    space = O.build_space(search_spec)
    assert "rules.long_entry.choose" not in space
    assert "rules.long_entry.options.reversion.all.0.right.const" in space       # inside the chosen option
    assert "features.upper.window" in space and "features.rsi.window" in space
    search_spec["rules"]["long_entry"]["choose"]["value"] = "breakout"
    assert "rules.long_entry.options.reversion.all.0.right.const" not in O.build_space(search_spec)


def test_conditional_space_nests_each_option(search_spec):
    space = O.build_space(search_spec, conditional=True)
    selector = space["rules.long_entry.choose"]
    assert selector["values"] == ["reversion", "breakout"]
    assert set(selector["next_level"]["reversion"]) == {"rules.long_entry.options.reversion.all.0.right.const"}
    assert "breakout" not in selector["next_level"]          # that option has no parameters of its own
    assert O.is_conditional(space)
    path = {"rules.long_entry.choose": "reversion", "rules.long_entry.options.reversion.all.0.right.const": 25.0,
            "features.rsi.window": 10}
    assert set(O.active_space(space, path)) == set(path)
    assert "next_level" not in O.butchc_space(space)["features.rsi.window"]
    assert set(O.butchc_space(space)["rules.long_entry.choose"]["next_level"]["reversion"]) == \
        {"rules.long_entry.options.reversion.all.0.right.const"}


def test_flat_backends_refuse_conditional_spaces(search_spec):
    space = O.build_space(search_spec, conditional=True)
    for backend in ("grid", "random", "scipy"):
        with pytest.raises(ValueError, match="flat spaces only"):
            O.BACKENDS[backend]["fn"](space, lambda batch: [0.0] * len(batch), 4, "maximize", 0, {})


def test_compiled_alternative_matches_the_written_out_rule(search_spec, ohlcv):
    pytest.importorskip("vectorbt")
    chosen = apply_params(search_spec, {"rules.long_entry.choose": "breakout"})
    direct = copy.deepcopy(chosen)
    direct["rules"]["long_entry"] = {"all": [{"left": {"price": "close"}, "op": ">",
                                              "right": {"feature": "upper", "shift": 1}}]}
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
