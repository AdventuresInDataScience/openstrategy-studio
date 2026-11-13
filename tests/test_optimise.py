"""Test 12: optimiser plumbing on a cheap synthetic objective with a known optimum."""


import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import optimise as O

SPACE = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 1, "max": 50, "int": True},
         "c": {"values": ["x", "y", "z"]}}
OPTIMUM = 1.0               # numbers at 30% of their range (0), plus 1 for c == "y"


def drive(backend, budget=200, seed=0, options=None):
    trials = []

    def evaluate_batch(params_list):
        values = [O.selftest_objective(SPACE, params) for params in params_list]
        trials.extend(zip(params_list, values))
        return values

    O.BACKENDS[backend]["fn"](SPACE, evaluate_batch, budget, "maximize", seed, options or {})
    finite = [(params, value) for params, value in trials if np.isfinite(value)]
    best = max(finite, key=lambda item: item[1])
    return trials, best


def check(backend, tolerance=0.05, **kwargs):
    trials, (params, value) = drive(backend, **kwargs)
    assert len(trials) <= kwargs.get("budget", 200)
    assert any(not np.isfinite(v) for _, v in trials) or backend == "grid", "the NaN trap was never visited"
    assert params["a"] <= 8.5, "a NaN trial must never be reported as the best"
    assert value >= OPTIMUM - tolerance, (backend, params, value)
    return trials


def test_grid():
    trials = check("grid", tolerance=0.01)
    assert len(trials) == 200


def test_random_sobol():
    check("random", tolerance=0.1)


def test_scipy_differential_evolution():
    check("scipy", tolerance=0.05, options={"method": "differential_evolution"})


def test_scipy_dual_annealing():
    check("scipy", tolerance=0.1, options={"method": "dual_annealing"})


def test_optuna():
    pytest.importorskip("optuna")
    check("optuna", tolerance=0.05, options={"batch": 4})


def test_butchc():
    pytest.importorskip("butchc")
    check("butchc", tolerance=0.1, options={"batch": 4})


def test_butchc_is_reproducible_with_an_executor():
    pytest.importorskip("butchc")
    first, _ = drive("butchc", budget=60, seed=7, options={"batch": 4})
    second, _ = drive("butchc", budget=60, seed=7, options={"batch": 4})
    # The selftest objective returns NaN in its trap, and NaN != NaN, so compare the
    # text: repr is exact for floats and spells every NaN the same way.
    assert repr(first) == repr(second)


def test_butchc_conditional_space():
    butchc = pytest.importorskip("butchc")
    space = {"family": {"values": ["ma_cross", "rsi"], "next_level": {
        "ma_cross": {"gap": {"min": 10, "max": 200, "int": True}},
        "rsi": {"level": {"min": 20.0, "max": 40.0}}}}}

    def objective(config):
        if config["family"] == "rsi":
            return -abs(config["level"] - 30.0)
        return -abs(config["gap"] - 50) / 10 - 1.0

    result = butchc.BUTChC_optimize(space, objective, 80, direction="maximize", seed=0, verbose=False)
    assert result["best_params"]["family"] == "rsi"
    assert "gap" not in result["best_params"]


def test_every_metric_scores_a_real_backtest(ma_spec, ohlcv, momentum_spec, ohlcv_multi):
    """Each objective on the Optimise page returns a finite number (phase 0: three used to raise every trial)."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    fast = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    for spec, data, params in ((ma_spec, ohlcv, fast), (momentum_spec, ohlcv_multi, None)):
        pf = engine.run_backtest(spec, data, params=params)["pf"]
        for name, metric in O.METRICS.items():
            value = metric["fn"](pf)
            assert isinstance(value, float) and np.isfinite(value), (name, value)


def test_marsopt():
    pytest.importorskip("marsopt")
    # marsopt 0.2.0 converges quickly and samples the top 15% of `a` (the NaN trap) in
    # under 1% of trials, so seed 0 never visits it and the NaN check would prove nothing.
    # Seed 1 (deterministic via random_state) visits it; the assertions are unchanged.
    check("marsopt", tolerance=0.1, budget=150, seed=1)


def test_space_helpers(ma_spec):
    space = O.build_space(ma_spec)
    assert space == {"features.fast.params.timeperiod": {"min": 10, "max": 100, "int": True, "step": 9},
                     "features.slow.params.timeperiod": {"min": 100, "max": 300, "int": True, "step": 20}}
    assert O.build_space(ma_spec, include=["features.fast.params.timeperiod"]).keys() ==         {"features.fast.params.timeperiod"}
    # A list stays a list, and a variable the strategy no longer has is not searched.
    listed = dict(ma_spec["optimise"]["variables"])
    listed["features.fast.params.timeperiod"] = {"values": [10, 20]}
    listed["features.gone.params.timeperiod"] = {"values": [1]}
    other = dict(ma_spec, optimise={"variables": listed})
    assert O.build_space(other)["features.fast.params.timeperiod"] == {"values": [10, 20]}
    assert "features.gone.params.timeperiod" not in O.build_space(other)
    assert O.butchc_space({"x": {"min": 1, "max": 2, "step": 0.5}}) == {"x": {"min": 1, "max": 2}}
    assert O.from_unit({"min": 1, "max": 100, "log": True}, 0.5) == pytest.approx(10.0)
    assert O.from_unit({"values": [1, 2, 3]}, 0.999) == 3
    assert isinstance(O.from_unit({"min": 1, "max": 9, "int": True}, 0.5), int)
    assert O.grid_values({"min": 10, "max": 20, "int": True, "step": 5}) == [10, 15, 20]
    assert O.grid_size(space) == 11 * 11


def test_walk_forward_folds_do_not_overlap():
    index = pd.bdate_range("2020-01-01", periods=500)
    for mode in ("rolling", "anchored"):
        folds = O.walk_forward_folds(index, window=200, step=50, mode=mode)
        assert len(folds) == 6
        for fold in folds:
            assert fold["in_sample"][1] < fold["out_of_sample"][0]
        for left, right in zip(folds, folds[1:]):
            assert left["out_of_sample"][1] < right["out_of_sample"][0]
        if mode == "anchored":
            assert all(fold["in_sample"][0] == index[0] for fold in folds)


def test_neighbourhood_and_stability():
    space = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 1, "max": 50, "int": True}, "c": {"values": [1, 2]}}
    points = O.neighbourhood(space, {"a": 5.0, "b": 20, "c": 2})
    assert len(points) == 9 and all(point["c"] == 2 for point in points)
    assert all(4.5 <= point["a"] <= 5.5 and 18 <= point["b"] <= 22 for point in points)
    summary = O.stability_summary([{"metric": 1.0, "total_return": 0.1}, {"metric": 0.5, "total_return": -0.1}])
    assert summary == {"n": 2, "median_metric": 0.75, "share_profitable": 0.5}


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0008, 0.01, 750))
    few = O.deflated_sharpe(1.2, list(rng.normal(0.5, 0.3, 5)), returns, "D")
    many = O.deflated_sharpe(1.2, list(rng.normal(0.5, 0.3, 500)), returns, "D")
    assert 0.0 <= many < few <= 1.0
    assert O.deflated_sharpe(1.2, [0.5], returns, "D") is None


def test_compare_summary():
    rows = []
    for backend, values in (("fast", [0.1, 0.9, 1.0, 1.0]), ("slow", [0.0, 0.2, 0.4, float("nan")])):
        for seed in (0, 1):
            for number, value in enumerate(values):
                rows.append({"phase": "search", "backend": backend, "seed": seed, "run_trial": number,
                             "objective": value})
    summary = O.compare_summary(pd.DataFrame(rows), "maximize")
    assert summary["target"] == pytest.approx(0.95)
    assert summary["backends"]["fast"]["median_trials_to_target"] == 3
    assert summary["backends"]["slow"]["reached_target"] == 0
    assert summary["backends"]["slow"]["median_final"] == pytest.approx(0.4)
    assert summary["backends"]["slow"]["valid_share"] == pytest.approx(0.75)
    assert list(O.best_so_far([1.0, float("nan"), 0.5, 2.0], "maximize")) == [1.0, 1.0, 1.0, 2.0]
