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


def check(backend, tolerance=0.05, visits_trap=True, **kwargs):
    """`visits_trap=False` for a search that homes in without ever sampling the trap (DIRECT, CMA-ES): its NaN
    handling is then checked by invalid_half instead."""
    trials, (params, value) = drive(backend, **kwargs)
    assert len(trials) <= kwargs.get("budget", 200)
    if visits_trap:
        assert any(not np.isfinite(v) for _, v in trials) or backend == "grid", "the NaN trap was never visited"
    assert params["a"] <= 8.5, "a NaN trial must never be reported as the best"
    assert value >= OPTIMUM - tolerance, (backend, params, value)
    return trials


def invalid_half(backend, budget=120, options=None):
    """Every trial with a above 5 is invalid (NaN): the search finishes, and its best is a valid one."""
    trials = []

    def evaluate_batch(params_list):
        values = [float("nan") if params["a"] > 5 else O.selftest_objective(SPACE, params) for params in params_list]
        trials.extend(zip(params_list, values))
        return values

    O.BACKENDS[backend]["fn"](SPACE, evaluate_batch, budget, "maximize", 0, options or {})
    finite = [(params, value) for params, value in trials if np.isfinite(value)]
    assert len(finite) < len(trials), "no invalid trial was tried"
    best = max(finite, key=lambda item: item[1])
    assert best[0]["a"] <= 5 and best[1] >= OPTIMUM - 0.2, best


def test_grid():
    trials = check("grid", tolerance=0.01)
    assert len(trials) == 200


def test_random_sobol():
    check("random", tolerance=0.1)


def test_scipy_differential_evolution():
    check("scipy", tolerance=0.05, options={"method": "differential_evolution"})


def test_scipy_dual_annealing():
    check("scipy", tolerance=0.1, options={"method": "dual_annealing"})


def test_scipy_direct_runs_each_point_once():
    """DIRECT, one trial at a time; a rounded whole number or list index that lands on a point already
    tried is not run again (DIRECT's thirds give many such repeats)."""
    trials = check("scipy", tolerance=0.1, visits_trap=False, options={"method": "direct"})
    tried = [tuple(sorted(params.items())) for params, _ in trials]
    assert len(tried) == len(set(tried))
    invalid_half("scipy", options={"method": "direct"})


@pytest.mark.parametrize("optimizer", ["TwoPointsDE", "PSO", "CMA", "PortfolioDiscreteOnePlusOne", "OnePlusOne",
                                       "TBPSA"])
def test_nevergrad(optimizer):
    """Every Nevergrad optimiser the Optimise page offers, on the cheap objective with its NaN trap, in batches
    as the job runs them; and a seed repeats it."""
    pytest.importorskip("nevergrad")
    assert optimizer in O.NEVERGRAD_OPTIMISERS
    options = {"optimizer": optimizer, "batch": 4}
    first = check("nevergrad", tolerance=0.15, visits_trap=optimizer != "CMA", options=options, budget=300)
    invalid_half("nevergrad", options=options)
    second, _ = drive("nevergrad", budget=300, options=options)
    assert [params for params, _ in first] == [params for params, _ in second]
    # A whole number is kept on its step, from its start.
    stepped = {"b": {"min": 3, "max": 30, "int": True, "step": 3}}
    seen = []
    O.run_nevergrad(stepped, lambda batch: seen.extend(batch) or [0.0] * len(batch), 40, "maximize", 0, options)
    assert {params["b"] for params in seen} <= set(range(3, 31, 3))


def test_nevergrad_offers_only_what_the_page_lists():
    """The page offers NEVERGRAD_OPTIMISERS and nothing else; every name is in Nevergrad's registry."""
    ng = pytest.importorskip("nevergrad")
    assert set(O.NEVERGRAD_OPTIMISERS) <= set(ng.optimizers.registry)
    assert "NGOpt" not in O.NEVERGRAD_OPTIMISERS       # its metamodels fail on numpy 2.5 (see the comment there)


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


def test_a_trial_keeps_what_vectorbtx_said_about_its_run(ma_spec, ohlcv):
    """Why a trial made few trades: the first sentence of each vectorbtX warning, beside the count
    (the owner's MA job, 20 November 2026, where every trial said only "only 1 trades")."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    ma_spec["instruments"]["EURUSD=X"]["min_size"] = 5.0        # risk 100 over 60 points is a stake of 1.67
    params = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    run = engine.run_backtest(ma_spec, ohlcv, params=params)
    job = {"metric": "sharpe", "constraints": {"min_trades": 30}}
    result = O.score(run["pf"], job, params, notes=run["warnings"])
    assert result["error"] == "only 0 trades"
    lines = result["notes"].split("\n")
    below = [line for line in lines if "minimum tradeable size" in line]
    assert below and below[0].startswith("EURUSD=X: ") and below[0].endswith("placed on those bars."), lines
    assert "Increase" not in result["notes"]          # vectorbtX's advice is left to the results page's hints
    # A run vectorbtX says nothing about records no notes.
    assert O.score(run["pf"], job, params)["notes"] is None


def test_the_range_check_runs_each_end_and_says_where_trades_fall_short(ma_spec, ohlcv):
    """Before a job: the strategy at its own settings, then each setting at both ends of its range, on its
    step as the searches keep it; a risk too small for the minimum stake makes no trade, and says why."""
    pytest.importorskip("vectorbtx")
    ma_spec["features"]["fast"]["params"]["timeperiod"] = 5
    ma_spec["features"]["slow"]["params"]["timeperiod"] = 20
    slow, risk = "features.slow.params.timeperiod", "risk.sizing.risk_amount"
    space = {slow: {"min": 10, "max": 42, "int": True, "step": 3},          # Optuna's own last value is 40
             risk: {"min": 1.0, "max": 200.0, "step": 0.5}}
    rows = O.range_check(ma_spec, ohlcv, space, min_trades=5)
    assert [(row["path"], row["value"]) for row in rows] == [(None, None), (slow, 10), (slow, 40), (risk, 1.0),
                                                            (risk, 200.0)]
    assert rows[0]["enough"] and rows[0]["trades"] >= 5, rows[0]
    # 1 at risk over a 60-point stop is a stake of 0.017 a point, under the smallest size of 1.
    assert rows[3]["trades"] == 0 and not rows[3]["enough"] and "minimum tradeable size" in rows[3]["notes"]
    assert rows[4]["enough"], rows[4]
    # A float range on its step: (0.2 - 0.01) / 0.01 is 18.999999999999996, still 0.2.
    rows = O.range_check(ma_spec, ohlcv, {"risk.stop.pct": {"min": 0.01, "max": 0.2, "step": 0.01}}, min_trades=0)
    assert [row["value"] for row in rows[1:]] == [0.01, 0.2] and all(row["enough"] for row in rows)


def test_the_range_check_runs_each_point_once(ma_spec, ohlcv, tmp_path, monkeypatch):
    """The owner (25 November 2026): every Start re-ran the check, at up to 30 s a point on a levered, risk-sized
    strategy. Each point's run is kept on disk: the same check again runs nothing, a changed minimum only
    re-judges, a widened range runs its new end only, and a changed strategy or data runs again."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    slow = "features.slow.params.timeperiod"
    space = {slow: {"min": 10, "max": 40, "int": True, "step": 5}}
    cache = tmp_path / "cache" / "range_check.json"
    first = O.range_check(ma_spec, ohlcv, space, min_trades=5, cache=cache)
    assert cache.exists() and not any(row["cached"] for row in first)
    runs = []
    real = engine.run_backtest
    monkeypatch.setattr(engine, "run_backtest", lambda *args, **kwargs: runs.append(kwargs["params"]) or real(*args, **kwargs))
    again = O.range_check(ma_spec, ohlcv, space, min_trades=5, cache=cache)
    assert not runs and all(row["cached"] for row in again)
    assert [(row["trades"], row["enough"]) for row in again] == [(row["trades"], row["enough"]) for row in first]
    # The minimum is not in the key: only whether each point is enough changes.
    most = max(row["trades"] for row in first)
    stricter = O.range_check(ma_spec, ohlcv, space, min_trades=most + 1, cache=cache)
    assert not runs and not any(row["enough"] for row in stricter)
    # A widened range: its new end only.
    O.range_check(ma_spec, ohlcv, {slow: {"min": 10, "max": 60, "int": True, "step": 5}}, min_trades=5, cache=cache)
    assert runs == [{slow: 60}]
    # A changed strategy (outside its optimise section) or changed data: every point again.
    runs.clear()
    ma_spec["features"]["fast"]["params"]["timeperiod"] = 6
    O.range_check(ma_spec, ohlcv, space, min_trades=5, cache=cache)
    assert len(runs) == 3
    runs.clear()
    ohlcv["close"] = ohlcv["close"] * 1.0001
    O.range_check(ma_spec, ohlcv, space, min_trades=5, cache=cache)
    assert len(runs) == 3
    # Its optimise section is not in the key: the variables' own ranges are the points.
    runs.clear()
    ma_spec["optimise"]["variables"] = {}
    O.range_check(ma_spec, ohlcv, space, min_trades=5, cache=cache)
    assert not runs


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
