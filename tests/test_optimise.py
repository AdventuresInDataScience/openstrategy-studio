"""Test 12: optimiser plumbing on a cheap synthetic objective with a known optimum."""


import json

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import optimise as O

SPACE = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 1, "max": 50, "int": True},
         "c": {"values": ["x", "y", "z"]}}
OPTIMUM = 1.0               # numbers at 30% of their range (0), plus 1 for c == "y"


def told(values):
    """Scores as a search is handed them (optimise's evaluate_batch): an invalid one (NaN) is as far from
    valid as a trial can be."""
    return [{"objective": value, "violation": 0.0 if np.isfinite(value) else 1.0} for value in values]


def drive(backend, budget=200, seed=0, options=None):
    trials = []

    def evaluate_batch(params_list):
        values = [O.selftest_objective(SPACE, params) for params in params_list]
        trials.extend(zip(params_list, values))
        return told(values)

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
        return told(values)

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
    # DIRECT ends its iteration past maxfun (13 on a budget of 12): the budget caps the backtests run.
    for budget in (12, 13, 30):
        seen = []
        O.run_scipy(SPACE, lambda batch: seen.extend(batch) or told([O.selftest_objective(SPACE, p) for p in batch]),
                    budget, "maximize", 0, {"method": "direct"})
        assert len(seen) <= budget, (budget, len(seen))


@pytest.mark.parametrize("optimizer", ["TwoPointsDE", "PSO", "CMA", "PortfolioDiscreteOnePlusOne", "OnePlusOne",
                                       "TBPSA"])
def test_nevergrad(optimizer):
    """Every Nevergrad optimiser the Optimise page offers, on the cheap objective with its NaN trap, in batches
    as the job runs them; and a seed repeats it."""
    pytest.importorskip("nevergrad")
    assert optimizer in O.NEVERGRAD_OPTIMISERS
    options = {"method": optimizer, "batch": 4}
    first = check("nevergrad", tolerance=0.15, visits_trap=optimizer != "CMA", options=options, budget=300)
    invalid_half("nevergrad", options=options)
    second, _ = drive("nevergrad", budget=300, options=options)
    assert [params for params, _ in first] == [params for params, _ in second]
    # A whole number is kept on its step, from its start.
    stepped = {"b": {"min": 3, "max": 30, "int": True, "step": 3}}
    seen = []
    O.run_nevergrad(stepped, lambda batch: seen.extend(batch) or told([0.0] * len(batch)), 40, "maximize", 0, options)
    assert {params["b"] for params in seen} <= set(range(3, 31, 3))


def test_nevergrad_offers_only_what_the_page_lists():
    """The page offers NEVERGRAD_OPTIMISERS and nothing else; every name is in Nevergrad's registry."""
    ng = pytest.importorskip("nevergrad")
    assert set(O.NEVERGRAD_OPTIMISERS) <= set(ng.optimizers.registry)
    assert "NGOpt" not in O.NEVERGRAD_OPTIMISERS       # its metamodels fail on numpy 2.5 (see the comment there)


def test_optuna():
    pytest.importorskip("optuna")
    check("optuna", tolerance=0.05, options={"batch": 4})


def test_optuna_learns_where_trials_are_invalid(tmp_path):
    """An invalid trial is told complete and infeasible, by how much, never FAIL (26 November 2026). TPE learns
    only from complete trials, so told FAIL it never learned the invalid region: here, where the best valid point
    is on its edge (as a best trial at exactly the minimum trades is), it spent every trial after the 50th there,
    on five seeds out of five. Told the constraint, 29% to 42%."""
    optuna = pytest.importorskip("optuna")
    space = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 0.0, "max": 10.0}}

    def result(params):
        if params["a"] > 4:                       # 60% of the box, more invalid further out
            return {"objective": float("nan"), "violation": (params["a"] - 4) / 6}
        return {"objective": params["a"] - abs(params["b"] - 7), "violation": 0.0}

    seen = []
    storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"
    O.run_optuna(space, lambda batch: seen.extend(batch) or [result(p) for p in batch], 150, "maximize", 0,
                 {"batch": 4, "storage": storage, "study_name": "job"})
    assert np.mean([params["a"] > 4 for params in seen[50:]]) < 0.5
    best = max((params for params in seen if params["a"] <= 4), key=lambda params: result(params)["objective"])
    assert best["a"] > 3.5 and abs(best["b"] - 7) < 1.0, best
    study = optuna.load_study(study_name="job", storage=storage)
    states = {trial.state for trial in study.trials}
    assert states == {optuna.trial.TrialState.COMPLETE}
    for trial, params in zip(study.trials, seen):
        assert trial.constraints["valid"] == pytest.approx(result(params)["violation"])


def test_a_search_hears_how_far_an_invalid_trial_was(ma_spec, ohlcv):
    """score() grades each broken constraint from 0 to 1 and adds them; the searches that need a number are told
    a penalty that grows with it, so 28 trades of 30 beats 4 of 30, and both lose to any valid trial."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    params = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    pf = engine.run_backtest(ma_spec, ohlcv, params=params)["pf"]
    made = O.score(pf, {"metric": "sharpe"}, params)
    trades, drawdown = made["n_trades"], made["max_drawdown"]
    assert made["violation"] == 0.0 and not made["screened"]
    short = O.score(pf, {"metric": "sharpe", "constraints": {"min_trades": 2 * trades}}, params)
    assert short["error"] == f"only {trades} trades" and short["violation"] == pytest.approx(0.5)
    both = O.score(pf, {"metric": "sharpe", "constraints": {"min_trades": 2 * trades,
                                                            "max_drawdown": drawdown / 2}}, params)
    assert both["error"] == short["error"]                   # the reason shown is the first broken
    assert both["violation"] == pytest.approx(0.5 + (drawdown / 2) / (1 - drawdown / 2))
    assert O.penalty({"violation": 0.1}) < O.penalty({"violation": 0.9}) < O.penalty({}) * 5
    assert O.penalty({}) == O.penalty({"violation": 1.0})     # nothing known: as far as a trial goes
    assert O.penalty({"violation": 0.0}) == O.penalty({})     # 0 is no violation, so an invalid one is told 1


def test_a_trial_with_too_few_entry_signals_is_not_run(ma_spec, ohlcv, monkeypatch):
    """A run makes at most as many trades as its entry signals in the window, so a search trial with fewer than
    the minimum trades is ruled out before its simulation (the owner's job #26: 99% of its trials). Its result
    says so; a trial with enough signals is run as before; and the bound holds where it is run."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    job = {"spec": ma_spec, "metric": "sharpe", "data_dir": None, "constraints": {"min_trades": 10_000},
           "settings": {"offline": True, "vectorbtx": {}}}
    monkeypatch.setitem(O.CONTEXT, "job", job)
    monkeypatch.setitem(O.CONTEXT, "data", ohlcv)
    params = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    ran = []
    real = engine.run_prepared
    monkeypatch.setattr(engine, "run_prepared",
                        lambda prepared, windows=None: ran.append(1) or real(prepared, windows=windows))
    out = O.evaluate_trial({"params": params, "window": None, "screen": True})
    entries = out["entries"]
    assert out["screened"] and not ran and not np.isfinite(out["objective"])
    assert out["error"] == f"at most {entries} trades: only {entries} entry signals (not run)"
    assert out["violation"] == pytest.approx((10_000 - entries) / 10_000)
    # Not a search trial (the held-back test, the stability check): run whatever its signals, as its numbers show.
    held = O.evaluate_trial({"params": params, "window": None})
    assert ran and not held["screened"] and held["error"] == f"only {held['n_trades']} trades"
    assert held["n_trades"] <= entries
    # Enough signals: run, and valid.
    job["constraints"] = {"min_trades": 1}
    ran.clear()
    valid = O.evaluate_trial({"params": params, "window": None, "screen": True})
    assert ran and not valid["screened"] and np.isfinite(valid["objective"])
    # A portfolio trades on its schedule, not on entry signals: never screened.
    prepared = engine.prepare_run(ma_spec, ohlcv, params=params)
    assert O.screen({**prepared, "plan": {**prepared["plan"], "kind": "portfolio"}},
                    {"constraints": {"min_trades": 10_000}}) is None


def test_the_optimise_page_says_how_much_of_the_search_the_leverage_cuts(ma_spec):
    """With a stop in % of the price, risk r over a stop s is a trade worth r / s of the account; above the
    leverage it is cut (or skipped), so that part of the search changes nothing. Named only where it is exact."""
    ma_spec["instruments"]["EURUSD=X"]["leverage"] = 10
    ma_spec["risk"]["sizing"] = {"risk_pct": 0.1, "risk_of": "balance"}
    ma_spec["risk"]["stop"]["unit"] = "pct"
    ma_spec["risk"]["stop"]["pct"] = 0.01
    ma_spec["optimise"]["variables"] = {"risk.sizing.risk_pct": {"values": [0.1, 0.2, 0.3]}}
    capped = O.leverage_cap(ma_spec)                    # 10x, 20x and 30x the account, at 10x leverage
    assert capped == {"share": pytest.approx(2 / 3), "leverage": 10.0, "skip": False, "what": "risk"}
    ma_spec["optimise"]["variables"]["risk.stop.pct"] = {"values": [0.01, 0.03]}
    assert O.leverage_cap(ma_spec)["share"] == pytest.approx(2 / 6)      # 10, 20, 30 and 3.3, 6.7, 10
    ma_spec["account"]["when_cash_short"] = "skip"
    assert O.leverage_cap(ma_spec)["skip"]
    # Whether there is a stop, or how it is measured, searched too: no share can be named.
    ma_spec["optimise"]["variables"]["risk.stop.unit"] = {"values": ["pct", "points"]}
    assert O.leverage_cap(ma_spec) is None
    # A stop in points: each trade's worth follows its price.
    del ma_spec["optimise"]["variables"]["risk.stop.unit"]
    ma_spec["risk"]["stop"]["unit"] = "points"
    assert O.leverage_cap(ma_spec) is None
    # A share of the account per trade, searched past the leverage.
    ma_spec["risk"]["sizing"] = {"notional_pct": 1.0, "risk_of": "equity"}
    ma_spec["optimise"]["variables"] = {"risk.sizing.notional_pct": {"values": [1.0, 5.0, 20.0]}}
    assert O.leverage_cap(ma_spec)["share"] == pytest.approx(1 / 3)
    ma_spec["optimise"]["variables"] = {"risk.sizing.notional_pct": {"values": [1.0, 5.0, 10.0]}}
    assert O.leverage_cap(ma_spec) is None                # up to the leverage: every one funded


def test_folds_cut_the_searched_history_back_to_back():
    index = pd.date_range("2000-01-01", periods=1000, freq="D")
    window = (index[100], index[699])
    folds = O.fold_windows(index, window, 6)
    assert len(folds) == 6 and folds[0][0] == index[100] and folds[-1][1] == index[699]
    for (_, end), (start, _) in zip(folds, folds[1:]):
        assert start == end + pd.Timedelta(days=1)          # back to back: no gap, no overlap
    lengths = [(end - start).days + 1 for start, end in folds]
    assert max(lengths) - min(lengths) <= 1
    assert O.fold_windows(index, None, 4)[-1][1] == index[-1]


def test_fold_scores_weigh_a_loss_and_a_gain_of_a_size_the_same():
    """The owner's point (26 November 2026): an average treats -50% and +50% as cancelling, when together they lose
    25%. As log growth they average to the compounded loss, and -50% mirrors +100%."""
    years = 1.0
    half, double = O.fold_score("total_return", -0.5, 5, years), O.fold_score("total_return", 1.0, 5, years)
    assert half == pytest.approx(-double)
    both = O.combine_folds([O.fold_score("total_return", -0.5, 5, years), O.fold_score("total_return", 0.5, 5, years)],
                           "mean", "maximize")
    assert O.fold_value("total_return", both) == pytest.approx(np.sqrt(0.5 * 1.5) - 1)     # -13.4% a fold, not 0
    # A year's growth: the same return over two years is half the log growth a year.
    assert O.fold_score("total_return", 0.21, 5, 2.0) == pytest.approx(np.log(1.1))
    # A profit factor as its log: 0.5 and 2 mirror each other; none losing is capped, not infinite.
    assert O.fold_score("profit_factor", 0.5, 5, years) == pytest.approx(-O.fold_score("profit_factor", 2.0, 5, years))
    assert O.fold_score("profit_factor", np.inf, 3, years) == pytest.approx(np.log(100))
    # No trades, or a ratio with nothing to divide by: no edge.
    assert O.fold_score("sharpe", np.inf, 0, years) == 0.0
    assert O.fold_score("sharpe", np.inf, 4, years) == 0.0
    assert O.fold_score("total_return", -1.0, 4, years) == pytest.approx(np.log(1e-4))  # lost everything: floored


def test_folds_combined_reward_consistency():
    steady, lumpy = [0.8, 0.9, 1.0, 0.9], [3.0, -0.5, 1.5, -0.3]          # the lumpy one has the higher average
    assert O.combine_folds(lumpy, "mean", "maximize") > O.combine_folds(steady, "mean", "maximize")
    assert O.combine_folds(steady, "mean_less_sd", "maximize") > O.combine_folds(lumpy, "mean_less_sd", "maximize")
    assert O.combine_folds(steady, "worst", "maximize") == 0.8
    assert O.combine_folds(steady, "median", "maximize") == 0.9
    # Minimising: worse is higher, so the spread is added and the worst is the highest.
    assert O.combine_folds([1.0, 3.0], "worst", "minimize") == 3.0
    assert O.combine_folds([1.0, 3.0], "mean_less_sd", "minimize") == pytest.approx(2.0 + np.std([1.0, 3.0], ddof=1))
    # Many trials at once, as the overfitting probability combines them.
    assert np.allclose(O.combine_folds(np.array([steady, lumpy]), "mean", "maximize"), [0.9, 0.925])


def test_the_probability_of_backtest_overfitting():
    """Settings with a real edge, the same in every fold, are not overfit (0); settings whose fold scores are pure
    luck are, half the time (0.5). An odd number of folds, or too few settings, gives none."""
    rng = np.random.default_rng(0)

    def trials(matrix):
        return pd.DataFrame({"phase": "search", "objective": matrix.mean(axis=1), "params_json": [
            f'{{"a": {number}}}' for number in range(len(matrix))],
            "folds_json": [json.dumps({"scores": list(row)}) for row in matrix]})

    edge = np.linspace(0, 2, 200)[:, None] + 0.01 * rng.standard_normal((200, 6))
    assert O.overfitting_probability(trials(edge), "mean_less_sd", "maximize")["probability"] == 0.0
    # Luck: half, on average. Its 20 splits share their folds, so one history's figure swings widely (a standard
    # deviation of 0.19 over 60 seeds, around 0.47): the average over many is the check.
    found = [O.overfitting_probability(trials(np.random.default_rng(seed).standard_normal((400, 6))), "mean",
                                       "maximize") for seed in range(20)]
    assert all(item["splits"] == 20 and item["trials"] == 400 for item in found)
    assert 0.35 <= np.mean([item["probability"] for item in found]) <= 0.6
    # Minimising: the best is the lowest, and an edge is still no overfitting.
    assert O.overfitting_probability(trials(-edge), "mean", "minimize")["probability"] == 0.0
    assert O.overfitting_probability(trials(edge[:, :5]), "mean", "maximize") is None
    assert O.overfitting_probability(trials(edge[:9]), "mean", "maximize") is None
    # Each setting once: a learning search tries the same one many times.
    repeated = trials(edge).assign(params_json='{"a": 1}')
    assert O.overfitting_probability(repeated, "mean", "maximize") is None


def test_a_trial_scored_across_folds(ma_spec, ohlcv):
    """Each fold a run of its own from the starting cash; the objective the folds' scores combined; the usual
    numbers the folds' together; the constraints on those, the minimum trades on the total."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    params = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    cuts = O.fold_windows(ohlcv["close"].index, None, 3)
    pfs = [engine.run_backtest(ma_spec, ohlcv, params=params, window=window)["pf"] for window in cuts]
    alone = [O.score(pf, {"metric": "sharpe"}, params) for pf in pfs]
    job = {"metric": "sharpe", "direction": "maximize", "constraints": {},
           "folds": {"count": 3, "summary": "mean_less_sd", "min_profitable": None}}
    result = O.score_folds(pfs, job, params)
    folds = json.loads(result["folds_json"])
    assert folds["metrics"] == pytest.approx([item["metric"] for item in alone])
    assert folds["trades"] == [item["n_trades"] for item in alone]
    assert result["objective"] == pytest.approx(O.combine_folds(folds["scores"], "mean_less_sd", "maximize"))
    assert result["metric"] == result["objective"]                         # a Sharpe ratio is combined as it is
    assert result["n_trades"] == sum(folds["trades"])
    assert result["total_return"] == pytest.approx(np.prod([1 + item["total_return"] for item in alone]) - 1)
    assert result["max_drawdown"] == max(item["max_drawdown"] for item in alone)
    # The minimum trades on the total, and consistency as a rule.
    total = result["n_trades"]
    assert np.isfinite(O.score_folds(pfs, {**job, "constraints": {"min_trades": total}}, params)["objective"])
    profitable = sum(1 for item in alone if item["total_return"] > 0)
    strict = O.score_folds(pfs, {**job, "folds": {**job["folds"], "min_profitable": profitable + 1}}, params)
    assert strict["error"] == f"{profitable} of 3 folds profitable"
    assert strict["violation"] == pytest.approx(1 / (profitable + 1))
    # A total return is combined as its log growth a year, and shown back as a return a year.
    growth = O.score_folds(pfs, {**job, "metric": "total_return", "folds": {**job["folds"], "summary": "mean"}}, params)
    assert growth["metric"] == pytest.approx(np.expm1(growth["objective"]))


def test_a_trials_folds_run_in_one_call_and_score_as_separate_runs(ma_spec, ohlcv, monkeypatch):
    """A search trial with folds runs them in one vectorbtX call (windows=), each its own account from the starting
    cash, and scores exactly as the same folds run one by one; a held-back test is still one run. With the range
    check, the strategy's own settings say how many trades each fold makes."""
    from conftest import windows_or_skip

    windows_or_skip()
    from openstrategy_studio import engine

    params = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}
    index = ohlcv["close"].index
    train = O.split_window(index, 0.75)[0]
    cuts = O.fold_windows(index, train, 4)
    job = {"spec": ma_spec, "metric": "sortino", "direction": "maximize", "data_dir": None,
           "settings": {"offline": True, "vectorbtx": {}}, "constraints": {"min_trades": 2},
           "folds": {"count": 4, "summary": "mean_less_sd", "min_profitable": None}}
    monkeypatch.setitem(O.CONTEXT, "job", job)
    monkeypatch.setitem(O.CONTEXT, "data", ohlcv)
    calls = []
    real = engine.run_prepared
    monkeypatch.setattr(engine, "run_prepared",
                        lambda prepared, windows=None: calls.append(windows) or real(prepared, windows=windows))
    together = O.evaluate_trial({"params": params, "window": train, "windows": cuts, "screen": True})
    assert calls == [cuts]                                            # one call for every fold
    alone = O.score_folds([engine.run_backtest(ma_spec, ohlcv, params=params, window=window)["pf"]
                           for window in cuts], job, params)
    assert together["objective"] == pytest.approx(alone["objective"])
    for key, values in json.loads(alone["folds_json"]).items():
        assert json.loads(together["folds_json"])[key] == pytest.approx(values), key
    for key in ("n_trades", "total_return", "max_drawdown", "sharpe", "profit_factor"):
        assert together[key] == pytest.approx(alone[key]), key
    held = O.evaluate_trial({"params": params, "window": train})
    assert "folds_json" not in held and calls[-1] is None
    # The range check runs the strategy's own settings: make them the trial's.
    ma_spec["features"]["fast"]["params"]["timeperiod"] = 5
    ma_spec["features"]["slow"]["params"]["timeperiod"] = 20
    rows = O.range_check(ma_spec, ohlcv, {}, min_trades=1, window=train, folds=4)
    assert rows[0]["fold_trades"] == json.loads(alone["folds_json"])["trades"]


def test_every_backend_has_its_methods_under_one_name():
    """The owner (25 November 2026): Optuna asked for a sampler, SciPy for a method, Nevergrad for an optimiser.
    Every backend now has methods, read from backend_options["method"], and a method that is not a backend's own
    (a comparison shares one job's options) runs its default."""
    assert set(O.METHODS) == set(O.BACKENDS)
    assert all(O.METHODS[backend] for backend in O.BACKENDS)
    assert O.method_of("optuna", {"method": "cmaes"}) == "cmaes"
    assert O.method_of("optuna", {"method": "direct"}) == "tpe"          # SciPy's, so Optuna's default
    assert O.method_of("scipy", {}) == "differential_evolution"
    assert set(O.METHODS["nevergrad"]) == set(O.NEVERGRAD_OPTIMISERS)
    assert O.search_label("optuna", {"method": "tpe"}) == "Optuna · TPE"
    assert O.search_label("scipy:direct") == "SciPy · DIRECT"           # as a comparison names its searches
    assert O.search_label("marsopt") == "marsopt · Mixed adaptive random search"
    # Jobs from before: the library's own words for the method, and the vectorised grid as a backend.
    assert O.search_label("optuna", {"sampler": "nsga2"}) == "Optuna · NSGA-II"
    assert O.search_label("nevergrad", {"optimizer": "PSO"}) == "Nevergrad · Particle swarm"
    assert O.search_label("grid_vectorised") == "Grid search · Many trials per backtest"
    # Optuna picks its sampler from the method.
    seen = []
    O.run_optuna({"a": {"min": 0.0, "max": 1.0}}, lambda batch: seen.extend(batch) or told([0.0] * len(batch)), 4,
                 "maximize", 0, {"method": "random", "batch": 2})
    assert len(seen) == 4


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
