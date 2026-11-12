"""Optimisation: search space, objective, metrics, backends, robustness tools.

Backends are interchangeable because they all receive the same three things:
a search space, an evaluation function and a budget.

    space            {param path: opt dict} in BUTChC's dialect (plan section 5)
    evaluate_batch   list of params dicts -> list of objective floats (NaN for an
                     invalid trial). The job worker supplies it: it fans the batch
                     out to a process pool, records every trial and raises
                     RuntimeError("cancelled") when the cancel file appears.
    budget           number of evaluations

`evaluate_trial` is the one top-level, picklable function that worker
processes run. It reads the job and its data from module globals set by
`init_worker`, so only small values (a job folder path, a params dict) ever
cross a process boundary.
"""

import itertools
import json
import math
import pickle
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import engine
from .spec import value_of

PENALTY = 1e12          # stands in for NaN where a backend needs a finite number

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def build_space(spec, include=None, conditional=False):
    """{param path: opt dict} for every optimisable node (or only those in `include`).

    Structural alternatives (choice nodes) are handled two ways:
    * conditional=False (every backend): the structure is fixed at the current
      choice, and only parameters inside the chosen options are searched.
    * conditional=True (BUTChC and Optuna only): each optimisable selector
      becomes a categorical whose next_level holds the parameters of each
      option, in BUTChC's dialect. The objective then receives only the
      parameters on the sampled path, and apply_params keeps the rest at their
      current values.
    """
    from .spec import is_choice, is_opt

    def wanted(path):
        return include is None or path in include

    def walk(node, path, out):
        if is_opt(node):
            if wanted(path):
                out[path] = dict(node["opt"])
            return
        if is_choice(node):
            selector = node["choose"]
            selector_path = f"{path}.choose" if path else "choose"
            if conditional and is_opt(selector) and wanted(selector_path):
                entry = dict(selector["opt"])
                branches = {}
                for label in entry["values"]:
                    branch = {}
                    walk(node["options"][label], f"{path}.options.{label}", branch)
                    if branch:
                        branches[label] = branch
                if branches:
                    entry["next_level"] = branches
                out[selector_path] = entry
            else:
                label = value_of(selector)
                walk(node["options"][label], f"{path}.options.{label}", out)
            return
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path else str(key), out)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}.{index}" if path else str(index), out)

    space = {}
    walk(spec, "", space)
    return dict(sorted(space.items()))


def is_conditional(space):
    return any("next_level" in opt for opt in space.values())


def active_space(space, params):
    """The flat {name: opt} of the parameters on one sampled path of a conditional space."""
    out = {}
    for name, opt in space.items():
        if name not in params:
            continue
        flat = {key: value for key, value in opt.items() if key != "next_level"}
        out[name] = flat
        branch = (opt.get("next_level") or {}).get(params[name])
        if branch:
            out.update(active_space(branch, params))
    return out


def _require_flat(space, backend):
    if is_conditional(space):
        raise ValueError(f"The {backend} backend searches flat spaces only. Fix the alternatives at their "
                         "current choice, or use BUTChC or Optuna.")


def butchc_space(space):
    # "step" is the Studio's grid-only key; BUTChC rejects unknown keys.
    out = {}
    for name, opt in space.items():
        clean = {key: value for key, value in opt.items() if key not in ("step", "next_level")}
        if "next_level" in opt:
            clean["next_level"] = {label: butchc_space(branch) for label, branch in opt["next_level"].items()}
        out[name] = clean
    return out


def _python(value):
    """numpy scalars to plain Python, so params serialise to JSON and print cleanly."""
    return value.item() if isinstance(value, np.generic) else value


def from_unit(opt, u):
    """Map u in [0, 1) onto one parameter's range (Sobol and scipy share this)."""
    if "values" in opt:
        values = opt["values"]
        return values[min(int(u * len(values)), len(values) - 1)]
    low, high = opt["min"], opt["max"]
    if opt.get("log"):
        value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
    else:
        value = low + u * (high - low)
    if opt.get("int"):
        return int(min(max(round(value), low), high))
    return float(value)


def grid_values(opt):
    """The values the grid backend tries for one parameter: values, or min..max by step."""
    if "values" in opt:
        return list(opt["values"])
    low, high = opt["min"], opt["max"]
    if opt.get("int"):
        step = int(opt.get("step") or max(1, (int(high) - int(low)) // 9))
        return list(range(int(low), int(high) + 1, step))
    if opt.get("log"):
        return [float(value) for value in np.geomspace(low, high, 10)]
    if opt.get("step"):
        return [float(value) for value in np.arange(low, high + opt["step"] / 2, opt["step"])]
    return [float(value) for value in np.linspace(low, high, 10)]


def grid_size(space):
    return math.prod(len(grid_values(opt)) for opt in space.values()) if space else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _scalar(value, how="mean"):
    """One float from a vectorbt result that may be a scalar or a one-per-column Series.

    ⚠️ Portfolios with several columns should already be reduced with
    group_by=True; the mean/sum here only guards against an ungrouped result.
    """
    array = np.asarray(value, dtype=float).ravel()
    if array.size == 0:
        return float("nan")
    if array.size == 1:
        return float(array[0])
    return float(np.nansum(array) if how == "sum" else np.nanmean(array))


def sqn(pf):
    # Neither library provides SQN: mean(pnl) / std(pnl) * sqrt(number of trades).
    pnl = np.asarray(pf.trades.pnl.values, dtype=float)
    if pnl.size < 2 or pnl.std(ddof=1) == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std(ddof=1) * math.sqrt(pnl.size))


METRICS = {
    "sharpe": {"label": "Sharpe ratio", "direction": "maximize",
               "fn": lambda pf: _scalar(pf.sharpe_ratio(group_by=True))},
    "sortino": {"label": "Sortino ratio", "direction": "maximize",
                "fn": lambda pf: _scalar(pf.sortino_ratio(group_by=True))},
    "calmar": {"label": "Calmar ratio", "direction": "maximize",
               "fn": lambda pf: _scalar(pf.calmar_ratio(group_by=True))},
    "total_return": {"label": "Total return", "direction": "maximize",
                     "fn": lambda pf: _scalar(pf.total_return(group_by=True))},
    # vectorbtX's metrics, by their accessors (phase 0: each equals its stats-table entry).
    # K-ratio, UPI and geometric Sharpe live on the returns accessor, recovery factor
    # and GHPR on the trades. ⚠️ Not pf.stats(tags="vectorbtx"): that table holds only
    # the last two, so a label lookup failed every trial for the first three.
    "k_ratio": {"label": "K-ratio (v3)", "direction": "maximize",
                "fn": lambda pf: _scalar(pf.get_returns_acc(group_by=True).k_ratio_v3())},
    "upi": {"label": "Ulcer performance index", "direction": "maximize",
            "fn": lambda pf: _scalar(pf.get_returns_acc(group_by=True).ulcer_performance_index())},
    "geometric_sharpe": {"label": "Geometric Sharpe", "direction": "maximize",
                         "fn": lambda pf: _scalar(pf.get_returns_acc(group_by=True).geometric_sharpe_ratio())},
    "recovery_factor": {"label": "Recovery factor", "direction": "maximize",
                        "fn": lambda pf: _scalar(pf.trades.recovery_factor(group_by=True))},
    "ghpr": {"label": "GHPR", "direction": "maximize", "fn": lambda pf: _scalar(pf.trades.ghpr(group_by=True))},
    "profit_factor": {"label": "Profit factor", "direction": "maximize",
                      "fn": lambda pf: _scalar(pf.trades.profit_factor(group_by=True))},
    "expectancy": {"label": "Expectancy", "direction": "maximize",
                   "fn": lambda pf: _scalar(pf.trades.expectancy(group_by=True))},
    "sqn": {"label": "SQN", "direction": "maximize", "fn": sqn},
}


def trial_extras(pf):
    """Numbers recorded for every trial whatever the objective, for constraints,
    the stability check and the deflated Sharpe ratio."""
    return {
        "sharpe": _scalar(pf.sharpe_ratio(group_by=True)),
        "total_return": _scalar(pf.total_return(group_by=True)),
        "max_drawdown": abs(_scalar(pf.max_drawdown(group_by=True))),
        "n_trades": int(_scalar(pf.trades.count(group_by=True), how="sum")),
        "profit_factor": _scalar(pf.trades.profit_factor(group_by=True)),
    }


# ---------------------------------------------------------------------------
# The objective, run inside worker processes
# ---------------------------------------------------------------------------

CONTEXT = {}            # set once per process by init_worker


def init_worker(job_dir):
    """Process initialiser: load the job, its data snapshot and its settings once."""
    from . import config

    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if job["metric"] == "selftest":
        config.apply_environment(job["settings"])       # a self-test job never imports vectorbtX
    else:
        config.apply_settings(job["settings"])
    CONTEXT.clear()
    CONTEXT["job"] = job
    with open(job_dir / "frames.pkl", "rb") as handle:
        CONTEXT["data"] = pickle.load(handle)
    if job["metric"] != "selftest":
        # Warm-up: the first backtest in a fresh process compiles numba code. Pay
        # it here, once per worker, rather than inside the first timed trial.
        evaluate_trial({"params": {}, "window": None})


def evaluate_trial(task):
    """One trial: {"params", "window"} -> {"params", "objective", extras..., "error"}.

    Anything vectorbtX rejects (BacktestError) and anything that breaks a
    constraint returns objective NaN, which every backend records but ignores.
    """
    job = CONTEXT["job"]
    params = {key: _python(value) for key, value in task["params"].items()}
    if job["metric"] == "selftest":
        # A cheap synthetic objective with a known optimum, for testing the job
        # plumbing (and an install) without running a backtest.
        if job.get("selftest_delay"):
            time.sleep(float(job["selftest_delay"]))      # lets the tests cancel a running job
        return {"params": params, "objective": selftest_objective(job["space"], params), "error": None}
    try:
        run = engine.run_backtest(job["spec"], CONTEXT["data"], params=params, window=task.get("window"),
                                  data_dir=job["data_dir"], offline=job["settings"].get("offline", False),
                                  static_rate=job["settings"]["vectorbtx"].get("static_rate"))
    except engine.backtest_error() as error:
        return {"params": params, "objective": float("nan"), "error": str(error)}
    return score(run["pf"], job, params, want_returns=task.get("want_returns"))


def score(pf, job, params, want_returns=False):
    """The objective, the extras and the constraints for one trial's portfolio."""
    extras = trial_extras(pf)
    value = METRICS[job["metric"]]["fn"](pf)
    result = {"params": params, "objective": value, "metric": value, **extras, "error": None}
    constraints = job.get("constraints") or {}
    if constraints.get("min_trades") and extras["n_trades"] < constraints["min_trades"]:
        # Without a trade-count floor, optimisers reliably find parameters that
        # trade twice and win both.
        result.update(objective=float("nan"), error=f"only {extras['n_trades']} trades")
    elif constraints.get("max_drawdown") and extras["max_drawdown"] > constraints["max_drawdown"]:
        result.update(objective=float("nan"), error=f"drawdown {extras['max_drawdown']:.1%}")
    elif constraints.get("min_profit_factor") and not extras["profit_factor"] >= constraints["min_profit_factor"]:
        result.update(objective=float("nan"), error=f"profit factor {extras['profit_factor']:.2f}")
    if want_returns:
        result["returns"] = pf.returns(group_by=True)
    if not np.isfinite(result["objective"]):
        result["objective"] = float("nan")
    return result


def evaluate_chunk(tasks):
    """Several trials in ONE backtest call (the vectorised grid). Same results as evaluate_trial each.

    If vectorbtX rejects the combined call, the chunk falls back to one call per
    trial, so a single invalid trial is isolated instead of failing them all.
    """
    job = CONTEXT["job"]
    if len(tasks) == 1 or tasks[0].get("want_returns") or job["metric"] == "selftest":
        return [evaluate_trial(task) for task in tasks]
    params_list = [{key: _python(value) for key, value in task["params"].items()} for task in tasks]
    try:
        trials = engine.run_vectorised(job["spec"], CONTEXT["data"], params_list, window=tasks[0].get("window"),
                                       data_dir=job["data_dir"], offline=job["settings"].get("offline", False),
                                       static_rate=job["settings"]["vectorbtx"].get("static_rate"))
    except engine.backtest_error():
        return [evaluate_trial(task) for task in tasks]
    return [score(pf, job, params) for params, pf in zip(params_list, trials)]


def selftest_objective(space, params):
    """Known optimum: every number at 30% of its range, every categorical at its
    second value (score 0 + number of categoricals). The top 15% of the FIRST
    numeric range is a trap that returns NaN: a backend that mishandled NaN
    (treating it as large) would wrongly report its best there."""
    total = 0.0
    numeric_seen = False
    for name, opt in space.items():
        value = params[name]
        if "values" in opt:
            total += 1.0 if value == opt["values"][min(1, len(opt["values"]) - 1)] else 0.0
            continue
        low, high = opt["min"], opt["max"]
        if not numeric_seen and value > low + 0.85 * (high - low):
            return float("nan")
        numeric_seen = True
        target = low + 0.3 * (high - low)
        total -= ((value - target) / (high - low)) ** 2
    return total


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
# Each takes (space, evaluate_batch, budget, direction, seed, options) and
# returns an optional dict of backend-specific results to save.

def run_grid(space, evaluate_batch, budget, direction, seed, options):
    _require_flat(space, "grid")
    # Loop-based: one backtest per point. "grid_vectorised" runs the same points
    # many-per-call instead (engine.run_vectorised).
    names = list(space)
    combos = itertools.product(*(grid_values(space[name]) for name in names))
    batch = max(1, int(options.get("batch", 8)))
    done = 0
    while done < budget:
        chunk = [dict(zip(names, values)) for values in itertools.islice(combos, min(batch, budget - done))]
        if not chunk:
            break
        evaluate_batch(chunk)
        done += len(chunk)
    return {}


def run_random(space, evaluate_batch, budget, direction, seed, options):
    """Scrambled Sobol points: the baseline and sanity check for every other backend."""
    _require_flat(space, "random")
    from scipy.stats import qmc

    names = list(space)
    points = qmc.Sobol(d=len(names), scramble=True, rng=seed).random_base2(m=max(0, math.ceil(math.log2(budget))))
    batch = max(1, int(options.get("batch", 8)))
    for start in range(0, budget, batch):
        chunk = [{name: from_unit(space[name], float(u)) for name, u in zip(names, row)}
                 for row in points[start:min(start + batch, budget)]]
        evaluate_batch(chunk)
    return {}


def run_optuna(space, evaluate_batch, budget, direction, seed, options):
    """Ask-and-tell in batches, so trials run in the job's process pool. Handles
    conditional spaces by asking for a branch's parameters only when it is chosen.

    Verified on Optuna 5.0.0: ask() with no distributions, trial.suggest_* on the
    returned trial, tell() with a value or state=FAIL, and SQLite storage holding one
    study per walk-forward fold in one file. Only the worker's main process asks and
    tells (the pool only evaluates), so SQLite never sees concurrent writers.
    """
    import optuna

    def suggest(trial, branch, params):
        # Define-by-run: a categorical's next_level is only asked for when chosen.
        for name, opt in branch.items():
            if "values" in opt:
                params[name] = trial.suggest_categorical(name, list(opt["values"]))
                nested = (opt.get("next_level") or {}).get(params[name])
                if nested:
                    suggest(trial, nested, params)
            elif opt.get("int"):
                params[name] = trial.suggest_int(name, int(opt["min"]), int(opt["max"]), step=int(opt.get("step") or 1),
                                                 log=bool(opt.get("log")))
            else:
                params[name] = trial.suggest_float(name, float(opt["min"]), float(opt["max"]), log=bool(opt.get("log")))
        return params

    samplers = {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed),        # continuous parameters only
        "nsga2": lambda: optuna.samplers.NSGAIISampler(seed=seed),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
        "qmc": lambda: optuna.samplers.QMCSampler(seed=seed),
    }
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=samplers[options.get("sampler", "tpe")](),
                                storage=options.get("storage"), study_name=options.get("study_name", "job"),
                                load_if_exists=True)
    batch = max(1, int(options.get("batch", 8)))
    done = 0
    while done < budget:
        trials = [study.ask() for _ in range(min(batch, budget - done))]
        values = evaluate_batch([suggest(trial, space, {}) for trial in trials])
        for trial, value in zip(trials, values):
            if np.isfinite(value):
                study.tell(trial, value)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
        done += len(trials)
    return {}


def run_scipy(space, evaluate_batch, budget, direction, seed, options):
    """differential_evolution (parallel through `workers`) or dual_annealing.

    scipy minimises, so the objective is negated when maximising. Categoricals
    are an integer index; log ranges are searched in log space.
    """
    _require_flat(space, "scipy")
    from scipy.optimize import differential_evolution, dual_annealing

    names = list(space)
    bounds = []
    integrality = []
    for name in names:
        opt = space[name]
        if "values" in opt:
            bounds.append((0, len(opt["values"]) - 1))
            integrality.append(True)
        elif opt.get("log"):
            bounds.append((math.log(opt["min"]), math.log(opt["max"])))
            integrality.append(False)
        else:
            bounds.append((opt["min"], opt["max"]))
            integrality.append(bool(opt.get("int")))
    sign = -1.0 if direction == "maximize" else 1.0

    def decode(x):
        params = {}
        for name, value in zip(names, x):
            opt = space[name]
            if "values" in opt:
                params[name] = opt["values"][int(round(value))]
            elif opt.get("log"):
                params[name] = float(math.exp(value))
            elif opt.get("int"):
                params[name] = int(round(value))          # dual_annealing has no integrality
            else:
                params[name] = float(value)
        return params

    def values_for(xs):
        values = evaluate_batch([decode(x) for x in xs])
        return [sign * value if np.isfinite(value) else PENALTY for value in values]

    if options.get("method", "differential_evolution") == "dual_annealing":
        dual_annealing(lambda x: values_for([x])[0], bounds, rng=seed, maxfun=budget, no_local_search=True)
        return {}
    # A population of popsize * d per generation; size both to fit the budget.
    dims = len(names)
    popsize = max(2, min(15, budget // (2 * dims)))
    maxiter = max(1, budget // (popsize * dims) - 1)
    # `workers` is handed our own map, which evaluates the population through
    # evaluate_batch (and so through the process pool) and ignores scipy's func.
    differential_evolution(_unused_objective, bounds, integrality=integrality, rng=seed, popsize=popsize,
                           maxiter=maxiter, workers=lambda func, xs: values_for(list(xs)),
                           updating="deferred", polish=False)
    return {}


def _unused_objective(x):
    # differential_evolution needs a func; ours is evaluated through `workers`.
    raise RuntimeError("differential_evolution should evaluate through the workers map")


def run_butchc(space, evaluate_batch, budget, direction, seed, options):
    """BUTChC 0.6.1. Handles conditional spaces natively (phase 3 here).

    It learns each parameter separately, so strongly interacting parameters
    should be reparameterised (fast and gap, with slow = fast + gap). batch up
    to 8 costs almost nothing in quality; past 16 it does.
    """
    from butchc import BUTChC_optimize

    # An order-preserving map that BUTChC calls with a batch of configs. It
    # ignores fn and evaluates through evaluate_batch, so the job's pool is used.
    executor = types.SimpleNamespace(map=lambda fn, configs: evaluate_batch(list(configs)))
    keywords = {"direction": direction, "seed": seed, "batch": max(1, int(options.get("batch", 4))),
                "executor": executor, "verbose": False}
    if options.get("n_warmup"):
        keywords["n_warmup"] = int(options["n_warmup"])
    if options.get("warm_start"):
        # ⚠️ The warm-start tree must come from a run with the SAME direction.
        with open(options["warm_start"], "rb") as handle:
            keywords["start_prob_tree"] = pickle.load(handle)
    result = BUTChC_optimize(butchc_space(space), lambda config: evaluate_batch([config])[0], budget, **keywords)
    return {"prob_tree": result.get("prob_tree")}


def run_marsopt(space, evaluate_batch, budget, direction, seed, options):
    """marsopt 0.2.0: sequential, flat spaces, string categories only."""
    _require_flat(space, "marsopt")
    from marsopt import Study

    # ⚠️ Categories must be strings, so map numbers and booleans to and from text.
    categories = {name: {str(value): value for value in opt["values"]} for name, opt in space.items()
                  if "values" in opt}

    def objective(trial):
        params = {}
        for name, opt in space.items():
            if "values" in opt:
                params[name] = categories[name][trial.suggest_categorical(name, list(categories[name]))]
            elif opt.get("int"):
                params[name] = int(trial.suggest_int(name, int(opt["min"]), int(opt["max"])))
            else:
                params[name] = float(trial.suggest_float(name, float(opt["min"]), float(opt["max"]),
                                                         log=bool(opt.get("log"))))
        value = evaluate_batch([params])[0]
        # NaN handling is not documented, so return a large penalty instead.
        if np.isfinite(value):
            return value
        return -PENALTY if direction == "maximize" else PENALTY

    study = Study(direction=direction, random_state=seed, verbose=False)
    study.optimize(objective, n_trials=budget)
    return {}


BACKENDS = {
    "optuna": {"label": "Optuna (default)", "fn": run_optuna},
    "grid": {"label": "Grid", "fn": run_grid},
    # The same points as "grid", evaluated many per backtest call (worker.py routes
    # its batches through evaluate_chunk). Eligibility: engine.vectorisable.
    "grid_vectorised": {"label": "Grid, many trials per call", "fn": run_grid},
    "random": {"label": "Random (Sobol)", "fn": run_random},
    "scipy": {"label": "scipy", "fn": run_scipy},
    "butchc": {"label": "BUTChC", "fn": run_butchc},
    "marsopt": {"label": "marsopt (optional extra)", "fn": run_marsopt},
}


# ---------------------------------------------------------------------------
# Robustness tools
# ---------------------------------------------------------------------------

def best_trial(trials, direction):
    """The best finite trial from a trials DataFrame, or None."""
    finite = trials[np.isfinite(trials["objective"].astype(float))]
    if finite.empty:
        return None
    index = finite["objective"].idxmax() if direction == "maximize" else finite["objective"].idxmin()
    return finite.loc[index]


def split_window(index, train_fraction):
    """(train window, test window) as (start, end) timestamp pairs on `index`."""
    cut = max(1, min(len(index) - 1, int(len(index) * train_fraction)))
    return (index[0], index[cut - 1]), (index[cut], index[-1])


def walk_forward_folds(index, window, step, mode="rolling"):
    """In-sample and out-of-sample windows, in bars, as timestamp pairs.

    anchored: every in-sample window starts at the first bar. rolling: each one
    is the `window` bars before its out-of-sample slice.
    """
    folds = []
    end = window
    while end + 1 <= len(index) - 1:
        start = 0 if mode == "anchored" else end - window
        out_end = min(end + step, len(index)) - 1
        folds.append({"in_sample": (index[start], index[end - 1]), "out_of_sample": (index[end], index[out_end])})
        end += step
    return folds


def neighbourhood(space, best, width=0.10, count=None, seed=0):
    """Points within ±width of the best, per numeric parameter; categoricals stay fixed.

    Up to four numeric parameters this is the full 3^d grid (best, -10%, +10%);
    beyond that, `count` random points in the box.
    """
    space = active_space(space, best)
    numeric = [name for name, opt in space.items() if "values" not in opt]

    def clip(name, value):
        opt = space[name]
        value = min(max(value, opt["min"]), opt["max"])
        return int(round(value)) if opt.get("int") else float(value)

    points = []
    if len(numeric) <= 4 and count is None:
        for factors in itertools.product((1 - width, 1.0, 1 + width), repeat=len(numeric)):
            point = dict(best)
            for name, factor in zip(numeric, factors):
                point[name] = clip(name, best[name] * factor)
            points.append(point)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(count or 50):
            point = dict(best)
            for name in numeric:
                point[name] = clip(name, best[name] * (1 + rng.uniform(-width, width)))
            points.append(point)
    unique = []
    for point in points:
        if point not in unique:
            unique.append(point)
    return unique


def stability_summary(results):
    """Median metric and share of profitable neighbours. A sharp peak with poor neighbours is overfit."""
    metric = np.array([result.get("metric", np.nan) for result in results], dtype=float)
    profit = np.array([result.get("total_return", np.nan) for result in results], dtype=float)
    finite = np.isfinite(profit)
    return {"n": len(results), "median_metric": float(np.nanmedian(metric)) if np.isfinite(metric).any() else None,
            "share_profitable": float((profit[finite] > 0).mean()) if finite.any() else None}


def freq_timedelta(freq):
    """A vectorbt freq string as a Timedelta ("D" -> 1 day, "4h" -> 4 hours).

    pandas 3 no longer turns a Day offset into a Timedelta, so the string is
    parsed directly, with an implied 1 when it has no number.
    """
    return pd.Timedelta(freq if freq[:1].isdigit() else "1" + freq)


def periods_per_year(freq):
    # vectorbt annualises with year_freq = 365 days by default, and vectorbtX keeps it
    # (phase 0: vbt.settings["returns"]["year_freq"] is 365 days).
    return pd.Timedelta(days=365) / freq_timedelta(freq)


def deflated_sharpe(best_sharpe, trial_sharpes, returns, freq):
    """Deflated Sharpe ratio (Bailey and López de Prado, 2014), from annualised inputs.

    All Sharpes are first de-annualised to per-bar values:
        SR0 = sqrt(V[SR]) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))
        DSR = Φ( (SR* - SR0) sqrt(T - 1) / sqrt(1 - g3 SR* + (g4 - 1) / 4 SR*²) )
    with N trials, V[SR] the variance of the trials' Sharpes, γ the
    Euler-Mascheroni constant, T the number of returns, and g3, g4 the skewness
    and (non-excess) kurtosis of the best strategy's returns. DSR is the
    probability that the true Sharpe is above zero after allowing for the number
    of configurations tried.
    """
    scale = math.sqrt(periods_per_year(freq))
    sharpes = np.asarray([value for value in trial_sharpes if np.isfinite(value)], dtype=float) / scale
    returns = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n_trials, n_obs = len(sharpes), len(returns)
    if n_trials < 2 or n_obs < 3 or not np.isfinite(best_sharpe):
        return None
    gamma = 0.5772156649015329
    best = best_sharpe / scale
    sr0 = math.sqrt(sharpes.var(ddof=1)) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
                                            + gamma * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    skew = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))
    denominator = 1 - skew * best + (kurtosis - 1) / 4 * best ** 2
    if denominator <= 0:
        return None
    return float(stats.norm.cdf((best - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denominator)))


# ---------------------------------------------------------------------------
# Backend comparison harness
# ---------------------------------------------------------------------------

def best_so_far(values, direction):
    """Running best of a sequence of objectives, ignoring NaN (invalid trials)."""
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.where(np.isfinite(series))
    running = series.cummax() if direction == "maximize" else series.cummin()
    return running.ffill()


def compare_summary(trials, direction):
    """Per backend: final best (median, best, worst over seeds), trials needed to get 95%
    of the way to the best value seen, and the share of valid trials.

    BUTChC's own benchmarks are synthetic and written with the optimiser, so this is
    how it earns its place against Optuna and random search on real strategies.
    """
    search = trials[trials["phase"] == "search"]
    values = search["objective"].astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {"target": None, "backends": {}}
    low, high = float(finite.min()), float(finite.max())
    target = low + 0.95 * (high - low) if direction == "maximize" else high - 0.95 * (high - low)
    reached = (lambda value: value >= target) if direction == "maximize" else (lambda value: value <= target)
    runs = {}
    for (backend, seed), run in search.groupby(["backend", "seed"], sort=False):
        run = run.sort_values("run_trial")
        curve = best_so_far(run["objective"], direction)
        hits = [number + 1 for number, value in enumerate(curve) if np.isfinite(value) and reached(value)]
        runs.setdefault(backend, []).append({
            "seed": int(seed), "final": float(curve.iloc[-1]) if np.isfinite(curve.iloc[-1]) else None,
            "trials_to_target": hits[0] if hits else None,
            "valid_share": float(np.isfinite(run["objective"].astype(float)).mean())})
    out = {}
    pick, worst = (max, min) if direction == "maximize" else (min, max)
    for backend, items in runs.items():
        finals = [item["final"] for item in items if item["final"] is not None]
        needed = [item["trials_to_target"] for item in items if item["trials_to_target"] is not None]
        out[backend] = {"runs": len(items), "median_final": float(np.median(finals)) if finals else None,
                        "best_final": pick(finals) if finals else None, "worst_final": worst(finals) if finals else None,
                        "reached_target": len(needed),
                        "median_trials_to_target": float(np.median(needed)) if needed else None,
                        "valid_share": float(np.mean([item["valid_share"] for item in items]))}
    return {"target": target, "backends": out}
