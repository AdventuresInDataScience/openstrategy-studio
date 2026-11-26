"""Optimisation: search space, objective, metrics, backends, robustness tools.

Backends are interchangeable because they all receive the same three things:
a search space, an evaluation function and a budget.

    space            {param path: opt dict} in BUTChC's dialect (plan section 5)
    evaluate_batch   list of params dicts -> list of results, one per params dict, each
                     with at least "objective" (NaN for an invalid trial) and "violation"
                     (how far an invalid trial was from valid, above 0; 0 for a valid one).
                     The job worker supplies it: it fans the batch out to a process pool,
                     records every trial and raises RuntimeError("cancelled") when the
                     cancel file appears.
    budget           number of evaluations

How a search hears of an invalid trial matters as much as that it does (26 November 2026): one that
learns nothing from it keeps sampling where trials are invalid. Optuna is told the trial is complete
and infeasible by how much (`trial.set_constraint`); Nevergrad, SciPy and marsopt a penalty that grows
with the violation (`penalty`), so an invalid region still has a way out; BUTChC gets NaN, which it
ranks as neutral (docs/UPSTREAM.md, BUTChC); grid and random search learn nothing either way.

`evaluate_trial` is the one top-level, picklable function that worker
processes run. It reads the job and its data from module globals set by
`init_worker`, so only small values (a job folder path, a params dict) ever
cross a process boundary.
"""

import hashlib
import itertools
import json
import math
import pickle
import re
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import engine

PENALTY = 1e12          # stands in for NaN where a backend needs a finite number (see penalty)


def violation_of(result):
    """How far a trial was from valid, above 0 for an invalid one: score's "violation", else 1.0 (as far as a
    trial can be: no trades, or a run vectorbtX refused)."""
    value = result.get("violation")
    if value is not None and np.isfinite(value) and value > 0:
        return float(value)
    return 1.0


def penalty(result):
    """An invalid trial's score for a search that needs a number, to MINIMISE: PENALTY, more the further the
    trial was from valid. A flat PENALTY made every invalid trial tie, so a search inside an invalid region
    had no direction out of it; this one prefers 28 trades to 4 when 30 are needed. Every one is far worse
    than any valid score, and the order is all the rank-based searches (CMA-ES, differential evolution,
    particle swarm) read. Under Nevergrad's 5e20, above which it clips and warns."""
    return PENALTY * (1.0 + min(violation_of(result), 9.0))

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def build_space(spec, include=None, conditional=False):
    """{param path: opt dict} in BUTChC's dialect, from the strategy's optimise variables.

    A variable {"start", "stop", "step"} becomes {"min", "max", "step"}, with "int": true
    for a whole-number value; {"values": [...]} stays a list. Only variables whose value
    the strategy still has are searched (validate_spec names the others), and only those
    in `include` when it is given.

    Rule versions (rules.<slot>.choose) are handled two ways:
    * conditional=False (every backend): the structure is fixed at the version in use,
      and only variables inside it (or outside any version) are searched.
    * conditional=True (BUTChC and Optuna only): a "choose" variable becomes a
      categorical whose next_level holds each version's own variables. The objective
      then receives only the values on the sampled path, and apply_params keeps the
      rest at their current values.
    """
    from .spec import collect_choices, tunable_inputs, variables

    rows = {row["path"]: row for row in tunable_inputs(spec)}

    def opt_for(path, variable):
        if "values" in variable:
            return {"values": list(variable["values"])}
        out = {"min": variable["start"], "max": variable["stop"]}
        if rows[path]["type"] == "int":
            out["int"] = True
        if variable.get("step"):
            out["step"] = variable["step"]       # the grid's spacing; stripped before BUTChC sees it
        return out

    wanted = {path: variable for path, variable in variables(spec).items()
              if path in rows and (include is None or path in include)}
    choices = collect_choices(spec)
    selectors = {f"{owner}.choose": owner for owner in choices}
    space = {}
    for path, variable in wanted.items():
        if path in selectors:
            continue                     # a version choice: searched only conditionally, below
        owner = next((owner for owner in choices if path.startswith(f"{owner}.options.")), None)
        if owner is None:
            space[path] = opt_for(path, variable)
            continue
        label = path[len(owner) + len(".options."):].split(".")[0]
        if conditional and f"{owner}.choose" in wanted:
            continue                     # goes into that version's branch
        if label == choices[owner]["choose"]:
            space[path] = opt_for(path, variable)
    if conditional:
        for selector, owner in selectors.items():
            if selector not in wanted:
                continue
            entry = opt_for(selector, wanted[selector])
            branches = {}
            for label in entry["values"]:
                prefix = f"{owner}.options.{label}."
                branch = {path: opt_for(path, variable) for path, variable in wanted.items() if path.startswith(prefix)}
                if branch:
                    branches[label] = branch
            if branches:
                entry["next_level"] = branches
            space[selector] = entry
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


def objective_label(metric, direction):
    """What a job searched for, in words: "Sharpe ratio, maximised". For the jobs table and the results."""
    if metric == "selftest":
        return "Self-test"
    label = METRICS.get(metric, {}).get("label", str(metric))
    return f"{label}, {'minimised' if direction == 'minimize' else 'maximised'}"


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
        value = selftest_objective(job["space"], params)
        return {"params": params, "objective": value, "error": None, "violation": 0.0 if np.isfinite(value) else 1.0,
                "screened": False}
    try:
        prepared = engine.prepare_run(job["spec"], CONTEXT["data"], params=params, window=task.get("window"),
                                      data_dir=job["data_dir"], offline=job["settings"].get("offline", False),
                                      static_rate=job["settings"]["vectorbtx"].get("static_rate"))
        if task.get("screen"):
            # Across folds too: each fold's trades open on its own entry signals, all inside the window.
            ruled_out = screen(prepared, job)
            if ruled_out is not None:
                return {"params": params, **ruled_out}
        # "windows": the folds, each its own run from the starting cash, in one vectorbtX call (score_folds).
        run = engine.run_prepared(prepared, windows=task.get("windows"))
    except engine.backtest_error() as error:
        return {"params": params, "objective": float("nan"), "error": str(error), "violation": 1.0, "screened": False}
    if task.get("windows"):
        return score_folds(run["pfs"], job, params, notes=run["warnings"])
    return score(run["pf"], job, params, want_returns=task.get("want_returns"), notes=run["warnings"])


def screen(prepared, job):
    """A search trial that cannot reach the minimum trades, found from its signals before its simulation; else None.

    A trade opens on an entry signal (an entry signal while a position is held opens nothing), so a run
    makes at most as many trades as its entry signals in the window. Held in all 325 of the owner's trials
    checked (26 November 2026, jobs #26 and #27). Counting them costs next to nothing after compiling, which
    the run needs anyway, while the simulation is ~97% of a trial: on job #26 (walk-forward, 500-bar windows,
    a 30-trade minimum) 99% of the trials had fewer entry signals than that. A trial ruled out here is a
    trial like any other: it counts against the budget (the owner's call: a warm start can add budget once
    the number ruled out shows), and the searches learn from its violation.

    Entries only on a rule strategy (a portfolio trades on its schedule), and not with vectorbt's
    `accumulate` in the engine details, where an entry adds to a position and exits can close it in parts.
    ⚠️ The entry frames are the ones the run reads, after order timing's masks; the last bar's signal can
    fill nowhere, so the bound is loose by at most one per column, never tight the wrong way.

    Not screened: an entry sized under the minimum size at the starting cash (every one would mean no
    trades). It cost ~18 ms a trial to size them with vectorbtX, against ~0.2 s for the runs it would
    save, which were 5% of the owner's 62,700 trials: about even, so not worth the code.
    """
    min_trades = (job.get("constraints") or {}).get("min_trades")
    plan = prepared["plan"]
    if not min_trades or plan["kind"] != "signals" or "accumulate" in plan["args"]:
        return None
    entries = 0
    for key in ("entries", "short_entries"):
        frame = prepared["compiled"].get(key)
        if frame is not None:
            entries += int(pd.DataFrame(frame).fillna(False).to_numpy(dtype=bool).sum())
    if entries >= min_trades:
        return None
    nan = float("nan")
    return {"objective": nan, "metric": nan, "sharpe": nan, "total_return": nan, "max_drawdown": nan,
            "n_trades": nan, "profit_factor": nan, "notes": None, "screened": True, "entries": entries,
            "error": f"at most {entries} trades: only {entries} entry signals (not run)",
            "violation": (min_trades - entries) / min_trades}


def score(pf, job, params, want_returns=False, notes=()):
    """The objective, the extras and the constraints for one trial's portfolio.

    `notes`: vectorbtX's warnings about the trial's run. The first sentence of each is kept, one
    per line, as "notes": what happened (entries below the minimum size were not placed, entries
    were cut to the leverage), which is the reason behind "only 4 trades" that the trade count
    alone cannot give (the owner's MA job, 20 November 2026). Only the first sentence, because the
    whole trials table is rewritten after every batch; the rest is vectorbtX's advice in its own
    arguments, which the results page replaces with the Studio's (analysis.WARNING_HINTS).
    """
    extras = trial_extras(pf)
    value = METRICS[job["metric"]]["fn"](pf)
    result = {"params": params, "objective": value, "metric": value, **extras, "error": None,
              "notes": first_sentences(notes), "violation": 0.0, "screened": False}
    judge(result, job)
    if want_returns:
        result["returns"] = pf.returns(group_by=True)
    return result


def first_sentences(notes):
    """The first sentence of each of vectorbtX's warnings, once each, one per line; None for none."""
    said = []
    for note in notes or ():
        # A sentence ends at ". " before a capital: "1.0 £/point" and "2013-12-09" do not end one.
        first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(note).strip(), maxsplit=1)[0]
        if first not in said:
            said.append(first)
    return "\n".join(said) or None


def judge(result, job, fold_returns=None):
    """A trial's constraints, in place: an invalid trial's objective becomes NaN, with the first constraint it broke as
    its "error" and how far from valid it was as its "violation". `fold_returns`: each fold's total return, for the
    constraint on profitable folds (score_folds)."""
    # Each constraint broken, with how far from valid the trial was, from 0 (just valid) to 1 (as far as it
    # goes: no trades at all, a drawdown of the whole account, no profit at all). The searches are told the
    # sum (`violation`), so they can tell nearly valid from hopeless; the reason shown is the first broken.
    constraints = job.get("constraints") or {}
    broken = []
    min_trades = constraints.get("min_trades")
    if min_trades and result["n_trades"] < min_trades:
        # Without a trade-count floor, optimisers reliably find parameters that
        # trade twice and win both.
        broken.append((f"only {result['n_trades']} trades", (min_trades - result["n_trades"]) / min_trades))
    most = constraints.get("max_drawdown")
    if most and result["max_drawdown"] > most:
        broken.append((f"drawdown {result['max_drawdown']:.1%}",
                       min(1.0, (result["max_drawdown"] - most) / max(1e-9, 1.0 - most))))
    least = constraints.get("min_profit_factor")
    if least and not result["profit_factor"] >= least:
        short = (least - result["profit_factor"]) / least if np.isfinite(result["profit_factor"]) else 1.0
        broken.append((f"profit factor {result['profit_factor']:.2f}", min(1.0, short)))
    needed = (job.get("folds") or {}).get("min_profitable")
    if needed and fold_returns is not None:
        profitable = sum(1 for value in fold_returns if value > 0)
        if profitable < needed:
            broken.append((f"{profitable} of {len(fold_returns)} folds profitable", (needed - profitable) / needed))
    if broken:
        result.update(objective=float("nan"), error=broken[0][0], violation=sum(far for _, far in broken))
    if not np.isfinite(result["objective"]):
        result["objective"] = float("nan")
        if not broken:
            # Valid by every constraint, but the metric has no value (a Sharpe ratio of a run with no trades,
            # docs/UPSTREAM.md issue 39): as far from valid as a trial can be.
            label = METRICS.get(job["metric"], {}).get("label", str(job["metric"]))
            result.update(error=f"{label} has no value", violation=1.0)
    return result


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------
# The owner (26 November 2026): a train/test split and a stability check leave a search free to pick settings
# that worked in one stretch of the searched history and not the others. With folds, the searched history is
# cut into back-to-back windows, each trial is run on every one from the starting cash (as the held-back test
# is), and its objective is the folds' scores combined. The held-back test stays the final, unseen check.
#
# How they are combined is the point:
# * Each fold's score is first put on a scale where a gain and a loss of the same size weigh the same, and
#   "no edge" is 0 (fold_score): a return as its log growth a year, so -50% and +50% are -0.69 and +0.41 (and
#   their average is their compounding); a profit factor as its log, so 0.5 and 2 mirror each other; the
#   ratios as they are. A fold with no trades scores 0, never vectorbt's +inf (docs/UPSTREAM.md issue 39).
# * On that scale the folds' average is close to the whole run's own score (back-to-back folds of a return
#   compound to the whole run's), so an average adds nothing to one run. What folds add is how much the score
#   varies between them, so the default combines them as the average less one standard deviation, the score
#   of a fairly bad period (about one fold in six, were they normal). The worst fold is stricter and rests on
#   one fold; the median ignores one disastrous fold; the plain average is offered for comparison.
# * ⚠️ With few trades a fold, a fold's score is mostly noise: the owner's job #27 made 4 to 8 trades in each of
#   five folds. The minimum trades applies to the total, and the Optimise page shows the trades per fold.

FOLD_SUMMARIES = {
    "mean_less_sd": {"label": "Average less one spread",
                     "help": "The folds' average less their standard deviation: the score of a fairly bad period. "
                             "Settings that score well in every fold beat settings that score very well in some and "
                             "badly in others."},
    "worst": {"label": "Worst fold", "help": "The lowest fold's score: the strictest, and it rests on one fold."},
    "median": {"label": "Median fold", "help": "The middle fold's score: not moved by one very good or one "
                                               "disastrous fold, so it does not punish a fold that loses a lot."},
    "mean": {"label": "Average", "help": "The folds' average: close to the whole searched history's own score, so "
                                         "it says little that one run does not. For comparison."},
}

# How a metric's fold scores are put on a scale where a gain and a loss of a size weigh the same, before they are
# combined (fold_score), and back (fold_value). Every other metric is combined as it is.
FOLD_SCALES = {"total_return": {"scale": "log_growth", "label": "Return a year", "format": "pct"},
               "profit_factor": {"scale": "log", "label": "Profit factor", "format": "factor"}}


def fold_windows(index, window, count):
    """`count` back-to-back windows of (nearly) equal bars over `window` of `index` (all of it for None), as (start,
    end) labels, each at least two bars."""
    bars = index if window is None else index[(index >= window[0]) & (index <= window[1])]
    cuts = np.linspace(0, len(bars), int(count) + 1).round().astype(int)
    out = []
    for low, high in zip(cuts[:-1], cuts[1:]):
        if high - low >= 2:
            out.append((bars[low], bars[high - 1]))
    return out


def fold_score(metric, value, trades, years):
    """One fold's score on its fold scale (FOLD_SCALES): 0 is no edge, and a fold with no trades scores 0."""
    if not trades or value is None or np.isnan(value):
        return 0.0
    scale = FOLD_SCALES.get(metric, {}).get("scale")
    if scale == "log_growth":
        # ⚠️ A fold that lost everything is ln(0): floored at -99.99%, as bad as a fold can be and still a number.
        return math.log(max(1.0 + float(value), 1e-4)) / max(float(years), 1 / 365.25)
    if scale == "log":
        # ⚠️ No losing trade is an infinite profit factor: capped at 100 (and no winning one floored at 0.01), or
        # one fold of three winners would outweigh every other.
        return math.log(min(max(float(value), 0.01), 100.0))
    # A ratio of a fold with no spread (no losing day, no drawdown) is infinite: scored as no edge, like no trades.
    return float(value) if np.isfinite(value) else 0.0


def fold_value(metric, score):
    """A fold score, or folds combined, back on the metric's own scale: a return a year, a profit factor."""
    scale = FOLD_SCALES.get(metric, {}).get("scale")
    if scale == "log_growth":
        return math.expm1(score)
    if scale == "log":
        return math.exp(score)
    return score


def combine_folds(scores, how, direction):
    """Fold scores combined (FOLD_SUMMARIES), along the last axis: a float for one trial's, an array for a matrix
    of trials (overfitting_probability). "Less one spread" and "worst" lean the way the search counts as worse."""
    scores = np.asarray(scores, dtype=float)
    worse = -1.0 if direction == "maximize" else 1.0
    if how == "mean":
        out = scores.mean(axis=-1)
    elif how == "median":
        out = np.median(scores, axis=-1)
    elif how == "worst":
        out = scores.min(axis=-1) if direction == "maximize" else scores.max(axis=-1)
    else:
        spread = scores.std(axis=-1, ddof=1) if scores.shape[-1] > 1 else np.zeros(scores.shape[:-1])
        out = scores.mean(axis=-1) + worse * spread
    return float(out) if np.ndim(out) == 0 else out


def score_folds(pfs, job, params, notes=()):
    """A trial scored across folds: each fold's run (`pfs`, one per window, each from the starting cash) scored by
    the metric, put on its fold scale and combined (job["folds"]["summary"]) into the objective.

    "metric" is the combination back on the metric's own scale (fold_value), what the results page shows; "folds_json"
    each fold's score, metric, trades and total return, for the page and the overfitting probability. The other
    numbers every trial records are the folds' together: trades summed, total returns compounded (a chain of fresh
    accounts), the worst fold's drawdown, the Sharpe ratio of the folds' returns joined end to end, and the profit
    factor of all their trades. The constraints are judged on those, the minimum trades on the total.
    """
    metric, how = job["metric"], (job.get("folds") or {}).get("summary", "mean_less_sd")
    scores, values, trades, returns, drawdowns = [], [], [], [], []
    joined, pnl = [], []
    for pf in pfs:
        extras = trial_extras(pf)
        value = METRICS[metric]["fn"](pf)
        index = pf.wrapper.index
        years = (index[-1] - index[0]) / pd.Timedelta(days=365.25)
        scores.append(fold_score(metric, value, extras["n_trades"], years))
        values.append(float(value))
        trades.append(int(extras["n_trades"]))
        returns.append(float(extras["total_return"]))
        drawdowns.append(float(extras["max_drawdown"]))
        joined.append(pf.returns(group_by=True))
        pnl.append(np.asarray(pf.trades.pnl.values, dtype=float))
    objective = combine_folds(scores, how, job["direction"])
    pnl = np.concatenate(pnl) if pnl else np.array([])
    losses = -pnl[pnl < 0].sum()
    together = pd.concat(joined)
    result = {"params": params, "objective": objective, "metric": fold_value(metric, objective),
              "sharpe": float(together.vbt.returns(freq=pfs[0].wrapper.freq).sharpe_ratio()),
              "total_return": float(np.prod(1.0 + np.asarray(returns)) - 1.0),
              "max_drawdown": max(drawdowns), "n_trades": sum(trades),
              "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses > 0 else (np.inf if len(pnl) else np.nan),
              "error": None, "notes": first_sentences(notes), "violation": 0.0, "screened": False,
              "folds_json": json.dumps({"scores": scores, "metrics": values, "trades": trades, "returns": returns})}
    return judge(result, job, fold_returns=returns)


def overfitting_probability(trials, how, direction):
    """The probability of backtest overfitting (Bailey, Borwein, López de Prado and Zhu, "The probability of backtest
    overfitting", 2017), by combinatorially symmetric cross-validation over the folds every trial was scored on.

    The folds are split into two halves in every way (6 folds: 20 ways). For each, the trial best on one half (its
    folds combined as the search combined them) is ranked among all the trials on the other half; overfitting is
    it ranking below the median there. The probability is the share of splits where it does: near 0 means the best
    on some history stays good on the rest, 0.5 or more means picking the best is no better than picking at random.
    No backtest is run: it reads the trials' fold scores. Each distinct setting once (a learning search tries the
    same one many times), from the valid search trials. An even number of folds, at least 4, and 10 settings.

    {"probability", "splits", "trials", "logits" (one per split: ln(w / (1 - w)), w the relative rank out of
    sample; below 0 is overfit)}, or None.
    """
    from itertools import combinations

    if trials is None or trials.empty or "folds_json" not in trials:
        return None
    rows = trials[(trials["phase"] == "search") & np.isfinite(trials["objective"].astype(float))
                  & trials["folds_json"].notna()]
    rows = rows.drop_duplicates(subset="params_json")
    if len(rows) < 10:
        return None
    matrix = np.array([json.loads(text)["scores"] for text in rows["folds_json"]], dtype=float)
    count = matrix.shape[1]
    if count < 4 or count % 2:
        return None
    better = 1.0 if direction == "maximize" else -1.0
    logits = []
    for chosen in combinations(range(count), count // 2):
        rest = [fold for fold in range(count) if fold not in chosen]
        inside = better * combine_folds(matrix[:, list(chosen)], how, direction)
        outside = better * combine_folds(matrix[:, rest], how, direction)
        best = int(np.argmax(inside))
        # The relative rank out of sample, 1 to N over N + 1, ties sharing the average rank.
        w = stats.rankdata(outside)[best] / (len(outside) + 1)
        logits.append(float(math.log(w / (1 - w))))
    logits = np.asarray(logits)
    # Exactly at the median (a tie) counts half.
    probability = float(np.mean(np.where(logits < 0, 1.0, np.where(logits == 0, 0.5, 0.0))))
    return {"probability": probability, "splits": len(logits), "trials": len(rows), "logits": logits.tolist()}


def evaluate_chunk(tasks):
    """Several trials in ONE backtest call (the vectorised grid). Same results as evaluate_trial each.

    If vectorbtX rejects the combined call, the chunk falls back to one call per
    trial, so a single invalid trial is isolated instead of failing them all.
    """
    job = CONTEXT["job"]
    # Folds run per trial (each trial's folds are one call of their own): the Optimise page offers no folds with
    # many trials per call, and a job that has both runs its trials one by one here.
    if len(tasks) == 1 or tasks[0].get("want_returns") or tasks[0].get("windows") or job["metric"] == "selftest":
        return [evaluate_trial(task) for task in tasks]
    params_list = [{key: _python(value) for key, value in task["params"].items()} for task in tasks]
    try:
        trials = engine.run_vectorised(job["spec"], CONTEXT["data"], params_list, window=tasks[0].get("window"),
                                       data_dir=job["data_dir"], offline=job["settings"].get("offline", False),
                                       static_rate=job["settings"]["vectorbtx"].get("static_rate"))
    except engine.backtest_error():
        return [evaluate_trial(task) for task in tasks]
    return [score(trial["pf"], job, params, notes=trial["warnings"]) for params, trial in zip(params_list, trials)]


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
    returned trial, tell() with a value, and SQLite storage holding one study per
    walk-forward fold in one file. Only the worker's main process asks and tells (the
    pool only evaluates), so SQLite never sees concurrent writers.

    ⚠️ An invalid trial is told as COMPLETE and infeasible (`trial.set_constraint`, Optuna 5.0's
    replacement for the sampler's deprecated `constraints_func`), never state=FAIL. TPE learns from
    complete and pruned trials only (optuna/samplers/_tpe/sampler.py, `_split_trials`), so a failed
    trial taught it nothing and it kept sampling where trials were invalid; an infeasible one goes in
    its "bad" group, ranked by how far it was from valid. NSGA-II reads the constraint too. CMA-ES,
    random and quasi-random ignore it, so the value told is `penalty`'s, which CMA-ES ranks by. Every
    trial carries the constraint (0 when valid), so the study is constrained throughout; the results
    page leaves the infeasible ones out of its parameter importance.
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
    study = optuna.create_study(direction=direction, sampler=samplers[method_of("optuna", options)](),
                                storage=options.get("storage"), study_name=options.get("study_name", "job"),
                                load_if_exists=True)
    batch = max(1, int(options.get("batch", 8)))
    worst = -1.0 if direction == "maximize" else 1.0       # penalty() is a score to minimise
    done = 0
    while done < budget:
        trials = [study.ask() for _ in range(min(batch, budget - done))]
        results = evaluate_batch([suggest(trial, space, {}) for trial in trials])
        for trial, result in zip(trials, results):
            value = result["objective"]
            if np.isfinite(value):
                trial.set_constraint("valid", 0.0)
                study.tell(trial, value)
            else:
                trial.set_constraint("valid", violation_of(result))
                study.tell(trial, worst * penalty(result))
        done += len(trials)
    return {}


def run_scipy(space, evaluate_batch, budget, direction, seed, options):
    """differential_evolution (parallel through `workers`), dual_annealing or direct (one trial at a time).

    scipy minimises, so the objective is negated when maximising. Categoricals
    are an integer index; log ranges are searched in log space. A setting rounded
    to a whole number or a list's index often lands on a point already tried
    (DIRECT samples the centres of thirds, and 3.5 and 3.9 are both 4), so each
    point is run once and its score reused: the budget caps scipy's calls, and the
    backtests run can be fewer.

    DIRECT (scipy.optimize.direct, its locally biased form): divides the box of
    ranges into thirds, again and again, where the scores are best or the boxes
    are largest. Deterministic, so the seed does nothing.
    """
    _require_flat(space, "scipy")
    from scipy.optimize import differential_evolution, direct, dual_annealing

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

    scores = {}             # each point's score, by its params, so a repeat is not run again
    ran = 0                 # backtests run, which the budget caps

    def values_for(xs):
        nonlocal ran
        points = [decode(x) for x in xs]
        texts = [json.dumps(params, sort_keys=True, default=str) for params in points]
        new = {}                    # the points not scored yet, each once
        for text, params in zip(texts, points):
            if text not in scores:
                new.setdefault(text, params)
        # ⚠️ DIRECT treats maxfun as a soft limit and ends its iteration past it (13 trials on a budget of 12,
        # found 25 November 2026), which the job counted as "13 of 12". A point past the budget is not run: it
        # scores the penalty, which no best is ever chosen from.
        allowed = dict(list(new.items())[:max(0, budget - ran)])
        if allowed:
            for text, result in zip(allowed, evaluate_batch(list(allowed.values()))):
                value = result["objective"]
                scores[text] = sign * value if np.isfinite(value) else penalty(result)
            ran += len(allowed)
        return [scores.get(text, penalty({})) for text in texts]

    method = method_of("scipy", options)
    if method == "dual_annealing":
        dual_annealing(lambda x: values_for([x])[0], bounds, rng=seed, maxfun=budget, no_local_search=True)
        return {}
    if method == "direct":
        # maxiter as large as the budget, so the budget is what stops it (or DIRECT's own tolerances,
        # once its boxes are smaller than any step that could change a trade).
        direct(lambda x: values_for([x])[0], bounds, maxfun=budget, maxiter=max(1000, budget))
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
    # ⚠️ An invalid trial reaches BUTChC as NaN, which it ranks as neutral (0.5), not bad, so it neither
    # seeks nor avoids invalid regions (butchc/optimizer.py; docs/UPSTREAM.md, BUTChC). The owner judges
    # that choice deliberate and will revisit it last (26 November 2026): no penalty is sent instead.
    def objectives(configs):
        return [result["objective"] for result in evaluate_batch(list(configs))]

    executor = types.SimpleNamespace(map=lambda fn, configs: objectives(configs))
    keywords = {"direction": direction, "seed": seed, "batch": max(1, int(options.get("batch", 4))),
                "executor": executor, "verbose": False}
    if options.get("n_warmup"):
        keywords["n_warmup"] = int(options["n_warmup"])
    if options.get("warm_start"):
        # ⚠️ The warm-start tree must come from a run with the SAME direction.
        with open(options["warm_start"], "rb") as handle:
            keywords["start_prob_tree"] = pickle.load(handle)
    result = BUTChC_optimize(butchc_space(space), lambda config: objectives([config])[0], budget, **keywords)
    return {"prob_tree": result.get("prob_tree")}


# Nevergrad's optimisers offered, by their name in its registry, with what each does in Nevergrad's own
# words (its docstrings, 1.0.12). ⚠️ Not NGOpt, its chooser, nor anything built on its metamodels: on
# numpy 2.5 they raise TypeError ("only 0-dimensional arrays can be converted to Python scalars", from
# float() of a one-element array in nevergrad/optimization/metamodel.py). tests/test_optimise.py runs
# every one listed here, so a library change that breaks one fails there.
NEVERGRAD_OPTIMISERS = {
    "TwoPointsDE": {"label": "Differential evolution, two-point crossover",
                    "help": "Mixes members of a population, mutating along the differences between them, with a "
                            "genetic two-point crossover. Copes well with many trials at once."},
    "PSO": {"label": "Particle swarm",
            "help": "A swarm of settings, each moving with its own inertia towards its own best and the swarm's "
                    "best. Robust, and copes well with many trials at once."},
    "CMA": {"label": "CMA-ES",
            "help": "An evolution strategy: samples from a bell curve that it reshapes towards the best "
                    "directions. Best on number settings with a budget large against their count."},
    "PortfolioDiscreteOnePlusOne": {"label": "(1+1), mixed mutations",
                                    "help": "Mutates the best settings so far, with a mix of mutation sizes. "
                                            "Suits whole numbers and lists of values, where a small step does "
                                            "not matter."},
    "OnePlusOne": {"label": "(1+1) evolution strategy",
                   "help": "Mutates the best settings so far by a step it adapts. Simple and robust on number "
                           "settings, best with fewer than about eight trials at once."},
    "TBPSA": {"label": "TBPSA, for noisy scores",
              "help": "An evolution strategy that adapts its population size, among the best in many noisy "
                      "problems. A backtest's score is not noisy (the same settings score the same), but it can "
                      "jump from one setting to the next."},
}


def run_nevergrad(space, evaluate_batch, budget, direction, seed, options):
    """Nevergrad 1.0.12, by ask and tell in batches, so a batch runs on the job's worker processes.

    Flat spaces only. A whole number is searched as a count of its steps from the start (as Optuna keeps
    it on its step), a list as a Choice, a log range as Log; a setting with one value is fixed. Nevergrad
    minimises, so the objective is negated when maximising, and an invalid trial is told `penalty` (under
    the 5e20 above which Nevergrad clips and warns). Each search is told it may have `batch` trials out
    at once (num_workers), which some of its optimisers size their populations by.
    """
    _require_flat(space, "nevergrad")
    import nevergrad as ng

    # Positional names: a setting's path has dots, which Nevergrad's names are not promised to allow.
    parts = {}
    fixed = {}
    for number, (name, opt) in enumerate(space.items()):
        label = f"p{number}"
        if "values" in opt:
            if len(opt["values"]) == 1:
                fixed[name] = opt["values"][0]
            else:
                parts[label] = ng.p.Choice(list(opt["values"]))
        elif opt.get("int"):
            steps = (int(opt["max"]) - int(opt["min"])) // int(opt.get("step") or 1)
            if steps <= 0:
                fixed[name] = int(opt["min"])
            else:
                parts[label] = ng.p.Scalar(lower=0, upper=steps).set_integer_casting()
        elif float(opt["max"]) <= float(opt["min"]):
            fixed[name] = float(opt["min"])
        elif opt.get("log"):
            parts[label] = ng.p.Log(lower=float(opt["min"]), upper=float(opt["max"]))
        else:
            parts[label] = ng.p.Scalar(lower=float(opt["min"]), upper=float(opt["max"]))
    names = {f"p{number}": name for number, name in enumerate(space)}

    def decode(value):
        params = dict(fixed)
        for label, chosen in value.items():
            name = names[label]
            opt = space[name]
            if "values" in opt:
                params[name] = chosen
            elif opt.get("int"):
                params[name] = int(opt["min"]) + int(chosen) * int(opt.get("step") or 1)
            else:
                params[name] = float(chosen)
        return params

    batch = max(1, int(options.get("batch", 8)))
    sign = -1.0 if direction == "maximize" else 1.0
    if not parts:
        evaluate_batch([dict(fixed)])           # nothing to search: the one point there is
        return {}
    parametrization = ng.p.Dict(**parts)
    parametrization.random_state = np.random.RandomState(seed)
    name = method_of("nevergrad", options)
    if name == "CMA" and parametrization.dimension == 1:
        # ⚠️ On one dimension Nevergrad's CMA-ES switches to the fcmaes package, which is not installed (a
        # compiled one): (1+1) instead. The Optimise page says so beside the choice.
        name = "OnePlusOne"
    optimiser = ng.optimizers.registry[name](parametrization=parametrization, budget=budget, num_workers=batch)
    done = 0
    while done < budget:
        candidates = [optimiser.ask() for _ in range(min(batch, budget - done))]
        results = evaluate_batch([decode(candidate.value) for candidate in candidates])
        for candidate, result in zip(candidates, results):
            value = result["objective"]
            optimiser.tell(candidate, sign * value if np.isfinite(value) else penalty(result))
        done += len(candidates)
    return {}


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
        result = evaluate_batch([params])[0]
        # NaN handling is not documented, so return a large penalty instead, graded like the others.
        if np.isfinite(result["objective"]):
            return result["objective"]
        return -penalty(result) if direction == "maximize" else penalty(result)

    study = Study(direction=direction, random_state=seed, verbose=False)
    study.optimize(objective, n_trials=budget)
    return {}


# Labels: a library by its own name as its project writes it (Optuna, SciPy, BUTChC, marsopt), a search
# that is no library's by what it does, in sentence case.
BACKENDS = {
    "optuna": {"label": "Optuna", "fn": run_optuna},
    # Two methods: every point in order, one backtest each, or the same points many per backtest call
    # (worker.py routes the batches of the "vectorised" method through evaluate_chunk; eligibility:
    # engine.vectorisable).
    "grid": {"label": "Grid search", "fn": run_grid},
    "random": {"label": "Random search", "fn": run_random},
    "scipy": {"label": "SciPy", "fn": run_scipy},
    "butchc": {"label": "BUTChC", "fn": run_butchc},
    "nevergrad": {"label": "Nevergrad", "fn": run_nevergrad},
    "marsopt": {"label": "marsopt", "fn": run_marsopt},
}


# Each backend's methods, the first its default: {backend: {method: {"label", "help"}}}. The Optimise page
# asks the same two things for every backend, which backend and which of its methods (the owner, 25 November
# 2026: Optuna asked for a sampler, SciPy for a method and Nevergrad for an optimiser), and a backend with one
# method shows it, fixed. A job keeps the choice as backend_options["method"], which each backend reads
# through method_of. Optuna's, SciPy's and Nevergrad's methods are their library's own names for them.
METHODS = {
    "optuna": {
        "tpe": {"label": "TPE", "help": "Tree-structured Parzen estimator: learns from the trials so far where "
                                        "the good settings are, and tries more there. The default."},
        "cmaes": {"label": "CMA-ES", "help": "Evolves a cloud of points over the number settings, reshaping it "
                                             "towards the best; a list of values is drawn at random."},
        "nsga2": {"label": "NSGA-II", "help": "A genetic algorithm: breeds new settings from the best so far."},
        "random": {"label": "Random", "help": "Every trial drawn at random: no learning."},
        "qmc": {"label": "Quasi-random", "help": "Trials spread evenly over the ranges (a Sobol sequence): no "
                                                 "learning."},
    },
    "grid": {
        "loop": {"label": "One trial per backtest", "help": "Every combination in order, one backtest each."},
        "vectorised": {"label": "Many trials per backtest", "help": "The same points, run many at once as "
                                                                    "columns of one vectorbtX call: faster, for "
                                                                    "settings that change only the signals."},
    },
    "random": {"sobol": {"label": "Sobol sequence", "help": "Scrambled Sobol points, spread evenly over the ranges "
                                                            "with no learning: the baseline any other search "
                                                            "should beat."}},
    "scipy": {
        "differential_evolution": {"label": "Differential evolution",
                                   "help": "Improves a population of settings by mixing them, a whole population "
                                           "side by side."},
        "dual_annealing": {"label": "Dual annealing",
                           "help": "Wanders widely at first, then settles near the best it has found (its local "
                                   "searches are off: they would spend the budget on steps too small to change a "
                                   "trade). One trial at a time."},
        "direct": {"label": "DIRECT", "help": "Divides the ranges into thirds, again and again, where the scores are "
                                              "best or the untried part is largest. One trial at a time, and it "
                                              "uses no seed."},
    },
    "butchc": {"tree": {"label": "Bayesian update tree",
                        "help": "Learns how good each value of each setting is, separately, from every trial so far, "
                                "and draws the next trials from that. Searches the versions of a rule too."}},
    "nevergrad": {name: {"label": item["label"], "help": item["help"]} for name, item in NEVERGRAD_OPTIMISERS.items()},
    "marsopt": {"mars": {"label": "Mixed adaptive random search",
                         "help": "Tries settings near the best ones so far, nearer as the search goes on. One trial "
                                 "at a time."}},
}


def method_of(backend, options):
    """The method `backend` runs: options["method"] when it is one of the backend's own, else its default.

    A comparison runs several backends on one job's options, so a method named for one backend means
    nothing to the next, which then runs its own default.
    """
    methods = METHODS[backend]
    method = (options or {}).get("method")
    return method if method in methods else next(iter(methods))


def search_label(backend, options=None):
    """A backend and its method in words, "Optuna · TPE": for job names, the jobs table and the results.

    `backend` may be "backend:method", as a comparison names each search it runs. A job from before 25
    November 2026 named its method by each library's own word (sampler, optimizer) and ran the grid's
    many-trials-per-call method as a backend of its own ("grid_vectorised"); those read the same way.
    """
    options = dict(options or {})
    backend, _, method = str(backend).partition(":")
    if backend == "grid_vectorised":
        backend, method = "grid", "vectorised"
    method = method or options.get("method") or options.get("sampler") or options.get("optimizer")
    if backend not in BACKENDS:
        return backend
    return f"{BACKENDS[backend]['label']} · {METHODS[backend][method_of(backend, {'method': method})]['label']}"


# ---------------------------------------------------------------------------
# Robustness tools
# ---------------------------------------------------------------------------

def setting_names(spec):
    """{param path: {"label", "labels"}}: each searched setting in words, for the results page.

    The same words the Optimise page lists the variables in (tunable_inputs' label, such as
    "jma7 › length", and its value labels, such as "crosses above"), so a result reads as the
    strategy does, not as "param:features.jma7.params.length". The self-test job has no
    strategy, and a path a strategy no longer holds keeps its path.
    """
    from .spec import tunable_inputs

    if not (spec or {}).get("data"):
        return {}
    return {row["path"]: {"label": row["label"], "labels": row.get("labels") or {}} for row in tunable_inputs(spec)}


def leverage_cap(spec):
    """How much of the search asks for trades worth more than the leverage allows, where sizing makes a trade's
    worth a fixed multiple of the account: {"share" (0 to 1), "leverage", "skip", "what" ("risk" or
    "notional")}, or None.

    With a stop in % of the price and risk sized as a share of the account (risk_pct), a trade is worth
    risk ÷ stop times the account, whatever the price (vectorbtX sizes it to lose the risk at the stop): job
    #27's best, 0.195 over 0.0325, is 6x. Above the leverage, vectorbtX cuts the entry to `leverage x` the equity
    at its signal bar (or skips it, under when_cash_short "skip"), so in that part of the search the risk makes
    no difference (or costs trades). A share of the account per trade (notional_pct) above the leverage is the
    same thing directly. With a stop in points or from a line each trade has a worth of its own, so no share of
    the search can be named, and None. Only when the sizing or the stop is searched: the Risk tab's trade-size
    check covers a strategy's own settings. The strictest leverage of the strategy's symbols.
    """
    from .spec import variables, when_cash_short

    risk = spec.get("risk") or {}
    sizing = risk.get("sizing") or {}
    stop = risk.get("stop") or {}
    searched = variables(spec)
    if not spec.get("data", {}).get("symbols") or not searched:
        return None
    leverage = min(engine.default_leverage(spec).values(), default=None)
    if not leverage:
        return None

    def values(path, current):
        # Every value the search may try: the list, or the range in fine steps (any searcher may pick between).
        variable = searched.get(path)
        if variable is None:
            return np.array([float(current)])
        if "values" in variable:
            return np.array(variable["values"], dtype=float)
        return np.linspace(float(variable["start"]), float(variable["stop"]), 201)

    if "notional_pct" in sizing and "risk.sizing.notional_pct" in searched:
        what, worth = "notional", values("risk.sizing.notional_pct", sizing["notional_pct"])
    elif ("risk_pct" in sizing and stop.get("kind") in ("fixed", "trailing") and stop.get("unit") == "pct"
          and stop.get("pct") and ({"risk.sizing.risk_pct", "risk.stop.pct"} & set(searched))
          and not ({"risk.stop.kind", "risk.stop.unit"} & set(searched))):
        risks = values("risk.sizing.risk_pct", sizing["risk_pct"])
        stops = values("risk.stop.pct", stop["pct"])
        with np.errstate(divide="ignore"):
            what, worth = "risk", (risks[:, None] / stops[None, :]).ravel()      # every pair the search may try
    else:
        return None
    share = float(np.mean(worth > leverage * (1 + 1e-9)))
    if share == 0:
        return None
    return {"share": share, "leverage": float(leverage), "skip": when_cash_short(spec) == "skip", "what": what}


def best_trial(trials, direction):
    """The best finite trial from a trials DataFrame, or None."""
    finite = trials[np.isfinite(trials["objective"].astype(float))]
    if finite.empty:
        return None
    index = finite["objective"].idxmax() if direction == "maximize" else finite["objective"].idxmin()
    return finite.loc[index]


RANGE_CHECK_KEPT = 2000        # points the range check's cache keeps, the newest


def frames_key(data):
    """A short hash of a run's data (every field's values, index and columns), for a cache key."""
    digest = hashlib.sha256()
    for name in sorted(data):
        frame = data[name]
        digest.update(name.encode("utf-8"))
        if isinstance(frame, (pd.DataFrame, pd.Series)):
            names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
            digest.update(str(names).encode("utf-8"))
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        else:
            digest.update(repr(frame).encode("utf-8"))
    return digest.hexdigest()[:16]


def range_check(spec, data, space, min_trades, window=None, data_dir=None, offline=False, static_rate=None,
                cache=None, folds=None):
    """Before a job starts: the strategy at its own settings, then each searched setting at both ends of its
    range, on its step as the searches keep it (Optuna moves a stop of 42 by 3 from 10 to 40), or at each of a
    list's options (at most eight, else the first and last), the others at their own; on the history the
    search uses (`window`).

    [{"path" (None: the strategy's own settings), "value", "trades", "enough" (at least `min_trades`),
      "error", "notes" (what vectorbtX said, as a trial records it), "cached"}]. A setting whose ends fall short
    may still have valid values between them; a strategy whose own settings and every end fall short is unlikely
    to have any (the owner's MA job, 20 November 2026: 5,000 trials, none valid, from a slippage of 20% a fill).
    Only the top level of a conditional space: a nested setting needs its branch chosen.

    `cache`: a JSON file that keeps each point's run (trades, error, notes), keyed by everything that decides
    it: the strategy less its optimise section, the point's value, the window, the data's own values, the
    vectorbtX settings in force and the libraries' versions. So a check repeated after a page visit, a restart
    or a change of backend runs nothing, and a widened range runs only its new end (the owner, 25 November
    2026: each run of a levered, risk-sized strategy can take 30 s). Whether a point is enough is worked out
    afresh from `min_trades`, which is not in the key.

    `folds`: a number of folds, when the job scores trials across them (score_folds). The strategy's own
    settings then also say how many trades each fold makes ("fold_trades"), which is what decides whether a
    fold's score means anything: a fold of five trades is mostly noise.
    """
    from .config import VECTORBTX_KEYS
    from .jobs import read_json, write_json
    from .spec import spec_hash

    vbt_settings = engine.vectorbtx().settings["vectorbtx"]
    # ⚠️ Everything a run reads, or a stale count is shown for a changed strategy. Rates fetched from FRED
    # while the run prepares are not in it; they move a trade count only through the account's carry.
    base = {"spec": {key: value for key, value in spec.items() if key != "optimise"},
            "window": None if window is None else [str(window[0]), str(window[1])], "data": frames_key(data),
            "offline": bool(offline), "static_rate": static_rate,
            "vectorbtx": {key: vbt_settings.get(key) for key in VECTORBTX_KEYS}, "versions": engine.versions()}
    known = read_json(cache, default={}) if cache else {}
    points = [(None, None)]
    for path, opt in space.items():
        if "values" in opt:
            values = list(opt["values"])
            ends = values if len(values) <= 8 else [values[0], values[-1]]
        else:
            low, high = opt["min"], opt["max"]
            if opt.get("step"):
                # 1e-9: (0.2 - 0.01) / 0.01 is 18.999999999999996, which would floor a step short.
                high = low + math.floor((high - low) / opt["step"] + 1e-9) * opt["step"]
            ends = [int(round(low)), int(round(high))] if opt.get("int") else [float(low), round(float(high), 12)]
        for value in ends:
            points.append((path, value))
    rows = []
    ran = False
    for path, value in points:
        params = {} if path is None else {path: _python(value)}
        # The folds are in the own settings' key only: no other point counts its trades per fold.
        key = spec_hash({**base, "params": params, **({"folds": int(folds)} if folds and path is None else {})})
        found = known.get(key)
        cached = found is not None
        if not cached:
            ran = True
            try:
                run = engine.run_backtest(spec, data, params=params, window=window, data_dir=data_dir,
                                          offline=offline, static_rate=static_rate)
            except engine.backtest_error() as error:
                # A point vectorbtX rejects is reported at that point, as a trial records it, rather than
                # aborting the check. (vectorbtX 5cbe635 raises BacktestError, not RuntimeError, when its
                # re-sizing loop cannot settle: docs/UPSTREAM.md issue 36.)
                found = {"trades": None, "error": f"could not run: {error}", "notes": None}
            else:
                # The trade count and the notes only: no constraint, and the cheapest metric.
                result = score(run["pf"], {"metric": "total_return"}, params, notes=run["warnings"])
                found = {"trades": result["n_trades"], "error": None, "notes": result["notes"]}
                if folds and path is None:
                    cuts = fold_windows(data["close"].index, window, int(folds))
                    split = engine.run_prepared(engine.prepare_run(spec, data, params=params, window=window,
                                                                   data_dir=data_dir, offline=offline,
                                                                   static_rate=static_rate), windows=cuts)
                    found["fold_trades"] = [trial_extras(pf)["n_trades"] for pf in split["pfs"]]
            known[key] = found          # appended, so the oldest points are the ones dropped
        rows.append({"path": path, "value": value, **found, "cached": cached,
                     "enough": found["trades"] is not None and (not min_trades or found["trades"] >= min_trades)})
    if cache and ran:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        write_json(cache, dict(list(known.items())[-RANGE_CHECK_KEPT:]))
    return rows


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
