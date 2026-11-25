"""Optimise: the variables to search and their ranges, then the backend and its method, budget, objective
and constraints; start, watch and cancel jobs.

Each part is a section that folds by its title (the owner, 25 November 2026: with many variables the page
grew too long to find the start button). Jobs run in a separate process, so they keep going if you close
the page.
"""

import json
import math

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, data as data_module, engine, jobs, optimise, ui
from openstrategy_studio.spec import (collect_choices, default_variable, spec_hash, tunable_inputs, validate_spec,
                                      variable_values, variables)

settings, data_dir = ui.boot("Optimise")
ui.page_help("optimise")
spec = ui.current()
st.title("Optimise")

problems = validate_spec(spec)
missing = ui.missing_data(spec, data_dir)
if missing:
    problems = problems + [f"Import {', '.join(missing)} at {spec['data']['timeframe']} (or finer) on the Data page."]
ui.show_problems(problems)


def write_variables():
    ui.save_draft()


def shown(row, value):
    """How a value reads: in words where the row has them (crosses above, not crosses_above)."""
    return (row.get("labels") or {}).get(value, value)


# --- variables to optimise ---------------------------------------------------------------
# Like MT5's inputs: every value the strategy has can be added, features, rule numbers and
# risk dials alike, each searched over a range (start, step, stop) or a list of values.
# They are stored in the spec's optimise.variables, keyed by the value's dotted path, so
# they are written directly here: set_value would split the path at its dots.
rows = tunable_inputs(spec)
by_path = {row["path"]: row for row in rows}
# Read, not created: rendering a page must not change the spec (the section appears on the first Add).
found = (spec.get("optimise") or {}).get("variables") or {}
counts = []
with ui.section(f"Variables to optimise ({len(found)})" if found else "Variables to optimise", "feature",
                "opt-variables"):
    # The one place a value is added to the search or taken out of it (the owner's call, 20 November
    # 2026): the Build page only shows, under each condition, what is searched here.
    st.caption("Pick what the optimiser searches here, from any part of the strategy. The Build page shows what "
               "is searched under each rule's conditions.")
    candidates = [row for row in rows if row["path"] not in found]
    if candidates:
        groups = sorted({row["group"] for row in candidates}, key=["Features", "Rules", "Risk", "Portfolio"].index)
        # Narrow a long list to one part of the strategy; nothing picked shows every part.
        part = st.segmented_control("Show", groups, default=None, key=ui.key("optimise", ":part"),
                                    help="Show only one part of the strategy's values in the list below: its "
                                         "features' settings, its rules' numbers and comparisons, or its risk dials. "
                                         "Press it again to show every part.")
        if part:
            candidates = [row for row in candidates if row["group"] == part]
        left, right = st.columns([5, 1], vertical_alignment="bottom")
        picked = left.selectbox("Add a variable", candidates,
                                format_func=lambda row: f"{row['group']}: {row['label']} (now {shown(row, row['value'])})",
                                key=ui.key("optimise", ":add"),
                                help=f"Every value of the strategy that can vary: {', '.join(groups).lower()}.")
        if right.button("Add", type="primary", key=ui.key("optimise", ":add:go"), icon=":material/add:",
                        width="stretch"):
            spec.setdefault("optimise", {}).setdefault("variables", {})[picked["path"]] = default_variable(picked)
            write_variables()
            ui.renew_widgets()
            st.rerun()
    if not found:
        st.info("Nothing to optimise yet. Add a variable above: a feature's setting, a number in a rule, or a risk "
                "dial such as the stop distance.")
    for path, variable in list(found.items()):
        row = by_path.get(path)
        with st.container(border=True):
            if row is None:
                left, right = st.columns([5, 1])
                left.warning(f"{path} no longer exists in the strategy.")
                if right.button("Remove", key=ui.key(f"opt:{path}", ":remove")):
                    del found[path]
                    write_variables()
                    ui.renew_widgets()
                    st.rerun()
                continue
            head = st.columns([4, 2, 1])
            # The search radio beside it has no label of its own, so its ⓘ sits on the name.
            head[0].markdown(f"**{row['label']}**  \n<small>{row['group']} · now {shown(row, row['value'])}</small>",
                             unsafe_allow_html=True,
                             help="A range: start, step and stop (the grid tries every step; other searchers any "
                                  "value in between). These values: a list to try, and nothing else.")
            ways = ["list"] if row["type"] == "choice" else ["range", "list"]
            how = head[1].radio("Search", ways, index=0 if "values" not in variable or len(ways) == 1 else 1,
                                format_func={"range": "A range", "list": "These values"}.get, horizontal=True,
                                key=ui.key(f"opt:{path}", ":how"), label_visibility="collapsed")
            if head[2].button("Remove", key=ui.key(f"opt:{path}", ":remove")):
                del found[path]
                write_variables()
                ui.renew_widgets()
                st.rerun()
            if how == "range" and "values" in variable:
                found[path] = default_variable(row)
                write_variables()
                ui.renew_widgets()
                st.rerun()
            if how == "list" and "values" not in variable:
                found[path] = {"values": variable_values(variable, row["type"])}
                write_variables()
                ui.renew_widgets()
                st.rerun()
            if how == "range":
                cast = int if row["type"] == "int" else float
                number = {"step": 1} if row["type"] == "int" else {"format": "%g"}
                columns = st.columns(3)
                start = columns[0].number_input("Start", value=cast(variable["start"]),
                                                key=ui.key(f"opt:{path}", ":start"), **number,
                                                help="The smallest value to try.")
                step = columns[1].number_input("Step", value=cast(variable.get("step") or 0), min_value=cast(0),
                                               key=ui.key(f"opt:{path}", ":step"), **number,
                                               help="The grid's spacing. Other searchers pick any value in the "
                                                    "range (whole numbers for a whole-number setting).")
                stop = columns[2].number_input("Stop", value=cast(variable["stop"]), key=ui.key(f"opt:{path}", ":stop"),
                                               **number, help="The largest value to try.")
                new = {"start": cast(start), "stop": cast(stop)}
                if step:
                    new["step"] = cast(step)
            else:
                if row["type"] == "choice":
                    chosen = st.multiselect("Values to try", row["options"],
                                            default=[value for value in variable["values"] if value in row["options"]],
                                            key=ui.key(f"opt:{path}", ":values"),
                                            format_func=lambda value, row=row: str(shown(row, value)),
                                            help="The choices the optimiser may pick between.")
                    new = {"values": chosen}
                else:
                    text = st.text_input("Values to try, separated by commas", key=ui.key(f"opt:{path}", ":values"),
                                         value=", ".join(str(value) for value in variable["values"]),
                                         help="Exactly these values and no others, such as 10, 20, 50, 100.")
                    parsed = []
                    for piece in text.split(","):
                        piece = piece.strip()
                        if piece:
                            try:
                                parsed.append(int(piece) if row["type"] == "int" else float(piece))
                            except ValueError:
                                st.error(f"'{piece}' is not a {'whole ' if row['type'] == 'int' else ''}number.")
                    new = {"values": parsed}
            if new != variable:
                found[path] = new
                write_variables()
            size = len(variable_values(found[path], row["type"])) if (found[path].get("values") or
                                                                        found[path].get("stop") is not None) else 0
            counts.append(size)
            st.caption(f"{size} value{'s' if size != 1 else ''} on the grid.")
    if counts:
        st.caption(f"{math.prod(counts):,} combinations in a full grid.")

    searchable = [path for path in collect_choices(spec) if f"{path}.choose" in variables(spec)]
    structure = False
    if searchable:
        structure = st.toggle("Also search the rule versions", key=ui.kept("structure", True),
                              help="The optimiser chooses between the versions of a rule, tuning each one's own "
                                   "numbers only when it is chosen. BUTChC and Optuna only.")
space = optimise.build_space(spec, conditional=structure)
conditional = optimise.is_conditional(space)

# --- the search: a backend and one of its methods ---------------------------------------
# Every search setting below keeps its value for the session, across visits to other pages
# (ui.kept): they are not the strategy's, so they are not in the spec. Every backend is asked the
# same two things, which backend and which of its methods (optimise.METHODS), the method kept for
# each backend; a backend with one method shows it, fixed.
workers_setting = max(1, int(settings.get("workers", 1)))
options = {}
with ui.section("Search", "setup", "opt-search"):
    modes = ["optimise", "walk_forward", "compare"]
    mode = st.radio("Run", modes, horizontal=True, key=ui.kept("mode", "optimise", modes),
                    format_func={"optimise": "One search", "walk_forward": "Walk-forward",
                                 "compare": "Compare searches"}.get,
                    help="One search finds the best settings. Walk-forward searches each in-sample window and "
                         "tests its best settings on the next. Compare searches runs each chosen backend and method "
                         "several times on the same budget, so you can see which one finds good settings fastest "
                         "on THIS strategy.")
    backends = ["optuna", "butchc"] if conditional else list(optimise.BACKENDS)
    if conditional:
        st.caption("Searching alternatives: only BUTChC and Optuna can do this.")
    row = st.columns(4)
    backend = row[0].selectbox("Backend" if mode != "compare" else "Options from",
                               backends, format_func=lambda name: optimise.BACKENDS[name]["label"],
                               key=ui.kept("backend", "optuna", backends),
                               help="The search library or kind. Optuna learns where good settings are. Grid search "
                                    "tries every combination in order. Random search spreads trials evenly (a "
                                    "baseline). SciPy runs an evolutionary, annealing or DIRECT search. BUTChC "
                                    "learns each setting separately. Nevergrad offers particle swarm, CMA-ES, "
                                    "differential evolution and others. marsopt tries settings near the best ones so "
                                    "far, one trial at a time. When comparing, this picks whose options below are "
                                    "used (each search compared runs the method it names).")
    methods = list(optimise.METHODS[backend])
    if mode != "compare":
        method = row[1].selectbox("Method", methods, key=ui.kept(f"method:{backend}", methods[0], methods),
                                  format_func=lambda name: optimise.METHODS[backend][name]["label"],
                                  disabled=len(methods) == 1,
                                  help=" ".join(f"{item['label']}: {item['help']}"
                                                for item in optimise.METHODS[backend].values()))
        options["method"] = method
    else:
        method = methods[0]
        row[1].caption("Each search compared names its own method, below.")
    default_budget = max(10 * len(space), 50) if backend == "butchc" else 200
    budget = row[2].number_input("Budget (trials)", min_value=1, step=10,
                                 key=ui.kept(f"budget:{backend}", default_budget),
                                 help="How many backtests the search may run. More finds better settings but takes "
                                      "longer, and makes a lucky result more likely.")
    seed = row[3].number_input("Seed", step=1, key=ui.kept("seed", 0),
                               help="Makes the search repeat: the same seed and settings try the same trials.")
    if mode != "compare":
        st.caption(optimise.METHODS[backend][method]["help"])
    # What only some backends or methods need, under the choice.
    if backend == "optuna" and budget < 50:
        st.caption("Under about 50 trials, Optuna's TPE is usually the better choice on smooth problems.")
    elif backend == "grid":
        size = optimise.grid_size(space)
        st.caption(f"The grid has {size:,} points"
                   + (f"; the first {int(budget):,} will be tried." if size > budget else "."))
        if method == "vectorised" and mode != "compare":
            eligible, reason = engine.vectorisable(spec, space)
            if eligible:
                options["batch"] = st.slider("Trials per call", 8, 256, key=ui.kept("vectorised_batch", 64),
                                             help="How many grid points one vectorbtX call runs at once, as columns "
                                                  "side by side. Larger is faster and uses more memory.")
            else:
                st.warning(reason)
    elif backend == "scipy" and method in ("dual_annealing", "direct") and mode != "compare":
        st.caption(f"{optimise.METHODS['scipy'][method]['label']} runs one trial at a time, so it uses one worker "
                   "process whatever the number set below. Differential evolution runs a whole population side by "
                   "side. A point it comes back to (a whole number rounds the same) is not run again, so it can "
                   "finish under its budget.")
    elif backend == "butchc":
        st.caption("BUTChC learns each parameter separately: optimise a fast window and a gap rather than two "
                   "coupled windows. A rough budget is ten trials per parameter.")
        # As many as the worker processes by default: BUTChC waits for its whole batch before it learns, so a
        # batch of 4 on 15 workers left 11 idle (the owner's job of 25 November 2026 ran at 3.5 s a trial,
        # against 1.4 s with a batch of 12, on the same strategy).
        options["batch"] = st.slider("Batch", 1, 16, key=ui.kept("butchc_batch", min(16, workers_setting)),
                                     help="How many trials BUTChC draws before it learns from them, run side by side "
                                          "on the worker processes. Fewer than the worker processes leaves some "
                                          "idle. Up to 8 costs almost nothing in quality; past 16 it does.")
        warm = {}
        for _, job in jobs.list_jobs(data_dir).iterrows():
            tree = data_dir / "jobs" / job["job_id"] / "prob_tree.pkl"
            if job["backend"] == "butchc" and tree.exists():
                # Only a job that searched these same settings: a tree learned over others has nothing to say here.
                earlier = jobs.read_job(tree.parent) or {}
                if set(earlier.get("space") or {}) == set(space):
                    number = f"#{job['number']} " if job["number"] else ""
                    warm[str(tree)] = f"{number}{job['name']} ({str(job['created_at'])[:16].replace('T', ' ')})"
        starts = [None] + list(warm)
        choice = st.selectbox("Warm start from", starts, key=ui.kept("butchc_warm_start", None, starts),
                              format_func=lambda path: "Nothing: start from scratch" if path is None else warm[path],
                              help="Start from what an earlier BUTChC job learned about these same settings, instead "
                                   "of from scratch. Only jobs that searched the same settings are listed.")
        if choice:
            options["warm_start"] = choice
            st.caption("Warm starts must come from a job with the same direction (maximise or minimise).")
    elif backend == "nevergrad" and method == "CMA" and len(space) == 1 and "values" not in next(iter(space.values())):
        st.caption("With one number setting to search, CMA-ES runs as (1+1): Nevergrad's CMA-ES needs at least two.")
    elif backend == "marsopt":
        st.caption("Runs one trial at a time, so it uses one worker process whatever the number set below.")
    if backend in ("optuna", "random", "nevergrad") or (backend == "grid" and method == "loop"):
        options["batch"] = st.slider("Trials per batch", 1, 32, key=ui.kept("batch", min(32, workers_setting)),
                                     help="How many trials are chosen and run together, spread over the worker "
                                          "processes. Larger keeps the workers busy; a learning search updates only "
                                          "between batches.")

    compare_settings = None
    if mode == "compare":
        # Every method of every backend the space allows, as "backend:method" (the worker runs each with its own
        # method); not the grid's many-trials-per-call, which not every strategy can run.
        offered = [f"{name}:{item}" for name in backends for item in optimise.METHODS[name]
                   if (name, item) != ("grid", "vectorised")]
        chosen = st.multiselect("Searches to compare", offered, format_func=optimise.search_label,
                                key=ui.kept("compare_searches", [name for name in ("optuna:tpe", "butchc:tree",
                                                                                   "random:sobol")
                                                                 if name in offered], offered),
                                help="The backends and methods to race against each other on the same budget.")
        seeds = st.slider("Runs per search (different seeds)", 1, 10, key=ui.kept("compare_runs", 3),
                          help="Each search runs this many times with different seeds, which shows how much of its "
                               "result is luck.")
        compare_settings = {"backends": chosen, "seeds": list(range(int(seed), int(seed) + seeds))}
        st.caption(f"{len(chosen) * seeds * int(budget):,} backtests in total: the budget above is per run.")

# --- objective ---------------------------------------------------------------------------
with ui.section("Objective and constraints", "entry", "opt-objective"):
    row = st.columns(4)
    metrics = list(optimise.METRICS)
    metric = row[0].selectbox("Metric", metrics, format_func=lambda m: optimise.METRICS[m]["label"],
                              key=ui.kept("metric", "sharpe", metrics),
                              help="The number each trial is scored by. Sharpe and Sortino weigh return against its "
                                   "swings, Calmar and the recovery factor against the worst drawdown; profit factor, "
                                   "expectancy and SQN judge the trades.")
    # Kept for each metric, so it still follows the metric: a new metric starts at its own direction.
    directions = ["maximize", "minimize"]
    direction = row[1].radio("Direction", directions,
                             key=ui.kept(f"direction:{metric}", optimise.METRICS[metric]["direction"], directions),
                             format_func={"maximize": "Maximise", "minimize": "Minimise"}.get,
                             help="Whether a higher or a lower metric is better. It follows the metric.")
    min_trades = row[2].number_input("Minimum trades", min_value=0, step=5, key=ui.kept("min_trades", 30),
                                     help="Without a floor, optimisers find settings that trade twice and win both.")
    max_drawdown = row[3].number_input("Maximum drawdown (0 = off)", min_value=0.0, max_value=1.0, step=0.05,
                                       format="%.2f", key=ui.kept("max_drawdown", 0.0),
                                       help="A trial whose worst fall from a peak is deeper than this (0.25 = 25%) "
                                            "counts as invalid.")
    min_profit_factor = st.number_input("Minimum profit factor (0 = off)", min_value=0.0, step=0.1,
                                        key=ui.kept("min_profit_factor", 0.0),
                                        help="A trial whose gross profit divided by gross loss is below this counts "
                                             "as invalid.")

walk = mode == "walk_forward"
walk_settings = None
holdout = 0.0
with ui.section("Robustness", "risk", "opt-robustness"):
    if walk:
        row = st.columns(3)
        windows = ["rolling", "anchored"]
        walk_settings = {"window": int(row[0].number_input("In-sample bars", min_value=20, step=50,
                                                           key=ui.kept("walk_window", 500),
                                                           help="Bars each fold optimises on.")),
                         "step": int(row[1].number_input("Out-of-sample bars", min_value=5, step=25,
                                                         key=ui.kept("walk_step", 125),
                                                         help="Bars each fold then tests its best settings on, "
                                                              "unseen. The windows move forward by this much each "
                                                              "fold.")),
                         "mode": row[2].radio("Windows", windows, horizontal=True,
                                              key=ui.kept("walk_mode", "rolling", windows),
                                              help="rolling: each in-sample window keeps the same length and moves "
                                                   "forward. anchored: every one starts at the first bar and grows.")}
        st.caption("The budget is shared between the folds. Signals are computed on the full history and only the "
                   "runs are sliced, so indicators are warmed up and nothing looks ahead.")
    else:
        holdout = st.slider("Hold back the last part of the history for testing", 0, 50, format="%d%%",
                            key=ui.kept("holdout", 30),
                            help="The optimiser never sees it; the best trial is then run on it once, from the "
                                 "starting cash, as each search trial is.") / 100
    # Kept for each way of running, as it only applies to one search: off and greyed out for the others.
    stability = st.checkbox("Check parameter stability (re-run ±10% around the best)", disabled=mode != "optimise",
                            key=ui.kept(f"stability:{mode}", mode == "optimise"),
                            help="Re-runs settings near the best one. Good neighbours mean a broad, real edge; poor "
                                 "ones mean a lucky spike. One search only.")

# --- start ---------------------------------------------------------------------------------
row = st.columns([2, 1])
# Not kept: it follows the strategy, the backend and its method.
name = row[0].text_input("Job name", value=f"{spec['name']}: {optimise.search_label(backend, options)}",
                         help="How the job is listed here and on the Optimiser results page, beside its number.")
workers = row[1].number_input("Worker processes", min_value=1, step=1, key=ui.kept("workers", workers_setting),
                              help="Processes running trials side by side. Up to your CPU's core count is faster; "
                                   "each one uses its own memory.")

blocked = bool(problems) or not space
if backend == "grid" and method == "vectorised" and mode != "compare" and not engine.vectorisable(spec, space)[0]:
    blocked = True
if mode == "compare" and not (compare_settings or {}).get("backends"):
    blocked = True


def start():
    frames = ui.load_frames_for(spec, data_dir)
    job = {"name": name, "kind": mode, "spec": spec, "space": space, "compare": compare_settings,
           "backend": backend, "backend_options": options, "budget": int(budget), "metric": metric,
           "direction": direction, "seed": int(seed), "workers": int(workers), "settings": settings,
           "constraints": {"min_trades": int(min_trades) or None, "max_drawdown": max_drawdown or None,
                           "min_profit_factor": min_profit_factor or None},
           "split": {"train_fraction": 1 - holdout} if holdout and not walk else None,  # compare uses it too
           "walk_forward": walk_settings, "stability": stability,
           "fingerprint": data_module.fingerprint(data_dir, spec, frames)}
    return jobs.start_job(data_dir, job, frames)


# --- the ranges' ends, checked before a job starts -------------------------------------------
# The owner's MA job (20 November 2026) ran 5,000 trials and found none valid: a slippage of 20% a
# fill meant every trial made 0 or 1 trades, which one backtest at the strategy's own settings would
# have shown. So Start first runs the strategy at its own settings and each setting at the ends of
# its range, on the history the search uses, unless that was just checked for these exact settings.
# A setting whose ends fall short only warns (trials there are invalid, nothing worse). When the
# strategy's own settings and every end fall short, the job is not started until "Start anyway".
# What the check shown was made for, so it is not shown for other settings. Each point's run is also kept
# on disk (optimise.range_check's cache), so checking again after leaving the page or a restart runs only
# the points it has not run before (none, for the same settings).
signature = json.dumps([spec_hash(spec), space, int(min_trades), holdout, walk_settings], sort_keys=True,
                       default=str)
checked = st.session_state.get("range_check")
if checked is not None and checked["signature"] != signature:
    checked = None


def check_ranges():
    frames = ui.load_frames_for(spec, data_dir)
    index = frames["close"].index
    if walk:
        folds = optimise.walk_forward_folds(index, walk_settings["window"], walk_settings["step"], walk_settings["mode"])
        window = folds[0]["in_sample"] if folds else None
    else:
        window = optimise.split_window(index, 1 - holdout)[0] if holdout else None
    rows = optimise.range_check(spec, frames, space, int(min_trades), window=window, data_dir=data_dir,
                                offline=settings.get("offline", False),
                                static_rate=settings["vectorbtx"].get("static_rate"),
                                cache=data_dir / "cache" / "range_check.json")
    return {"signature": signature, "rows": rows, "hopeless": bool(min_trades) and not any(r["enough"] for r in rows)}


row = st.columns([1, 1, 2])
start_pressed = row[0].button("Start optimisation", type="primary", disabled=blocked,
                              help="Checks the ranges first (below) if they have not been checked for these settings, "
                                   "then starts the job in the background.")
check_pressed = row[1].button("Check the ranges", disabled=blocked, icon=":material/rule:",
                              help="Runs the strategy at its own settings, then each variable at the start and at the "
                                   "end of its range with the others at their own, on the history the search uses, "
                                   "and says where a trial cannot reach the minimum trades. A few backtests; a point "
                                   "checked before, on the same strategy and data, is not run again.")
if check_pressed or (start_pressed and checked is None):
    checked = ui.run_guarded("Checking the ranges", check_ranges)
    if checked is not None:
        st.session_state["range_check"] = checked

if checked is not None:
    names = optimise.setting_names(spec)
    shown_rows = []
    short = []
    for item in checked["rows"]:
        if item["path"] is None:
            setting, value = "Its own settings", ""
        else:
            named = names.get(item["path"], {})
            setting = named.get("label", item["path"])
            value = (named.get("labels") or {}).get(item["value"], item["value"])
        if not item["enough"] and item["path"] is not None:
            short.append(f"{setting} at {value} ({item['error'] or 'too few trades'})")
        shown_rows.append({"Setting": setting, "Value": str(value), "Trades": item["trades"],
                           "Enough": "yes" if item["enough"] else "no",
                           "What the backtest said": item["error"] if item["trades"] is None else (item["notes"] or "")})
    own = checked["rows"][0]
    notes = [item["notes"] for item in checked["rows"] if item["notes"]]
    example = notes[0].split("\n")[0] if notes else None
    if checked["hopeless"]:
        made = (own["error"] if own["trades"] is None
                else f"makes {own['trades']:,} trade{'' if own['trades'] == 1 else 's'} on the history the search uses")
        st.error(f"No trial is likely to be valid: at its own settings the strategy {made}, and no end of any range "
                 f"reaches the minimum of {int(min_trades):,} trades either."
                 + (f" The backtests said: \"{example}\"" if example else "")
                 + " Check its costs and sizes on the Build page (a slippage or spread typed in the wrong unit, a "
                   "size under the smallest size, a stop the risk cannot fund), or lower the minimum trades.",
                 icon=":material/block:")
        hint = analysis.warning_hints([example])[0][1] if example else None
        if hint:
            st.caption(hint)
    elif not own["enough"] or short:
        parts = []
        if own["trades"] is None:
            parts.append(f"At its own settings the strategy {own['error']}")
        elif not own["enough"]:
            parts.append(f"At its own settings the strategy makes {own['trades']:,} "
                         f"trade{'' if own['trades'] == 1 else 's'}, under the minimum of {int(min_trades):,}.")
        if short:
            parts.append(f"These ends of the ranges fall short of the minimum: {'; '.join(short)}. Trials there "
                         "count as invalid, so a narrower range spends the budget where trials can be valid.")
        st.warning(" ".join(parts), icon=":material/warning:")
    else:
        st.caption(f"Checked: the strategy at its own settings and every end of every range make at least "
                   f"{int(min_trades):,} trades on the history the search uses.")
    with st.expander("The ranges' ends, checked", expanded=checked["hopeless"]):
        ui.table(pd.DataFrame(shown_rows), column_config={
            "What the backtest said": st.column_config.TextColumn(width="large")})
        kept = sum(1 for item in checked["rows"] if item.get("cached"))
        if kept:
            st.caption(f"{kept} of {len(checked['rows'])} from earlier checks of the same strategy and data, so not "
                       "run again.")

start_now = start_pressed and checked is not None and not checked["hopeless"]
if checked is not None and checked["hopeless"]:
    start_now = start_now or st.button("Start anyway", icon=":material/play_arrow:", disabled=blocked,
                                       help="Start the job although the check found no setting that reaches the "
                                            "minimum trades: every trial is then likely to be invalid.")
if start_now:
    started = ui.run_guarded("Starting the job", start)
    if started is not None:
        st.success("Started, in the background: it keeps going if you leave this page. Its progress is under Jobs "
                   "below. The first trials take longest, while each worker process loads the backtest engine.")

# --- jobs ---------------------------------------------------------------------------------
listing = jobs.list_jobs(data_dir)
running = not listing.empty and listing["status"].isin(["starting", "running"]).any()


def took(seconds):
    """A duration in words: 40 s, 3 min, 1 h 20 min."""
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60)} min"


def since(stamp):
    """Seconds since an ISO time a job wrote (UTC), or None."""
    if not isinstance(stamp, str) or not stamp:
        return None
    return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(stamp)).total_seconds()


def trials_done(job):
    """A finished job's trials in one cell, with how it ended when that was not done: the jobs table's status and
    budget columns were one too many (the owner, 25 November 2026). A done job that ran under its budget (SciPy
    does not repeat a point) says of how many."""
    done, budget = int(job.done or 0), int(job.budget or 0)
    count = f"{done:,}" if job.status == "done" and done >= budget else f"{done:,} of {budget:,}"
    ended = {"done": "", "cancelled": ", cancelled", "failed": ", failed: see the Optimiser results page",
             "stopped": ", stopped from outside before it finished"}.get(job.status, f", {job.status}")
    return count + ended


@st.fragment(run_every="3s" if running else None)
def job_list():
    """Each running job as a card: what it is doing, how far it has got, how long is left. Redrawn every
    few seconds while one runs; the rest in a table, with the results a click away."""
    current = jobs.list_jobs(data_dir)
    if current.empty:
        st.write("No jobs yet.")
        return
    for job in current.itertuples():
        if job.status not in ("starting", "running"):
            continue
        with ui.card(f"job-{job.job_id}", "setup"):
            done, total = int(job.done or 0), max(1, int(job.budget or 1))
            stage = "Starting the worker" if job.status == "starting" else (job.stage or "Searching")
            if isinstance(job.fold, str) and job.fold:
                stage += f" (walk-forward {job.fold})"
            title = f"#{job.number} {job.name}" if job.number else job.name
            ui.card_header(f"job-{job.job_id}", title, meta=stage)
            # How long it has run, and how long is left at the pace of the trials so far (once a few
            # are in: the first ones also carry each process's start).
            parts = [f"{done:,} of {total:,} trials"]
            if job.best_value is not None and not pd.isna(job.best_value):
                parts.append(f"best so far {job.best_value:.4g}")
            elapsed = since(job.started_at)
            if elapsed is not None:
                parts.append(f"running for {took(elapsed)}")
            searching = since(job.searching_since)
            if searching and done >= 3 and done < total:
                parts.append(f"about {took((total - done) * searching / done)} left")
            elif done < total:
                parts.append("working out how long it will take")
            st.progress(min(1.0, done / total), text=" · ".join(parts))
            if st.button("Cancel", key=f"cancel:{job.job_id}", icon=":material/stop_circle:",
                         help="Stops the job after the trials already running; what it found so far is kept."):
                jobs.cancel_job(job.path)
                st.toast("Cancelling after the trials already running.")
    finished = current[~current["status"].isin(["starting", "running"])]
    if not finished.empty:
        # A job that stopped from outside (its process ended) says so, with what to do, until another job
        # is started: then it is history, and its row says it stopped (the owner, 25 November 2026: a job
        # that died on 19 November still headed the page under every later run). The newest first, so a
        # restart that stopped several jobs at once shows each of them.
        for job in current.itertuples():
            if job.status != "stopped":
                break
            if isinstance(job.error, str):
                st.warning(f"**{job.name}**: {job.error}")
        # Its number to keep track of it, what it searched with (the backend AND its method) and for (the
        # objective, without which "Best" means nothing), and one trials cell (the owner, 25 November 2026).
        rows = []
        for job in finished.itertuples():
            compared = job.searches if job.kind == "compare" and isinstance(job.searches, list) else None
            rows.append({"#": job.number, "Job": job.name,
                         "Search": ("Compared: " + ", ".join(optimise.search_label(item) for item in compared)
                                    if compared else optimise.search_label(job.backend, job.backend_options)),
                         "Objective": optimise.objective_label(job.metric, job.direction) if job.metric else "",
                         # No valid trial (or none yet) reads as a dash, not "None".
                         "Best": "—" if job.best_value is None or pd.isna(job.best_value) else f"{job.best_value:.4g}",
                         "Trials": trials_done(job),
                         "Started": str(job.created_at)[:16].replace("T", " ")})
        ui.table(pd.DataFrame(rows), column_config={"#": st.column_config.NumberColumn(width="small", format="%d")})
        # Words, not st.page_link: a link needs the whole multipage app, which a page tested on its own lacks.
        st.caption("Each job's best settings, in and out of sample, and its backtest are on the **Optimiser "
                   "results** page in the menu, under its number.")


with ui.section("Jobs", "account", "opt-jobs"):
    job_list()
