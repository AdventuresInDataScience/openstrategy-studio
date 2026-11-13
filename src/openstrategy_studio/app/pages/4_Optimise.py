"""Optimise: the variables to search and their ranges, then the backend, budget, objective and
constraints; start, watch and cancel jobs.

Jobs run in a separate process, so they keep going if you close the page.
"""

import importlib.util
import math

import numpy as np
import pandas as pd
import streamlit as st

from openstrategy_studio import data as data_module, engine, jobs, optimise, ui
from openstrategy_studio.spec import collect_choices, tunable_inputs, validate_spec, variable_values, variables

settings, data_dir = ui.boot("Optimise")
ui.page_help("optimise")
spec = ui.current()
st.title("Optimise")

problems = validate_spec(spec)
missing = ui.missing_data(spec, data_dir)
if missing:
    problems = problems + [f"Import {', '.join(missing)} at {spec['data']['timeframe']} (or finer) on the Data page."]
ui.show_problems(problems)

# --- variables to optimise ---------------------------------------------------------------
# Like MT5's inputs: every value the strategy has can be added, features, rule numbers and
# risk dials alike, each searched over a range (start, step, stop) or a list of values.
# They are stored in the spec's optimise.variables, keyed by the value's dotted path, so
# they are written directly here: set_value would split the path at its dots.
st.subheader("Variables to optimise")
rows = tunable_inputs(spec)
by_path = {row["path"]: row for row in rows}
# Read, not created: rendering a page must not change the spec (the section appears on the first Add).
found = (spec.get("optimise") or {}).get("variables") or {}


def default_variable(row):
    """A starting range around the current value: half to double it, ten steps."""
    if row["type"] == "choice":
        return {"values": list(row["options"])}
    value = row["value"] if isinstance(row["value"], (int, float)) and row["value"] else 1
    start, stop = (value / 2, value * 2) if value > 0 else (value * 2, value / 2)
    if row["type"] == "int":
        start, stop = int(math.floor(start)), int(math.ceil(stop))
        if stop <= start:
            stop = start + 1
        return {"start": start, "stop": stop, "step": max(1, (stop - start) // 10)}
    return {"start": round(start, 6), "stop": round(stop, 6), "step": round((stop - start) / 10, 6)}


def write_variables():
    ui.save_draft()


candidates = [row for row in rows if row["path"] not in found]
if candidates:
    left, right = st.columns([5, 1])
    groups = sorted({row["group"] for row in candidates}, key=["Features", "Rules", "Risk", "Portfolio"].index)
    picked = left.selectbox("Add a variable", candidates,
                            format_func=lambda row: f"{row['group']}: {row['label']} (now {row['value']})",
                            key=ui.key("optimise", ":add"),
                            help=f"Every value of the strategy that can vary: {', '.join(groups).lower()}.")
    right.write("")
    if right.button("Add", type="primary", key=ui.key("optimise", ":add:go")):
        spec.setdefault("optimise", {}).setdefault("variables", {})[picked["path"]] = default_variable(picked)
        write_variables()
        ui.renew_widgets()
        st.rerun()
if not found:
    st.info("Nothing to optimise yet. Add a variable above: a feature's setting, a number in a rule, or a risk dial "
            "such as the stop distance.")
counts = []
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
        head[0].markdown(f"**{row['label']}**  \n<small>{row['group']} · now {row['value']}</small>",
                         unsafe_allow_html=True)
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
            start = columns[0].number_input("Start", value=cast(variable["start"]), key=ui.key(f"opt:{path}", ":start"),
                                            **number)
            step = columns[1].number_input("Step", value=cast(variable.get("step") or 0), min_value=cast(0),
                                           key=ui.key(f"opt:{path}", ":step"), **number,
                                           help="The grid's spacing. Other searchers pick any value in the range "
                                                "(whole numbers for a whole-number setting).")
            stop = columns[2].number_input("Stop", value=cast(variable["stop"]), key=ui.key(f"opt:{path}", ":stop"),
                                           **number)
            new = {"start": cast(start), "stop": cast(stop)}
            if step:
                new["step"] = cast(step)
        else:
            if row["type"] == "choice":
                chosen = st.multiselect("Values to try", row["options"], default=[value for value in variable["values"]
                                                                                   if value in row["options"]],
                                        key=ui.key(f"opt:{path}", ":values"))
                new = {"values": chosen}
            else:
                text = st.text_input("Values to try, separated by commas", key=ui.key(f"opt:{path}", ":values"),
                                     value=", ".join(str(value) for value in variable["values"]))
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
    structure = st.toggle("Also search the rule versions", value=True,
                          help="The optimiser chooses between the versions of a rule, tuning each one's own numbers "
                               "only when it is chosen. BUTChC and Optuna only.")
space = optimise.build_space(spec, conditional=structure)
conditional = optimise.is_conditional(space)

# --- backend ------------------------------------------------------------------------
st.subheader("Search")
mode = st.radio("Run", ["optimise", "walk_forward", "compare"], horizontal=True,
                format_func={"optimise": "One search", "walk_forward": "Walk-forward",
                             "compare": "Compare backends"}.get,
                help="Compare backends runs each chosen backend several times on the same budget, so you can see "
                     "which one finds good settings fastest on THIS strategy.")
backends = ["optuna", "butchc"] if conditional else list(optimise.BACKENDS)
if conditional:
    st.caption("Searching alternatives: only BUTChC and Optuna can do this.")
row = st.columns(3)
backend = row[0].selectbox("Backend" if mode != "compare" else "Options from",
                           backends, format_func=lambda name: optimise.BACKENDS[name]["label"])
default_budget = max(10 * len(space), 50) if backend == "butchc" else 200
budget = row[1].number_input("Budget (trials)", min_value=1, value=default_budget, step=10, key=f"budget:{backend}")
seed = row[2].number_input("Seed", value=0, step=1)
options = {}
if backend == "optuna":
    options["sampler"] = st.selectbox("Sampler", ["tpe", "cmaes", "nsga2", "random", "qmc"],
                                      format_func={"tpe": "TPE", "cmaes": "CMA-ES (numbers only)",
                                                   "nsga2": "NSGA-II", "random": "Random", "qmc": "Quasi-random"}.get)
    if budget < 50:
        st.caption("Under about 50 trials, Optuna's TPE is usually the better choice on smooth problems.")
elif backend in ("grid", "grid_vectorised"):
    size = optimise.grid_size(space)
    st.caption(f"The grid has {size:,} points" + (f"; the first {int(budget):,} will be tried." if size > budget else "."))
    if backend == "grid_vectorised":
        eligible, reason = engine.vectorisable(spec, space)
        if eligible:
            options["batch"] = st.slider("Trials per call", 8, 256, 64)
        else:
            st.warning(reason)
elif backend == "scipy":
    options["method"] = st.radio("Method", ["differential_evolution", "dual_annealing"], horizontal=True,
                                 format_func=lambda m: m.replace("_", " ").capitalize())
elif backend == "butchc":
    st.caption("BUTChC learns each parameter separately: optimise a fast window and a gap rather than two coupled "
               "windows. A rough budget is ten trials per parameter.")
    options["batch"] = st.slider("Batch", 1, 16, 4, help="Up to 8 costs almost nothing in quality; past 16 it does.")
    warm = []
    for _, job in jobs.list_jobs(data_dir).iterrows():
        tree = data_dir / "jobs" / job["job_id"] / "prob_tree.pkl"
        if job["backend"] == "butchc" and tree.exists():
            warm.append(str(tree))
    choice = st.selectbox("Warm start from", [None] + warm, format_func=lambda path: "None" if path is None else path)
    if choice:
        options["warm_start"] = choice
        st.caption("Warm starts must come from a job with the same direction (maximise or minimise).")
elif backend == "marsopt" and importlib.util.find_spec("marsopt") is None:
    st.warning("marsopt is an optional extra: install it with `uv sync --extra marsopt`.")
if backend in ("optuna", "grid", "random"):
    options["batch"] = st.slider("Trials per batch", 1, 32, max(1, int(settings.get("workers", 1))))

# --- objective --------------------------------------------------------------------------
st.subheader("Objective and constraints")
row = st.columns(4)
metric = row[0].selectbox("Metric", list(optimise.METRICS), format_func=lambda m: optimise.METRICS[m]["label"])
direction = row[1].radio("Direction", ["maximize", "minimize"],
                         index=["maximize", "minimize"].index(optimise.METRICS[metric]["direction"]),
                         format_func={"maximize": "Maximise", "minimize": "Minimise"}.get)
min_trades = row[2].number_input("Minimum trades", min_value=0, value=30, step=5,
                                 help="Without a floor, optimisers find settings that trade twice and win both.")
max_drawdown = row[3].number_input("Maximum drawdown (0 = off)", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                   format="%.2f")
min_profit_factor = st.number_input("Minimum profit factor (0 = off)", min_value=0.0, value=0.0, step=0.1)

compare_settings = None
if mode == "compare":
    st.subheader("Comparison")
    offered = [name for name in backends if name not in ("grid_vectorised",)]
    if importlib.util.find_spec("marsopt") is None and "marsopt" in offered:
        offered.remove("marsopt")
    chosen = st.multiselect("Backends", offered, default=[name for name in ("optuna", "butchc", "random") if name in offered],
                            format_func=lambda name: optimise.BACKENDS[name]["label"])
    seeds = st.slider("Runs per backend (different seeds)", 1, 10, 3)
    compare_settings = {"backends": chosen, "seeds": list(range(int(seed), int(seed) + seeds))}
    st.caption(f"{len(chosen) * seeds * int(budget):,} backtests in total: the budget above is per run.")

st.subheader("Robustness")
walk = mode == "walk_forward"
walk_settings = None
holdout = 0.0
if walk:
    row = st.columns(3)
    walk_settings = {"window": int(row[0].number_input("In-sample bars", min_value=20, value=500, step=50)),
                     "step": int(row[1].number_input("Out-of-sample bars", min_value=5, value=125, step=25)),
                     "mode": row[2].radio("Windows", ["rolling", "anchored"], horizontal=True)}
    st.caption("The budget is shared between the folds. Signals are computed on the full history and only the "
               "runs are sliced, so indicators are warmed up and nothing looks ahead.")
else:
    holdout = st.slider("Hold back the last part of the history for testing", 0, 50, 30, format="%d%%",
                        help="The optimiser never sees it; the best trial is then run on it once.") / 100
stability = st.checkbox("Check parameter stability (re-run ±10% around the best)", value=mode == "optimise",
                        disabled=mode != "optimise")

row = st.columns([2, 1])
name = row[0].text_input("Job name", value=f"{spec['name']}: {backend}")
workers = row[1].number_input("Worker processes", min_value=1, value=int(settings.get("workers", 1)), step=1)

blocked = bool(problems) or not space
if backend == "grid_vectorised" and mode != "compare" and not engine.vectorisable(spec, space)[0]:
    blocked = True
if mode == "compare" and not (compare_settings or {}).get("backends"):
    blocked = True
if st.button("Start optimisation", type="primary", disabled=blocked):
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

    started = ui.run_guarded("Starting the job", start)
    if started is not None:
        st.success("Started. Progress is shown below; results are on the Results page.")

with st.expander("Check the job runner"):
    st.write("Runs a quick job on a synthetic objective (no backtest) to confirm that background jobs work on this machine.")
    if st.button("Run a self-test job"):
        index = pd.bdate_range("2020-01-01", periods=10)
        frames = {"close": pd.DataFrame({"X": np.ones(10)}, index=index)}
        job = {"name": "Self-test", "kind": "optimise", "spec": {}, "space": {"a": {"min": 0.0, "max": 10.0},
               "b": {"min": 1, "max": 50, "int": True}, "c": {"values": ["x", "y", "z"]}},
               "backend": "random", "backend_options": {"batch": 4}, "budget": 64, "metric": "selftest",
               "direction": "maximize", "seed": 0, "workers": 1, "settings": settings, "constraints": {},
               "split": None, "walk_forward": None, "stability": False}
        jobs.start_job(data_dir, job, frames)
        st.success("Self-test started. It should finish within seconds with a best value near 1.0.")

# --- jobs ---------------------------------------------------------------------------------
st.subheader("Jobs")
listing = jobs.list_jobs(data_dir)
running = not listing.empty and listing["status"].isin(["starting", "running"]).any()


@st.fragment(run_every="3s" if running else None)
def job_list():
    current = jobs.list_jobs(data_dir)
    if current.empty:
        st.write("No jobs yet.")
        return
    for job in current.itertuples():
        if job.status in ("starting", "running"):
            with st.container(border=True):
                done, total = int(job.done or 0), max(1, int(job.budget or 1))
                st.progress(min(1.0, done / total), text=f"{job.name}: {done} of {total} trials, best "
                            f"{'—' if job.best_value is None or pd.isna(job.best_value) else f'{job.best_value:.4g}'}")
                if st.button("Cancel", key=f"cancel:{job.job_id}"):
                    jobs.cancel_job(job.path)
                    st.toast("Cancelling after the current batch.")
    finished = current[~current["status"].isin(["starting", "running"])]
    if not finished.empty:
        ui.table(finished[["name", "backend", "status", "done", "budget", "best_value", "created_at"]])


job_list()
