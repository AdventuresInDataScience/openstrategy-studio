"""Optimiser results: each optimisation job's best settings first, then how the search went.

The page leads with the answer (the owner's call, 20 November 2026): the best settings in the
strategy's own words, the objective in sample and out of sample, and the best trial's backtest
with the held-back part marked. The trials, the search's charts and the robustness checks follow.

Saved backtests are compared on the Backtest results page (moved there on 19 November 2026,
when the menu names said which results each page holds).
"""

import copy
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, jobs, optimise, ui
from openstrategy_studio.spec import get_path, set_path, spec_hash

settings, data_dir = ui.boot("Optimiser results")
ui.page_help("results")
ui.font_css()
st.title("Optimiser results")


def load_frames(job_dir):
    """The data the job ran on: its own snapshot, so the backtest matches the job's numbers."""
    with open(job_dir / "frames.pkl", "rb") as handle:
        return pickle.load(handle)


def shown_value(path, value):
    """A setting's value in words where it has them (crosses above, not crosses_above)."""
    return (names.get(path, {}).get("labels") or {}).get(value, value)


def searched(variable):
    """What a variable searched, in words: 3 to 14 by 1, or 10, 20, 50."""
    if not variable:
        return ""
    if "values" in variable:
        return ", ".join(str(value) for value in variable["values"])
    return f"{variable['start']:g} to {variable['stop']:g}" + (f" by {variable['step']:g}" if variable.get("step")
                                                                else "")


listing = jobs.list_jobs(data_dir)
if listing.empty:
    st.write("No optimisations yet. Start one on the Optimise page.")
    st.stop()

labels = {row.job_id: f"{row.name} ({row.status}, {row.created_at})" for row in listing.itertuples()}
job_id = st.selectbox("Job", list(labels), format_func=labels.get,
                      help="An optimisation started on the Optimise page, with its status and start time.")
job_dir = Path(listing.set_index("job_id").loc[job_id, "path"])
job = jobs.read_job(job_dir)
progress = jobs.read_progress(job_dir)
summary = jobs.read_summary(job_dir)
trials = jobs.read_trials(job_dir)
direction = job["direction"]
metric_label = optimise.METRICS.get(job["metric"], {}).get("label", job["metric"])
# The same inside a sentence: lower case, but a ratio named after a person keeps its capital.
metric_words = metric_label if metric_label.split(" ")[0] in ("Sharpe", "Sortino", "Calmar") else metric_label.lower()
# Each searched setting in the strategy's own words (jma7 › length), for every table and chart.
names = optimise.setting_names(job.get("spec"))
variables = ((job.get("spec") or {}).get("optimise") or {}).get("variables") or {}

if progress["status"] == "failed":
    st.error("The job failed.")
    st.code(progress.get("error", ""), language="text")
    log = job_dir / "worker.log"
    if log.exists():
        with st.expander("Worker log"):
            st.code(log.read_text(encoding="utf-8")[-5000:], language="text")
elif progress["status"] == "stopped":
    # Its process ended from outside (jobs.read_progress); what it finished, if anything, is below.
    st.warning(progress.get("error") or "The job stopped without finishing.")
elif progress["status"] in ("starting", "running"):
    st.info(f"Still running: {progress.get('done', 0)} of {progress.get('budget', 0)} trials. "
            "Refresh to see the latest.")

if trials.empty:
    st.stop()
search = trials[trials["phase"] == "search"]
params_columns = analysis.param_columns(search)
theme, glow = ui.theme_picker(settings, "results")

# --- the answer first: the best settings, in and out of sample ------------------------------
best = optimise.best_trial(search, direction) if job["kind"] != "walk_forward" else None
if best is None and job["kind"] != "walk_forward" and not search.empty and progress["status"] == "done":
    # Nothing to lead with: say why, from the trials' own reasons ("only 4 trades" and the like,
    # grouped whatever the number), rather than leave an empty page.
    reasons = search["error"].dropna().astype(str).str.replace(r"[\d.,]+", "N", regex=True).value_counts()
    top = f" Most often: {reasons.index[0]} ({reasons.iloc[0]} of {len(search)} trials)." if len(reasons) else ""
    most = (f" The most trades any trial made was {int(search['n_trades'].max()):,}."
            if "n_trades" in search and search["n_trades"].notna().any() else "")
    # Why, in the backtests' own words: what vectorbtX said most often about the trials' runs
    # (entries below the minimum size, cut to the leverage), counted once per trial whatever its
    # numbers, with one trial's words as the example and the Studio's hint for it. Jobs from before
    # 20 November 2026 kept no notes.
    said, hint = "", None
    counts, examples = {}, {}
    for note in (search["notes"].dropna().astype(str) if "notes" in search else []):
        seen = set()
        for line in note.split("\n"):
            kind = re.sub(r"[\d.,]+", "N", line)
            if kind not in seen:
                seen.add(kind)
                counts[kind] = counts.get(kind, 0) + 1
                examples.setdefault(kind, line)
    if counts:
        commonest = max(counts, key=counts.get)
        said = (f" The backtests said, in {counts[commonest]:,} of {len(search):,} trials, for example: "
                f"\"{examples[commonest]}\"")
        hint = analysis.warning_hints([examples[commonest]])[0][1]
    st.warning(f"No trial was valid: every one broke a constraint or could not run, so there are no best settings.{top}"
               f"{most}{said} Loosen the constraints or widen the ranges on the Optimise page, or check the "
               "strategy's costs and sizes on the Build page; the table at the foot of this page gives each trial's "
               "reason.")
    if hint:
        st.caption(hint)
if best is not None and job["metric"] != "selftest":
    params = jobs.trial_params(best)
    held = summary.get("held_out") or {}
    # The raw score: the objective is the same number for a valid trial (older jobs record only it).
    in_sample = float(best.get("metric", best["objective"]))
    with ui.card("best-trial", "entry"):
        ui.card_header("best-trial", "Best trial", meta=f"the best {metric_words} of "
                       f"{int(summary.get('n_trials', len(search))):,} tried, on the part of the history the "
                       "optimiser searched",
                       help="In sample is the history the optimiser searched, and the best trial is the best "
                            "score there. Out of sample is the history held back from it, run once with the best "
                            "settings afterwards: the honest test. A big drop from one to the other means the "
                            "settings fit the past rather than found an edge.")
        columns = st.columns(4)
        columns[0].metric(f"{metric_label}, in sample", f"{in_sample:.4g}",
                          help="The best trial's score on the history the optimiser searched.")
        if held.get("metric") is not None:
            out_of_sample = float(held["metric"])
            columns[1].metric(f"{metric_label}, out of sample", f"{out_of_sample:.4g}",
                              delta=f"{out_of_sample - in_sample:+.4g} from in sample",
                              delta_color="normal" if direction == "maximize" else "inverse",
                              help=f"The same settings on the held-back history, {held['window'][0][:10]} to "
                                   f"{held['window'][1][:10]}, which the optimiser never saw.")
        else:
            columns[1].metric(f"{metric_label}, out of sample", "—",
                              help="Nothing was held back from this job: set a share to hold back on the Optimise "
                                   "page to test the best settings on history the optimiser never saw.")
        columns[2].metric("Trades, in / out of sample",
                          f"{int(best['n_trades']):,} / " + ("—" if held.get("n_trades") is None
                                                             else f"{int(held['n_trades']):,}"),
                          help="How many trades the best settings made in each part. A handful is not evidence.")
        columns[3].metric("Total return, in / out of sample",
                          f"{float(best['total_return']):.1%} / " + ("—" if held.get("total_return") is None
                                                                     else f"{float(held['total_return']):.1%}"),
                          help="Each part is run from the starting cash on its own, as the optimiser scored it.")

        # The best settings, each named as the strategy names it, beside the value the job started
        # from and the range it searched.
        rows = []
        for path, value in params.items():
            started = get_path(job["spec"], path) if names else None
            rows.append({"Setting": names.get(path, {}).get("label", path), "Best": str(shown_value(path, value)),
                         "When the job ran": "" if started is None else str(shown_value(path, started)),
                         "Searched": searched(variables.get(path))})
        ui.table(pd.DataFrame(rows))

        buttons = st.container(horizontal=True, gap="small")
        apply = buttons.button("Apply to the strategy", type="primary", icon=":material/check:",
                               help="Writes these values into the job's strategy, keeping the optimisation "
                                    "ranges, and opens it on the Build page.")
        show = buttons.button("Show its backtest", icon=":material/show_chart:",
                              help="Draws the best settings over the whole history, with a dashed line where the "
                                   "held-back part starts, and its numbers in sample, out of sample and over the "
                                   "whole period. The job saved it as it finished; a job from before 25 November "
                                   "2026 runs it now.")
        check = buttons.button("Check it reproduces", icon=":material/verified:",
                               help="Re-runs the best trial to check the stored result still comes out the same, "
                                    "and works out the deflated Sharpe ratio.")
    if apply:
        # The values go where the variables point; the variables (and their ranges) stay.
        updated = copy.deepcopy(job["spec"])
        for path, value in params.items():
            set_path(updated, path, value)
        if spec_hash(ui.current()) != spec_hash(job["spec"]):
            st.info("The strategy open now differs from the one this job optimised; the job's version was used.")
        ui.replace_spec(updated)
        st.success("Applied. The Build page now shows these values.")

    if show:
        def best_backtest():
            # The job's own record (worker.save_best_backtest), when it is of these settings: a job that
            # is still running may have found a better trial since. Otherwise it is run now.
            saved_path = job_dir / "best_backtest.pkl"
            if saved_path.exists():
                with open(saved_path, "rb") as handle:
                    saved = pickle.load(handle)
                if saved["params"] == params:
                    return {key: saved[key] for key in ("results", "split", "table")}
            return analysis.split_backtest(job, load_frames(job_dir), params, settings)

        shown = ui.run_guarded("Opening the best trial's backtest", best_backtest)
        if shown is not None:
            st.session_state["best_backtest"] = {"job": job_id, **shown}
    if check:
        def rerun_best():
            frames = load_frames(job_dir)
            window = None
            if job.get("split"):
                window = optimise.split_window(frames["close"].index, float(job["split"]["train_fraction"]))[0]
            run = engine.run_backtest(job["spec"], frames, params=params, window=window, data_dir=job["data_dir"],
                                      offline=bool(settings.get("offline")),
                                      static_rate=settings["vectorbtx"].get("static_rate"))
            pf = run["pf"]
            value = optimise.METRICS[job["metric"]]["fn"](pf)
            returns = pf.returns(group_by=True)
            dsr = optimise.deflated_sharpe(optimise.trial_extras(pf)["sharpe"], search["sharpe"].tolist(),
                                           returns, job["spec"]["account"]["freq"])
            return value, dsr

        outcome = ui.run_guarded("Re-running the best trial", rerun_best)
        if outcome is not None:
            value, dsr = outcome
            match = abs(value - in_sample) <= 1e-9 * max(1.0, abs(value))
            (st.success if match else st.warning)(
                f"Re-run gives {value:.6g}; the job recorded {in_sample:.6g}."
                + ("" if match else " They differ: the data or the libraries changed since the job ran."))
            if dsr is not None:
                st.write(f"Deflated Sharpe ratio: **{dsr:.1%}**, the probability that the true Sharpe is above "
                         f"zero after allowing for {int(search['objective'].notna().sum())} trials.")

    # The best trial's backtest, kept while this job stays picked.
    backtest = st.session_state.get("best_backtest")
    if backtest is not None and backtest["job"] == job_id:
        st.subheader("The best trial's backtest")
        row = st.columns([2, 2])
        view = row[0].selectbox("View", list(analysis.SPLIT_VIEWS), format_func=analysis.SPLIT_VIEWS.get,
                                help="Which chart of the best trial's backtest to draw. Each marks where the "
                                     "held-back history starts.")
        symbol = None
        if view == "price":
            symbols = backtest["results"]["meta"]["symbols"]
            symbol = row[1].selectbox("Symbol", symbols, help="Which symbol's prices and trades to show.")
        ui.chart(analysis.fig_split(backtest["results"], backtest["split"], theme, glow, view, symbol))
        if backtest["split"] is not None:
            st.caption("Left of the dashed line is the history the optimiser searched; right of it, shaded, the "
                       "history held back from it. The curve is one account from the first bar to the last; the "
                       "table runs each part from the starting cash on its own, as the optimiser scored it.")
        # The row names as a column of their own, wide enough to read whole.
        ui.table(backtest["table"].reset_index(names=""), column_config={
            "": st.column_config.TextColumn(width="medium")})

# --- walk-forward: each fold's best settings and how they held up -----------------------------
if summary.get("walk_forward"):
    with ui.card("walk-forward", "entry"):
        walk = summary["walk_forward"]
        ratio = walk.get("oos_to_is_ratio")
        ui.card_header("walk-forward", "Walk-forward",
                       meta="each fold searched its in-sample window, then ran its best settings on the next",
                       help="The out-of-sample to in-sample ratio compares the folds' average scores: near 1 means "
                            "the in-sample result carried over; well below it means the settings fit each window "
                            "rather than found an edge.")
        st.metric(f"{metric_label}: out of sample / in sample", "—" if ratio is None else f"{ratio:.2f}")
        rows = []
        for fold in walk["folds"]:
            row = {"Fold": fold["fold"] + 1, "In sample": " to ".join(str(t)[:10] for t in fold["in_sample"]),
                   "Out of sample": " to ".join(str(t)[:10] for t in fold.get("out_of_sample", [])),
                   f"{metric_label}, in": fold.get("in_sample_metric"),
                   f"{metric_label}, out": fold.get("out_of_sample_metric")}
            for path, value in (fold.get("params") or {}).items():
                row[names.get(path, {}).get("label", path)] = str(shown_value(path, value))
            if fold.get("error"):
                row["Note"] = fold["error"]
            rows.append(row)
        ui.table(pd.DataFrame(rows))
        stitched = job_dir / "walk_forward_returns.parquet"
        if stitched.exists():
            ui.chart(analysis.fig_walk_forward(pd.read_parquet(stitched)["returns"], theme, glow))

# --- how the search went ----------------------------------------------------------------------
st.subheader("How the search went")
if job["kind"] == "compare" and summary.get("compare"):
    compare = summary["compare"]
    st.write("Each backend ran several times (different seeds) with the same budget. "
             + ("" if compare["target"] is None else
                f"The target is 95% of the way to the best value any run found ({compare['target']:.4g})."))
    rows = []
    for backend_name, item in compare["backends"].items():
        rows.append({"Backend": optimise.BACKENDS.get(backend_name, {}).get("label", backend_name),
                     "Runs": item["runs"], "Median best": item["median_final"], "Best": item["best_final"],
                     "Worst": item["worst_final"], "Runs reaching the target": item["reached_target"],
                     "Median trials to the target": item["median_trials_to_target"],
                     "Valid trials": f"{item['valid_share']:.0%}"})
    ui.table(pd.DataFrame(rows))
    ui.chart(analysis.fig_compare_backends(trials, direction, theme))
    st.caption("BUTChC's published benchmarks are synthetic; this is the comparison that counts for your strategy.")
columns = st.columns(3)
# Always show how many configurations were tried: it is what the best is selected from.
columns[0].metric("Configurations tried", int(summary.get("n_trials", len(search))),
                  help="Every set of settings the search ran. The more it tries, the more likely the best is luck.")
columns[1].metric("Valid trials", int(search["objective"].notna().sum()),
                  help="Trials that ran and met the constraints (minimum trades, maximum drawdown, minimum profit "
                       "factor). The rest are in the table with the reason.")
if summary.get("stability"):
    stable = summary["stability"]
    columns[2].metric("Neighbours profitable", "—" if stable["share_profitable"] is None
                      else f"{stable['share_profitable']:.0%}",
                      help=f"{stable['n']} settings within ±10% of the best were re-run; their median "
                           f"{metric_words} is "
                           f"{'—' if stable['median_metric'] is None else format(stable['median_metric'], '.4g')}. "
                           "A sharp peak with poor neighbours is likely overfit.")

# The trials with each setting named in words, so the charts' axes read as the strategy does.
named = trials.rename(columns={column: "param:" + names.get(column.removeprefix("param:"), {}).get(
    "label", column.removeprefix("param:")) for column in params_columns})
named_columns = analysis.param_columns(named[named["phase"] == "search"])
if named_columns:
    ui.chart(analysis.fig_convergence(trials, direction, theme))
    view = st.radio("Explore", ["Pair heatmap", "Parallel coordinates", "One setting"], horizontal=True,
                    help="Pair heatmap: the score over two settings at once. Parallel coordinates: every trial as a "
                         "line across all settings. One setting: the score against one setting.")
    if view == "Pair heatmap" and len(named_columns) >= 2:
        left, right = st.columns(2)
        x = left.selectbox("Across", named_columns, index=0, format_func=lambda c: c.removeprefix("param:"),
                           help="The setting along the horizontal axis.")
        y = right.selectbox("Down", named_columns, index=1, format_func=lambda c: c.removeprefix("param:"),
                            help="The setting along the vertical axis.")
        ui.chart(analysis.fig_pair_heatmap(named, x, y, theme))
    elif view == "Parallel coordinates" or (view == "Pair heatmap" and len(named_columns) < 2):
        ui.chart(analysis.fig_parallel(named, theme))
    else:
        column = st.selectbox("Setting", named_columns, format_func=lambda c: c.removeprefix("param:"),
                              help="The setting to plot the score against.")
        ui.chart(analysis.fig_param_scatter(named, column, theme))
    optuna_db = job_dir / "optuna.db"
    if job["backend"] == "optuna" and optuna_db.exists() and st.checkbox(
            "Show parameter importance",
            help="Optuna's estimate of how much each setting moved the score across the trials."):
        def importance():
            import optuna
            study = optuna.load_study(study_name="job", storage=f"sqlite:///{optuna_db.resolve().as_posix()}")
            found = optuna.importance.get_param_importances(study)
            return {names.get(path, {}).get("label", path): value for path, value in found.items()}
        found = ui.run_guarded("Computing importance", importance)
        if found:
            ui.chart(analysis.fig_importance(found, theme))

# --- every trial, best first ------------------------------------------------------------------
st.subheader("Every trial")
ordered = named.sort_values("objective", ascending=direction == "minimize", na_position="last")
# The other numbers every trial records, less the one that is the objective (a Sharpe ratio search
# would list its Sharpe ratio twice).
wanted = (["trial", "objective"] + named_columns
          + [column for column in ("n_trades", "total_return", "max_drawdown", "sharpe", "profit_factor")
             if column in ordered and column != job["metric"]]
          + [column for column in (("backend", "seed") if job["kind"] == "compare" else ()) if column in ordered]
          + [column for column in ("fold",) if job["kind"] == "walk_forward" and column in ordered]
          + [column for column in ("phase", "error", "notes") if column in ordered])
table = ordered[wanted].rename(columns={column: column.removeprefix("param:") for column in named_columns})
table = table.rename(columns={"trial": "Trial", "objective": metric_label, "n_trades": "Trades",
                              "total_return": "Total return", "max_drawdown": "Max drawdown",
                              "sharpe": "Sharpe ratio", "profit_factor": "Profit factor", "backend": "Backend",
                              "seed": "Seed", "fold": "Fold", "phase": "Part", "error": "Why it is not valid",
                              "notes": "What the backtest said"})
ui.table(table, height=360, column_config={
    "Total return": st.column_config.NumberColumn(format="percent"),
    "Max drawdown": st.column_config.NumberColumn(format="percent"),
    "What the backtest said": st.column_config.TextColumn(width="large")})
st.caption(f"Best first by {metric_words}. A blank score broke a constraint (Why it is not valid says which) or "
           "could not run. What the backtest said: vectorbtX's notes on the trial's run, such as entries below the "
           "minimum size that were not placed. Part: search trials, the held-back check, and the stability re-runs.")
