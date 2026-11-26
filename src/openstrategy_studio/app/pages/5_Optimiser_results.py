"""Optimiser results: each optimisation job's best settings first, then its backtest or how the search went.

The page leads with the answer (the owner's call, 20 November 2026): the best settings in the
strategy's own words and the objective in sample and out of sample. Under it, one of two views.
The best trial's backtest comes first and opens on its own (the owner, 26 November 2026: it is
not optional, so it is not behind a button): its numbers by part with the held-back part marked,
a whole tear sheet for each part (25 November 2026), and the best settings over periods drawn at
random. The other view is how the search went: its charts, its robustness checks and every trial.
A toggle, not two folding boxes: the tear sheet holds expanders of its own, which Streamlit cannot
nest, and only the view shown is drawn.

The job picked can be saved, renamed or deleted beside the picker (the owner, 26 November 2026:
the list of jobs was just always there). Saved backtests are compared on the Backtest results
page (moved there on 19 November 2026, when the menu names said which results each page holds).
"""

import copy
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, jobs, optimise, store, ui
from openstrategy_studio.spec import get_path, set_path, spec_hash

settings, data_dir = ui.boot("Optimiser results")
ui.page_help("results")
ui.font_css()
st.title("Optimiser results")

# The two views under the best trial.
BACKTEST_VIEW = "The best trial's backtest"
SEARCH_VIEW = "How the search went"


def load_frames(job_dir):
    """The data the job ran on: its own snapshot, so the backtest matches the job's numbers."""
    with open(job_dir / "frames.pkl", "rb") as handle:
        return pickle.load(handle)


def shown_value(path, value):
    """A setting's value in words where it has them (crosses above, not crosses_above), a number to six figures
    (0.120255, not 0.12025478525617365: Apply writes the exact one)."""
    value = (names.get(path, {}).get("labels") or {}).get(value, value)
    return f"{value:.6g}" if isinstance(value, float) else value


def searched(variable):
    """What a variable searched, in words: 3 to 14 by 1, or 10, 20, 50."""
    if not variable:
        return ""
    if "values" in variable:
        return ", ".join(str(value) for value in variable["values"])
    return f"{variable['start']:g} to {variable['stop']:g}" + (f" by {variable['step']:g}" if variable.get("step")
                                                                else "")


def job_label(row):
    """A job in the picker: its number and name (★ when saved), what it searched for and when, and how it ended
    if not done."""
    parts = [(f"#{row.number} {row.name}" if row.number else row.name) + (" ★" if row.saved else "")]
    if row.metric:
        parts.append(optimise.objective_label(row.metric, row.direction))
    parts.append(str(row.created_at)[:16].replace("T", " "))
    if row.status != "done":
        parts.append(row.status)
    return " · ".join(parts)


listing = jobs.list_jobs(data_dir)
if listing.empty:
    st.write("No optimisations yet. Start one on the Optimise page.")
    st.stop()

labels = {row.job_id: job_label(row) for row in listing.itertuples()}
# The job picked stays picked when it is saved or renamed, which changes its label ("results:picked", set
# just before the rerun); after a delete the newest is picked. Leaving the page forgets the pick, as Streamlit
# does a widget's, so the page opens on the newest job: usually the one just started.
picked = st.session_state.pop("results:picked", None)
if picked in labels:
    st.session_state["results:job"] = picked
elif st.session_state.get("results:job") not in labels:
    st.session_state.pop("results:job", None)
job_id = st.selectbox("Job", list(labels), format_func=labels.get, key="results:job",
                      help="An optimisation started on the Optimise page: its number, name (★ when saved), objective "
                           "and start time, newest first.")
listed = listing.set_index("job_id").loc[job_id]
job_dir = Path(listed["path"])

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
min_trades = (job.get("constraints") or {}).get("min_trades") or 0
if job["kind"] == "compare":
    searches = ", ".join(optimise.search_label(item) for item in (job.get("compare") or {}).get("backends") or [])
    st.caption(f"Compared {searches} · {optimise.objective_label(job['metric'], direction)}")
else:
    st.caption(f"{optimise.search_label(job['backend'], job.get('backend_options'))} · "
               f"{optimise.objective_label(job['metric'], direction)}")

# --- save, rename or delete the job picked -------------------------------------------------
# Jobs are recorded as they start; saving marks the ones to keep, so the Optimise page can delete the rest in
# one go. Renaming changes the name in every list (the index's; job.json keeps the one it started with).
actions = st.container(horizontal=True, gap="small")
running_now = listed["status"] in ("starting", "running")
if listed["saved"]:
    if actions.button("Unsave", icon=":material/bookmark_remove:",
                      help="Takes off the saved mark (★), so deleting the jobs not saved, on the Optimise page, "
                           "deletes this one too."):
        store.save_job(data_dir, job_id, False)
        st.session_state["results:picked"] = job_id
        st.rerun()
elif actions.button("Save", icon=":material/bookmark_add:",
                    help="Marks this job as one to keep (★ in the lists). Deleting the jobs not saved, on the "
                         "Optimise page, leaves it alone. Every job is on disk whether saved or not."):
    store.save_job(data_dir, job_id, True)
    st.session_state["results:picked"] = job_id
    st.rerun()
with actions.popover("Rename", icon=":material/edit:", help="Gives the job another name in every list."):
    new_name = st.text_input("New name", value=listed["name"], key=f"rename:{job_id}",
                             help="What the job is called on this page and on the Optimise page. Its number stays.")
    if st.button("Rename", key=f"rename-button:{job_id}", type="primary",
                 disabled=not new_name.strip() or new_name.strip() == listed["name"]):
        store.rename_job(data_dir, job_id, new_name.strip())
        st.session_state["results:picked"] = job_id
        st.rerun()
with actions.popover("Delete", icon=":material/delete:", disabled=running_now,
                     help="Cancel the job on the Optimise page first: it is still running." if running_now else
                          "Deletes the job, with its trials, its backtest and its log, once you confirm."):
    number_text = f"#{listed['number']} " if listed["number"] else ""
    st.write(f"Delete {number_text}{listed['name']}, with its trials, its backtest and its log? This cannot be undone.")
    if st.button("Delete for good", key=f"delete:{job_id}", type="primary", icon=":material/delete_forever:"):
        if ui.run_guarded("Deleting the job", lambda: jobs.delete_job(data_dir, job_id)) is not None:
            st.session_state.pop("best_backtest", None)
            st.rerun()

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


def said_most(notes):
    """(what vectorbtX said most often, in how many runs, its words in one of them), counting each kind of note
    once per run whatever its numbers; None when nothing was said."""
    counts, examples = {}, {}
    for note in notes:
        seen = set()
        for line in str(note).split("\n"):
            kind = re.sub(r"[\d.,]+", "N", line)
            if kind not in seen:
                seen.add(kind)
                counts[kind] = counts.get(kind, 0) + 1
                examples.setdefault(kind, line)
    if not counts:
        return None
    commonest = max(counts, key=counts.get)
    return commonest, counts[commonest], examples[commonest]


# --- the answer first: the best settings, in and out of sample ------------------------------
best = optimise.best_trial(search, direction) if job["kind"] != "walk_forward" else None
if best is None and job["kind"] != "walk_forward" and not search.empty and progress["status"] == "done":
    # Nothing to lead with: say why, from the trials' own reasons ("only 4 trades" and the like,
    # grouped whatever the number), rather than leave an empty page.
    reasons = search["error"].dropna().astype(str).str.replace(r"[\d.,]+", "N", regex=True).value_counts()
    top = f" Most often: {reasons.index[0]} ({reasons.iloc[0]} of {len(search)} trials)." if len(reasons) else ""
    most = (f" The most trades any trial made was {int(search['n_trades'].max()):,}."
            if "n_trades" in search and search["n_trades"].notna().any() else "")
    if not most and "entries" in search and search["entries"].notna().any():
        # Every trial ruled out from its signals (optimise.screen), so none was run and none has a trade count.
        most = (f" The most entry signals any trial had was {int(search['entries'].max()):,}, and every trade opens "
                "on one.")
    # Why, in the backtests' own words: what vectorbtX said most often about the trials' runs
    # (entries below the minimum size, cut to the leverage), with one trial's words as the example and
    # the Studio's hint for it. Jobs from before 20 November 2026 kept no notes.
    said, hint = "", None
    found = said_most(search["notes"].dropna().astype(str) if "notes" in search else [])
    if found:
        _, count, example = found
        said = f" The backtests said, in {count:,} of {len(search):,} trials, for example: \"{example}\""
        hint = analysis.warning_hints([example])[0][1]
    st.warning(f"No trial was valid: every one broke a constraint or could not run, so there are no best settings.{top}"
               f"{most}{said} Loosen the constraints or widen the ranges on the Optimise page, or check the "
               "strategy's costs and sizes on the Build page; the table at the foot of this page gives each trial's "
               "reason.")
    if hint:
        st.caption(hint)
# Trials ruled out from their signals, with no backtest (optimise.screen; jobs from 26 November 2026 on).
screened = int(search["screened"].fillna(False).astype(bool).sum()) if "screened" in search else 0
if screened:
    st.caption(f"{screened:,} of {len(search):,} trials were ruled out from their signals alone, with no backtest: "
               f"each had fewer entry signals than the {int(min_trades):,} trades required, and every trade opens "
               "on one. They count against the budget; a warm start with a larger budget, or a lower minimum, "
               "gives the search more to work with.")
# Only the search can be shown when there is no best trial (a walk-forward job, or none valid).
view = SEARCH_VIEW
if best is not None and job["metric"] != "selftest":
    params = jobs.trial_params(best)
    held = summary.get("held_out") or {}
    # A job scored across folds (optimise.score_folds): its best trial's score is its folds combined, so the
    # in-sample numbers beside the held-back test's are its run over the whole searched history (the worker's
    # "in_sample"), one run each side; the folds have a row of their own below.
    whole_in = summary.get("in_sample")
    folded = isinstance(best.get("folds_json"), str)
    # The raw score: the objective is the same number for a valid trial (older jobs record only it).
    in_sample = float(whole_in["metric"]) if whole_in else float(best.get("metric", best["objective"]))
    in_trades = whole_in["n_trades"] if whole_in else best["n_trades"]
    in_return = whole_in["total_return"] if whole_in else best["total_return"]
    with ui.card("best-trial", "entry"):
        ui.card_header("best-trial", "Best trial", meta=f"the best {metric_words} of "
                       f"{int(summary.get('n_trials', len(search))):,} tried, on the part of the history the "
                       "optimiser searched",
                       help="In sample is the history the optimiser searched, and the best trial is the best "
                            "score there. Out of sample is the history held back from it, run once with the best "
                            "settings afterwards, from the starting cash as each search trial was: the honest "
                            "test. A big drop from one to the other means the settings fit the past rather than "
                            "found an edge.")
        columns = st.columns(4)
        columns[0].metric(f"{metric_label}, in sample", analysis.shown_number(in_sample, "ratio"),
                          help="The best trial's score on the history the optimiser searched.")
        if held.get("metric") is not None:
            out_of_sample = float(held["metric"])
            # ⚠️ A held-back test with no trades has a Sharpe ratio of inf from vectorbt (a flat account), which
            # is no score at all: "—", and no change from in sample (analysis.shown_number).
            changed = np.isfinite(out_of_sample) and bool(held.get("n_trades"))
            columns[1].metric(f"{metric_label}, out of sample",
                              analysis.shown_number(out_of_sample if held.get("n_trades") else np.nan, "ratio"),
                              delta=f"{out_of_sample - in_sample:+.4g} from in sample" if changed else None,
                              delta_color="normal" if direction == "maximize" else "inverse",
                              help=f"The same settings on the held-back history, {held['window'][0][:10]} to "
                                   f"{held['window'][1][:10]}, which the optimiser never saw, from the starting "
                                   "cash.")
        else:
            columns[1].metric(f"{metric_label}, out of sample", "—",
                              help="Nothing was held back from this job: set a share to hold back on the Optimise "
                                   "page to test the best settings on history the optimiser never saw.")
        columns[2].metric("Trades, in / out of sample",
                          f"{int(in_trades):,} / " + ("—" if held.get("n_trades") is None
                                                      else f"{int(held['n_trades']):,}"),
                          help="How many trades the best settings made in each part. A handful is not evidence.")
        columns[3].metric("Total return, in / out of sample",
                          f"{float(in_return):.1%} / " + ("—" if held.get("total_return") is None
                                                          else f"{float(held['total_return']):.1%}"),
                          help="Each part is run from the starting cash on its own, as the optimiser scored it.")
        if folded:
            # The folds: what the search scored, each fold's own number, and how far choosing the best was luck.
            fold = json.loads(best["folds_json"])
            settings_of_folds = summary.get("folds") or job.get("folds") or {}
            how = settings_of_folds.get("summary", "mean_less_sd")
            scale = optimise.FOLD_SCALES.get(job["metric"], {})
            shown_as = scale.get("format", "ratio")
            combined_label = scale.get("label", metric_label)
            row = st.columns(4)
            row[0].metric(f"{combined_label}, across {len(fold['metrics'])} folds",
                          analysis.shown_number(float(best.get("metric", np.nan)), shown_as),
                          help=f"What the search scored: the folds combined by \"{optimise.FOLD_SUMMARIES[how]['label']}\". "
                               f"{optimise.FOLD_SUMMARIES[how]['help']} Each fold is a run of its own from the "
                               "starting cash." + (" A fold's return counts as its growth a year." if
                                                   job["metric"] == "total_return" else ""))
            overfit = optimise.overfitting_probability(search, how, direction)
            if overfit is not None:
                row[1].metric("Probability of backtest overfitting", f"{overfit['probability']:.0%}",
                              help=f"Over {overfit['splits']} ways of splitting the folds in two: how often the best of "
                                   f"the {overfit['trials']:,} settings tried on one half ranked below the median on "
                                   "the other half (Bailey, Borwein, López de Prado and Zhu). Near 0, the best on some "
                                   "history stays good on the rest; 50% or more, picking the best was no better than "
                                   "picking at random. From the trials' fold scores: no backtest is run.")
            else:
                row[1].metric("Probability of backtest overfitting", "—",
                              help="Needs an even number of folds, at least 4, and at least 10 valid settings tried.")
            each = ", ".join(analysis.shown_number(value, "pct" if job["metric"] == "total_return" else "ratio")
                             for value in fold["metrics"])
            st.caption(f"Each fold's {metric_words}: {each}. Trades in each: "
                       f"{', '.join(f'{count:,}' for count in fold['trades'])}. In sample is the best settings over the "
                       "whole searched history in one run, beside the held-back test's one run.")
        # A held-back test with too few trades says nothing, and why it made so few is the useful part (the
        # owner, 25 November 2026: no trade out of sample, and a Sharpe ratio of inf, looked like a bug). It is
        # run from the starting cash, as every trial is; a risk-sized account that grew in the searched years
        # trades there in the whole-history chart, but a new one may not reach the minimum size.
        if held.get("n_trades") is not None and int(held["n_trades"]) < max(1, int(min_trades)):
            count = int(held["n_trades"])
            made = "made no trades" if count == 0 else (f"made only {count:,} trade{'' if count == 1 else 's'}, under "
                                                        f"the {int(min_trades):,} the search required")
            notes = [line for line in str(held.get("notes") or "").split("\n") if line]
            st.warning(f"The held-back test {made}, so its numbers say little about these settings. It runs from "
                       f"the starting cash ({job['spec']['account']['init_cash']:,.0f}), as each search trial did."
                       + (f" The backtest said: \"{notes[0]}\"" if notes else
                          " The held-back part's tear sheet, below, says what the backtest said about it.")
                       + " The whole-history backtest below may still trade there: its account had grown or shrunk "
                         "by then (its column \"Held back, inside the whole run\").", icon=":material/warning:")
            hint = analysis.warning_hints(notes[:1])[0][1] if notes else None
            if hint:
                st.caption(hint)

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

    # --- one of two views: the best trial's backtest, or how the search went ----------------------
    views = [BACKTEST_VIEW, SEARCH_VIEW]
    view = st.segmented_control("Show", views, key=ui.kept("results_view", BACKTEST_VIEW, views),
                                help="The best trial's backtest: its numbers by part, a whole tear sheet for each, "
                                     "and the best settings over random periods. How the search went: the search's "
                                     "charts, its robustness checks and every trial.") or BACKTEST_VIEW

    def best_backtest():
        """The job's own record (worker.save_best_backtest) when it is of these settings and of this version of
        the page's backtest (one from before 25 November 2026 has no tear sheet per part; a running job may have
        found a better trial since). Otherwise it is run now, and a finished job keeps it, so it is run once."""
        saved_path = job_dir / "best_backtest.pkl"
        if saved_path.exists():
            with open(saved_path, "rb") as handle:
                saved = pickle.load(handle)
            if saved.get("version") == analysis.SPLIT_BACKTEST_VERSION and saved["params"] == params:
                return {key: value for key, value in saved.items() if key != "params"}
        made = analysis.split_backtest(job, load_frames(job_dir), params, settings)
        if progress["status"] == "done":
            # Atomically, as the worker writes it: its worker has gone, but another tab may be reading it.
            temporary = saved_path.with_suffix(".pkl.tmp")
            with open(temporary, "wb") as handle:
                pickle.dump({"params": params, **made}, handle)
            jobs.retry_while_locked(lambda: os.replace(temporary, saved_path))
        return made

    # The best trial's backtest opens on its own and is kept while this job and its best stay the same,
    # so the page's other widgets do not open it again. In a card (not an expander: the tear sheet holds
    # expanders of its own, which Streamlit cannot nest).
    backtest = None
    if view == BACKTEST_VIEW:
        wanted = [job_id, json.dumps(params, sort_keys=True, default=str)]
        backtest = st.session_state.get("best_backtest")
        if backtest is None or backtest.get("key") != wanted:
            shown = ui.run_guarded("Opening the best trial's backtest", best_backtest)
            backtest = None if shown is None else {"key": wanted, **shown}
            if backtest is not None:
                st.session_state["best_backtest"] = backtest
    if backtest is not None:
        with ui.card("best-backtest", "entry"):
            ui.card_header("best-backtest", "The best trial's backtest", meta="its numbers by part, then the tear "
                           "sheet of each")
            # The row names as a column of their own, wide enough to read whole.
            ui.table(backtest["table"].reset_index(names=""), column_config={
                "": st.column_config.TextColumn(width="medium")})
            if backtest["split"] is not None:
                st.caption("In sample and out of sample are each run from the starting cash, as the optimiser "
                           "scored them; the whole period is one account from the first bar to the last. "
                           "\"Held back, inside the whole run\" is that account's held-back part: it starts "
                           "from what the account was worth there, so a risk-sized account that grew trades "
                           "bigger than a new one could, and may trade where a new one cannot. A blank is a "
                           "number the run does not have, such as a Sharpe ratio with no trades.")
            folds_shown = backtest.get("folds")
            if folds_shown is not None:
                # The spread one in-sample number hides (the owner, 26 November 2026): the best settings on each
                # fold, as the search scored them, beside the searched history's one run (dashed).
                rows = folds_shown["rows"]
                st.markdown(f"**The {len(rows)} folds**",
                            help="The best settings on each fold of the searched history, each a run of its own from "
                                 "the starting cash, as the search scored them: a box per number, each fold a point, "
                                 "and the searched history in one run as the dashed line. A narrow spread is settings "
                                 "that worked throughout; a wide one, settings that worked in some stretches only.")
                numbers = [column for column, _, _ in analysis.PERIOD_NUMBERS]
                objective = None
                if job["metric"] not in numbers:
                    objective = ("metric", metric_label, "pct" if job["metric"] == "total_return" else "ratio")
                idle = int((rows["n_trades"].fillna(0) == 0).sum())
                if idle:
                    st.caption(f"{idle} of {len(rows)} folds made no trades, so they are left out of the boxes.")
                if idle < len(rows):
                    ui.chart(analysis.fig_periods(rows, folds_shown["reference"], theme, objective,
                                                  title=f"The best settings on each of the {len(rows)} folds",
                                                  reference_label="Searched history", item="Fold"))
            parts = list(backtest["windows"])
            part = st.segmented_control("Tear sheet of", parts, key=ui.kept("best_backtest_part",
                                                                          analysis.WHOLE_PERIOD, parts),
                                        help="Every chart and metric of one part's own run: the whole period "
                                             "(the held-back part marked on the charts along the dates), or "
                                             "the searched or held-back history on its own, from the starting "
                                             "cash.") or analysis.WHOLE_PERIOD
            ui.tear_sheet(backtest["windows"][part], "best", f"{job_id}:{part}", theme=theme, glow=glow,
                          split=backtest["split"] if part == analysis.WHOLE_PERIOD else None)

    if view == BACKTEST_VIEW:
        # --- the best settings over random periods -------------------------------------------------
        # The owner (25 November 2026): a drawdown of 22% over the run is one number; the same settings over
        # many periods drawn at random give its spread. Each period is run from the starting cash, as a trial
        # is, and kept with the job (random_periods.pkl, by its settings), so it is run once.
        with ui.card("random-periods", "risk"):
            ui.card_header("random-periods", "Over random periods", meta="the best settings on periods drawn at random",
                           fold=("results:random-periods", False, "random-periods:open", "the random periods"),
                           help="Runs the best settings on many periods of the history drawn at random, each from the "
                                "starting cash, and shows how each number spreads over them. A narrow spread near the "
                                "whole history's value is a strategy that does not depend on when it started.")
            if ui.is_open("results:random-periods", False):
                places = ["whole"] + (["searched", "held_back"] if job.get("split") else [])
                row = st.columns(3)
                count = int(row[0].number_input("Periods", min_value=5, max_value=200, step=5,
                                                key=ui.kept("periods_count", 20),
                                                help="How many periods to draw. Each is one backtest."))
                length = row[1].slider("Each period's length", 10, 90, format="%d%%", key=ui.kept("periods_length", 40),
                                       help="How long each period is, as a share of the history they are drawn from. "
                                            "Shorter periods give more varied numbers.") / 100
                within = row[2].selectbox("Drawn from", places, key=ui.kept("periods_within", "whole", places),
                                          format_func={"whole": "The whole history", "searched": "The searched history",
                                                       "held_back": "The held-back history"}.get,
                                          help="Where the periods may fall. The held-back history is the one the "
                                               "optimiser never saw.")
                saved_path = job_dir / "random_periods.pkl"
                wanted = json.dumps([params, count, length, within, int(job.get("seed") or 0)], sort_keys=True,
                                    default=str)
                stored = {}
                if saved_path.exists():
                    with open(saved_path, "rb") as handle:
                        stored = pickle.load(handle)
                if st.button("Run the periods", icon=":material/casino:", disabled=wanted in stored,
                             help="Runs the best settings on each period. Periods run once are kept with the job, so "
                                  "the same choices show at once next time."):
                    bar = st.progress(0.0, text=f"0 of {count} periods")

                    def run_periods():
                        return analysis.random_periods(job, load_frames(job_dir), params, settings, count=count,
                                                       length=length, within=within, seed=int(job.get("seed") or 0),
                                                       progress=lambda done, total: bar.progress(
                                                           done / total, text=f"{done} of {total} periods"))

                    outcome = ui.run_guarded("Running the periods", run_periods)
                    bar.empty()
                    if outcome is not None:
                        stored[wanted] = outcome
                        with open(saved_path, "wb") as handle:
                            pickle.dump(stored, handle)
                outcome = stored.get(wanted)
                if outcome is not None:
                    periods, whole = outcome["periods"], outcome["whole"]
                    spans = pd.to_datetime(periods["to"]) - pd.to_datetime(periods["from"])
                    traded = periods[periods["n_trades"].fillna(0) > 0] if "n_trades" in periods else periods.iloc[0:0]
                    st.caption(f"{len(periods)} periods of {outcome['bars']:,} bars each (about "
                               f"{spans.mean().days / 365.25:.1f} years), each from the starting cash.")
                    idle = len(periods) - len(traded)
                    if idle:
                        # Why a period never traded, in the backtest's words (the commonest note among them).
                        found = said_most(periods.loc[~periods.index.isin(traded.index), "notes"].dropna()
                                          if "notes" in periods else [])
                        st.warning(f"{idle} of {len(periods)} periods made no trades, so they are left out of the "
                                   "boxes."
                                   + (f" The backtest said, in {found[1]} of them: \"{found[2]}\"" if found else ""),
                                   icon=":material/warning:")
                    if not traded.empty:
                        # The objective has a panel of its own unless it is one of the usual numbers (the Sharpe
                        # ratio); its value is each period's "metric", as a trial records it.
                        numbers = list(analysis.PERIOD_NUMBERS)
                        objective = None
                        if job["metric"] not in [column for column, _, _ in numbers]:
                            objective = ("metric", metric_label, "pct" if job["metric"] == "total_return" else "ratio")
                            numbers.insert(0, objective)
                        ui.chart(analysis.fig_periods(periods, whole, theme, objective))
                        # The spread in numbers, beside the whole history's value.
                        summary_rows = []
                        for column, label, how in numbers:
                            values = traded[column].astype(float)
                            values = values[np.isfinite(values)]
                            quantiles = values.quantile([0, 0.25, 0.5, 0.75, 1]) if not values.empty else None
                            item = {"Number": label, "Whole history": analysis.shown_number(whole.get(column), how)}
                            for name, q in zip(("Lowest", "25%", "Median", "75%", "Highest"), (0, 0.25, 0.5, 0.75, 1)):
                                item[name] = analysis.shown_number(None if quantiles is None else quantiles[q], how)
                            summary_rows.append(item)
                        ui.table(pd.DataFrame(summary_rows))
                        st.caption("The whole history's own value is the dashed line. Its drawdown is over a longer "
                                   "time than any period's, so it is usually the deeper; the annualised return and "
                                   "the Sharpe ratio compare across lengths.")
                    with st.expander("Every period"):
                        table = periods.rename(columns={"period": "Period", "from": "From", "to": "To",
                                                        "n_trades": "Trades", "annualised_return": "Annualised return",
                                                        "max_drawdown": "Max drawdown", "sharpe": "Sharpe ratio",
                                                        "error": "Why it is not valid",
                                                        "notes": "What the backtest said"})
                        wanted_columns = [column for column in ("Period", "From", "To", "Trades", "Annualised return",
                                                                "Max drawdown", "Sharpe ratio", "Why it is not valid",
                                                                "What the backtest said") if column in table]
                        table = table[wanted_columns]
                        if "Sharpe ratio" in table:
                            # A period with no trades has a Sharpe ratio of inf from vectorbt: blank, it has none.
                            table["Sharpe ratio"] = table["Sharpe ratio"].where(table["Trades"].fillna(0) > 0)
                        table["From"] = table["From"].astype(str).str[:10]
                        table["To"] = table["To"].astype(str).str[:10]
                        ui.table(table, column_config={
                            "Annualised return": st.column_config.NumberColumn(format="percent"),
                            "Max drawdown": st.column_config.NumberColumn(format="percent"),
                            "What the backtest said": st.column_config.TextColumn(width="large")})

if view == SEARCH_VIEW:
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
                log = st.checkbox("Log scale", key=ui.kept("walk_forward:log", False), help=ui.LOG_SCALE_HELP)
                figure = analysis.fig_walk_forward(pd.read_parquet(stitched)["returns"], theme, glow)
                ui.chart(analysis.log_scale(figure) if log else figure)

    # --- how the search went ----------------------------------------------------------------------
    st.subheader("How the search went")
    if job["kind"] == "compare" and summary.get("compare"):
        compare = summary["compare"]
        st.write("Each search ran several times (different seeds) with the same budget. "
                 + ("" if compare["target"] is None else
                    f"The target is 95% of the way to the best value any run found ({compare['target']:.4g})."))
        rows = []
        for backend_name, item in compare["backends"].items():
            rows.append({"Search": optimise.search_label(backend_name),
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
                               f"{'—' if stable['median_metric'] is None else format(stable['median_metric'], '.4g')}"
                               ". "
                               "A sharp peak with poor neighbours is likely overfit.")

    # The trials with each setting named in words, so the charts' axes read as the strategy does.
    named = trials.rename(columns={column: "param:" + names.get(column.removeprefix("param:"), {}).get(
        "label", column.removeprefix("param:")) for column in params_columns})
    named_columns = analysis.param_columns(named[named["phase"] == "search"])
    if named_columns:
        ui.chart(analysis.fig_convergence(trials, direction, theme))
        view = st.radio("Explore", ["Pair heatmap", "Parallel coordinates", "One setting"], horizontal=True,
                        help="Pair heatmap: the score over two settings at once. Parallel coordinates: every trial "
                             "as a "
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
                # ⚠️ Only the valid trials: an invalid one is a complete trial told a penalty (optimise.run_optuna,
                # from 26 November 2026), and Optuna's importance reads every complete trial, so the penalties
                # would make it say which settings decide validity rather than the score.
                valid = [trial for trial in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
                         if not any(value > 0 for value in trial.constraints.values())]
                scored = optuna.create_study(direction=direction)
                scored.add_trials(valid)
                found = optuna.importance.get_param_importances(scored)
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
    score_label = metric_label
    if "folds_json" in ordered and ordered["folds_json"].notna().any():
        # Across folds, a search trial's score is its folds combined: shown on the metric's own scale (a return a
        # year, not its log growth: optimise.fold_value), in the order the search ranked it.
        how = (summary.get("folds") or job.get("folds") or {}).get("summary", "mean_less_sd")
        score_label = (f"{optimise.FOLD_SCALES.get(job['metric'], {}).get('label', metric_label)}, "
                       f"{optimise.FOLD_SUMMARIES[how]['label'].lower()} of the folds")
        table["objective"] = ordered["metric"].where(np.isfinite(ordered["objective"].astype(float)))
    if "sharpe" in table and "n_trades" in table:
        # ⚠️ vectorbt's Sharpe ratio of a run with no trades is inf (a flat account): blank, as it has none.
        table["sharpe"] = table["sharpe"].where(table["n_trades"].fillna(0) > 0)
    if "backend" in table:
        table["backend"] = table["backend"].map(optimise.search_label)
    table = table.rename(columns={"trial": "Trial", "objective": score_label, "n_trades": "Trades",
                                  "total_return": "Total return", "max_drawdown": "Max drawdown",
                                  "sharpe": "Sharpe ratio", "profit_factor": "Profit factor", "backend": "Search",
                                  "seed": "Seed", "fold": "Fold", "phase": "Part", "error": "Why it is not valid",
                                  "notes": "What the backtest said"})
    ui.table(table, height=360, column_config={
        "Total return": st.column_config.NumberColumn(format="percent"),
        "Max drawdown": st.column_config.NumberColumn(format="percent"),
        "What the backtest said": st.column_config.TextColumn(width="large")})
    st.caption(f"Best first by {metric_words}. A blank score broke a constraint (Why it is not valid says which) or "
               "could not run. What the backtest said: vectorbtX's notes on the trial's run, such as entries below the "
               "minimum size that were not placed. Part: search trials, the held-back check, and the stability "
               "re-runs." + (" Across folds, a search or stability trial's score is its folds combined, and its "
                             "other numbers are the folds' together (trades summed, returns compounded, the worst "
                             "fold's drawdown); the held-back check and the in-sample run are one run each."
                             if score_label != metric_label else ""))
