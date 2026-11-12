"""Results: optimisation results (trials, best, robustness) and saved runs compared side by side."""

import copy
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, jobs, optimise, store, ui
from openstrategy_studio.spec import get_path, is_opt, set_path, spec_hash

settings, data_dir = ui.boot("Results")
ui.font_css()
st.title("Results")
theme, glow = ui.theme_picker(settings, "results")
optimisation_tab, compare_tab = st.tabs(["Optimisation results", "Compare saved runs"])

with optimisation_tab:
    listing = jobs.list_jobs(data_dir)
    if listing.empty:
        st.write("No optimisations yet. Start one on the Optimise page.")
    else:
        labels = {row.job_id: f"{row.name} ({row.status}, {row.created_at})" for row in listing.itertuples()}
        job_id = st.selectbox("Job", list(labels), format_func=labels.get)
        job_dir = Path(listing.set_index("job_id").loc[job_id, "path"])
        job = jobs.read_job(job_dir)
        progress = jobs.read_progress(job_dir)
        summary = jobs.read_summary(job_dir)
        trials = jobs.read_trials(job_dir)
        direction = job["direction"]

        if progress["status"] == "failed":
            st.error("The job failed.")
            st.code(progress.get("error", ""), language="text")
            log = job_dir / "worker.log"
            if log.exists():
                with st.expander("Worker log"):
                    st.code(log.read_text(encoding="utf-8")[-5000:], language="text")
        elif progress["status"] in ("starting", "running"):
            st.info(f"Still running: {progress.get('done', 0)} of {progress.get('budget', 0)} trials. "
                    "Refresh to see the latest.")

        if trials.empty:
            st.stop()
        search = trials[trials["phase"] == "search"]
        if job["kind"] == "compare" and summary.get("compare"):
            st.subheader("Backends compared")
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
        best = optimise.best_trial(search, direction)
        columns = st.columns(4)
        # Always show how many configurations were tried: it is what the best is selected from.
        columns[0].metric("Configurations tried", int(summary.get("n_trials", len(search))))
        columns[1].metric("Valid trials", int(search["objective"].notna().sum()))
        columns[2].metric(f"Best {optimise.METRICS.get(job['metric'], {}).get('label', job['metric'])}",
                          "—" if best is None else f"{best['objective']:.4g}")
        if summary.get("held_out"):
            held = summary["held_out"]
            columns[3].metric("Same settings on held-back data", "—" if held.get("metric") is None else f"{held['metric']:.4g}",
                              help=f"{held['window'][0]} to {held['window'][1]}; never seen by the optimiser.")
        if summary.get("stability"):
            stable = summary["stability"]
            median = "—" if stable["median_metric"] is None else f"{stable['median_metric']:.4g}"
            share = "—" if stable["share_profitable"] is None else f"{stable['share_profitable']:.0%}"
            st.write(f"Stability: {stable['n']} neighbours within ±10% of the best have a median metric of {median}, "
                     f"and {share} of them are profitable. A sharp peak with poor neighbours is likely overfit.")

        if best is not None and job["metric"] != "selftest":
            params = jobs.trial_params(best)
            st.subheader("Best settings")
            st.json(params)
            left, right = st.columns(2)
            if left.button("Apply to the strategy",
                           help="Writes these values into the job's strategy, keeping the optimisation ranges, "
                                "and opens it on the Build page."):
                updated = copy.deepcopy(job["spec"])
                for path, value in params.items():
                    node = get_path(updated, path)
                    if is_opt(node):
                        node["value"] = value
                    else:
                        set_path(updated, path, value)
                if spec_hash(ui.current()) != spec_hash(job["spec"]):
                    st.info("The strategy open now differs from the one this job optimised; the job's version was used.")
                ui.replace_spec(updated)
                st.success("Applied. The Build page now shows these values.")

            def rerun_best():
                with open(job_dir / "frames.pkl", "rb") as handle:
                    frames = pickle.load(handle)
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

            if right.button("Re-run the best trial", help="Checks that the stored result reproduces, and computes "
                                                          "the deflated Sharpe ratio."):
                outcome = ui.run_guarded("Re-running the best trial", rerun_best)
                if outcome is not None:
                    value, dsr = outcome
                    match = abs(value - float(best["objective"])) <= 1e-9 * max(1.0, abs(value))
                    (st.success if match else st.warning)(
                        f"Re-run gives {value:.6g}; the job recorded {float(best['objective']):.6g}."
                        + ("" if match else " They differ: the data or the libraries changed since the job ran."))
                    if dsr is not None:
                        st.write(f"Deflated Sharpe ratio: **{dsr:.1%}**, the probability that the true Sharpe is above "
                                 f"zero after allowing for {int(search['objective'].notna().sum())} trials.")

        st.subheader("Trials")
        ordered = trials.sort_values("objective", ascending=direction == "minimize", na_position="last")
        ui.table(ordered.drop(columns=["params_json"], errors="ignore"), height=320)

        params_columns = analysis.param_columns(search)
        if params_columns:
            ui.chart(analysis.fig_convergence(trials, direction, theme))
            view = st.radio("Explore", ["Pair heatmap", "Parallel coordinates", "One parameter"], horizontal=True)
            if view == "Pair heatmap" and len(params_columns) >= 2:
                left, right = st.columns(2)
                x = left.selectbox("Across", params_columns, index=0, format_func=lambda c: c.removeprefix("param:"))
                y = right.selectbox("Down", params_columns, index=1, format_func=lambda c: c.removeprefix("param:"))
                ui.chart(analysis.fig_pair_heatmap(trials, x, y, theme))
            elif view == "Parallel coordinates":
                ui.chart(analysis.fig_parallel(trials, theme))
            else:
                column = st.selectbox("Parameter", params_columns, format_func=lambda c: c.removeprefix("param:"))
                ui.chart(analysis.fig_param_scatter(trials, column, theme))
            optuna_db = job_dir / "optuna.db"
            if job["backend"] == "optuna" and optuna_db.exists() and st.checkbox("Show parameter importance"):
                def importance():
                    import optuna
                    study = optuna.load_study(study_name="job", storage=f"sqlite:///{optuna_db.resolve().as_posix()}")
                    return optuna.importance.get_param_importances(study)
                found = ui.run_guarded("Computing importance", importance)
                if found:
                    ui.chart(analysis.fig_importance(found, theme))

        if summary.get("walk_forward"):
            st.subheader("Walk-forward")
            walk = summary["walk_forward"]
            ratio = walk.get("oos_to_is_ratio")
            st.write("Out-of-sample to in-sample metric ratio: " + ("—" if ratio is None else f"**{ratio:.2f}**")
                     + ". Near 1 means the in-sample result carried over.")
            ui.table(pd.DataFrame(walk["folds"]).astype(str))
            stitched = job_dir / "walk_forward_returns.parquet"
            if stitched.exists():
                ui.chart(analysis.fig_walk_forward(pd.read_parquet(stitched)["returns"], theme, glow))

with compare_tab:
    runs = store.list_runs(data_dir)
    if runs.empty:
        st.write("No saved runs yet. Save a run on the Backtest page.")
    else:
        labels = {row.run_id: f"{row.name} ({row.created_at})" for row in runs.itertuples()}
        chosen = st.multiselect("Runs", list(labels), default=list(labels)[:2], format_func=labels.get)
        if chosen:
            curves = []
            table = {}
            for run_id in chosen:
                results = analysis.load_results(runs.set_index("run_id").loc[run_id, "path"])
                curves.append((labels[run_id], results["series"]["value"]))
                table[labels[run_id]] = results["stats"]["portfolio"]["all"]
            ui.chart(analysis.fig_compare(curves, theme, glow))
            ui.table(pd.DataFrame(table).astype(str), hide_index=False)
