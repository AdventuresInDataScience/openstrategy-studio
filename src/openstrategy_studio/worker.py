"""Entry point of a background job: python -m openstrategy_studio.worker <job_dir>.

The worker creates its own process pool (each process loads the job's data
once in its initialiser), runs the backend, rewrites progress.json as each trial
finishes (with a stage in words, for the Optimise page's progress bar) and
appends to trials.parquet after every batch. A thread touches `heartbeat` every
few seconds, so a page can tell a worker that died (its window closed, the
machine out of memory) from a slow one (jobs.read_progress). Cancel creates a
`cancel` file that is checked before every batch.

A job with a best trial ends by saving that trial's backtest (best_backtest.pkl), which the
Optimiser results page opens instead of re-running it.

Job kinds:
    optimise       one search, then the held-back test and the stability check
    walk_forward   one search per in-sample window, each tested on the next slice
    compare        every chosen backend, for several seeds, on the same space and
                   budget: the backend comparison harness (plan phase 3)
"""

import json
import os
import pickle
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import jobs, optimise


def main(job_dir):
    job_dir = Path(job_dir)
    job = jobs.read_job(job_dir)
    direction = job["direction"]
    rows = []
    # `done` counts search trials only, so it is measured against the budget (the held-back test
    # and the stability re-runs are extra). `stage` says in words what is happening now.
    progress = {"status": "running", "done": 0, "budget": job["budget"], "best_value": None,
                "best_params": None, "pid": os.getpid(), "started_at": jobs.now(),
                "stage": "Loading the data and the backtest engine", "searching_since": None}
    summary = {}
    pool = None
    # The heartbeat: touched every few seconds from a thread of its own, whatever the search is
    # doing, so a stale one means the process has gone (jobs.read_progress checks).
    beating = threading.Event()

    def heartbeat():
        while not beating.wait(jobs.HEARTBEAT_SECONDS):
            (job_dir / "heartbeat").touch()

    (job_dir / "heartbeat").touch()
    threading.Thread(target=heartbeat, name="openstrategy-heartbeat", daemon=True).start()
    # Which backend run the next trials belong to (the comparison harness runs several). "options" are the
    # backend's own (its method among them); "name" is what the trials table records it as: the backend, or
    # "backend:method" when a comparison picked the method (optimise.search_label reads both).
    current = {"backend": job["backend"], "options": dict(job.get("backend_options") or {}), "name": job["backend"],
               "seed": job["seed"], "run_trial": 0, "fold": None}

    def flush():
        # The whole table is rewritten each batch: simple, and fast at these sizes.
        # Atomically, because the Results page may be reading it at that moment.
        if rows:
            jobs.write_parquet(job_dir / "trials.parquet", pd.DataFrame(rows))
        write_progress()

    def write_progress():
        progress["updated_at"] = jobs.now()
        jobs.write_json(job_dir / "progress.json", progress)

    def finished(result, phase):
        """One trial back: count it (search trials only), keep the best so far, and say so at once."""
        if phase != "search":
            return
        progress["done"] += 1
        if progress["searching_since"] is None:
            progress["searching_since"] = jobs.now()
        progress["stage"] = "Searching"
        value = result["objective"]
        if np.isfinite(value):
            best = progress["best_value"]
            if best is None or (value > best if direction == "maximize" else value < best):
                progress["best_value"] = float(value)
                progress["best_params"] = result["params"]
        write_progress()

    def evaluate(params_list, window=None, phase="search", want_returns=False):
        """Run a batch through the pool, record every trial, return the full results."""
        if (job_dir / "cancel").exists():
            raise RuntimeError("cancelled")
        tasks = [{"params": params, "window": window, "want_returns": want_returns} for params in params_list]
        if current["backend"] == "grid" and optimise.method_of("grid", current["options"]) == "vectorised":
            # Many trials per backtest call: one contiguous chunk per worker process.
            size = max(1, -(-len(tasks) // max(1, int(job.get("workers", 1)))))
            chunks = [tasks[start:start + size] for start in range(0, len(tasks), size)]
            if pool is not None and len(chunks) > 1:
                results = [result for chunk in pool.map(optimise.evaluate_chunk, chunks) for result in chunk]
            else:
                results = [result for chunk in chunks for result in optimise.evaluate_chunk(chunk)]
            for result in results:
                finished(result, phase)
        elif pool is not None and len(tasks) > 1:
            # Each trial is counted as it comes back, not when the whole batch has, so the page's
            # progress bar moves steadily; the results keep the batch's order.
            results = [None] * len(tasks)
            futures = {pool.submit(optimise.evaluate_trial, task): number for number, task in enumerate(tasks)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                finished(results[futures[future]], phase)
        else:
            results = []
            for task in tasks:
                results.append(optimise.evaluate_trial(task))
                finished(results[-1], phase)
        for result in results:
            row = {key: value for key, value in result.items() if key not in ("params", "returns")}
            row.update(trial=len(rows), phase=phase, fold=current["fold"], params_json=json.dumps(result["params"]),
                       backend=current["name"], seed=current["seed"], run_trial=current["run_trial"])
            if phase == "search":
                current["run_trial"] += 1
            for name, value in result["params"].items():
                row[f"param:{name}"] = value
            rows.append(row)
        flush()
        return results

    def search(space, budget, window, seed, choice=None, label=None):
        """Run one backend once; returns the best search trial of this run.

        `choice`: a comparison's "backend" or "backend:method"; None for the job's own. A comparison's
        backend runs the method it names, else its own default (optimise.method_of): the job's method is
        another backend's. The grid's many-trials-per-call method is the job's own only, as a comparison
        cannot know every strategy can run it (engine.vectorisable).
        """
        backend, _, method = (choice or job["backend"]).partition(":")
        options = dict(job.get("backend_options") or {})
        if choice is not None:
            options.pop("method", None)
            if method and method != "vectorised":
                options["method"] = method
        current.update(backend=backend, options=options, name=choice or job["backend"], seed=seed, run_trial=0)
        if current["backend"] == "optuna":
            options["storage"] = f"sqlite:///{(job_dir / 'optuna.db').resolve().as_posix()}"
            options["study_name"] = label or "job"
        start = len(rows)
        extra = optimise.BACKENDS[current["backend"]]["fn"](
            space, lambda params_list: [r["objective"] for r in evaluate(params_list, window, "search")],
            budget, direction, seed, options)
        if extra.get("prob_tree") is not None:
            with open(job_dir / f"prob_tree{'' if label is None else '_' + label}.pkl", "wb") as handle:
                pickle.dump(extra["prob_tree"], handle)
        found = [row for row in rows[start:] if row["phase"] == "search" and np.isfinite(row["objective"])]
        if not found:
            return None
        pick = max if direction == "maximize" else min
        return pick(found, key=lambda row: row["objective"])

    try:
        write_progress()
        optimise.init_worker(job_dir)           # this process evaluates too
        if int(job.get("workers", 1)) > 1:
            pool = ProcessPoolExecutor(max_workers=int(job["workers"]), initializer=optimise.init_worker,
                                       initargs=(str(job_dir),))
            # The pool's processes start with the first batch, and each loads the data and the
            # backtest engine before its first trial: the slow part of a job's start.
            progress["stage"] = (f"Starting {int(job['workers'])} worker processes: each loads the backtest "
                                 "engine before its first trial")
        else:
            progress["stage"] = "Searching"
        flush()
        space = job["space"]
        index = optimise.CONTEXT["data"]["close"].index
        train = test = None
        if job.get("split"):
            train, test = optimise.split_window(index, float(job["split"]["train_fraction"]))

        if job["kind"] == "walk_forward":
            settings = job["walk_forward"]
            folds = optimise.walk_forward_folds(index, int(settings["window"]), int(settings["step"]),
                                                settings.get("mode", "rolling"))
            per_fold = max(1, job["budget"] // max(1, len(folds)))
            progress["budget"] = per_fold * len(folds)          # search trials; each fold's test is extra
            fold_rows = []
            stitched = []
            for number, fold in enumerate(folds):
                current["fold"] = number
                progress["fold"] = f"fold {number + 1} of {len(folds)}"
                best = search(space, per_fold, fold["in_sample"], job["seed"] + number, label=f"fold-{number}")
                if best is None:
                    fold_rows.append({"fold": number, "in_sample": [str(t) for t in fold["in_sample"]],
                                      "error": "no valid in-sample trial"})
                    continue
                params = json.loads(best["params_json"])
                # Signals are compiled on the full history and only the run is
                # sliced, so indicators are warm and nothing looks ahead.
                outcome = evaluate([params], fold["out_of_sample"], "out_of_sample", want_returns=True)[0]
                stitched.append(outcome["returns"])
                fold_rows.append({"fold": number, "in_sample": [str(t) for t in fold["in_sample"]],
                                  "out_of_sample": [str(t) for t in fold["out_of_sample"]], "params": params,
                                  "in_sample_metric": best.get("metric", best["objective"]),
                                  "out_of_sample_metric": outcome.get("metric", outcome["objective"])})
            if stitched:
                returns = pd.concat(stitched)
                returns = returns[~returns.index.duplicated(keep="first")]
                jobs.write_parquet(job_dir / "walk_forward_returns.parquet", returns.to_frame("returns"))
            ins = [row["in_sample_metric"] for row in fold_rows if "in_sample_metric" in row]
            outs = [row["out_of_sample_metric"] for row in fold_rows if "out_of_sample_metric" in row]
            ratio = None
            if ins and outs:
                denominator = float(np.nanmean(ins))
                if np.isfinite(denominator) and denominator != 0:
                    ratio = float(np.nanmean(outs) / denominator)
            summary["walk_forward"] = {"folds": fold_rows, "oos_to_is_ratio": ratio}

        elif job["kind"] == "compare":
            # The same space and budget for every backend and seed, so the curves compare.
            settings = job["compare"]
            progress["budget"] = int(job["budget"]) * len(settings["backends"]) * len(settings["seeds"])
            for choice in settings["backends"]:
                for seed in settings["seeds"]:
                    # A file-safe label (Optuna's study, BUTChC's tree): "optuna:tpe" has a colon.
                    search(space, int(job["budget"]), train, int(seed), choice=choice,
                           label=f"{choice.replace(':', '-')}-{seed}")
            summary["compare"] = optimise.compare_summary(pd.DataFrame(rows), direction)

        else:
            best = search(space, int(job["budget"]), train, job["seed"])
            if best is not None:
                params = json.loads(best["params_json"])
                summary["best_params"] = params
                summary["best_value"] = float(best["objective"])
                if test is not None:
                    progress["stage"] = "Testing the best settings on the held-back history"
                    write_progress()
                    outcome = evaluate([params], test, "held_out")[0]
                    # With what vectorbtX said about the run ("notes") and the constraint it broke ("error"):
                    # a held-back test with no trades is otherwise a Sharpe ratio of inf with no reason
                    # given (the owner's JMA jobs, 25 November 2026: every entry below the minimum stake).
                    summary["held_out"] = {key: outcome.get(key) for key in
                                           ("metric", "sharpe", "total_return", "max_drawdown", "n_trades", "notes",
                                            "error")}
                    summary["held_out"]["window"] = [str(test[0]), str(test[1])]
                if job.get("stability"):
                    progress["stage"] = "Re-running settings near the best (the stability check)"
                    write_progress()
                    # Re-evaluate a ±10% neighbourhood of the best point.
                    points = optimise.neighbourhood(space, params, seed=job["seed"])
                    results = []
                    batch = max(1, int(job.get("workers", 1)))
                    for start in range(0, len(points), batch):
                        results += evaluate(points[start:start + batch], train, "stability")
                    summary["stability"] = optimise.stability_summary(results)
        if job["kind"] != "walk_forward" and job["metric"] != "selftest":
            # The best trial's backtest, as the Optimiser results page draws it (whole history, in sample,
            # out of sample), made once here and kept with the job, so Show its backtest opens at once
            # instead of re-running it (the owner, 25 November 2026). The best search trial of the whole
            # job, as the page picks it (a comparison's is the best of every backend's runs).
            searched = pd.DataFrame([row for row in rows if row["phase"] == "search"])
            top = optimise.best_trial(searched, direction) if not searched.empty else None
            if top is not None:
                progress["stage"] = "Saving the best trial's backtest for the results page"
                write_progress()
                save_best_backtest(job_dir, job, json.loads(top["params_json"]), summary)
        summary["n_trials"] = sum(1 for row in rows if row["phase"] == "search")
        progress["status"] = "done"
    except RuntimeError as error:
        if str(error) != "cancelled":
            progress["status"] = "failed"
            progress["error"] = traceback.format_exc()
        else:
            progress["status"] = "cancelled"
    except Exception:   # noqa: BLE001  (record any failure so the page can show it)
        progress["status"] = "failed"
        progress["error"] = traceback.format_exc()
    finally:
        beating.set()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        summary.setdefault("n_trials", sum(1 for row in rows if row.get("phase") == "search"))
        jobs.write_json(job_dir / "summary.json", summary)
        flush()
    return progress["status"]


def save_best_backtest(job_dir, job, params, summary):
    """analysis.split_backtest of the best trial, pickled to best_backtest.pkl with its params.

    On the job's own data snapshot and its own settings, so it is the run its trials were scored on.
    A failure here is recorded in the summary and leaves the job's results whole: the results page
    then runs the backtest itself, as it does for a job from before 25 November 2026.
    """
    from . import analysis

    try:
        shown = analysis.split_backtest(job, optimise.CONTEXT["data"], params, job["settings"])
    except Exception:   # noqa: BLE001  (the search's results are complete; this is a convenience)
        summary["best_backtest_error"] = traceback.format_exc()
        return
    # Atomically, like every file the page and the worker share.
    path = job_dir / "best_backtest.pkl"
    temporary = path.with_suffix(".pkl.tmp")
    with open(temporary, "wb") as handle:
        pickle.dump({"params": params, **shown}, handle)
    jobs.retry_while_locked(lambda: os.replace(temporary, path))


if __name__ == "__main__":
    # The guard matters on Windows, where pool processes re-import this module.
    sys.exit(0 if main(sys.argv[1]) in ("done", "cancelled") else 1)
