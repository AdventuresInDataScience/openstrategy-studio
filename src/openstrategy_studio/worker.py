"""Entry point of a background job: python -m openstrategy_studio.worker <job_dir>.

The worker creates its own process pool (each process loads the job's data
once in its initialiser), runs the backend, and after every batch appends to
trials.parquet and rewrites progress.json. Cancel creates a `cancel` file that
is checked before every batch.

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
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import jobs, optimise


def main(job_dir):
    job_dir = Path(job_dir)
    job = jobs.read_job(job_dir)
    direction = job["direction"]
    rows = []
    progress = {"status": "running", "done": 0, "budget": job["budget"], "best_value": None,
                "best_params": None, "pid": os.getpid(), "started_at": jobs.now()}
    summary = {}
    pool = None
    # Which backend run the next trials belong to (the comparison harness runs several).
    current = {"backend": job["backend"], "seed": job["seed"], "run_trial": 0, "fold": None}

    def flush():
        # The whole table is rewritten each batch: simple, and fast at these sizes.
        # Atomically, because the Results page may be reading it at that moment.
        if rows:
            jobs.write_parquet(job_dir / "trials.parquet", pd.DataFrame(rows))
        progress["updated_at"] = jobs.now()
        jobs.write_json(job_dir / "progress.json", progress)

    def evaluate(params_list, window=None, phase="search", want_returns=False):
        """Run a batch through the pool, record every trial, return the full results."""
        if (job_dir / "cancel").exists():
            raise RuntimeError("cancelled")
        tasks = [{"params": params, "window": window, "want_returns": want_returns} for params in params_list]
        if current["backend"] == "grid_vectorised":
            # Many trials per backtest call: one contiguous chunk per worker process.
            size = max(1, -(-len(tasks) // max(1, int(job.get("workers", 1)))))
            chunks = [tasks[start:start + size] for start in range(0, len(tasks), size)]
            if pool is not None and len(chunks) > 1:
                results = [result for chunk in pool.map(optimise.evaluate_chunk, chunks) for result in chunk]
            else:
                results = [result for chunk in chunks for result in optimise.evaluate_chunk(chunk)]
        elif pool is not None and len(tasks) > 1:
            results = list(pool.map(optimise.evaluate_trial, tasks))
        else:
            results = [optimise.evaluate_trial(task) for task in tasks]
        for result in results:
            row = {key: value for key, value in result.items() if key not in ("params", "returns")}
            row.update(trial=len(rows), phase=phase, fold=current["fold"], params_json=json.dumps(result["params"]),
                       backend=current["backend"], seed=current["seed"], run_trial=current["run_trial"])
            if phase == "search":
                current["run_trial"] += 1
            for name, value in result["params"].items():
                row[f"param:{name}"] = value
            rows.append(row)
            progress["done"] += 1
            value = result["objective"]
            if phase == "search" and np.isfinite(value):
                best = progress["best_value"]
                if best is None or (value > best if direction == "maximize" else value < best):
                    progress["best_value"] = float(value)
                    progress["best_params"] = result["params"]
        flush()
        return results

    def search(space, budget, window, seed, backend=None, label=None):
        """Run one backend once; returns the best search trial of this run."""
        current.update(backend=backend or job["backend"], seed=seed, run_trial=0)
        options = dict(job.get("backend_options") or {})
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
        optimise.init_worker(job_dir)           # this process evaluates too
        if int(job.get("workers", 1)) > 1:
            pool = ProcessPoolExecutor(max_workers=int(job["workers"]), initializer=optimise.init_worker,
                                       initargs=(str(job_dir),))
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
            progress["budget"] = per_fold * len(folds) + len(folds)
            fold_rows = []
            stitched = []
            for number, fold in enumerate(folds):
                current["fold"] = number
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
            for backend in settings["backends"]:
                for seed in settings["seeds"]:
                    search(space, int(job["budget"]), train, int(seed), backend=backend, label=f"{backend}-{seed}")
            summary["compare"] = optimise.compare_summary(pd.DataFrame(rows), direction)

        else:
            best = search(space, int(job["budget"]), train, job["seed"])
            if best is not None:
                params = json.loads(best["params_json"])
                summary["best_params"] = params
                summary["best_value"] = float(best["objective"])
                if test is not None:
                    outcome = evaluate([params], test, "held_out")[0]
                    summary["held_out"] = {key: outcome.get(key) for key in
                                           ("metric", "sharpe", "total_return", "max_drawdown", "n_trades")}
                    summary["held_out"]["window"] = [str(test[0]), str(test[1])]
                if job.get("stability"):
                    # Re-evaluate a ±10% neighbourhood of the best point.
                    points = optimise.neighbourhood(space, params, seed=job["seed"])
                    results = []
                    batch = max(1, int(job.get("workers", 1)))
                    for start in range(0, len(points), batch):
                        results += evaluate(points[start:start + batch], train, "stability")
                    summary["stability"] = optimise.stability_summary(results)
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
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        summary.setdefault("n_trials", sum(1 for row in rows if row.get("phase") == "search"))
        jobs.write_json(job_dir / "summary.json", summary)
        flush()
    return progress["status"]


if __name__ == "__main__":
    # The guard matters on Windows, where pool processes re-import this module.
    sys.exit(0 if main(sys.argv[1]) in ("done", "cancelled") else 1)
