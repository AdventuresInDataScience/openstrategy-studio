"""Background optimisation jobs.

Streamlit reruns the script on every interaction, so a long optimisation cannot
live in the session. A job is a folder, <data_dir>/jobs/<job_id>/:

    job.json         what to run: spec, space, backend, budget, objective, constraints, seed
    frames.pkl       the data snapshot every worker process loads once
    progress.json    rewritten by the worker after every batch (done, budget, best so far)
    trials.parquet   one row per evaluated trial
    summary.json     best trial, held-out result, stability, walk-forward results
    cancel           created by Cancel; the worker checks it before every batch
    worker.log       the worker's stdout and stderr

Results survive closing the page and restarting Streamlit, because everything
is on disk.
"""

import json
import os
import pickle
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import store


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    """Write atomically (temp file, then replace), so a reader never sees half a file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def start_job(data_dir, job, frames):
    """Write the job folder and launch `openstrategy-studio worker <job_dir>` as a detached process."""
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = Path(data_dir) / "jobs" / job_id
    job_dir.mkdir(parents=True)
    job = dict(job, job_id=job_id, created_at=now(), data_dir=str(data_dir))
    write_json(job_dir / "job.json", job)
    with open(job_dir / "frames.pkl", "wb") as handle:
        pickle.dump(frames, handle)
    write_json(job_dir / "progress.json", {"status": "starting", "done": 0, "budget": job["budget"],
                                          "updated_at": now()})
    options = {"stdout": None, "stderr": subprocess.STDOUT, "cwd": str(job_dir)}
    # Detach, so the job outlives the Streamlit session that started it.
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        options["start_new_session"] = True
    from .cli import self_command    # `python -m ...` normally; the bundle's own executable when frozen

    with open(job_dir / "worker.log", "w", encoding="utf-8") as log:
        options["stdout"] = log
        subprocess.Popen(self_command("worker", str(job_dir)), **options)
    store.add_job(data_dir, job_id, job.get("name", ""), job["kind"], job["backend"], job["created_at"], job_dir)
    return job_dir


def read_job(job_dir):
    return read_json(Path(job_dir) / "job.json")


STARTUP_GRACE_SECONDS = 60


def read_progress(job_dir):
    """The worker's progress. A worker that died before it could write any shows as failed, with its log."""
    job_dir = Path(job_dir)
    progress = read_json(job_dir / "progress.json", default={"status": "unknown", "done": 0, "budget": 0})
    if progress["status"] == "starting":
        log = job_dir / "worker.log"
        age = datetime.now(timezone.utc) - datetime.fromisoformat(progress["updated_at"])
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if text.strip() and age.total_seconds() > 2 and ("Error" in text or "Traceback" in text):
            progress.update(status="failed", error=text[-4000:])
        elif age.total_seconds() > STARTUP_GRACE_SECONDS:
            progress.update(status="failed", error=text[-4000:] or "The worker process never started.")
    return progress


def read_summary(job_dir):
    return read_json(Path(job_dir) / "summary.json", default={})


def read_trials(job_dir):
    """Trials as a DataFrame with one param:<path> column per parameter."""
    path = Path(job_dir) / "trials.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def trial_params(row):
    """The exact params dict of one trials row (types preserved through JSON)."""
    return json.loads(row["params_json"])


def cancel_job(job_dir):
    (Path(job_dir) / "cancel").touch()


def list_jobs(data_dir):
    """Every job with its current progress, newest first."""
    rows = []
    for _, job in store.list_jobs(data_dir).iterrows():
        progress = read_progress(job["path"])
        rows.append({"job_id": job["job_id"], "name": job["name"], "kind": job["kind"],
                     "backend": job["backend"], "created_at": job["created_at"], "status": progress["status"],
                     "done": progress.get("done", 0), "budget": progress.get("budget", 0),
                     "best_value": progress.get("best_value"), "path": job["path"]})
    return pd.DataFrame(rows)
