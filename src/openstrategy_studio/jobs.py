"""Background optimisation jobs.

Streamlit reruns the script on every interaction, so a long optimisation cannot
live in the session. A job is a folder, <data_dir>/jobs/<job_id>/:

    job.json         what to run: spec, space, backend, budget, objective, constraints, seed
    frames.pkl       the data snapshot every worker process loads once
    progress.json    rewritten by the worker as each trial finishes (done, budget, best so far, stage)
    heartbeat        touched by the worker every HEARTBEAT_SECONDS, so a dead one can be told from a slow one
    trials.parquet   one row per evaluated trial
    summary.json     best trial, held-out result, stability, walk-forward results
    best_backtest.pkl  the best trial's backtest as the results page draws it (worker.save_best_backtest)
    cancel           created by Cancel; the worker checks it before every batch
    worker.log       the worker's stdout and stderr

Results survive closing the page and restarting Streamlit, because everything
is on disk.
"""

import json
import os
import pickle
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import store


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def retry_while_locked(action, attempts=100):
    """Run `action`, retrying for up to about two seconds while Windows reports the file locked.

    ⚠️ On Windows a file the other process has open cannot be replaced, and a file
    that is being replaced cannot be opened, for a moment: both raise PermissionError.
    The page and the worker poll each other's files, so without this a page refresh
    could fail, or worse, kill a running job. Elsewhere this never retries.
    """
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02)


def write_json(path, value):
    """Write atomically (temp file, then replace), so a reader never sees half a file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    retry_while_locked(lambda: os.replace(temporary, path))


def write_parquet(path, frame):
    """Write a frame atomically, like write_json: the page reads trials while the worker writes them."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    retry_while_locked(lambda: os.replace(temporary, path))


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(retry_while_locked(lambda: path.read_text(encoding="utf-8")))


def start_job(data_dir, job, frames):
    """Write the job folder and launch `openstrategy-studio worker <job_dir>` as a detached process."""
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = Path(data_dir) / "jobs" / job_id
    job_dir.mkdir(parents=True)
    # Its # (1, 2, 3 ...), shorter to keep track of than the id, kept in the job too (the owner, 25 November 2026).
    number = store.next_job_number(data_dir)
    job = dict(job, job_id=job_id, number=number, created_at=now(), data_dir=str(data_dir))
    write_json(job_dir / "job.json", job)
    with open(job_dir / "frames.pkl", "wb") as handle:
        pickle.dump(frames, handle)
    write_json(job_dir / "progress.json", {"status": "starting", "done": 0, "budget": job["budget"],
                                          "updated_at": now()})
    options = {"stdout": None, "stderr": subprocess.STDOUT, "cwd": str(job_dir)}
    # Its own process group, so the job outlives the Streamlit session that started it.
    # ⚠️ On Windows, CREATE_NO_WINDOW, never DETACHED_PROCESS: a venv's python.exe is a launcher that
    # starts the real interpreter as a child, and a detached launcher's child opens a console window of
    # its own, the empty black window a job used to bring up (the owner, 20 November 2026). Closing that
    # window killed the job with nothing recorded. CREATE_NO_WINDOW gives the worker a console with no
    # window, which the interpreter and the job's pool processes inherit, so nothing shows; and as it is
    # not the Streamlit console, closing that one does not stop the job.
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    from .cli import self_command    # `python -m ...` normally; the bundle's own executable when frozen

    with open(job_dir / "worker.log", "w", encoding="utf-8") as log:
        options["stdout"] = log
        subprocess.Popen(self_command("worker", str(job_dir)), **options)
    store.add_job(data_dir, job_id, job.get("name", ""), job["kind"], job["backend"], job["created_at"], job_dir,
                  number=number)
    return job_dir


def read_job(job_dir):
    return read_json(Path(job_dir) / "job.json")


STARTUP_GRACE_SECONDS = 60
# The worker touches its heartbeat file this often, from a thread of its own (worker.main), and a
# job whose heartbeat is this much older, and whose process has gone, has stopped.
HEARTBEAT_SECONDS = 5
STALE_SECONDS = 30


def process_alive(pid):
    """Whether a process with this id is running, without touching it.

    ⚠️ Never os.kill(pid, 0) on Windows: there it terminates the process. OpenProcess asks only to
    read its exit code; a process another user owns refuses even that, and counts as alive.
    """
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5                     # ERROR_ACCESS_DENIED: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _seconds_since(stamp):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


def read_progress(job_dir):
    """The worker's progress, with a worker that has died shown as such rather than running for ever.

    A worker that died before it could write any shows as failed, with its log. One that dies
    later (its window closed, the machine out of memory, a hard crash) writes nothing at all, so
    it is judged from outside: its heartbeat has gone quiet AND its process has gone (a quiet
    heartbeat alone may be a machine that slept). Then it is "stopped", with what is known, and
    that is written back, so the verdict survives the process id being reused. A job from before
    the heartbeat is judged by its process alone.
    """
    job_dir = Path(job_dir)
    progress = read_json(job_dir / "progress.json", default={"status": "unknown", "done": 0, "budget": 0})
    if progress["status"] == "starting":
        log = job_dir / "worker.log"
        age = _seconds_since(progress["updated_at"])
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if text.strip() and age > 2 and ("Error" in text or "Traceback" in text):
            progress.update(status="failed", error=text[-4000:])
        elif age > STARTUP_GRACE_SECONDS:
            progress.update(status="failed", error=text[-4000:] or "The worker process never started.")
    elif progress["status"] == "running":
        beat = job_dir / "heartbeat"
        quiet = (datetime.now().timestamp() - beat.stat().st_mtime if beat.exists()
                 else _seconds_since(progress.get("updated_at") or progress["started_at"]))
        if quiet > STALE_SECONDS and not process_alive(progress.get("pid")):
            last = datetime.fromtimestamp(beat.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") \
                if beat.exists() else progress.get("updated_at", "")
            progress.update(status="stopped", stopped_at=last, error=(
                f"The job stopped without finishing, after {progress.get('done', 0)} of "
                f"{progress.get('budget', 0)} trials (last heard from at {last[:19].replace('T', ' ')} UTC). "
                "Nothing went wrong in the search itself, or it would say so: the worker process ended from "
                "outside, as when its window is closed, the computer runs out of memory (try fewer worker "
                "processes) or it restarts. Start the job again."))
            write_json(job_dir / "progress.json", progress)
    return progress


def read_summary(job_dir):
    return read_json(Path(job_dir) / "summary.json", default={})


def read_trials(job_dir):
    """Trials as a DataFrame with one param:<path> column per parameter."""
    path = Path(job_dir) / "trials.parquet"
    if not path.exists():
        return pd.DataFrame()
    return retry_while_locked(lambda: pd.read_parquet(path))


def trial_params(row):
    """The exact params dict of one trials row (types preserved through JSON)."""
    return json.loads(row["params_json"])


def cancel_job(job_dir):
    (Path(job_dir) / "cancel").touch()


def delete_job(data_dir, job_id):
    """A job off the index, and its folder (its trials, its backtest, its log) off the disk. It cannot be undone.

    A job still starting or running is refused: its worker would go on writing into a folder that is gone.
    Cancel it first. ⚠️ Only a folder inside <data_dir>/jobs is removed, whatever the index says, so a damaged
    index can never point the delete at anything else.
    """
    import shutil

    row = store.list_jobs(data_dir).set_index("job_id").loc[job_id]
    job_dir = Path(row["path"])
    if read_progress(job_dir)["status"] in ("starting", "running"):
        raise ValueError(f"{row['name']} is still running. Cancel it on the Optimise page, then delete it.")
    jobs_root = (Path(data_dir) / "jobs").resolve()
    if job_dir.exists() and job_dir.resolve().parent == jobs_root:
        # A file the page has just read can be briefly locked on Windows.
        retry_while_locked(lambda: shutil.rmtree(job_dir))
    store.delete_job(data_dir, job_id)
    _ASKED.pop(str(job_dir), None)
    return job_dir


# What a job was asked to do, by its folder: job.json never changes once written, and the Optimise page
# lists every job every few seconds while one runs.
_ASKED = {}


def _asked(job_dir):
    """{"metric", "direction", "backend_options", "searches"} from a job's job.json, read once per folder.
    "searches": a comparison's backends and methods ("optuna:tpe"), else None."""
    key = str(job_dir)
    if key not in _ASKED:
        job = read_job(job_dir) or {}
        _ASKED[key] = {"metric": job.get("metric"), "direction": job.get("direction"),
                       "backend_options": job.get("backend_options") or {},
                       "searches": (job.get("compare") or {}).get("backends")}
    return _ASKED[key]


def list_jobs(data_dir):
    """Every job with its current progress and what it searched for, newest first.

    "number" is the job's # (store.next_job_number; 0 for none, which the index's own numbering leaves no
    job without); "saved" whether it is marked to keep (store.save_job); "metric", "direction",
    "backend_options" and "searches" are its own (_asked), for the jobs table's objective and method.
    """
    rows = []
    for _, job in store.list_jobs(data_dir).iterrows():
        progress = read_progress(job["path"])
        number = job.get("number")
        rows.append({"job_id": job["job_id"], "number": 0 if pd.isna(number) else int(number), "name": job["name"],
                     "saved": bool(job.get("saved") == 1),          # NaN (no row value) is not saved
                     "kind": job["kind"], **_asked(job["path"]),
                     "backend": job["backend"], "created_at": job["created_at"], "status": progress["status"],
                     "done": progress.get("done", 0), "budget": progress.get("budget", 0),
                     "best_value": progress.get("best_value"), "path": job["path"],
                     # For the Optimise page's progress card: what it is doing, and since when.
                     "stage": progress.get("stage"), "fold": progress.get("fold"),
                     "started_at": progress.get("started_at"), "searching_since": progress.get("searching_since"),
                     "error": progress.get("error")})
    return pd.DataFrame(rows)
