"""Test 13: the job runner. Uses the self-test objective, so no backtest library is needed."""

import time

import pytest

from openstrategy_studio import config, jobs, store

from conftest import make_ohlcv

SPACE = {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 1, "max": 50, "int": True}, "c": {"values": ["x", "y", "z"]}}


def selftest_job(budget, delay=0.0, workers=1, batch=1):
    return {"name": "test", "kind": "optimise", "spec": {}, "space": SPACE, "backend": "random",
            "backend_options": {"batch": batch}, "budget": budget, "metric": "selftest", "direction": "maximize",
            "seed": 0, "workers": workers, "settings": config.load_settings(), "constraints": {}, "split": None,
            "walk_forward": None, "stability": False, "selftest_delay": delay}


def wait_for(job_dir, statuses, timeout=120, minimum_done=0):
    deadline = time.time() + timeout
    progress = jobs.read_progress(job_dir)
    while time.time() < deadline:
        progress = jobs.read_progress(job_dir)
        if progress["status"] in statuses and progress.get("done", 0) >= minimum_done:
            return progress
        time.sleep(0.2)
    raise AssertionError(f"the job did not reach {statuses}: {progress}")


@pytest.fixture
def home(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path / "home"))
    return tmp_path / "data"


def test_job_runs_to_completion(home):
    job_dir = jobs.start_job(home, selftest_job(5), make_ohlcv(20))
    progress = wait_for(job_dir, {"done", "failed", "cancelled"})
    assert progress["status"] == "done", progress.get("error")
    assert progress["done"] == 5
    trials = jobs.read_trials(job_dir)
    assert len(trials) == 5 and {"param:a", "param:b", "param:c", "objective"} <= set(trials.columns)
    assert jobs.read_summary(job_dir)["best_params"] == progress["best_params"]
    assert jobs.list_jobs(home)["status"].tolist() == ["done"]


def test_progress_advances_and_cancel_stops_the_worker(home):
    job_dir = jobs.start_job(home, selftest_job(1000, delay=0.05), make_ohlcv(20))
    first = wait_for(job_dir, {"running"}, minimum_done=1)
    later = wait_for(job_dir, {"running"}, minimum_done=first["done"] + 3)
    assert later["done"] > first["done"]
    jobs.cancel_job(job_dir)
    progress = wait_for(job_dir, {"cancelled", "done", "failed"})
    assert progress["status"] == "cancelled"
    assert progress["done"] < 1000
    assert len(jobs.read_trials(job_dir)) == progress["done"]     # trials.parquet readable after exit


def test_job_with_a_process_pool(home):
    job_dir = jobs.start_job(home, selftest_job(16, workers=2, batch=4), make_ohlcv(20))
    progress = wait_for(job_dir, {"done", "failed", "cancelled"})
    assert progress["status"] == "done", progress.get("error")
    assert len(jobs.read_trials(job_dir)) == 16


def test_backend_comparison_job(home):
    """A comparison may name each search's method ("scipy:direct"), recorded as named; a plain backend runs its
    default, whatever method the job's own options hold (Optuna's TPE means nothing to SciPy)."""
    job = selftest_job(12, batch=4)
    job["backend_options"]["method"] = "tpe"
    job.update(kind="compare", compare={"backends": ["random", "grid", "scipy", "scipy:direct"], "seeds": [0, 1]})
    job_dir = jobs.start_job(home, job, make_ohlcv(20))
    progress = wait_for(job_dir, {"done", "failed", "cancelled"})
    assert progress["status"] == "done", progress.get("error")
    trials = jobs.read_trials(job_dir)
    assert set(trials["backend"]) == {"random", "grid", "scipy", "scipy:direct"}
    assert trials.groupby(["backend", "seed"]).size().max() <= 12
    summary = jobs.read_summary(job_dir)["compare"]
    assert set(summary["backends"]) == {"random", "grid", "scipy", "scipy:direct"}
    assert all(item["runs"] == 2 for item in summary["backends"].values())
    # DIRECT is deterministic and differential evolution is not: the two SciPy searches are different searches.
    direct = trials[trials["backend"] == "scipy:direct"]
    assert direct.groupby("seed")["params_json"].apply(tuple).nunique() == 1


def test_jobs_are_numbered_in_the_order_they_start(home):
    """A # to keep track of each job by (the owner, 25 November 2026), kept in the job and the index."""
    first = jobs.start_job(home, selftest_job(4), make_ohlcv(20))
    second = jobs.start_job(home, selftest_job(4), make_ohlcv(20))
    for job_dir in (first, second):
        wait_for(job_dir, {"done", "failed", "cancelled"})
    assert jobs.read_job(first)["number"] == 1 and jobs.read_job(second)["number"] == 2
    assert jobs.list_jobs(home)["number"].tolist() == [2, 1]            # newest first
    assert store.next_job_number(home) == 3


def test_jobs_from_before_numbers_are_numbered_once_by_when_they_started(tmp_path):
    """An index made before jobs had numbers gains the column, its jobs numbered in the order they were started."""
    import sqlite3

    path = tmp_path / "catalogue.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, name TEXT, kind TEXT, backend TEXT, created_at TEXT, "
                     "path TEXT)")
        conn.execute("INSERT INTO jobs VALUES ('b', 'later', 'optimise', 'optuna', '2026-11-25T10:00:00+00:00', 'x')")
        conn.execute("INSERT INTO jobs VALUES ('a', 'earlier', 'optimise', 'optuna', '2026-11-19T10:00:00+00:00', 'y')")
    listed = store.list_jobs(tmp_path).set_index("job_id")["number"]
    assert listed.to_dict() == {"a": 1, "b": 2}
    assert store.next_job_number(tmp_path) == 3


def running_job_dir(tmp_path, pid, heartbeat_age, done=10, budget=40):
    """A job folder as a running worker leaves it: progress with its process id, and a heartbeat this old."""
    import os
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    jobs.write_json(job_dir / "progress.json", {"status": "running", "done": done, "budget": budget, "pid": pid,
                                                "started_at": jobs.now(), "updated_at": jobs.now()})
    (job_dir / "heartbeat").touch()
    old = time.time() - heartbeat_age
    os.utime(job_dir / "heartbeat", (old, old))
    return job_dir


def a_finished_process_id():
    import subprocess
    import sys
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_a_worker_that_died_shows_as_stopped_not_running_for_ever(tmp_path):
    """Its heartbeat gone quiet and its process gone: stopped, with what to do, and written back (the
    owner's job, 19 November 2026, died with its window and showed "running, 0 of 200" for hours)."""
    job_dir = running_job_dir(tmp_path, a_finished_process_id(), heartbeat_age=120)
    progress = jobs.read_progress(job_dir)
    assert progress["status"] == "stopped"
    assert "after 10 of 40 trials" in progress["error"] and "Start the job again" in progress["error"]
    assert jobs.read_json(job_dir / "progress.json")["status"] == "stopped"      # it survives the pid's reuse


def test_a_quiet_worker_whose_process_lives_is_still_running(tmp_path):
    """A quiet heartbeat alone may be a machine that slept: while the process lives, the job runs."""
    import os
    assert jobs.read_progress(running_job_dir(tmp_path, os.getpid(), heartbeat_age=600))["status"] == "running"


def test_a_dead_worker_with_a_fresh_heartbeat_is_given_time(tmp_path):
    """Within the heartbeat's grace a job is never judged, whatever its process."""
    assert jobs.read_progress(running_job_dir(tmp_path, a_finished_process_id(), heartbeat_age=1))["status"] == "running"


def test_a_job_starts_with_no_window(home, monkeypatch):
    """⚠️ CREATE_NO_WINDOW, never DETACHED_PROCESS: a detached venv launcher's interpreter opened an empty
    console window, and closing it killed the job (the owner, 20 November 2026)."""
    import os
    import subprocess
    if os.name != "nt":
        pytest.skip("Windows only: the window flags")
    seen = {}
    monkeypatch.setattr(subprocess, "Popen", lambda command, **options: seen.update(options))
    jobs.start_job(home, selftest_job(1), make_ohlcv(20))
    assert seen["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert not seen["creationflags"] & subprocess.DETACHED_PROCESS


def test_a_running_job_beats_and_says_what_it_is_doing(home):
    """The heartbeat file appears, progress counts search trials against the budget, and says its stage."""
    job_dir = jobs.start_job(home, selftest_job(8), make_ohlcv(20))
    progress = wait_for(job_dir, {"done", "failed", "cancelled"})
    assert progress["status"] == "done", progress.get("error")
    assert (job_dir / "heartbeat").exists()
    assert progress["done"] == progress["budget"] == 8 and progress["stage"] == "Searching"
    assert progress["searching_since"] is not None
