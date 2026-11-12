"""Test 13: the job runner. Uses the self-test objective, so no backtest library is needed."""

import time

import pytest

from openstrategy_studio import config, jobs

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
    job = selftest_job(12, batch=4)
    job.update(kind="compare", compare={"backends": ["random", "grid", "scipy"], "seeds": [0, 1]})
    job_dir = jobs.start_job(home, job, make_ohlcv(20))
    progress = wait_for(job_dir, {"done", "failed", "cancelled"})
    assert progress["status"] == "done", progress.get("error")
    trials = jobs.read_trials(job_dir)
    assert set(trials["backend"]) == {"random", "grid", "scipy"}
    assert trials.groupby(["backend", "seed"]).size().max() <= 12
    summary = jobs.read_summary(job_dir)["compare"]
    assert set(summary["backends"]) == {"random", "grid", "scipy"}
    assert all(item["runs"] == 2 for item in summary["backends"].values())
