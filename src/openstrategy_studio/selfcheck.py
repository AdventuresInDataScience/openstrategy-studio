"""`openstrategy-studio selfcheck`: the phase 4 gate on any machine.

"A clean machine runs build, backtest and optimise." Everything happens in a
temporary data folder with synthetic prices and offline mode on, so it needs no
network and touches none of your data. Exit code 0 means every check passed.
"""

import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

REQUIRED = ["vectorbt", "vectorbtx", "numba", "pandas", "numpy", "plotly", "plotly_themes", "streamlit",
            "ta_patterns", "optuna", "butchc", "scipy", "pyarrow", "platformdirs"]


def main(quick=False, keep=False):
    folder = Path(tempfile.mkdtemp(prefix="openstrategy-selfcheck-"))
    previous = os.environ.get("OPENSTRATEGY_STUDIO_HOME")
    os.environ["OPENSTRATEGY_STUDIO_HOME"] = str(folder)
    results = []
    failures = []
    state = {}

    def check(name, function, needs=()):
        print(f"  {name} ...", end="", flush=True)
        missing = [item for item in needs if item not in state]
        if missing:
            results.append((name, False, 0.0, "skipped: an earlier check failed"))
            print(" skipped: an earlier check failed")
            return
        started = time.time()
        try:
            detail = function() or ""
            results.append((name, True, time.time() - started, detail))
            print(f" ok ({time.time() - started:.1f}s) {detail}")
        except Exception as error:   # noqa: BLE001  (the harness reports every failure and carries on)
            results.append((name, False, time.time() - started, f"{type(error).__name__}: {error}"))
            failures.append((name, traceback.format_exc()))
            print(f" FAILED: {type(error).__name__}: {error}")

    from . import config

    def installed(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return name in sys.modules

    def environment():
        missing = [name for name in REQUIRED if not installed(name)]
        if missing:
            raise RuntimeError(f"not installed: {', '.join(missing)} (run `uv sync`)")
        settings = config.load_settings()
        settings["offline"] = True              # nothing may reach the network
        config.save_settings(settings)
        config.apply_settings(settings)         # imports vectorbtX and applies its settings
        cache = Path(os.environ["NUMBA_CACHE_DIR"])
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "write-test").write_text("ok", encoding="utf-8")
        state["settings"] = settings
        state["data_dir"] = config.data_dir(settings)
        from . import engine
        versions = engine.versions()
        return f"Python {sys.version.split()[0]}, vectorbt {versions['vectorbt']}, vectorbtX {versions['vectorbtx']}"

    def data():
        from . import data as data_module
        from .spec import list_templates, load_template

        templates = {name: load_template(name) for name in list_templates()}
        symbols = sorted({symbol for spec in templates.values() for symbol in spec["data"]["symbols"]})
        synthetic = data_module.synthetic_ohlcv(symbols, n=900, seed=7)
        for symbol in symbols:
            frame = synthetic["close"][[symbol]].rename(columns={symbol: "close"})
            for field in ("open", "high", "low", "volume"):
                frame[field] = synthetic[field][symbol]
            clean, info = data_module.normalise_ohlcv(frame, "1d")
            problems = data_module.validate_ohlcv(clean, "1d")
            if any(problem["check"].startswith(("High", "Low", "Zero")) for problem in problems):
                raise RuntimeError(f"synthetic data for {symbol} failed validation")
            data_module.store_dataset(state["data_dir"], symbol, "1d", clean, "selfcheck", tz=info["tz"])
        state["frames"] = {}
        for name, spec in templates.items():
            spec["data"].update(start="2018-01-01", end="2030-12-31")
            state["frames"][name] = (spec, data_module.load_for_spec(spec, state["data_dir"]))
        return f"{len(symbols)} symbols, {len(templates)} templates"

    def build():
        from .compile import compile_signals, compile_weights, export_script
        from .spec import apply_params, strategy_kind, validate_spec

        for name, (spec, frames) in state["frames"].items():
            problems = validate_spec(spec)
            if problems:
                raise RuntimeError(f"{name}: {problems[0]}")
            applied = apply_params(spec)
            if strategy_kind(applied) == "signals":
                compile_signals(applied, frames)
            elif strategy_kind(applied) == "portfolio":
                compile_weights(applied, frames)
            ast.parse(export_script(spec, state["data_dir"]))
        return "validated, compiled and exported"

    def backtest():
        from . import analysis, data as data_module, engine

        lines = []
        for name, (spec, frames) in state["frames"].items():
            keywords = {"data_dir": state["data_dir"], "offline": True,
                        "static_rate": state["settings"]["vectorbtx"].get("static_rate")}
            run = engine.run_backtest(spec, frames, **keywords)
            benchmark = engine.run_backtest(spec, frames, benchmark=True, **keywords)
            results = analysis.extract_results(run, benchmark)
            run_dir = analysis.save_run(state["data_dir"], results, run["spec"], state["settings"],
                                        data_module.fingerprint(state["data_dir"], spec, frames))
            loaded = analysis.load_results(run_dir)
            for figure in analysis.available_figures(loaded):
                analysis.FIGURES[figure]["fn"](loaded, "plotly_white")
            lines.append(f"{name}: {len(results['trades'])} trades")
        state["backtested"] = True
        return "; ".join(lines)

    def wait(job_dir, timeout):
        from . import jobs
        deadline = time.time() + timeout
        while time.time() < deadline:
            progress = jobs.read_progress(job_dir)
            if progress["status"] in ("done", "failed", "cancelled"):
                return progress
            time.sleep(0.5)
        raise RuntimeError(f"the job did not finish within {timeout} seconds")

    def job_runner():
        import pandas as pd

        from . import jobs
        job = {"name": "selfcheck: job runner", "kind": "optimise", "spec": {}, "backend": "random",
               "space": {"a": {"min": 0.0, "max": 10.0}, "b": {"min": 1, "max": 50, "int": True}},
               "backend_options": {"batch": 4}, "budget": 16, "metric": "selftest", "direction": "maximize",
               "seed": 0, "workers": 2, "settings": state["settings"], "constraints": {}, "split": None,
               "walk_forward": None, "stability": False}
        frames = {"close": pd.DataFrame({"X": [1.0, 1.0]}, index=pd.bdate_range("2020-01-01", periods=2))}
        progress = wait(jobs.start_job(state["data_dir"], job, frames), 120)
        if progress["status"] != "done":
            raise RuntimeError(progress.get("error") or progress["status"])
        return f"{progress['done']} trials across 2 processes"

    def optimise():
        from . import jobs, optimise as optimise_module

        spec, frames = state["frames"]["ma_cross"]
        job = {"name": "selfcheck: optimise", "kind": "optimise", "spec": spec,
               "space": optimise_module.build_space(spec), "backend": "random", "backend_options": {"batch": 2},
               "budget": 6, "metric": "sharpe", "direction": "maximize", "seed": 0, "workers": 2,
               "settings": state["settings"], "constraints": {}, "split": {"train_fraction": 0.7},
               "walk_forward": None, "stability": False}
        job_dir = jobs.start_job(state["data_dir"], job, frames)
        progress = wait(job_dir, 900)
        if progress["status"] != "done":
            raise RuntimeError((progress.get("error") or progress["status"]).strip().splitlines()[-1])
        trials = jobs.read_trials(job_dir)
        return f"{len(trials)} backtests in the background, held-back test included"

    def pages():
        from streamlit.testing.v1 import AppTest

        from .spec import load_template
        app = Path(__file__).parent / "app"
        names = [app / "Home.py"] + sorted((app / "pages").glob("*.py"))
        for page in names:
            test = AppTest.from_file(str(page), default_timeout=300)
            test.session_state["spec"] = load_template("ma_cross")
            test.session_state["spec_rev"] = 0
            test.run()
            if test.exception:
                raise RuntimeError(f"{page.name}: {test.exception[0].value}")
        return f"{len(names)} pages"

    print(f"OpenStrategy Studio self-check (temporary data in {folder})")
    check("Environment", environment)
    check("Data: store and load", data, needs=("settings",))
    check("Build: validate, compile, export every template", build, needs=("frames",))
    check("Backtest every template, save and redraw it", backtest, needs=("frames",))
    check("Background jobs", job_runner, needs=("settings",))
    if not quick:
        check("Optimise a strategy in the background", optimise, needs=("frames", "backtested"))
        check("App pages load", pages, needs=("settings",))

    passed = all(ok for _, ok, _, _ in results)
    print()
    print("All checks passed." if passed else f"{sum(not ok for _, ok, _, _ in results)} check(s) failed.")
    for name, text in failures:
        print(f"\n--- {name} ---\n{text}")
    if previous is None:
        os.environ.pop("OPENSTRATEGY_STUDIO_HOME", None)
    else:
        os.environ["OPENSTRATEGY_STUDIO_HOME"] = previous
    if keep or not passed:
        print(f"\nThe temporary data folder was kept: {folder}")
    else:
        shutil.rmtree(folder, ignore_errors=True)
    return 0 if passed else 1
