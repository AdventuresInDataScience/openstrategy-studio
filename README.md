# OpenStrategy Studio

Define a trading strategy through forms, a rule builder and expressions, then backtest it, optimise it and read a tear sheet, all without writing code. Everything runs locally on [vectorbtX](https://github.com/AdventuresInDataScience/vectorbtx) (`vbt.x.SingleBacktest` and `vbt.x.PortfolioBacktest`), vectorbt 1.1.0, ta_patterns, BUTChC, Optuna, marsopt and plotly-themes.

**Status: all five phases of the implementation plan are written. Phase 0, confirming the code against the real libraries, is yours to run.** The code was written without the backtest libraries available, so start with [`docs/VERIFY.md`](docs/VERIFY.md). It is the checklist of every item to confirm, with the test that answers each one. Working on this with Claude Code? [`CLAUDE.md`](CLAUDE.md) is its orientation file.

## Install and run

The environment is managed with [uv](https://docs.astral.sh/uv/). Nothing is installed with bare `pip`.

```bash
uv lock                       # first time only: resolves and writes uv.lock (commit it)
uv sync                       # creates .venv from the lock file
uv run openstrategy-studio    # opens the app in your browser
uv run pytest -rs             # the test suite; -rs lists anything skipped
uv run openstrategy-studio selfcheck   # can this machine build, backtest and optimise? (offline)
uv run openstrategy-studio desktop     # the app in its own window (uv sync --extra desktop)
```

For people who are not developers, the first-run installers in `packaging/installer/` do all of this. They end with a desktop shortcut that works offline. See [`docs/PACKAGING.md`](docs/PACKAGING.md), which also covers Docker and the experimental bundled app.

Notes:

* **marsopt** is an optional extra, because it builds a Cython extension when no wheel matches your platform. Install it with `uv sync --extra marsopt`.
* **Proxies.** Behind a TLS-inspecting proxy or antivirus, `uv` fails every fetch with `UnknownIssuer` unless you use `uv sync --native-tls`. The Yahoo and FRED downloaders also need `CURL_CA_BUNDLE`, `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE`; see the vectorbtX README.
* **FRED rates** need a free key in the `FRED_API_KEY` environment variable, a `.env` file in the data folder, or the Settings page. The starter templates use fixed rates, so they run without one.
* **Moving everything.** `OPENSTRATEGY_STUDIO_HOME=<folder>` puts settings and data under one folder, which is useful for a portable install.

## Using it

1. **Data:** download from Yahoo, or import CSV or Parquet (MT4/MT5 exports included). Every import is checked, and nothing is stored until you accept the report.
2. **Build:** start from a template, or build up the instruments, features, rules (or the portfolio), risk and account. The live preview marks every signal and warns about rules that never fire. Turn on **Optimise** beside any number to give it a search range. **Turn into alternatives** lets the optimiser choose between whole rules (structure search). The Risk tab adds a time stop, breakeven and step trailing. The Pipeline tab draws the strategy as a diagram.
3. **Backtest:** run, then read the stats tables and fourteen figures in any plotly-themes theme. You can save a run and reopen it later.
4. **Optimise:** use grid (or the vectorised grid, many trials per call), Sobol, Optuna, scipy, BUTChC or marsopt, with constraints, a held-back test period, a stability check or walk-forward. **Compare backends** runs several of them, with several seeds each, on the same budget. Jobs run in the background and survive closing the page.
5. **Results:** see the trials, the best settings with **Apply to the strategy**, convergence, a pair heatmap, parallel coordinates, a re-run check, the deflated Sharpe ratio, walk-forward folds and the backend comparison. You can also compare saved runs.
6. **Patterns:** see how each of the 300 detectors was followed on your data, then add one to a rule in one click.

The **Spec and code** tab shows the strategy as JSON, and as a standalone script of plain vectorbtX calls that reproduces the backtest.

## Layout

```
src/openstrategy_studio/
  spec.py         the strategy spec: defaults, validate_spec, apply_params, JSON, hashing
  blocks.py       BLOCKS: every feature block's parameters, compile function and export snippet
  compile.py      compile_signals, compile_weights, export_script
  engine.py       run_backtest and call_plan: the ONLY file that imports vectorbtx
  exit_rules.py   numba templates: time stop, breakeven, step trailing
  expressions.py  the expression editor (whitelisted ast, never eval)
  research.py     pattern research
  data.py         normalise, validate, import, store and load bar data
  store.py        the SQLite index: catalogue, runs, jobs
  config.py       settings.toml and environment
  analysis.py     results, saved runs, stats tables, figures, themes
  optimise.py     search space, metrics, six backends, walk-forward, stability, deflated Sharpe
  jobs.py         start, read and cancel background jobs
  worker.py       python -m openstrategy_studio.worker <job_dir>
  ui.py           Streamlit helpers: spec-bound widgets, the rule builder, guarded runs
  cli.py          the openstrategy-studio command (run, desktop, selfcheck, worker)
  desktop.py      the app in a native window (pywebview)
  selfcheck.py    the phase 4 gate, runnable on any machine
  app/Home.py, app/pages/1_Data.py ... 7_Patterns.py
  templates/*.json
tests/            plan section 12, tests 1 to 14, plus fixtures
docs/VERIFY.md    the phase 0 checklist
docs/PACKAGING.md the phase 4 spikes, the decision and the gate
packaging/        installers, Docker, the PyInstaller spike
CLAUDE.md         orientation for Claude Code
```

## Known limits

* **The extra exits and the vectorised grid are experimental** until their gates pass (see `docs/VERIFY.md`).
* **The bundled app is a spike.** The installer is the shipped route.
* **The instrument grid** warns about fields a mode rejects instead of greying them out.
* **Cap weighting and long/short selection** are not offered yet.

## Licence

vectorbt carries the Commons Clause, so OpenStrategy Studio is for personal and internal use. Selling hosting is not allowed.
