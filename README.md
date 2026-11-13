# OpenStrategy Studio

Define a trading strategy from indicators, patterns and rules, then backtest it, optimise it and read a tear sheet, all without writing code. Everything runs locally on [vectorbtX](https://github.com/AdventuresInDataScience/vectorbtx) (`vbt.x.SingleBacktest` and `vbt.x.PortfolioBacktest`), vectorbt 1.1.0, TA-Lib, pandas-ta-classic, ta_patterns, BUTChC, Optuna, marsopt and plotly-themes.

**Status: all five phases of the implementation plan are written, and phase 0 (confirming the code against the real libraries) has been run.** The test suite and `openstrategy-studio selfcheck` pass on vectorbt 1.1.0 and vectorbtX `9f4cf54`, which fixes every issue phase 0 found in it. [`docs/VERIFY.md`](docs/VERIFY.md) records every answer, what changed, and the two items still open (a real MT4/MT5 export and a clean-machine install). The issues found in vectorbtX itself are in [`docs/UPSTREAM.md`](docs/UPSTREAM.md), each with a standalone reproduction; one found since phase 0 (issue 15: holding costs left out of the total return) is open. Working on this with Claude Code? [`CLAUDE.md`](CLAUDE.md) is its orientation file.

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

* **marsopt** is an optional extra, because it builds a Cython extension when no wheel matches your platform. Install it with `uv sync --extra marsopt`. Extras are not remembered: a later plain `uv sync` removes them, so name every extra you want each time (`uv sync --extra marsopt --extra desktop`).
* **Proxies.** Behind a TLS-inspecting proxy or antivirus, `uv` fails every fetch with `UnknownIssuer` unless you use `uv sync --native-tls`. The Yahoo and FRED downloaders also need `CURL_CA_BUNDLE`, `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE`; see the vectorbtX README.
* **FRED rates** need a free key in the `FRED_API_KEY` environment variable, a `.env` file in the data folder, or the Settings page. The starter templates use fixed rates, so they run without one.
* **Moving everything.** `OPENSTRATEGY_STUDIO_HOME=<folder>` puts settings and data under one folder, which is useful for a portable install.

## Using it

1. **Data:** download from Yahoo, or import CSV or Parquet (MT4/MT5 exports included). Every import is checked, and nothing is stored until you accept the report.
2. **Build**, step by step, with the whole strategy **at a glance** beside the form:
   * **Setup:** the symbols and dates, then what you are building (trading on signals, or a portfolio: one symbol never rebalanced is buy and hold) and how you trade it (buying outright, CFDs, spread bets or FX spot). That choice sets the costs, how shorts are charged and how positions are sized, for every symbol.
   * **Features:** pick from a nested menu (category, family, item) holding every TA-Lib indicator, the pandas-ta-classic ones TA-Lib lacks, every ta_patterns candlestick and chart pattern, and transforms such as a percentage band or offset around any line. Every output is a line you can use (`kc.upperband`), and every input can read another feature.
   * **Rules:** lettered conditions combined by a logic line such as `(A and (B or C)) and not D`, with `xor`, `nand`, `nor` and `xnor` too; brackets are required where operators mix. **Make versions of this rule** lets the optimiser choose between whole rules.
   * **Risk:** sizing labelled for how you trade, stops, targets, extra exits (after a number of bars, breakeven, step trailing) and, on intraday bars, entries good for the day and closing before the day ends.
   * **Preview** marks every signal; **Pipeline** draws the strategy; **Strategy file** shows the JSON and a standalone vectorbtX script.
3. **Backtest:** run, then read the stats tables and fourteen figures in any plotly-themes theme. You can save a run and reopen it later.
4. **Optimise:** add the **variables to optimise**, any value of the strategy (a feature's period, a number in a rule, the stop, the target, the bars held), each over a range (start, step, stop) or a list, MT5-style. Then use grid (or the vectorised grid, many trials per call), Sobol, Optuna, scipy, BUTChC or marsopt, with constraints, a held-back test period, a stability check or walk-forward. **Compare backends** runs several of them, with several seeds each, on the same budget. Jobs run in the background and survive closing the page.
5. **Results:** see the trials, the best settings with **Apply to the strategy**, convergence, a pair heatmap, parallel coordinates, a re-run check, the deflated Sharpe ratio, walk-forward folds and the backend comparison. You can also compare saved runs.
6. **Patterns:** see how each of the 300 detectors was followed on your data, then add one to a rule in one click.

Every page has a **How to use this page** guide in the sidebar.

## Changing the look

Pick a page theme on the **Settings** page (Daylight, Midnight, Paper, Nord or Terminal), save, and restart the Studio. Charts set to **Match the page theme** follow it. To make your own theme, press **Make an editable copy** and edit the `.toml` file it creates. The Settings page shows where it is, and the comments in a copy of Daylight explain every setting.

A theme is one file in `src/openstrategy_studio/themes/`:

* `[theme]` is Streamlit's own theme table (colours, fonts, corners, heading sizes, sidebar colours, chart colours). Any option `streamlit config show` lists under `[theme]` works. Changes need a restart.
* `[studio]` holds the Studio's extras, which Streamlit ignores: a second accent, the gradient, card surfaces and shadows, the page glow, and the matching chart theme. They feed the one stylesheet, `themes/studio.css`. Changes to either show on the next rerun.

## Layout

```
src/openstrategy_studio/
  spec.py         the strategy spec (version 2): validate_spec, tunable_inputs, apply_params, JSON, hashing
  indicators.py   the feature catalogue: menu, exclusions, compute; indicator_catalogue.json is its snapshot
  compile.py      compile_signals, compile_weights, export_script
  engine.py       run_backtest and call_plan: the ONLY file that imports vectorbtx
  exit_rules.py   numba templates: time stop, breakeven, step trailing
  expressions.py  the logic line (parsed, never eval), conditions in words, the At a glance summary
  research.py     pattern research
  data.py         normalise, validate, import, store and load bar data
  store.py        the SQLite index: catalogue, runs, jobs
  config.py       settings.toml and environment
  analysis.py     results, saved runs, stats tables, figures, chart themes
  style.py        the look: page themes (themes/*.toml), the stylesheet (themes/studio.css), the logo
  optimise.py     search space, metrics, six backends, walk-forward, stability, deflated Sharpe
  jobs.py         start, read and cancel background jobs
  worker.py       python -m openstrategy_studio.worker <job_dir>
  ui.py           Streamlit helpers: spec-bound widgets, the rule builder, guarded runs
  cli.py          the openstrategy-studio command (run, desktop, selfcheck, worker)
  desktop.py      the app in a native window (pywebview)
  selfcheck.py    the phase 4 gate, runnable on any machine
  app/Home.py, app/pages/1_Data.py ... 7_Patterns.py
  templates/*.json
  themes/*.toml, themes/studio.css
  help/*.md       each page's sidebar guide
tests/            plan section 12, tests 1 to 14, plus fixtures
docs/VERIFY.md    the phase 0 record: answers and findings
docs/UPSTREAM.md  issues to fix in vectorbtX, with docs/upstream/vectorbtx_issues.py reproducing each
docs/PACKAGING.md the phase 4 spikes, the decision and the gate
packaging/        installers, Docker, the PyInstaller spike
CLAUDE.md         orientation for Claude Code
```

## Known limits

* **The buy-and-hold benchmark is before costs.** It is the market itself, held unlevered, which is what beta and alpha are measured against.
* **The bundled app is a spike.** The installer is the shipped route.
* **The instrument grid** warns about fields a mode rejects instead of greying them out.
* **Cap weighting and long/short selection** are not offered yet.

## Licence

vectorbt carries the Commons Clause, so OpenStrategy Studio is for personal and internal use. Selling hosting is not allowed.
