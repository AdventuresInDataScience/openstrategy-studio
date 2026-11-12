# CLAUDE.md: working on OpenStrategy Studio

A Streamlit app in which a person defines a trading strategy without code, then backtests and optimises it on vectorbtX (`vbt.x.SingleBacktest`, `vbt.x.PortfolioBacktest`) and vectorbt 1.1.0. All five phases of the implementation plan are written, and **phase 0 has been run** against the real libraries (vectorbt 1.1.0, vectorbtX `ace29e2`): the suite and `selfcheck` pass. `docs/VERIFY.md` is the record of every answer and every change it caused; `docs/UPSTREAM.md` lists what is to be fixed in vectorbtX. See "Status and next steps" below before starting.

## Commands

```bash
uv lock                          # once, if uv.lock is missing; commit it
uv sync --extra marsopt --extra desktop   # name every extra each time: a plain `uv sync` removes them
uv run pytest -rs                # -rs lists skips; with every extra installed only the bash check skips on Windows
uv run pytest tests/test_engine.py -x -q
uv run openstrategy-studio       # the app; also: desktop, selfcheck, worker <dir>
uv run openstrategy-studio selfcheck --quick
```

Never `pip install`. Add dependencies with `uv add`.

## Status and next steps (as of 12 November 2026)

**Where we are.** Phase 0 is done and recorded in `docs/VERIFY.md`: against vectorbt 1.1.0 and vectorbtX `ace29e2`, `uv run pytest -rs` gives 131 passed and 1 skipped (the bash installer check on Windows), and `selfcheck` passes. Phase 0 found 13 items that belong in vectorbtX: twelve bugs and one required feature (shared cash with an entry priority). They are in `docs/UPSTREAM.md`. The owner writes vectorbtX and is fixing them there. **Work in this repo is paused until that new vectorbtX is installed.**

The owner's rule: behaviour that is wrong in vectorbtX gets fixed in vectorbtX and flagged in `docs/UPSTREAM.md` (with a check in `docs/upstream/vectorbtx_issues.py`), not hidden behind a Studio workaround. UX and app logic are fixed here. Ask before editing the vectorbtX checkout (`../vectorbtx`) or opening issues on its repo.

**Next, in this order:**

1. **Install the new vectorbtX.** After it is pushed, run `uv lock --upgrade-package vectorbtx`, then `uv sync --extra marsopt --extra desktop`, and commit `uv.lock`. To iterate on an unpushed checkout, point `[tool.uv.sources]` at `vectorbtx = { path = "../vectorbtx", editable = true }` for the session, and restore the git source before committing.
2. **See what is fixed.** `uv run python -I docs/upstream/vectorbtx_issues.py` prints PRESENT or FIXED for each item; its exit code is the number still present. Check 13 is the acceptance test for the shared-cash API: if vectorbtX's argument names differ from the proposal, rename `SHARED_CASH_ARGUMENTS` in the script first.
3. **Remove the workarounds of the fixed issues.** Each issue's "Studio workaround" line in `docs/UPSTREAM.md` says exactly which code, tests and docs change. ⚠️ Issue 1 (short signals not shifted) must be removed in the same change as the bump, or shorts are shifted twice. Keep the workaround of anything still PRESENT.
4. **Re-run** `uv run pytest -rs` and `uv run openstrategy-studio selfcheck`, and read each failure as a finding about the library. Update the versions table and answers in `docs/VERIFY.md`, and mark the fixed items in `docs/UPSTREAM.md`. The phase 3 gates (`test_gate_matches_the_reference_simulator`, `test_vectorised_equals_one_by_one`) must keep passing; never loosen them.
5. **Build the Studio side of shared cash** (UPSTREAM.md issue 13, "Studio side"):
   * spec fields for the account (replicated or shared, the entry priority, and skip or shrink), checked in `validate_spec`;
   * mapped by `engine.call_plan`, so the export follows automatically;
   * a priority picker on the Build page's Account tab (symbol order, random with a seed, or by a feature, which compiles to a per-bar score frame);
   * results that show the replicated case as N accounts with a per-symbol table. `analysis.stats_tables` currently averages the columns silently.
   * `engine.vectorisable` must reject shared cash, because the vectorised grid gives each trial its own column and cash.
6. **Other Studio to-dos from phase 0:**
   * require both currencies on an FX instrument (UPSTREAM.md issue 3);
   * save golden numbers from vectorbtX's `examples/strategies/01_ma_cross_spreadbet_fx.py` once its `base_currency` line is fixed (`docs/VERIFY.md` section 4).
7. **Still open from phase 0:** `data.apply_mapping` against a real MT4/MT5 export, and the installers on a clean machine (`docs/PACKAGING.md`).

## Layout (one module per concept)

| File | Concept |
|---|---|
| `spec.py` | The strategy spec (a plain dict): validate_spec, apply_params, choices, JSON, hashing |
| `blocks.py` | `BLOCKS`: every feature block's params, compile function and export snippet |
| `compile.py` | compile_signals, compile_weights, export_script (code generation) |
| `engine.py` | call_plan (pure) and run_backtest, vectorisable, run_vectorised: **the only importer of vectorbtx** |
| `exit_rules.py` | Numba templates for the time stop, breakeven and step trailing |
| `expressions.py` | Text to and from the rule tree (whitelisted `ast`, never `eval`) |
| `data.py` | Normalise, validate, import, store, load, synthetic data |
| `store.py` | The SQLite index of datasets, runs and jobs |
| `config.py` | settings.toml, environment, `OPENSTRATEGY_STUDIO_HOME` |
| `analysis.py` | Results extraction, saved runs, stats, every figure, themes, the pipeline diagram |
| `optimise.py` | Spaces (flat or conditional), metrics, backends, walk-forward, stability, DSR, backend comparison |
| `jobs.py`, `worker.py` | Background jobs (files on disk, a detached process) |
| `research.py` | Pattern research |
| `ui.py` | Streamlit helpers: spec-bound widgets, the rule builder, guarded runs |
| `cli.py`, `desktop.py`, `selfcheck.py` | The command line, the webview window, the phase 4 gate |
| `app/Home.py`, `app/pages/*.py` | Thin pages: no business logic |

## Code style (the owner's rules; they override your defaults)

1. **Functional, no classes.** Use functions and plain dicts and lists. No dataclasses, no pydantic. The only classes are the ones vectorbt and vectorbtX provide. Where a library wants an object (BUTChC's executor), use `types.SimpleNamespace`.
2. **Do not over-modularise.** A helper earns its place only when it is called from three or more places, or the function would otherwise exceed about 100 lines. A reader should follow a feature top to bottom in one file.
3. **Explicit over clever.** Registries are dict literals. Prefer a visible `for` loop to a dense comprehension chain.
4. **Comment heavily.** Explain the why. Mark anything that can silently give a wrong number with `# ⚠️`, and anything unverified with `⚠️ VERIFY`.
5. **Few exception cases.** Validate once at the boundary: loading a spec, an import, the user pressing Run. Pages have ONE `try/except`, via `ui.run_guarded`. Do not wrap calls that cannot fail.
6. **Keyword arguments** at every vectorbt and vectorbtX call site.

UI text: sentence case, plain verbs, and errors that say what to do. No ALL-CAPS labels.

## Invariants: do not break these

* **One source of truth.** `st.session_state["spec"]` is the spec. Widgets are seeded from it on every run and write back through `ui.set_value`. Keys come from `ui.key(path)`, which includes a revision number that `ui.replace_spec` bumps.
* **The compiler never sees optimisation markup.** `apply_params` strips every `{"value", "opt"}` node and resolves every `{"choose", "options"}` node.
* **Short signals are shifted by one bar in `compile_signals`.** The front door shifts only entries, exits and schedule. Long signals must NOT be shifted twice.
* **Always the four-array form.** Never `direction="both"`.
* **Stops are measured from the fill and pay the spread.** `call_plan` passes `stop_entry_price="price"` and `stop_exit_price="stopmarket"` whenever there is a stop or target. vectorbt's defaults anchor stops at the entry bar's close and fill them with no slippage.
* **An FX pair never uses `SingleBacktest`.** Its flat `base_currency=` is read as the account currency, which drops the pair and zeroes carry. `call_plan` routes any instrument with a base currency through `PortfolioBacktest(instruments=...)`.
* **The benchmark is `pf.benchmark_value()`.** Never a second front-door call with no signals: that hold is funded at buying power (30x on an FX spread bet). Never pass `benchmark_rets=` to `returns_stats`.
* **Select one column as `pf[[column]]`, never `pf[column]`.** The scalar form crashes every metric on a leveraged vectorbtX portfolio.
* **Never filter exits.** Filters are allowed in entry rules only; `validate_spec` enforces it.
* **No higher-timeframe leakage.** The filter needs `label="left", closed="left"` and then `.shift(1)`. `tests/test_lookahead.py` uses dense truncation cut points; keep them dense.
* **Portfolio weights stay full-shape.** Never forward-fill them. Always pass `schedule=` to `drifted_weights`.
* **One cost path.** Costs come only from instrument fields, never a raw `fees=` (it collides with commission and raises).
* **Charts.** `st.plotly_chart(fig, theme=None)` everywhere (via `ui.chart`). Every figure ends with `analysis.finish(fig, theme)`.
* **Export and in-app runs share one plan.** `export_script` and `run_backtest` both use `engine.call_plan`. `tests/test_export.py` keeps them equal.
* **Rendering must not change the spec.** A page rendered with defaults must leave the spec unchanged. Number widgets keep an integer range integer (`ui.param_input`).
* **Paths.** Use pathlib everywhere, never shell commands. All user data lives under `config.data_dir`.
* **Starting processes.** Use `cli.self_command`, which also works in a frozen bundle.
* **Files two processes share** (job folders) are written atomically with `jobs.write_json` / `jobs.write_parquet` and read through `jobs.retry_while_locked`: on Windows a file being replaced, or held open by the other process, briefly refuses access.
* **No HTTPS context before `config.apply_environment`.** On Windows, uv's CPython aborts the whole process when `SSLKEYLOGFILE` is set and any HTTPS context is created; `apply_environment` drops it. If a test run dies with no summary and `OPENSSL_Uplink` in the output, that is the cause.

## Testing conventions

Use plain pytest functions with no classes or mocking frameworks, and no network. Synthetic data comes from `tests/conftest.py::make_ohlcv`. Tests that need a library call `pytest.importorskip` inside the test.

## Known gaps

* The instrument grid cannot grey out individual cells, so it warns about fields a mode rejects instead.
* Cap weighting and long/short selection are not offered: there is no market-cap source, and negative weights are unverified.
* Conditional spaces work with BUTChC and Optuna only; the other backends fix alternatives at their current choice.
* The PyInstaller bundle is a spike (`docs/PACKAGING.md`). The first-run installer is the shipped route.
* The FX conversion markup and the carry markup are never charged (vectorbtX issues 9 and 4 in `docs/UPSTREAM.md`). Both are left to vectorbtX rather than worked around here.
