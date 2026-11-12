# CLAUDE.md: working on OpenStrategy Studio

A Streamlit app in which a person defines a trading strategy without code, then backtests and optimises it on vectorbtX (`vbt.x.SingleBacktest`, `vbt.x.PortfolioBacktest`) and vectorbt 1.1.0. All five phases of the implementation plan are written, and **phase 0 has been run** against the real libraries (vectorbt 1.1.0, vectorbtX `ace29e2`, then re-run on `16293a2`, which fixes what phase 0 found): the suite and `selfcheck` pass. `docs/VERIFY.md` is the record of every answer and every change it caused; `docs/UPSTREAM.md` lists what is to be fixed in vectorbtX. See "Status and next steps" below before starting.

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

**Where we are.** Phase 0 is done and recorded in `docs/VERIFY.md`. It found 13 items that belong in vectorbtX (twelve bugs and one required feature, shared cash with an entry priority), listed in `docs/UPSTREAM.md`. **All 13 are fixed in vectorbtX `16293a2`, which `uv.lock` pins**, and the Studio's workarounds for them are removed. Against it, `uv run pytest -rs` gives 133 passed and 1 skipped (the bash installer check on Windows), both phase 3 gates pass unchanged, and `selfcheck` passes. One new limitation came with the fixes: with an extra exit active, a stop is still not checked on its entry bar (`docs/UPSTREAM.md`, issue 14, open).

The owner's rule: behaviour that is wrong in vectorbtX gets fixed in vectorbtX and flagged in `docs/UPSTREAM.md` (with a check in `docs/upstream/vectorbtx_issues.py`), not hidden behind a Studio workaround. UX and app logic are fixed here. Ask before editing the vectorbtX checkout (`../vectorbtx`) or opening issues on its repo.

**Updating vectorbtX** (done for `16293a2`; repeat for the next bump): `uv lock --upgrade-package vectorbtx`, `uv sync --extra marsopt --extra desktop`, then `uv run python -I docs/upstream/vectorbtx_issues.py` (PRESENT or FIXED per item; exit code = number present), remove the workaround of each fixed item as its `docs/UPSTREAM.md` entry says, re-run `uv run pytest -rs` and `selfcheck`, and read each failure as a finding about the library. The phase 3 gates (`test_gate_matches_the_reference_simulator`, `test_vectorised_equals_one_by_one`) must keep passing; never loosen them. To iterate on an unpushed checkout, point `[tool.uv.sources]` at `vectorbtx = { path = "../vectorbtx", editable = true }` for the session, and restore the git source before committing.

**Next, in this order:**

1. **Decide on issue 14** with the owner: it needs a vectorbtX change (callbacks opting in to split bars) and then the `exit_rules.py` templates indexing by bar. `test_extra_exits_say_the_entry_bar_is_not_checked` pins the warning until then.
2. **Build the Studio side of shared cash** (UPSTREAM.md issue 13, "Studio side"; vectorbtX's API is `cash=`, `entry_priority=`, `when_cash_short=`):
   * spec fields for the account (replicated or shared, the entry priority, and skip or shrink), checked in `validate_spec`;
   * mapped by `engine.call_plan`, so the export follows automatically;
   * a priority picker on the Build page's Account tab (symbol order, random with a seed, or by a feature, which compiles to a per-bar score frame);
   * results that show the replicated case as N accounts with a per-symbol table. `analysis.stats_tables` currently averages the columns silently.
   * `engine.vectorisable` must reject shared cash, because the vectorised grid gives each trial its own column and cash.
3. **Other Studio to-dos from phase 0:**
   * require both currencies on an FX instrument (UPSTREAM.md issue 3);
   * save golden numbers from vectorbtX's `examples/strategies/01_ma_cross_spreadbet_fx.py`, now that its `base_currency` line reads as the pair (`docs/VERIFY.md` section 4).
4. **Still open from phase 0:** `data.apply_mapping` against a real MT4/MT5 export, and the installers on a clean machine (`docs/PACKAGING.md`).

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
* **The compiler shifts no signal.** The front door shifts entries, exits, short entries, short exits and the schedule to the next bar's open itself (vectorbtX `16293a2`). A shift in `compile_signals` or `export_script` would delay them two bars.
* **Always the four-array form.** Never `direction="both"`.
* **Stops are measured from the fill, pay the spread, and are checked on the entry bar.** vectorbtX's front door sets `stop_entry_price="price"` and `stop_exit_price="stopmarket"` and splits each bar into its open and the rest of the bar whenever a stop or target is active. `call_plan` must not pass either price; vectorbt's own defaults anchor stops at the entry bar's close and fill them with no slippage. ⚠️ The entry-bar check does not happen with an extra exit (issue 14).
* **An FX pair never uses `SingleBacktest`.** `call_plan` routes any instrument with a base currency through `PortfolioBacktest(instruments=...)`, which keeps every field on the instrument. The reporting currency (`account.base_currency` in the spec) is passed as `account_currency=`, never `base_currency=`: on FX terms vectorbtX reads a flat `base_currency=` as the pair's base.
* **The benchmark is `pf.benchmark_value()`.** Never a second front-door call with no signals: that hold is funded at buying power (30x on an FX spread bet). Never pass `benchmark_rets=` to `returns_stats`.
* **Select one column as `pf[[column]]`.** It returns the same one-column shape a single run does. (The scalar form crashed every metric on a leveraged portfolio until vectorbtX `16293a2`.)
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
* With an extra exit (time stop, breakeven, step trailing) a stop is not checked on the bar the trade was entered, and vectorbtX warns (issue 14 in `docs/UPSTREAM.md`). Left to vectorbtX rather than worked around here.
