# CLAUDE.md: working on OpenStrategy Studio

A Streamlit app in which a person defines a trading strategy without code, then backtests and optimises it on vectorbtX (`vbt.x.SingleBacktest`, `vbt.x.PortfolioBacktest`) and vectorbt 1.1.0. All five phases of the implementation plan are written, and **phase 0 has been run** against the real libraries (vectorbt 1.1.0, vectorbtX `ace29e2`, then re-run on `16293a2`, `9f4cf54` and `9fa427e`, which fix everything phase 0 and its follow-ups found): the suite and `selfcheck` pass. `docs/VERIFY.md` is the record of every answer and every change it caused; `docs/UPSTREAM.md` lists what was found in vectorbtX and how each item was fixed. See "Status and next steps" below before starting.

## Commands

```bash
uv lock                          # once, if uv.lock is missing; commit it
uv sync --extra marsopt --extra desktop   # name every extra each time: a plain `uv sync` removes them
uv run pytest -rs                # -rs lists skips; with every extra installed only the bash check skips on Windows
uv run pytest tests/test_engine.py -x -q
uv run openstrategy-studio       # the app; also: desktop, selfcheck, worker <dir>
uv run openstrategy-studio selfcheck --quick
uv run python -m openstrategy_studio.indicators   # rewrite indicator_catalogue.json after a TA-Lib, pandas-ta or ta_patterns bump
```

Never `pip install`. Add dependencies with `uv add`.

## Status and next steps (as of 13 November 2026)

**Where we are.** Phase 0 is done and recorded in `docs/VERIFY.md`. It found 13 items that belong in vectorbtX (twelve bugs and one required feature, shared cash with an entry priority), listed in `docs/UPSTREAM.md`. **All 13 are fixed in vectorbtX `16293a2`**, and the Studio's workarounds for them are removed. Issue 14 (with an extra exit active, a stop was not checked on its entry bar) is fixed in `9f4cf54`. **Issues 15 to 19 are fixed in `9fa427e`, which `uv.lock` pins**: holding costs in the totals (15), holding costs per instrument in a shared account (16), borrowing without a margin loan and leverage on the replay path (17), the borrow fee on a direct short (18, a feature), and the per-instrument `financing` field, which was never read and is now reserved (19). `docs/upstream/vectorbtx_issues.py` reports 0 of 18 present.

**Also done on 13 November 2026: the redesign's styling, then the Build page's UX** (the owner's points, in "Next" item 1). Spec version 2, with no backward compatibility (the owner's call: still in development; the templates were converted):
* **Setup asks two questions:** what is being built (`setup.kind`: trading on signals, or a portfolio; buy and hold is a portfolio that never rebalances, `schedule.rebalance = "never"`) and how it is traded (`setup.trading`: one vectorbtX mode for every instrument). The instrument grid shows only the fields that mode accepts, and sizing is labelled for it.
* **The feature catalogue** (`indicators.py`, `indicator_catalogue.json`): every TA-Lib function but its candlesticks (176, TA-Lib 0.8), the pandas-ta-classic indicators TA-Lib lacks (69), every ta_patterns detector (295), and the Studio's own (MA envelopes, previous day/week/month, % offset, % band, distance, bars ago, bars since, Python). A nested picker (category, family, item) shows each entry's outputs; every input can read a price or another feature.
* **Rules are lettered conditions and a logic line** (`and or xor nand nor xnor not`, brackets required where operators mix), with rule versions for structure search.
* **Optimisation moved to the Optimise page's variables** (`spec.optimise.variables`, MT5-style start/step/stop or a list), any value: features, rule numbers, risk dials.
* **Order timing** for intraday bars: entries good for the day, flat by the day's end (approximations, `docs/UPSTREAM.md` issue 20).
* **The At a glance panel** (sticky, beside the Build form) and **a guide per page** in the sidebar (`help/*.md`).
* Findings: ta_patterns' flag and pennant detectors look ahead (`docs/UPSTREAM.md` P1; left out of the catalogue). vectorbtX has no market-on-close or expiring orders (issue 20, a feature request). The RSI templates now trade less: TA-Lib's RSI is Wilder's, smoother than vectorbt's simple one.
* `uv run pytest -rs`: 191 passed, 1 skipped (the bash check on Windows); both phase 3 gates and the golden numbers pass (TA-Lib's SMA reproduces them); `selfcheck` passes.

**Done on 13 November 2026:** the Studio side of shared cash (issue 13: spec fields, `call_plan`, the Account tab's picker, per-account results, `vectorisable`), both currencies required on an FX instrument (issue 3), and the golden numbers from vectorbtX's `01_ma_cross_spreadbet_fx.py` (`tests/fixtures/golden/`, which the Studio's template matches exactly on real Yahoo data). Then the bump to `9fa427e`: the golden numbers regenerated on the same prices (`make_golden.py --offline`; only End Value and Total Return moved, by the summed carry), `borrow_rate` in the Build page's instrument grid and in `validate_spec` (equity mode only), the grid's financing column removed, and vectorbtX's warnings shown on the Backtest page. `uv run pytest -rs` gives 151 passed and 1 skipped (the bash installer check on Windows), both phase 3 gates pass, and `selfcheck` passes.

The owner's rule: behaviour that is wrong in vectorbtX gets fixed in vectorbtX and flagged in `docs/UPSTREAM.md` (with a check in `docs/upstream/vectorbtx_issues.py`), not hidden behind a Studio workaround. UX and app logic are fixed here. Ask before editing the vectorbtX checkout (`../vectorbtx`) or opening issues on its repo.

**Updating vectorbtX** (done for `16293a2`, `9f4cf54` and `9fa427e`; repeat for the next bump): `uv lock --upgrade-package vectorbtx`, `uv sync --extra marsopt --extra desktop`, then `uv run python -I docs/upstream/vectorbtx_issues.py` (PRESENT or FIXED per item; exit code = number present), remove the workaround of each fixed item as its `docs/UPSTREAM.md` entry says, re-run `uv run pytest -rs` and `selfcheck`, and read each failure as a finding about the library. If the bump changes results on purpose, `test_golden_numbers_from_the_vectorbtx_example` fails: regenerate with `uv run python -I tests/fixtures/golden/make_golden.py ../vectorbtx --offline` (the saved prices, no download) and check that only the numbers the bump's changelog names moved. The phase 3 gates (`test_gate_matches_the_reference_simulator`, `test_vectorised_equals_one_by_one`) must keep passing; never loosen them. To iterate on an unpushed checkout, point `[tool.uv.sources]` at `vectorbtx = { path = "../vectorbtx", editable = true }` for the session, and restore the git source before committing.

**Next, in this order:**

1. **The redesign** (the owner's call, 13 November 2026). The owner's original vision for the Build page was an fxDreema-style canvas of blocks and links; the implementation plan (section 6) swapped it for stage-by-stage forms, which is what was built. Two parts:
   * **Styling: done on 13 November 2026.** The look lives in `themes/` (one `.toml` per page theme, plus the one stylesheet `studio.css`) and `style.py`; see the invariant "The look lives in one place" below. Five themes: Daylight (default), Midnight, Paper, Nord, Terminal. The Settings page picks one, previews it and makes an editable copy in the user's themes folder. Charts default to "Match the page theme" (`analysis.PAGE_THEME`), and the pipeline diagram takes the theme's colours. Checked in Edge (headless, 1440x900) on Home, Build (with its Pipeline tab), Backtest and Settings in all five themes, and on Data, Optimise, Results and Patterns in Midnight. Not done: page layout (Home is still the plain two-column page); `st.navigation` (icons and sections in the sidebar), which would move the entry point off `app/Home.py`; and the Build preview chart's long title, which runs into its legend in a narrow column whatever the theme (`fig_preview` and `finish`'s legend placement).
   * **The Build page's UX: done on 13 November 2026** (above). Next from the owner: ⓘ tooltips on every setting (Streamlit's `help=`), and more points to come.
   * **A blocks-and-links canvas** for features and rules: the owner asked to judge first whether the logic line and the At a glance panel make it unnecessary. If built, it is another view of the same spec: feature inputs are the links, conditions and logic the rule side. Typed ports (number series vs true/false), so only compilable links can be drawn. Start with a spike: a Streamlit custom component (`streamlit-flow-component` on React Flow, or `barfi` on Rete.js; ⚠️ VERIFY both against Streamlit 1.67) and a round-trip test that a spec goes onto the canvas and back unchanged. Style it from the page theme's tokens (`style.load_theme`), as the pipeline diagram is.
   * Small follow-ups: the Build preview chart's long title runs into its legend in its column; `st.navigation` for sidebar icons and sections.
2. **Still open from phase 0:** `data.apply_mapping` against a real MT4/MT5 export, and the installers on a clean machine (`docs/PACKAGING.md`).

## Layout (one module per concept)

| File | Concept |
|---|---|
| `spec.py` | The strategy spec (version 2, a plain dict): validate_spec, tunable_inputs, apply_params, choices, JSON, hashing |
| `indicators.py`, `indicator_catalogue.json` | The feature catalogue: menu placement and exclusions (dict literals), the generated snapshot, compute and its export source |
| `compile.py` | compile_signals, compile_weights, export_script (code generation) |
| `engine.py` | call_plan (pure) and run_backtest, vectorisable, run_vectorised: **the only importer of vectorbtx** |
| `exit_rules.py` | Numba templates for the time stop, breakeven and step trailing |
| `expressions.py` | The logic line (a tokenizer and recursive descent, never `eval`), conditions in words, the At a glance summary |
| `data.py` | Normalise, validate, import, store, load, synthetic data |
| `store.py` | The SQLite index of datasets, runs and jobs |
| `config.py` | settings.toml, environment, `OPENSTRATEGY_STUDIO_HOME` |
| `analysis.py` | Results extraction, saved runs, stats, every figure, chart themes (including "page"), the pipeline diagram |
| `style.py`, `themes/` | The look: page theme files, the stylesheet, the logo, the Settings preview, the diagram's palette |
| `optimise.py` | Spaces (flat or conditional), metrics, backends, walk-forward, stability, DSR, backend comparison |
| `jobs.py`, `worker.py` | Background jobs (files on disk, a detached process) |
| `research.py` | Pattern research |
| `ui.py` | Streamlit helpers: spec-bound widgets, the feature picker and editor, the rule editor, the glance panel, page help, guarded runs |
| `help/*.md` | The sidebar guide for each page (packaged, so easy to edit) |
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
* ⚠️ **A structural change renews every widget.** Adding or removing a condition, a feature or a variable, or switching a kind, calls `ui.renew_widgets` (inside `ui.rerun_after`): Streamlit keeps a keyed widget's own state, so without it the logic line keeps its old text and a reused letter shows (and writes back) a deleted condition's settings. `tests/test_app.py::test_build_steps_edit_the_spec` drives this.
* **Every value is a literal; ranges live in `optimise.variables` only,** keyed by the value's dotted path. `apply_params` writes a trial's values at those paths, drops `optimise`, and resolves every `{"choose", "options"}`, so the compiler never sees a range. Write variables directly (their keys contain dots), never through `set_path`.
* **One trading mode per strategy.** `setup.trading` is the vectorbtX mode; instruments carry none (`validate_spec` rejects one), and `engine.instrument_fields` adds it. A universe file keeps its own modes.
* **Every feature comes from the catalogue.** `indicator_catalogue.json` is generated, never edited: change the dict literals in `indicators.py` and rerun `python -m openstrategy_studio.indicators`; `tests/test_indicators.py` fails when the snapshot and the libraries disagree. Patterns come from ta_patterns only. pandas-ta's `offset` is never shown (a negative one is lookahead).
* ⚠️ **No catalogue entry looks ahead.** `test_no_entry_looks_ahead` computes every entry on full, truncated and different-future data; a failure goes to `EXCLUDED` with its reason (and upstream when it is a library's bug), never into the menu.
* **The logic line is parsed, never evaluated,** and brackets are required where two different operators meet. Every condition must appear in it.
* **Order timing masks signals on their own bar** (good for the day, flat by the day's end): never a shift. They are approximations until vectorbtX has the orders (`docs/UPSTREAM.md` issue 20).
* **The compiler shifts no signal.** The front door shifts entries, exits, short entries, short exits and the schedule to the next bar's open itself (vectorbtX `16293a2`). A shift in `compile_signals` or `export_script` would delay them two bars.
* **Always the four-array form.** Never `direction="both"`.
* **Stops are measured from the fill, pay the spread, and are checked on the entry bar.** vectorbtX's front door sets `stop_entry_price="price"` and `stop_exit_price="stopmarket"` and splits each bar into its open and the rest of the bar whenever a stop or target is active. `call_plan` must not pass either price; vectorbt's own defaults anchor stops at the entry bar's close and fill them with no slippage. The extra exits' callbacks run on the split bars too, through vectorbtX's bar view (called once per bar at the open, with bar indices): ⚠️ keep `exit_rules.py` templates written for whole bars, indexing their own arrays by `c.i`.
* **An FX pair never uses `SingleBacktest`.** `call_plan` routes any instrument with a base currency through `PortfolioBacktest(instruments=...)`, which keeps every field on the instrument. The reporting currency (`account.base_currency` in the spec) is passed as `account_currency=`, never `base_currency=`: on FX terms vectorbtX reads a flat `base_currency=` as the pair's base.
* **The benchmark is `pf.benchmark_value()`.** Never a second front-door call with no signals: that hold is funded at buying power (30x on an FX spread bet). Never pass `benchmark_rets=` to `returns_stats`.
* **Select one column as `pf[[column]]`.** It returns the same one-column shape a single run does. (The scalar form crashed every metric on a leveraged portfolio until vectorbtX `16293a2`.)
* **Never filter exits.** Filters are allowed in entry rules only; `validate_spec` enforces it.
* **Cash across symbols is always named.** For a rule strategy on two or more symbols (`spec.cash_model`), `call_plan` passes `cash=` (default `per_instrument`), plus `entry_priority=` and `when_cash_short=` when shared. Never pass `cash_sharing`, `group_by` or `call_seq` through `engine_extra`; `validate_spec` rejects them. A feature priority is a score frame from `compile_signals`, unshifted like the signals.
* **Separate accounts are reported as accounts.** `analysis.stats_tables` regroups ungrouped columns into one book of N x `init_cash` (vectorbt's `stats()` would average them silently); `per_symbol_table` shows each account. Never show the average.
* **No higher-timeframe leakage.** The filter needs `label="left", closed="left"` and then `.shift(1)`. `tests/test_lookahead.py` uses dense truncation cut points; keep them dense.
* **Portfolio weights stay full-shape.** Never forward-fill them. Always pass `schedule=` to `drifted_weights`.
* **One cost path.** Costs come only from instrument fields, never a raw `fees=` (it collides with commission and raises).
* **The borrow fee follows how a thing is traded.** `borrow_rate` belongs to equity mode only (a direct short borrows what it sells); a CFD or spread bet short pays its funding instead, and `validate_spec` says so. Never put a default rate in a template: none is published, and a blank one warns, which is the point.
* **Financing is one rule for every instrument:** the account's rates plus the Settings page's markup. There is no per-instrument financing (vectorbtX keeps the field reserved); `engine.instrument_fields` drops one an older spec holds, and the grid has no such column.
* **vectorbtX's warnings reach the page.** It reports what it could not price (a short borrowed for free, a carry leg with no currency) as Python warnings, which never reach Streamlit on their own. `run_backtest` collects each run's (`"always"`, or a long-lived server would show each once), `extract_results` keeps them in `meta["warnings"]`, and the Backtest page shows them. Never silence or drop them.
* **Charts.** `st.plotly_chart(fig, theme=None)` everywhere (via `ui.chart`). Every figure ends with `analysis.finish(fig, theme)`. The "page" chart theme is an ordinary template in plotly's registry: `analysis` registers it from the default page theme at import, and `ui.boot` re-points it at the page's own on every run.
* **The look lives in one place.** Colours, fonts, corners and spacing come only from the page theme file (`themes/<name>.toml`: `[theme]` for Streamlit, `[studio]` for the Studio's tokens) and the one stylesheet (`themes/studio.css`), which names no colour of its own, only `--os-*` variables. Pages set no colour and inject no CSS; anything that draws (charts, the diagram, a future canvas) reads the tokens through `style`. The stylesheet targets `data-testid` attributes, never `st-emotion-cache-*` classes.
* **The page theme is fixed per server start.** `cli.run_app` passes `--theme.base <file>` (never `.streamlit/config.toml`, which is read from the working directory) and records the file in `style.ACTIVE_ENV`; pages read that same file, so the stylesheet always matches Streamlit's colours. `st.set_option` cannot change theme options in 1.67, so a theme picked in Settings applies at the next start. A missing or broken theme file falls back to Daylight with a message; it never stops the Studio.
* ⚠️ **No `<` before a letter in the stylesheet**, even in a comment: `st.html`'s DOMPurify silently drops the whole `<style>`. `style.stylesheet` escapes `<` as `\3c `, and `tests/test_style.py` checks.
* **Export and in-app runs share one plan.** `export_script` and `run_backtest` both use `engine.call_plan`. `tests/test_export.py` keeps them equal.
* **Rendering must not change the spec.** A page rendered with defaults must leave the spec unchanged. Number widgets keep an integer range integer (`ui.param_input`).
* **Paths.** Use pathlib everywhere, never shell commands. All user data lives under `config.data_dir`.
* **Starting processes.** Use `cli.self_command`, which also works in a frozen bundle.
* **Files two processes share** (job folders) are written atomically with `jobs.write_json` / `jobs.write_parquet` and read through `jobs.retry_while_locked`: on Windows a file being replaced, or held open by the other process, briefly refuses access.
* **No HTTPS context before `config.apply_environment`.** On Windows, uv's CPython aborts the whole process when `SSLKEYLOGFILE` is set and any HTTPS context is created; `apply_environment` drops it. If a test run dies with no summary and `OPENSSL_Uplink` in the output, that is the cause.

## Testing conventions

Use plain pytest functions with no classes or mocking frameworks, and no network. Synthetic data comes from `tests/conftest.py::make_ohlcv`. Tests that need a library call `pytest.importorskip` inside the test.

## Known gaps

* The instrument grid shows only the fields the strategy's trading mode accepts; it still cannot grey out single cells.
* Cap weighting and long/short selection are not offered: there is no market-cap source, and negative weights are unverified.
* Conditional spaces work with BUTChC and Optuna only; the other backends fix a rule at its version in use.
* Only the grid uses a variable's step; the other searchers pick any value in the range (whole numbers for a whole-number setting).
* The PyInstaller bundle is a spike (`docs/PACKAGING.md`). The first-run installer is the shipped route.
