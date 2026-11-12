# CLAUDE.md: working on OpenStrategy Studio

A Streamlit app in which a person defines a trading strategy without code, then backtests and optimises it on vectorbtX (`vbt.x.SingleBacktest`, `vbt.x.PortfolioBacktest`) and vectorbt 1.1.0. All five phases of the implementation plan are written. **None of the code that calls vectorbt, vectorbtX, ta_patterns, BUTChC, Optuna, Streamlit or plotly has been executed yet**: it was written without those libraries or network access. Your first job is phase 0 (below).

## Commands

```bash
uv lock                          # once, if uv.lock is missing; commit it
uv sync                          # add --extra marsopt / --extra desktop as needed
uv run pytest -rs                # -rs lists skips; with everything installed only marsopt may skip
uv run pytest tests/test_engine.py -x -q
uv run openstrategy-studio       # the app; also: desktop, selfcheck, worker <dir>
uv run openstrategy-studio selfcheck --quick
```

Never `pip install`. Add dependencies with `uv add`.

## Do this first: phase 0

1. `uv sync`, then `uv run pytest -rs`. Read every failure as a finding about the libraries, not just a bug: the tests encode the plan's TEST items.
2. Expected failure: `test_engine.py::test_same_bar_stop_and_target` reports which of stop or target wins inside one bar. Set `engine.SAME_BAR_STOP_RULE` to that answer.
3. Work through `docs/VERIFY.md`. Every `⚠️ VERIFY` comment in the code is listed there with the test that settles it. Fix each mismatch in the ONE place it lives (usually `engine.py`), then fill in the Answer column.
4. The phase 3 gates: `tests/test_exit_rules.py::test_gate_matches_the_reference_simulator` (the numba exit templates against vectorbt's real stop handling) and `tests/test_vectorised.py::test_vectorised_equals_one_by_one`. If a gate fails, fix the template or keep the feature labelled experimental. Do not loosen the test.
5. `uv run openstrategy-studio selfcheck` must pass. That is the automated half of the phase 4 gate (`docs/PACKAGING.md`).

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
* **Never filter exits.** Filters are allowed in entry rules only; `validate_spec` enforces it.
* **No higher-timeframe leakage.** The filter needs `label="left", closed="left"` and then `.shift(1)`. `tests/test_lookahead.py` uses dense truncation cut points; keep them dense.
* **Portfolio weights stay full-shape.** Never forward-fill them. Always pass `schedule=` to `drifted_weights`.
* **One cost path.** Costs come only from instrument fields, never a raw `fees=` (it collides with commission and raises).
* **Charts.** `st.plotly_chart(fig, theme=None)` everywhere (via `ui.chart`). Every figure ends with `analysis.finish(fig, theme)`.
* **Export and in-app runs share one plan.** `export_script` and `run_backtest` both use `engine.call_plan`. `tests/test_export.py` keeps them equal.
* **Rendering must not change the spec.** A page rendered with defaults must leave the spec unchanged. Number widgets keep an integer range integer (`ui.param_input`).
* **Paths.** Use pathlib everywhere, never shell commands. All user data lives under `config.data_dir`.
* **Starting processes.** Use `cli.self_command`, which also works in a frozen bundle.

## Testing conventions

Use plain pytest functions with no classes or mocking frameworks, and no network. Synthetic data comes from `tests/conftest.py::make_ohlcv`. Tests that need a library call `pytest.importorskip` inside the test.

## Known gaps

* The instrument grid cannot grey out individual cells, so it warns about fields a mode rejects instead.
* Cap weighting and long/short selection are not offered: there is no market-cap source, and negative weights are unverified.
* Conditional spaces work with BUTChC and Optuna only; the other backends fix alternatives at their current choice.
* The PyInstaller bundle is a spike (`docs/PACKAGING.md`). The first-run installer is the shipped route.
