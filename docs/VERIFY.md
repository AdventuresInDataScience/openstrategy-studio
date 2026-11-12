# Phase 0: confirm before relying on it

This code was written without vectorbt, vectorbtX, ta_patterns, BUTChC, Optuna, Streamlit, plotly or pyarrow installed, and without network access. The pure pandas, numpy and scipy parts were run and tested. Everything that calls those libraries was written from the plan's verified facts and the libraries' documented signatures, but has not been executed.

The roadmap's phase 0 gate is: **every item below has a passing test or a written answer.** Work through it once after the first `uv sync`.

## 1. Run the suite and read the skips

```bash
uv sync
uv run pytest -rs
```

With every library installed, nothing should skip except `test_marsopt` (unless you installed the extra with `uv sync --extra marsopt`). Each test that fails is a finding to record here, not just a bug to fix: the plan's TEST items are exactly the places where the source was read but not run.

The one failure that is expected until you act on it is `test_same_bar_stop_and_target` (item 2.4).

## 2. The plan's TEST items (section 3)

| # | Question | Test that answers it | Answer |
|---|---|---|---|
| 2.1 | Short signals are not shifted by the front door, so the Studio shifts them. Does a short signal at bar t fill at bar t+1's open? | `test_engine.py::test_short_signal_fills_next_open` | |
| 2.2 | Long signals fill at the next open and are not shifted twice. | `test_engine.py::test_long_signal_fills_next_open` | |
| 2.3 | A bar that opens beyond the stop fills at the open. | `test_engine.py::test_gap_through_stop_fills_at_open` | |
| 2.4 | When one bar holds both the stop and the target, which wins? Read `simulate_from_signals_nb` (around `vectorbt/portfolio/nb.py` line 2381). | `test_engine.py::test_same_bar_stop_and_target` fails with the observed answer. Set `engine.SAME_BAR_STOP_RULE` to it, and the Risk tab will document it. | |
| 2.5 | What does a weight matrix with gross exposure above 1 actually do? Leverage stays off the portfolio path until this is known (`validate_spec` rejects fixed weights above 100%). | No test yet: try a 1.5-gross weight matrix by hand. | |
| 2.6 | Adjusted equity prices: close from `adj_close`, with open, high and low scaled by the same factor. Check against a stock with a dividend. | Run AAPL through the Data page and compare with Yahoo's adjusted series. | |
| 2.7 | How should buy-and-hold benchmark returns be handed to `returns_stats`? (`benchmark_rets=` is used now.) | `analysis.stats_tables`; check that m2 and the Treynor ratio are no longer NaN. | |

## 3. Unverified call sites in the code

Each is marked `⚠️ VERIFY` at the line named.

| Where | Assumption |
|---|---|
| `engine.mode_rules` | The shape of `MODE_RULES` in `vectorbtx/frontend/prepare.py`. Keys containing "require" are read as required fields, and keys containing "reject", "forbid" or "disallow" as rejected fields. If the shape differs, every instrument field simply stays enabled. |
| `engine.instrument_fields` | `price_field` is consumed by the loader and NOT sent to the front door. Confirm the front door does not need it. |
| `engine.load_universe` | `vbt.x.load_universe` accepts a .csv or .xlsx path, as the front door's `instruments=` does. |
| `engine.benchmark_plan` | A front-door call with neither signals nor weights is buy and hold, and spread-bet or CFD modes size it sensibly without a sizing argument. |
| `engine.prepare_data` | `to_base_currency` is called with the frame positionally, then `base=`, `fx=` and `instruments=`. `fx_markup` is left at the function's default; check whether it should read `vbt.settings["vectorbtx"]["fx_markup"]`. |
| `engine` construct wrappers | `drifted_weights(weights, prices, schedule=)` and `threshold_schedule(drifted, weights, tolerance=)` take their first arguments positionally, as the examples do. |
| `optimise._from_stats` | The vectorbtX metrics (K-ratio v3, Ulcer performance index, geometric Sharpe, recovery factor, GHPR) are read from `pf.stats(tags="vectorbtx")` by label words. Replace them with direct accessors once their names are confirmed. |
| `optimise.METRICS` and `trial_extras` | `group_by=True` is accepted by `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `total_return`, `max_drawdown` and the trades methods on vectorbtX's portfolio. |
| `optimise.periods_per_year` | vectorbtX keeps vectorbt's default `year_freq` of 365 days (used to de-annualise Sharpe ratios for the deflated Sharpe ratio). |
| `optimise.run_optuna` | Ask-and-tell with `study.ask(distributions)`, and SQLite storage per job (one study per walk-forward fold). |
| `optimise.run_butchc` | An executor that is a `SimpleNamespace` with an order-preserving `map` is accepted, and `result["prob_tree"]` exists. |
| `optimise.run_marsopt` | `Study(direction=, random_state=, verbose=)` and the `Trial.suggest_*` names. |
| `data.download_yahoo` | vectorbt 1.1.0's `YFData.download` passes `interval=` and `auto_adjust=False` on to yfinance, and a single symbol's `get()` returns the field frame. |
| `data.apply_mapping` | Against a real MT4/MT5 history export (the fixture in `tests/fixtures/` is hand-made). |
| `blocks.pattern_catalogue` | `ta_patterns.scanner`'s BULLISH, BEARISH, DIRECTIONAL and NON_DIRECTIONAL sets hold candlestick names, and `ta_patterns.chart_patterns`' sets hold chart names. |
| `analysis.stats_tables` | `pf.stats(settings={"init_cash": pf.get_init_cash()})` is how a resolved `init_cash` reaches `absolute_drawdown` and `recovery_factor`. |
| `analysis.extract_results` | `pf.x.intrabar_drawdown()`, `pf.x.user_size()`, `pf.x.margin_level()`, `pf.x.borrowed_portion()`, `pf.x.drift()` and `pf.holding_costs`. Missing ones are hidden by design, so check that the figures you expect actually appear. |
| `analysis.mae_mfe` | vectorbt 1.1.0 really has no native MAE/MFE. |
| `analysis.fig_rolling` | Rolling metrics are computed in pandas. Swap in vectorbtX's `rolling_*` methods (API.md section 11) once their names are confirmed. |
| `ui.chart`, `ui.table` | `st.plotly_chart(..., width="stretch")` and `st.dataframe(..., width="stretch")` on the installed Streamlit (pinned at 1.50 or later). |
| `tests/test_app.py` | `streamlit.testing.v1.AppTest` is still the API for app tests. |

### Phase 3 and 4 additions

| Where | Assumption |
|---|---|
| `exit_rules.adjust_sl_nb` / `adjust_tp_nb` | vectorbt 1.1.0's callback contexts carry the 0.2x field names: `i, col, position_now, init_i, init_price, curr_price, curr_stop, curr_trail` for the stop, and `i, col, position_now, init_i, init_price, curr_stop` for the target. |
| `exit_rules` | The adjustment runs BEFORE bar i's stop check. A trailing stop's `curr_price` is updated after the check, from completed bars. Negative stop fractions are rejected, which is why stops above the entry use the trailing reference. |
| `engine.call_plan` (extra exits) | The front door passes `adjust_sl_func_nb`, `adjust_sl_args`, `adjust_tp_func_nb` and `adjust_tp_args` through to `Portfolio.from_signals`, and accepts a far `target_points` (1e9) so the time stop's target callback runs. |
| `engine.run_vectorised` | `PortfolioBacktest` accepts `cash_sharing=False` and `group_by=False`, takes instruments keyed by arbitrary column names (`trial_0`, `trial_1` ...), and `pf[column]` selects one trial. |
| `optimise.run_optuna` | `study.ask()` with no distributions, then `trial.suggest_*` on the returned trial, before `study.tell`. |
| `optimise.run_butchc` (conditional) | String option labels as `values` and as `next_level` keys. |
| `ui.rule_editor`, Build page | `st.graphviz_chart` renders offline (Streamlit bundles its renderer). |
| `desktop.py` | pywebview 5's `create_window(..., min_size=)` and `start()`; Streamlit's `/_stcore/health` endpoint. |
| `packaging/` | Everything in `docs/PACKAGING.md`, checked on a clean machine. |

The two phase 3 gates are tests:

* `tests/test_exit_rules.py::test_gate_matches_the_reference_simulator` runs hand-built bars through vectorbtX and compares them with a simulator that encodes the assumptions above. A failure means vectorbt's order of operations differs: adjust the templates, not the test.
* `tests/test_vectorised.py::test_vectorised_equals_one_by_one` requires each trial in one call to equal that trial run alone. Until it passes, do not use the "Grid, many trials per call" backend.

## 4. Golden numbers

`test_golden_ma_cross` compares the compiled MA-cross spec with the same strategy written as a direct vectorbtX call. When you run `examples/strategies/01_ma_cross_spreadbet_fx.py` on your machine, save its `pf.stats()`, `pf.trades.stats()` and `pf.returns_stats()` as text in `tests/fixtures/golden/`, and add a test that runs the template on the same Yahoo data and compares the two.

Note that the template uses a static rate table so that it runs without a FRED key; the example uses FRED. Switch the template's `account.rates` to `{"source": "fred"}` for that comparison.

## 5. Decisions taken where the plan left a choice

* **compile.py and vectorbtX.** compile.py needs vectorbtX's schedule and weighting helpers, but engine.py is the only file allowed to import vectorbtX. They are therefore thin wrappers in engine.py.
* **Reporting currency.** A single-symbol spec with a reporting currency (`account.base_currency`) runs through `PortfolioBacktest`. `SingleBacktest`'s flat `base_currency` keyword is the instrument's own (EUR for EURUSD).
* **Filters.** Filters are leaves allowed only in entry rules, and `validate_spec` rejects them in exits.
* **Selection is optional.** A portfolio with no selection holds every asset, which is what the 60/40 template needs. Fixed weights are used as given and are not re-normalised.
* **Weights left out.** Cap weighting (there is no market-cap data source) and long/short selection (negative weights are unverified) are not offered yet.
* **References can shift.** A reference can carry its own `shift` (`{"feature": "upper", "shift": 1}`), so a breakout can compare with the previous bar's channel. This is in addition to the leaf-level `shift`.
* **Structure search uses choice nodes.** A choice is `{"choose": label or optimisable categorical, "options": {label: subtree}}`, and it can sit anywhere in the spec. `apply_params` resolves choices, so nothing downstream sees them. `validate_spec` checks every alternative. BUTChC and Optuna search the conditional space; the other backends fix the structure at the current choice.
* **The extra exits need a fixed stop.** They adjust an existing stop, and vectorbt only calls the adjustment while a stop is active.
* **The vectorised grid is limited.** It only varies parameters that change the signals, on a single symbol.
