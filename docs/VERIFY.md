# Phase 0: confirm before relying on it

This code was first written without vectorbt, vectorbtX, ta_patterns, BUTChC, Optuna, Streamlit, plotly or pyarrow installed, and without network access. **Phase 0 has now been run** (12 November 2026, Windows 11, Python 3.12.11) against:

| Package | Version |
|---|---|
| vectorbt | 1.1.0 |
| vectorbtX | 0.1.0, commit `9fa427e` (the commit `uv.lock` pins; phase 0 itself ran on `ace29e2`) |
| numba / numpy / pandas | 0.68.1 / 2.5.4 / 3.1.1 |
| Streamlit / plotly | 1.67.0 / 6.9.0 |
| ta_patterns / BUTChC / Optuna / marsopt | 1.2.1 / 0.6.1 / 5.0.0 / 0.2.0 |
| pywebview | 6.2.1 (the `desktop` extra resolves to 6, not 5) |

Result: `uv run pytest -rs` passes with every extra installed (`uv sync --extra marsopt --extra desktop`). The only skip is the bash installer check on Windows. `uv run openstrategy-studio selfcheck` prints "All checks passed". Both phase 3 gates pass, so the extra exits and the vectorised grid are no longer labelled experimental.

**Re-run on vectorbtX `16293a2`** (12 November 2026), which fixes all 13 items in `docs/UPSTREAM.md`: `docs/upstream/vectorbtx_issues.py` reports 0 of 13 present. With the new library and the old workarounds still in place, 4 tests failed, each an expected consequence: the short double shift (issue 1, two tests, one of them the exit-rules gate's short case), the intrabar drawdown now agreeing with the Studio's (issue 11), and carry now including its markup (issue 4). With the workarounds removed (issues 1, 5, 6, 9 and 11), `uv run pytest -rs` gives 133 passed and 1 skipped, both phase 3 gates pass unchanged, and `selfcheck` passes. One new limitation came with the fix: see 2.10 and `docs/UPSTREAM.md` issue 14.

**Re-run on vectorbtX `9f4cf54`** (12 November 2026), which fixes issue 14. On it, only the test pinning that gap failed, as intended. It is replaced by a positive test, and the exit-rules gate gained four entry-bar cases, which pass on `9f4cf54` and fail on `16293a2`. `uv run pytest -rs` gives 134 passed and 1 skipped, both phase 3 gates pass, and `selfcheck` passes.

**Re-run on vectorbtX `9fa427e`** (13 November 2026), which fixes `docs/UPSTREAM.md` issues 15 to 17 and 19 and builds 18 (the borrow fee on a direct short): the script reports 0 of 18 present, and 15 to 19 present against `9f4cf54`. With the new library, only `test_golden_numbers_from_the_vectorbtx_example` failed, by design (issue 15 puts carry into End Value and Total Return); regenerated on the same prices (section 4), exactly those two numbers moved. Three Studio tests short a synthetic equity instrument with no `borrow_rate`, so vectorbtX now warns in them, as intended. With the Studio side of 18 and 19 done (the `borrow_rate` grid column, no financing column, vectorbtX's warnings on the Backtest page), `uv run pytest -rs` gives 151 passed and 1 skipped, both phase 3 gates pass, and `selfcheck` passes.

The phase 0 gate is: **every item below has a passing test or a written answer.** Two items need things this machine does not have (a real MT4/MT5 export, and a clean machine for the installers); they are marked ⏳.

## 1. Run the suite and read the skips

```bash
uv sync                          # add --extra marsopt --extra desktop for everything
uv run pytest -rs
```

The first run found 14 failures. Each one was a finding about the libraries or about Windows, and section 6 lists what changed.

## 2. The plan's TEST items (section 3)

| # | Question | Test that answers it | Answer |
|---|---|---|---|
| 2.1 | Does a short signal at bar t fill at bar t+1's open? | `test_engine.py::test_short_signal_fills_next_open` | ✅ Yes, at bar t+1's open. The first failure was the spread: equity mode defaults to `spread_points=1.0`, charged as half a point on each fill. The timing tests now use a zero-spread instrument, and `test_spread_is_paid_on_every_fill` pins the cost. On `ace29e2` the front door did not shift shorts (`docs/UPSTREAM.md`, issue 1), so the compiler did; since `16293a2` the front door shifts all four arrays and the compiler shifts none (`test_lookahead.py::test_the_compiler_shifts_no_signal`). |
| 2.2 | Long signals fill at the next open and are not shifted twice. | `test_engine.py::test_long_signal_fills_next_open` | ✅ Yes (same spread note as 2.1). |
| 2.3 | A bar that opens beyond the stop fills at the open. | `test_engine.py::test_gap_through_stop_fills_at_open` | ✅ Yes. With the stop fix of 2.9, it also pays half the spread. |
| 2.4 | When one bar holds both the stop and the target, which wins? | `test_engine.py::test_same_bar_stop_and_target` | ✅ **The stop.** vectorbt computes the stop-loss price first and only looks at the take-profit when no stop-loss price was hit (in both `simulate_from_signals_nb` and `simulate_from_signal_func_nb`). `engine.SAME_BAR_STOP_RULE = "stop"`; the Risk tab says so. |
| 2.5 | What does a weight matrix with gross exposure above 1 actually do? | By hand (0.75 + 0.75 on two assets) | ✅ vectorbtX warns ("Target weights sum to 1.50 ... the excess will not fill"), then fills **in column order until the cash runs out**: the first asset got its full 0.75, the second about 0.25, for 1.0x gross in total. So weights above 100% do not give leverage on the portfolio path; they skew the book. `validate_spec` keeps rejecting them. Since `16293a2` vectorbtX scales such rows pro rata instead (0.50 / 0.50) and says so; the rejection stays. |
| 2.6 | Adjusted equity prices: close from `adj_close`, with open, high and low scaled by the same factor. | AAPL 2024 through `data.download_yahoo`, `normalise_ohlcv`, `store_dataset`, `load_frames(adjusted=...)`, against yfinance's `auto_adjust=True` | ✅ Open, high, low and close all match Yahoo's adjusted series to within 3.3e-7 (float rounding) over a window with four dividends. |
| 2.7 | How should buy-and-hold benchmark returns be handed to `returns_stats`? | `test_engine.py::test_benchmark_is_the_market_unlevered` | ✅ **They should not be.** `returns_stats()` already benchmarks against `pf.benchmark_returns()`: the market, an unlevered equal-weight hold of the closes. Beta, alpha, M2 and the Treynor ratio come out finite. The Studio's old benchmark (a front-door call with no signals) was **funded at buying power**: an FX spread bet "held" at 30x lost 247% while the market fell 6.5%, and its returns turned M2 into 786% and the Treynor ratio into -120. `engine.benchmark_plan` is gone; the buy-and-hold line is `pf.benchmark_value()`. |
| 2.8 | *(new)* Is a stop in points measured from the price the order filled at? | `test_engine.py::test_stop_is_measured_from_the_fill_open` | ❌→✅ **Not by default.** vectorbt's `stop_entry_price` defaults to `"close"`, so the stop fraction (which vectorbtX computed from the fill open) was anchored at the entry bar's CLOSE. A 2-point stop on a buy filled at 100, on a bar that closed at 103, sat at 100.94: above the entry. The engine passed `stop_entry_price="price"`; since `16293a2` vectorbtX's front door sets it, and the engine passes nothing. |
| 2.9 | *(new)* Does a stopped-out trade pay the spread on exit? | `test_engine.py::test_spread_is_paid_on_every_fill` | ❌→✅ **Not by default.** vectorbt's `stop_exit_price` defaults to `"stoplimit"`, which fills with zero slippage, and vectorbtX puts the spread into slippage. A stopped-out trade paid half the spread a signal exit pays, which flatters every strategy that exits by stops, spread bets most of all. The engine passed `stop_exit_price="stopmarket"`; since `16293a2` vectorbtX's front door sets it. |
| 2.10 | *(new, `16293a2`)* Is a stop checked on the bar the position is entered? | `test_engine.py::test_stop_is_checked_on_the_entry_bar`; `test_exit_rules.py::test_extra_exits_check_the_entry_bar_without_a_warning` and the entry-bar cases of `test_gate_matches_the_reference_simulator` | ✅ **Yes, with or without an extra exit.** vectorbt checks a position's stops from the bar after entry, so a stop the entry bar traded through was missed (always in the strategy's favour, worst for tight stops on daily bars). vectorbtX now simulates each bar as its open plus the rest of the bar whenever a stop or target is active, so a buy at 100 on a bar that trades to 97 is stopped at 98 that bar. In `16293a2` that did not apply with a custom callback, so the extra exits kept the bias (`docs/UPSTREAM.md`, issue 14): a time-stop trade whose entry bar fell to 79 against a 20-point stop was reported flat at 100 four bars later instead of stopped at 80. `9f4cf54` wraps each callback so it is still called once per bar at the open, with bar indices, and the entry bar is checked. An exit signal at the open now also beats an intrabar stop on the same bar. |

## 3. Call sites that were unverified

Each was marked `⚠️ VERIFY` in the code. The markers are gone where the answer is ✅; the code comments now state the verified fact.

| Where | Assumption | Answer |
|---|---|---|
| `engine.mode_rules` | The shape of `MODE_RULES`. | ✅ `{mode: {"required": (...), "rejected": (...)}}`. Reasons are kept separately in `_REJECT_REASONS`. `base_currency` is rejected outside `fx_spot` unless `asset_class == "fx"`. |
| `engine.instrument_fields` | `price_field` is consumed by the loader, not sent. | ✅ vectorbtX uses it only for defaults and validation. |
| `engine.load_universe` | `vbt.x.load_universe` accepts a .csv or .xlsx path. | ✅ .csv, .xlsx and .xls. |
| `engine.benchmark_plan` | A call with neither signals nor weights is buy and hold, sized sensibly. | ❌ It holds at buying power: 30x on the FX spread bet, 2x on equities. Removed; see 2.7. |
| `engine.prepare_data` | `to_base_currency(frame, base=, fx=, instruments=)`; `fx_markup` left at its default. | ✅ The signature matches. On `ace29e2` the markup was never charged (`docs/UPSTREAM.md`, issue 9). ✅ Since `16293a2` it is, from `vbt.settings["vectorbtx"]["fx_markup"]` (the Settings page), including in a walk-forward window: `test_fx_conversion_markup_is_charged_in_a_reporting_currency`. The account currency is now passed as `account_currency=`. |
| `engine` construct wrappers | `drifted_weights` and `threshold_schedule` take their first arguments positionally. | ✅ `tests/test_portfolio.py`. |
| `engine.call_plan` (single symbol) | *(new)* `SingleBacktest`'s flat `base_currency` keyword is the instrument's. | ❌→✅ **It is the ACCOUNT currency.** `PortfolioBacktest` names `base_currency` as the account currency, so the flat form swallows it. An EURUSD spread bet then lost its pair: carry resolved to zero (with a "no currency pair" warning) and it was charged financing instead, 3x the correct carry at EUR 3% and USD 4.5%. An instrument that names its base currency now goes through `PortfolioBacktest(instruments=...)`. That affects the ma_cross, donchian_breakout, candlestick_reversal and structure_search templates. Pinned by `test_fx_pair_routes_to_portfolio` and `test_fx_carry_uses_the_pair`. `16293a2` fixes the flat form (on an FX instrument it now names the pair); the route stays, and carry now includes the carry markup (issue 4). |
| `optimise._from_stats` | The vectorbtX metrics can be read from `pf.stats(tags="vectorbtx")` by label. | ❌→✅ That table holds only recovery factor and GHPR, so **K-ratio, the Ulcer performance index and geometric Sharpe failed every trial**. All five now use direct accessors (`pf.get_returns_acc(group_by=True).k_ratio_v3()` and so on; `pf.trades.recovery_factor(group_by=True)`, `pf.trades.ghpr(group_by=True)`), each equal to its stats-table entry. `test_every_metric_scores_a_real_backtest`. |
| `optimise.METRICS` and `trial_extras` | `group_by=True` is accepted everywhere. | ✅ |
| `optimise.periods_per_year` | vectorbtX keeps `year_freq` at 365 days. | ✅ `vbt.settings["returns"]["year_freq"]` is 365 days. |
| `optimise.run_optuna` | Ask-and-tell, and SQLite storage per job (one study per fold). | ✅ On Optuna 5.0.0, two fold studies share one SQLite file. Only the worker's main process asks and tells, so there are no concurrent writers. |
| `optimise.run_butchc` | A `SimpleNamespace` executor is accepted, and `result["prob_tree"]` exists. | ✅ Both. The reproducibility test compared lists holding NaN (`NaN != NaN`); it now compares the `repr`. |
| `optimise.run_marsopt` | `Study(direction=, random_state=, verbose=)` and the `Trial.suggest_*` names. | ✅ marsopt 0.2.0 installs from a wheel on Windows / Python 3.12. It samples the NaN trap region in under 1% of trials, so the test uses seed 1, which visits it. |
| `data.download_yahoo` | `interval=` and `auto_adjust=False` reach yfinance; one symbol's `get()` is the field frame. | ✅ `YFData.download_symbol` passes `**kwargs` to `Ticker.history`; `get()` returns Open ... Adj Close, Volume, Dividends, Stock Splits. |
| `data.apply_mapping` | Against a real MT4/MT5 history export. | ⏳ Open: the fixture in `tests/fixtures/` is hand-made. Export one from a terminal and add it. |
| `blocks.pattern_catalogue` | The ta_patterns sets hold candlestick and chart names. | ✅ 106 candlestick names and 194 chart names. The picker offers 299, because `chart_patterns` lists `double_bottom_ugly` but the function is `ugly_double_bottom` (a ta_patterns naming bug, listed in `docs/UPSTREAM.md`). |
| `analysis.stats_tables` | `settings={"init_cash": pf.get_init_cash()}` reaches `absolute_drawdown` and `recovery_factor`. | ✅ Both finite. ❌→✅ `returns_stats(tags="mt5")` returns `None`, not an empty table (no returns metric carries that tag); it is now read as empty. |
| `analysis.extract_results` | The `pf.x` extras and `pf.holding_costs`. | ✅ `user_size`, `margin_level`, `borrowed_portion` and `drift` exist with the assumed shapes; `pf.holding_costs` is a property. ❌→✅ On `ace29e2` `pf.x.intrabar_drawdown()` returned one float, not a curve, and was wrong for shared cash and shorts (`docs/UPSTREAM.md`, issue 11), so the Studio drew its own. Since `16293a2` it draws vectorbtX's `pf.x.intrabar_drawdown_series()`, which agrees with the old curve exactly. |
| `analysis.mae_mfe` | vectorbt 1.1.0 has no native MAE/MFE. | ✅ None. |
| `analysis.fig_rolling` | Swap in vectorbtX's `rolling_*` methods. | ✅ They are `rolling_ulcer_index`, `rolling_geometric_sharpe_ratio`, `rolling_snr` and `rolling_k_ratio` on the returns accessor. The figure keeps its pandas rolling Sharpe and volatility: vectorbtX has neither, and the figure is drawn from saved runs, which hold returns, not a portfolio. |
| `ui.chart`, `ui.table` | `width="stretch"` on the installed Streamlit. | ✅ It is the default on 1.67. |
| `tests/test_app.py` | `streamlit.testing.v1.AppTest` is still the API. | ✅ |

### Phase 3 and 4 additions

| Where | Assumption | Answer |
|---|---|---|
| `exit_rules.adjust_sl_nb` / `adjust_tp_nb` | The callback contexts carry the 0.2x field names. | ✅ `AdjustSLContext` and `AdjustTPContext` have them (plus `val_price_now` and `curr_i`, unused). The reference simulator called the `@njit` functions with a `SimpleNamespace`, which numba cannot type; it now calls `.py_func`. |
| `exit_rules` | The adjustment runs before bar i's stop check; the trailing reference updates after it from completed bars. | ✅ The callbacks exist only in `simulate_from_signal_func_nb`, which `from_signals` uses whenever an adjust function is given. Order: adjust SL, adjust TP, stop check, then the trailing reference from bar i's high or low. |
| `engine.call_plan` (extra exits) | The front door passes the callbacks through and accepts a far target. | ✅ The gate passes. |
| `engine.run_vectorised` | `cash_sharing=False`, `group_by=False`, arbitrary column names, and `pf[column]` selects one trial. | ✅ except the last: **`pf[column]` crashes any metric** on a leveraged instrument (`docs/UPSTREAM.md`, issue 10, fixed in `16293a2`). `run_vectorised` returns one portfolio per trial, selected as `pf[[column]]`. With the FX fix above, each trial equals the trial run alone. |
| `optimise.run_optuna` | `study.ask()` with no distributions, then `trial.suggest_*`. | ✅ |
| `optimise.run_butchc` (conditional) | String option labels as `values` and `next_level` keys. | ✅ `test_butchc_conditional_space`. |
| `ui.rule_editor`, Build page | `st.graphviz_chart` renders offline. | ✅ The renderer is bundled in Streamlit's static files (no CDN). Not watched in a browser with the network off. |
| `desktop.py` | pywebview's `create_window(..., min_size=)` and `start()`; the `/_stcore/health` endpoint. | ✅ pywebview 6.2.1 keeps both; the health poll is tested. A window was not opened during phase 0. The health check now uses `http.client`: `urllib.request.urlopen` builds an HTTPS context even for `http://`, which aborts the process on Windows when `SSLKEYLOGFILE` is set (section 6). |
| `packaging/` | Everything in `docs/PACKAGING.md`, checked on a clean machine. | ⏳ Open: needs a clean machine. |

## 4. Golden numbers

`test_golden_ma_cross` compares the compiled MA-cross spec with the same strategy written as a direct vectorbtX call, and passes. The direct call is `PortfolioBacktest(instruments={"EURUSD=X": {...}})`, with no stop price overrides: since `16293a2` the front door sets them.

`examples/strategies/01_ma_cross_spreadbet_fx.py` in vectorbtX passes `base_currency="EUR"` to `SingleBacktest`. On `ace29e2` that hit `docs/UPSTREAM.md` issue 2 and charged a long the whole USD rate. `16293a2` reads it as the pair, as the example's comment says.

**Saved on 13 November 2026** (vectorbtX `9f4cf54`). `tests/fixtures/golden/make_golden.py` runs the example file itself, unmodified, on real Yahoo data (1,564 EURUSD bars, 2019 to 2024), with the one change the example offers for running without a FRED key: its static rate table, EUR 4% and USD 5.3%. It saves the prices it downloaded (`eurusd_2019_2024_yahoo.csv`) and the 104 numbers in `pf.stats()`, `pf.trades.stats()` and `pf.returns_stats()` (`ma_cross_01.json`). `test_golden_numbers_from_the_vectorbtx_example` runs the Studio's MA-cross template on those prices with the same rates, offline, and agrees with every number to 1e-9. The example's flat `SingleBacktest` and the Studio's `PortfolioBacktest(instruments=...)` therefore give the same result on real data.

**Regenerated on 13 November 2026** (vectorbtX `9fa427e`, which fixes `docs/UPSTREAM.md` issue 15: `stats()` left carry out of Total Return and End Value). Run with `--offline`, which serves the saved prices to the example's own Yahoo download, so the bars are the same and Yahoo's revisions cannot move anything. As intended, exactly two of the 104 numbers moved, by exactly the summed carry: End Value 9,756.00 to 9,725.28, and Total Return [%] −2.44 to −2.75, which now equals `returns_stats()`'s.

## 5. Decisions taken where the plan left a choice

* **compile.py and vectorbtX.** compile.py needs vectorbtX's schedule and weighting helpers, but engine.py is the only file allowed to import vectorbtX. They are therefore thin wrappers in engine.py.
* **Which front door.** `SingleBacktest` is used only for one symbol whose instrument names no base currency, with no reporting currency. Everything else goes through `PortfolioBacktest`, which is the same implementation with one column. That covers any FX pair (finding in section 3) and a reporting currency (`account.base_currency`).
* **Stops.** Measured from the open the order filled at, before the spread (`stop_entry_price="price"`), and charged the spread on exit (`stop_exit_price="stopmarket"`). `engine.call_plan` set both until vectorbtX `16293a2`, whose front door now sets them itself, so neither the in-app run nor the export passes them.
* **Benchmark.** The buy-and-hold line and every benchmark-relative statistic use vectorbt's own benchmark: the same instruments, equal-weighted, unlevered, from the first bar, before costs. A cost-inclusive unlevered hold is not offered: at 1x a £10,000 account cannot reach a spread bet's £1/pt minimum stake on EURUSD, so the hold may not trade at all.
* **Filters.** Filters are leaves allowed only in entry rules, and `validate_spec` rejects them in exits.
* **Selection is optional.** A portfolio with no selection holds every asset, which is what the 60/40 template needs. Fixed weights are used as given and are not re-normalised.
* **Weights left out.** Cap weighting (there is no market-cap data source) and long/short selection (negative weights are unverified) are not offered yet.
* **References can shift.** A reference can carry its own `shift` (`{"feature": "upper", "shift": 1}`), so a breakout can compare with the previous bar's channel. This is in addition to the leaf-level `shift`.
* **Structure search uses choice nodes.** A choice is `{"choose": label or optimisable categorical, "options": {label: subtree}}`, and it can sit anywhere in the spec. `apply_params` resolves choices, so nothing downstream sees them. `validate_spec` checks every alternative. BUTChC and Optuna search the conditional space; the other backends fix the structure at the current choice.
* **The extra exits need a fixed stop.** They adjust an existing stop, and vectorbt only calls the adjustment while a stop is active.
* **The vectorised grid is limited.** It only varies parameters that change the signals, on a single symbol, and never a shared account.
* **Cash across symbols.** A rule strategy on several symbols names its cash model explicitly (`account.cash`, default `per_instrument`, vectorbt's own behaviour). Separate accounts are reported as one book of N x `init_cash`, the sum the equity curve already drew and the optimiser already scored, with a per-symbol table beside it. A random entry priority must be seeded. Buy and hold is not offered in a shared account: without a size, the first symbol would take all the cash.

## 6. Windows findings in the Studio itself (fixed)

* **`SSLKEYLOGFILE` aborts the process.** On Windows, the standalone CPython that uv installs aborts with `OPENSSL_Uplink ... no OPENSSL_Applink` as soon as any HTTPS context is created while `SSLKEYLOGFILE` is set (some IDEs, proxies and debugging tools set it). That killed the test run silently, and it would take down the Streamlit server on the first FRED or Yahoo download. `config.apply_environment` now drops the variable on Windows, and `desktop.wait_until_up` no longer creates an HTTPS context for a plain HTTP poll. `test_sslkeylogfile_is_dropped_on_windows`.
* **Job files race.** On Windows, a file the other process has open cannot be replaced, and a file being replaced cannot be opened, for a moment. The page polling `progress.json` failed intermittently with `PermissionError`, and the same race could kill a running worker. `jobs.retry_while_locked` retries briefly, and `trials.parquet` is now written atomically like the JSON files (`jobs.write_parquet`).
* **Text encoding.** `Path.read_text()` without `encoding=` reads the ANSI code page on Windows. The app code always passed `utf-8`; one test did not.

## 7. Upstream issues

The items that belong in vectorbtX (twelve bugs and one required feature, shared cash with an entry priority, each with a standalone reproduction or acceptance check in `docs/upstream/vectorbtx_issues.py`) are in [`UPSTREAM.md`](UPSTREAM.md), with the line numbers, suggested fixes, and the Studio workaround to remove once each is fixed. That document also covers the ta_patterns naming bug and the shared-cash design question.

All 13 are fixed in vectorbtX `16293a2`, and the Studio's workarounds for them are removed (each issue's status is in `UPSTREAM.md`). Issue 14, a limitation that came with the entry-bar stop fix, is fixed in `9f4cf54`. Nothing found in phase 0 is open in vectorbtX.

Issue 15 was found after phase 0, on `9f4cf54`, while checking the per-symbol results for shared cash: holding costs reach `value()` and `returns()` but not `total_profit()`, `final_value()`, `total_return()` or the End Value and Total Return rows of `stats()`. It is open, with check 15 in the script.
