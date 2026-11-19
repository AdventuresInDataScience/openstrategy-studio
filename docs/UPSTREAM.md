# vectorbtX issues found by OpenStrategy Studio's phase 0

Found while running the Studio against vectorbtX `ace29e2` (identical to the local checkout apart from line endings) and vectorbt 1.1.0. Line numbers below are for that commit.

Every issue has a standalone reproduction that imports only vectorbtX, pandas and numpy:

```bash
uv run python -I docs/upstream/vectorbtx_issues.py     # prints PRESENT or FIXED per issue; exit code = number present
```

Run it in vectorbtX's own environment while patching, or copy the checks into vectorbtX's tests. At `ace29e2`, all 13 are present.

## Status (19 November 2026)

**All 13 are fixed in vectorbtX `16293a2`; issue 14, found with those fixes, in `9f4cf54`; and issues 15 to 19 in `9fa427e`**, which `uv.lock` now pins. 15 was found on `9f4cf54` while building the per-symbol results for shared cash, and 16 to 19 while fixing it (18 is a feature: the borrow fee, a documented gap in vectorbtX's own audit). Against `9fa427e` the script now reports **8 of 26 present** (exit code 8): issues 21 (found 16 November 2026) and 22 (found 17 November 2026) are open, and so are 23 (a required feature: risk sizing from the balance or the equity), 24 (unlevered defaults) and 25 (point-size defaults), all asked for by the owner on 17 November 2026; from 18 November 2026, 26 (a required feature: stops and targets as a vector, with risk sizing from them) and 27 (a trailing stop in points widens as the price rises); and from 19 November 2026, 28 (a required feature: risk sizing with no stop, from the leverage). Against `9f4cf54` it reports 15 to 19 present as well. vectorbtX carries the same checks in `tests/test_upstream_fixes.py`, `tests/test_borrow.py` and `tests/test_holding_sim.py`. The Studio workarounds below are removed, except where an issue says it can stay:

| Issue | Studio side now |
|---|---|
| 1 | Removed: `compile_signals` and `export_script` shift nothing; `shift_shorts` is gone. |
| 2 | Kept by choice: an FX pair still goes through `PortfolioBacktest(instruments=...)`. The account currency is now passed as `account_currency=`, since vectorbtX's flat `base_currency=` names a pair's base on FX terms. |
| 3 | Done: `validate_spec` requires both currencies on an FX instrument (`asset_class="fx"` or `mode="fx_spot"`), and the instrument grid warns beside the cell. vectorbtX skips carry for a missing leg, with a warning. |
| 4 | Carry now includes `carry_markup`, which the Settings page can set. Saved FX runs change by the markup. |
| 5, 6 | Removed: `engine.STOP_ENTRY_PRICE` / `STOP_EXIT_PRICE` and their `call_plan` lines. |
| 7, 8 | Nothing to remove; `validate_spec` still rejects fixed weights above 100%. |
| 9 | Removed: the Settings caption. The markup is charged, from the Settings page. |
| 10 | `pf[[column]]` stays in `run_vectorised` (harmless). |
| 11 | Removed: `analysis.extract_results` draws `pf.x.intrabar_drawdown_series()`. |
| 12 | Nothing to remove. |
| 13 | Built (13 November 2026); see "Studio side" under issue 13. |
| 14 | Fixed in `9f4cf54`; nothing to remove. The pinning test became `test_extra_exits_check_the_entry_bar_without_a_warning`, and the exit-rules gate gained entry-bar cases. |
| 15 | Fixed in `9fa427e`; there was no workaround to remove. The golden numbers were regenerated on the same prices (`make_golden.py --offline`): End Value and Total Return [%] moved by exactly the summed carry (−30.73), and nothing else did. |
| 16 | Fixed in `9fa427e`; nothing to remove. Shared-account runs with rates now charge each symbol its own carry, and a shared share-dealing account is financed on its debit, split by long value. |
| 17 | Fixed in `9fa427e`; nothing to remove. A portfolio (weights) of shares with rates no longer pays financing on a book it paid for. |
| 18 | Built (13 November 2026): `borrow_rate` in the Build page's instrument grid (equity mode only; `validate_spec` says so elsewhere), and vectorbtX's warnings now reach the Backtest page (`run_backtest` collects them; `extract_results` keeps them in the run's meta). |
| 19 | Done: the grid's financing column is gone, and `engine.instrument_fields` drops a per-instrument `financing` value an older spec may hold. |
| 21 | **Open** (found 16 November 2026 on `9fa427e`): `min_trade` never skips a trade. No workaround; the Build page's tooltip states what happens. |
| 22 | **Open** (found 17 November 2026 on `9fa427e`): the buying-power and minimum-size warnings measure every bar, not the bars an order is placed on. No workaround needed: the Risk tab's trade-size check measures at the entries. |
| 23 | **Open** (a required feature, 17 November 2026): risk sizing from the balance or the equity. No workaround (it needs sizing inside the simulation); the Risk tab says the risk is a fixed share of the starting cash. |
| 24 | **Open** (17 November 2026): bought outright and FX spot default to 2x. The Studio names 1x itself (`engine.usual_leverage`), so its runs are right either way. |
| 25 | **Open** (17 November 2026): an index's point size defaults to 0.01, not 1.0, and a share priced in pence is not recognised. The Studio names the usual point size itself (`engine.usual_point_size`). |
| 26 | **Open** (a required feature, 18 November 2026): stops and targets as a vector (per symbol, per bar from the signal bar), and risk sizing from a stop in %, a per-bar stop or a trailing stop. No workaround. Stops in % of the price need nothing new (vectorbt's own `sl_stop`, `tp_stop`, `sl_trail`) and are built; a stop from a feature waits on this. |
| 27 | **Open** (18 November 2026): a trailing stop in points widens as the price rises (it becomes a fraction of the fill price). No workaround; the Build page's tooltip says what happens. |
| 28 | **Open** (a required feature, 19 November 2026): risk sizing with no stop, from the leverage (the trade's margin is the risk: worth R × L, an implied stop 1 / L of the price away). No workaround; the Risk tab's tooltip says it waits on vectorbtX. |

## Where each phase 0 finding belongs

| Belongs in | Findings |
|---|---|
| **vectorbtX: bugs** | 1 to 12 below. Ten give wrong numbers without an error, one crashes, and one is housekeeping. Later, all fixed: 14, 15, 16 and 17 (wrong numbers), and 19 (a setting accepted in silence and never read). Open: 21 (a setting that does not do what it says), 22 (warnings that measure bars with no order), 24 (defaults that borrow where nothing should be borrowed), 25 (point sizes that are not the market's) and 27 (a trailing stop in points that widens as the price rises: wrong numbers). |
| **vectorbtX: required feature** | 13, shared cash with an entry priority. Portfolios need it, and the Studio builds its account options on it. Check 13 in the script is its acceptance test. Later: 18, the borrow fee on a direct short (check 18). Open: 23, risk sizing from the balance or the equity (check 23 is its acceptance test); 26, stops and targets as a vector (per symbol, per bar from the signal bar), with risk sizing from them (check 26); and 28, risk sizing with no stop, from the leverage (check 28). |
| **ta_patterns** | `chart_patterns.BULLISH` and `CHART_PATTERNS` list `double_bottom_ugly`, but the function is `ugly_double_bottom`. |
| **Environment** (not a library) | On Windows, uv's standalone CPython aborts when `SSLKEYLOGFILE` is set and an HTTPS context is created. vectorbtX's `FREDData` would hit it too. |
| **The Studio** (fixed there; no action for you) | A label lookup in `pf.stats(tags="vectorbtx")` for returns metrics, which vectorbtX's docs correctly place in `returns_stats`. Reading `returns_stats(tags="mt5")` returning `None`. Choosing a no-signal front-door call as the benchmark (the 30x funding behind it is issue 7). A Windows file race in job folders, a text-encoding slip, the desktop health check, and two test-harness bugs. Still to add: the instrument grid should require both currencies for an FX instrument. |

## The issues

Severity: **wrong numbers** means a plausible result with no error, which is the worst kind. The Studio workaround column says what to delete in the Studio once each one is fixed.

### 1. Short signals are not shifted to the next bar: lookahead

* **Where.** `frontend/backtest.py:266-267` shifts `entries` and `exits` with `_next_bar`, but `short_entries` and `short_exits` arrive in `**kwargs`. They reach `from_signals` unshifted through `**sim_kwargs` (`:211`, `:274`).
* **What happens.** With a signal at bar 1, a long fills at bar 2's open (102), and a short fills at bar 1's open (101), the bar whose close produced the signal.
* **Why it is wrong.** The docstring (`:123`) promises "Orders fill at the next bar's open, always". Any user who passes short arrays gets lookahead on every short.
* **Fix.** Pop both from `sim_kwargs` and pass `_next_bar(...)` beside `entries` and `exits`. ⚠️ This changes results for anyone who pre-shifted, so it needs a changelog note.
* **Studio workaround, and what to change when it is fixed.** Do this in the SAME change as the vectorbtX bump, or shorts are shifted twice: remove the shift in `compile.compile_signals` (and its `shift_shorts` argument, whose only caller is the live preview at `app/pages/2_Build.py:577`), and the four lines `export_script` emits for it (two comments and two `.shift(1, ...)` lines). Then rewrite `tests/test_lookahead.py::test_short_signals_are_shifted_one_bar` to check that the compiler no longer shifts shorts, drop the assertion at `tests/test_export.py:22`, and update the CLAUDE.md invariant "Short signals are shifted by one bar". `tests/test_engine.py::test_short_signal_fills_next_open` must still pass unchanged: it checks the outcome, a fill at bar t+1.

### 2. `SingleBacktest(base_currency=...)` is read as the account currency

* **Where.** `PortfolioBacktest` names `base_currency` as the account currency (`frontend/backtest.py:73`). `SingleBacktest` forwards flat instrument fields in `**kwargs` (`frontend/single.py:88`), so a flat `base_currency` never reaches the instrument.
* **What happens.** A long EURUSD spread bet at EUR 3% / USD 4.5% is charged **4.50% a year** (issue 3's mechanism, with a "no currency pair" warning). The same instrument through `PortfolioBacktest(instruments=...)` is charged the correct 1.50% differential.
* **Also affected.** `examples/strategies/01_ma_cross_spreadbet_fx.py:84` does exactly this, with the comment "the pair, so carry can use the differential".
* **Fix.** Either route a flat `base_currency` to the instrument when `asset_class == "fx"` or the mode is `fx_spot`, or rename the account argument (`account_currency=`). Renaming is clearer and breaking. Fix the example either way.
* **Studio workaround.** `engine.call_plan` sends any instrument with a base currency through `PortfolioBacktest`. It can stay once this is fixed (one column through `PortfolioBacktest` is the same implementation). If you revert it, `tests/test_engine.py::test_fx_pair_routes_to_portfolio` and the golden test's hand call change with it; `test_fx_carry_uses_the_pair` must keep passing.

### 3. A missing FX currency leg is priced at 0%, not skipped

* **Where.** `holding/callback.py:265-268`: `_rate_for(rates, None, index)` returns zeros, and the carry branch (`:230-238`) then computes `0% - quote rate`.
* **What happens.** An FX spread bet without `base_currency` warns "Carry will resolve to zero" (`instruments/validate.py:129`), but a long is charged **the whole USD rate (4.50%)**, and a short is credited it.
* **Fix.** When either currency is missing, skip carry as the warning says, or (better, since zero carry is also wrong) raise at construction for an FX instrument with carry active and no pair.
* **Studio workaround.** None. The Studio should validate that an FX instrument names both currencies anyway (a Studio to-do), but the library should not price a missing leg at zero.

### 4. `PortfolioBacktest`'s `carry_markup=0.0` overrides the setting

* **Where.** `frontend/backtest.py:80` defaults `carry_markup=0.0` and passes it on (`:300`). `apply_holding_costs` only falls back to `settings["vectorbtx"]["carry_markup"]` when it receives `None` (`holding/callback.py:191-194`). `simulate_with_holding_costs` (`callback.py:394`, the `Portfolio.from_holding_costs` path) has the same `0.0` default at `:401`.
* **What happens.** With the setting at 1.00%, a long is charged the bare 1.50% differential with no markup. `core/settings.py:16` says that leaving it at zero "flatters carry strategies specifically".
* **Fix.** Default `carry_markup=None` in both signatures.
* **Studio workaround.** None. Until it is fixed, Studio backtests charge no carry markup. When it is fixed, remove the carry note from the README's known limits and from CLAUDE.md's known gaps, and expect FX carry figures in saved runs to change by the markup.

### 5. Point stops are anchored at the entry bar's close

* **Where.** `frontend/prepare.py:438-462` converts points into a fraction of the **fill price** (the next open), as the docstring at `backtest.py:130` promises. But the front door never sets `stop_entry_price`, so vectorbt's default `"close"` applies the fraction to the entry bar's **close**.
* **What happens.** A buy fills at 100 on a bar that closes at 103. Its 2-point stop sits at 100.94, above the entry, and fires the next bar.
* **Fix.** `sim_kwargs.setdefault("stop_entry_price", "price")` in the signals branch. `"price"` is the order price before slippage, which is exactly what the points were converted against. `"fillprice"` would anchor after the spread instead. Trailing stops start from the same anchor, so they are fixed too.
* **Studio workaround.** `engine.STOP_ENTRY_PRICE`, passed by `engine.call_plan`. When vectorbtX sets it itself, delete the constant, its `call_plan` line, the `stop_entry_price=` in `tests/test_engine.py::test_golden_ma_cross`'s hand call and the assertion in `test_single_symbol_plan`. Keep `test_stop_is_measured_from_the_fill_open`: it checks the outcome.

### 6. Stop exits pay no spread

* **Where.** vectorbt's default `stop_exit_price="stoplimit"` returns zero slippage for a stop fill (`vectorbt/portfolio/nb.py:1616-1630`). vectorbtX puts the spread into `slippage` (`costs/base.py:11`, `costs/compose.py:101-103`).
* **What happens.** With a 1-point spread, a signal exit pays half a point (99.5), while a stop gapped to 95 fills at 95.0 instead of 94.5. Every strategy that exits by stops is under-costed, spread bets most of all, where the spread is the whole cost.
* **Fix, two options.**
  * One line: `sim_kwargs.setdefault("stop_exit_price", "stopmarket")`. That charges spread plus slippage on stops and targets. A take-profit is a limit order, so it arguably should pay the spread but not slippage.
  * The root cause: the spread and slippage share vectorbt's single `slippage` field. Charging the spread as a fee (a fraction of notional) would apply it to every fill whatever `stop_exit_price` is, with fills then reported at mid.
* **Studio workaround.** `engine.STOP_EXIT_PRICE` (`"stopmarket"`). Remove it as for issue 5, and keep `test_spread_is_paid_on_every_fill`. If you choose the fee route instead, that test's expected fill prices move to mid, with the spread in `Fees`.

### 7. Buy and hold is funded at buying power

* **Where.** `frontend/backtest.py:229` hands the engine `equity x leverage`. The weights branch resets that to equity (`:233-250`), but the hold branch (`from_holding`, `:282`) does not.
* **What happens.** A call with neither signals nor weights on a 30x spread bet holds **30x the account's equity**. With synthetic EURUSD the "buy and hold" lost 247% while the market fell 6.5%.
* **Fix.** Fund the hold at equity, as the weights branch does. At 1x a small account may not reach a spread bet's minimum stake (£10,000 cannot buy £1/pt of EURUSD), so warn when the hold cannot trade.
* **Studio workaround.** The Studio no longer makes this call. Its benchmark is `pf.benchmark_value()`. Nothing to remove.

### 8. Weights above 100% fill in column order

* **Where.** `frontend/backtest.py:321-336` (`_warn_if_weights_exceed_cash`).
* **What happens.** For targets of 0.75 / 0.75, the book holds 0.75 / 0.25. The warning says "the excess will not fill and the book will sit below target", which reads as an even shortfall. In fact the first column fills in full and the last is starved.
* **Fix.** Scale the weights pro rata to the cash that is there and warn, or raise. At the least, make the warning describe what happens.
* **Studio workaround.** `validate_spec` rejects fixed weights above 100%. Keep it either way.

### 9. The FX conversion markup is never charged

* **Where.** `frontend/backtest.py:167` (`open = pd.DataFrame(open)...`) drops `ConvertedFrame.fx_markup` before `prepare`, so `costs/compose.py:71` reads `None`. `fx_markup` is listed in `_FRONTEND_ARGS` (`:33`) but is not a parameter, so `fx_markup=` falls into `**kwargs` and vectorbt raises `TypeError`. `fx/convert.py:141` also hard-codes `0.005`, ignoring `settings["vectorbtx"]["fx_markup"]`.
* **What happens.** A 0.5% and a 5% markup fill at the same price (80.0).
* **Fix.** Read `getattr(open, "fx_markup", None)` before coercing and pass it through explicitly. Make `fx_markup=` a real parameter. Default `to_base_currency(fx_markup=None)` to the setting.
* **Studio workaround.** None possible without a second cost path. The Settings page says the field has no effect yet. When it is fixed, remove that caption (`app/pages/6_Settings.py`), the comment in `engine.prepare_data`, and the README and CLAUDE.md notes.

### 10. `pf["column"]` crashes every metric on a leveraged portfolio

* **Where.** `core/portfolio.py:221`: `np.asarray(margin_loan)[idxs]` with a scalar label leaves a 0-d array. `get_init_cash` (`:126-129`) passes it to vectorbt's `init_cash_grouped_nb`, which needs 1-D.
* **What happens.** `pf["A"].total_return()` raises a numba `TypingError`; `pf[["A"]]` works.
* **Fix.** `np.atleast_1d(...)` at `:221` (or at `:126`).
* **Studio workaround.** `engine.run_vectorised` selects `pf[[column]]`. It can stay; it is harmless.

### 11. `pf.x.intrabar_drawdown()`: shared cash counted once per column, and shorts valued as longs

* **Where.** `core/accessors.py:95-102`.
* **What happens.**
  * The function sums `pf.cash(group_by=False)`, which under cash sharing gives each column the whole starting cash. A 4-asset weights book reports **0.88% intrabar against a 2.71% close-to-close drawdown**: shallower, which the docstring says cannot happen. Counting the shared cash once gives 3.48%.
  * It values every position at the highs for the peak and the lows for the trough, so a short of 10 at 100 with a spike to 104 reports **-0.100%** (a gain) instead of 0.450%.
* **Fix.**
  * Use the group-level `pf.cash()`. It reconciles exactly with `pf.value()` and already nets out the margin loan.
  * Bracket each side by its own extremes: longs best at the high, shorts best at the low.
  * Consider returning the series (or adding one): the maximum alone cannot be plotted.
* **Studio workaround.** `analysis.extract_results` computes the curve itself. If vectorbtX returns or adds a series, switch to it. `tests/test_portfolio.py::test_intrabar_drawdown_brackets_the_close_drawdown` asserts that the Studio's figure is deeper than the library's on shared cash, so that line must change to "equal".

### 12. Housekeeping: pandas 4 removes `DatetimeIndex.weekday`

* **Where.** `holding/accrual.py:48`. Use `.day_of_week`; the numbering is the same, with Monday as 0.
* vectorbt 1.1.0's own `DataFrame.rename(inplace=True)` (`vectorbt/records/base.py:614`) is deprecated too. That one is vectorbt's to fix; `pyproject.toml`'s `pandas<4` pin covers both for now.

### 13. Shared cash with an entry priority (required feature)

* **Why it is needed.** A portfolio of signal strategies is one account trading several instruments. Today the front door replicates `init_cash` per column: three symbols on 10,000 trade 30,000, and nothing says so. The Studio needs both models, with the user choosing between them. The shared one needs a rule for who gets filled when entries compete for cash.
* **Why it belongs in vectorbtX, not the Studio.**
  * It is simulation semantics. It must behave the same in the Studio, in the Studio's exported scripts (plain vectorbtX calls), and in anyone's notebook.
  * It needs the signals *after* the front door's one-bar shift, so that same-bar exits can be placed first. Only the front door has them, and issue 1 shows how easily two shifts disagree.
  * Funding (`buying_power(..., cash_sharing=...)`, `backtest.py:223-229`) and sizing (`risk_pct` off `init_cash`) already live there and interact with shared cash.
* **What works today.** The front door forwards `cash_sharing`, `group_by`, `call_seq`, `seed` and `allow_partial` to vectorbt. Tested with three same-bar entries, each wanting 40% of the account:

  | Setting | Result |
  |---|---|
  | Default | Each column gets its **own** full `init_cash`. |
  | `cash_sharing=True, group_by=True` | Column order: A 40, B 40, **C 20 (a silent partial fill)**. |
  | `call_seq` array (C, A, B) | C 40, A 40, B 20. |
  | `call_seq="random"`, `seed=` | Works, and is reproducible per seed. |
  | `allow_partial=False` | A 40, B 40, C skipped. |

* **The gap.** A `call_seq` order ranks *every* order in the bar, exits included. With priority C-first, C's entry ran before A's same-bar exit freed cash, so C got 20 when 40 would have fit. `call_seq="auto"` puts exits first but then ranks entries by order value, not by priority.
* **Required behaviour.** Check 13 in the script is the acceptance test, written against these proposed names; rename `SHARED_CASH_ARGUMENTS` there if yours differ.
  * `cash="per_instrument" | "shared"`. `"per_instrument"` is today's behaviour. `"shared"` is one account (`cash_sharing=True, group_by=True`, funded as one pool).
  * `entry_priority=None | [columns] | DataFrame | "random"`, with `seed=`:
    * a list is a static order;
    * a DataFrame is a per-bar score, higher first, aligned with the signals (the bar the signal is computed on) and shifted with them;
    * `"random"` is a seeded permutation of the competing entries on each bar;
    * `None` keeps column order.
  * `when_cash_short="skip" | "partial"`, mapping to `allow_partial`. Skip is the better default: a risk-sized position should not silently shrink.
  * Priority decides only between entries. On each bar, the orders that close or reduce positions at the open go first, so their cash is available, then entries in priority order, then the rest. In the check's scenario, A exits while C and D enter with room for one: priority D-first gives "A sells 40, D buys 40, C does not trade".
* **Reference sketch** (passes check 13 when wrapped around today's front door; built only from existing vectorbt arguments):

  ```python
  def shared_call_seq(entries, exits, columns, entry_priority=None, seed=None):
      """Per-bar call order: exits first, then entries by priority, then the rest. Signals already shifted."""
      rng = np.random.default_rng(seed)
      order = np.zeros((len(entries), len(columns)), dtype=np.int64)
      for i in range(len(entries)):
          exiting = [k for k in range(len(columns)) if exits.iloc[i, k]]
          entering = [k for k in range(len(columns)) if entries.iloc[i, k] and k not in exiting]
          if isinstance(entry_priority, list):
              entering.sort(key=lambda k: entry_priority.index(columns[k]))
          elif isinstance(entry_priority, pd.DataFrame):          # shifted by one bar, like the signals
              entering.sort(key=lambda k: -entry_priority.iloc[i, k])
          elif entry_priority == "random":
              entering = list(rng.permutation(entering))
          rest = [k for k in range(len(columns)) if k not in exiting and k not in entering]
          order[i] = exiting + entering + rest
      return order    # pass as call_seq=, with cash_sharing=True, group_by=True, allow_partial=(when_cash_short == "partial")
  ```

  The short arrays (once issue 1 shifts them) and reversals (`upon_opposite_entry="reverse"`) belong in the same exits-first and entries-by-priority groups.
* ⚠️ **One subtlety the sketch does not handle.** vectorbt processes each column completely (its stop check, then its signal) before the next. A stop hit *intrabar* on a column ordered before an entry frees cash the entry could not have had at the open. That is a small optimistic bias. Either order columns with only intrabar stops after the entries, or document it.
* **Studio side, built 13 November 2026** (no vectorbtX action):
  * the spec's `account` holds `cash` (`per_instrument` or `shared`), `entry_priority` (`{"by": "symbols", "order": [...]}`, `{"by": "random"}` seeded by `engine_extra.seed`, or `{"by": "feature", "score": ref, "highest_first": bool}`) and `when_cash_short` (`skip` or `partial`), checked by `validate_spec`. `spec.cash_model` says when the question arises at all: a rule strategy on two or more symbols. One symbol is one account either way, and a weights portfolio is shared by construction;
  * `engine.call_plan` always names `cash=` for such a strategy, so the export says which model it is. A feature priority compiles to a per-bar score frame in `compile_signals`, on the signals' bar and unshifted (the front door shifts it with them; `test_entry_priority_score_is_causal_and_unshifted`);
  * the Build page's Account tab has the picker, and the Pipeline tab names the account;
  * results: `analysis.stats_tables` regroups separate accounts into one book of N x `init_cash`, matching the equity curve and the optimiser (`group_by=True`), instead of vectorbt's silent average. The Backtest page says "N separate accounts" and shows a per-symbol table, and each account in it equals that symbol run alone (`test_separate_accounts_are_reported_as_accounts`). A symbol in a shared account shows only what it traded;
  * `engine.vectorisable` refuses a shared account;
  * `test_shared_cash_fills_competing_entries_by_priority` checks every priority and both cash-short rules through the Studio, and the `shared_account_ma_cross` template exercises the whole path in every per-template test and in `selfcheck`.

### 14. Stops on the entry bar are not checked when an adjust callback is given (fixed in `9f4cf54`)

* **What was done.** Not the opt-in suggested below: vectorbtX wraps each `adjust_sl_func_nb` / `adjust_tp_func_nb` in a bar view, automatically. On the split index the wrapper calls the callback only on a bar's open step, with `c.i`, `c.init_i` and `c.curr_i` divided back to bars and its arguments untouched, and returns the stop unchanged on the rest of the bar. A callback therefore sees exactly the calls it saw on whole bars, so the Studio's templates needed no change, and the rest of the entry bar is checked. Only a custom `signal_func_nb`, or templates (`vbt.Rep`) in a callback's arguments, still fall back to whole bars with the warning; the Studio uses neither.
* **Found while fixing it.** Handed the split index unwrapped, a callback's `c.i` runs to twice the bar count, and because numba does not bounds-check, the templates would have written past the end of their `state_*` arrays and corrupted memory. The whole-bar fallback in `16293a2` was therefore necessary, not just conservative.
* **Studio side.** `test_extra_exits_check_the_entry_bar_without_a_warning` replaces the pinning test. The exit-rules gate gained four entry-bar cases (each template, long and short), checked against the reference simulator, which always checked the entry bar: they pass on `9f4cf54` and fail on `16293a2`.

The original report:

* **Where.** vectorbtX `16293a2`, `frontend/split_bars.py` (`can_split`) and the signals branch of `frontend/backtest.py`.
* **What happens.** vectorbtX now checks a stop on the bar a position is entered, by simulating each bar as its open plus the rest of the bar, and mapping the orders back. A custom `adjust_sl_func_nb`, `adjust_tp_func_nb` or `signal_func_nb` takes opaque arguments that cannot be re-indexed to the two-step index, so with one the front door falls back to whole bars and warns ("stops are NOT checked on the bar a position is entered"). The Studio's extra exits (`exit_rules.py`: time stop, breakeven, step trailing) are exactly such callbacks, so **every strategy with an extra exit keeps the entry-bar bias**: a stop the entry bar trades through is missed. It always flatters the result, most for tight stops on daily bars.
* **Fix, a suggestion.** Let a callback opt in to split bars, for example `split_bars=True` on the front door (or a marker attribute on the function), documented as: on the split index `c.i` and `c.init_i` are steps, the bar is `c.i // 2`, even steps are the open (O = H = L = C = open) and odd steps the rest of the bar; per-bar arrays in the callback's arguments stay per bar. The Studio's templates would then index by `c.i // 2` and count bars as `c.i // 2 - c.init_i // 2`, and the time stop would fire on the open step. The phase 3 gate (`tests/test_exit_rules.py::test_gate_matches_the_reference_simulator`) would need its reference simulator to check the entry bar too.
* **Studio side, until then.** `test_exit_rules.py::test_extra_exits_say_the_entry_bar_is_not_checked` pins the warning, so the gap stays visible; delete it when this is fixed.

### 15. Holding costs reach `value()` but not `total_profit()`, `final_value()`, `total_return()` or `stats()` (fixed in `9fa427e`)

* **What was done.** As suggested: vectorbtX's `Portfolio.total_profit` adds each column's summed holding costs on the ungrouped branch (upstream's grouped branch re-enters it and sums per group), so `final_value()` equals `value()[-1]` and stats' End Value and Total Return follow. It is not `@cached_method`: that decorator keys its cache on the method name, so an override and the cached parent it calls would share one slot and recurse. Trade P&L stays before holding costs, by design and now documented: `trades.pnl` and the MT5 trade metrics built from it (Total Net Profit, Recovery Factor, the balance drawdowns) leave carry out, because a charge accrues per bar and has no trade to belong to; `pf.holding_costs` holds the difference.
* **Studio side, done.** The golden numbers were regenerated with `make_golden.py --offline`, which replays the saved prices through the example's own download call: only End Value (9,756.00 to 9,725.28) and Total Return [%] (−2.44 to −2.75) moved, by exactly the summed carry.

The original report:

* **Where.** vectorbtX `9f4cf54`, `core/portfolio.py`: `Portfolio.cash_flow` folds `_holding_costs` in (its docstring says this "is what makes holding costs visible at all"), but vectorbt's `total_profit` is computed from the order records (`total_profit_nb` over close and orders), not from `cash_flow`. `final_value` (`init_cash + total_profit`) and `total_return` (`total_profit / init_cash`) inherit it, and so do the `End Value` and `Total Return [%]` rows of `pf.stats()`.
* **What happens.** A long EURUSD spread bet in a flat market, carry only (check 15 in the script): carry charged −11.46, `pf.value()[-1]` = 9,988.54, but `final_value()` and stats' End Value = 10,000.00 and `total_return()` = 0.00% (`value()` gives −0.11%). On the Studio's MA-cross template on 400 synthetic bars: `returns_stats()` Total Return −1.24% with carry, `stats()` Total Return −0.98% whether rates are given or not. So within one tear sheet the equity curve, Sharpe, drawdown and `returns_stats` include carry, while End Value and Total Return leave it out; the optimiser's `total_return` metric leaves it out too. Every strategy with `rates=` is affected, and leveraged FX carry is where it matters most.
* **Fix, a suggestion.** Override `total_profit` on vectorbtX's `Portfolio` to add the summed `_holding_costs` (per column ungrouped, summed per group when grouped), as `cash_flow` does, so `final_value`, `total_return` and the stats rows follow. Check `trades.pnl` is meant to stay gross (per-trade P&L before swaps is the MT5 convention, with swaps reported separately), and say so in the docs either way.
* **Studio side.** No workaround. The per-symbol table shows `Holding costs` beside `Trade P&L`, so the carry is visible. When it is fixed, `test_golden_numbers_from_the_vectorbtx_example` fails by design: re-run `tests/fixtures/golden/make_golden.py` and check that only the carry-dependent numbers moved.

### 16. Holding costs in a shared account are charged on the wrong amounts (fixed in `9fa427e`)

Found while fixing 15. Two faults, in opposite directions, both only with `cash="shared"` and `rates=`.

* **Carry and full-notional financing were over-charged.** `apply_holding_costs` read position values from `pf.asset_value()`, which a shared account reports once for the whole account, and `build_charges` gave that one series to every instrument in it. Each paid on the whole account's positions: an account of N instruments paid up to N times its carry, and an instrument holding nothing paid as much as the others (check 16: A holds a EURUSD spread bet, B holds nothing, and B paid A's 11.46). Now each instrument is charged on its own position, so a shared account pays what separate accounts placing the same orders pay; `build_charges` raises when an input has no column for an instrument instead of sharing out an account's series.
* **Margin share-dealing financing was under-charged.** Financing on the borrowed basis (`mode="equity"`) read each instrument's borrowing as if it had the account's whole equity to itself, so A and B buying 8,000 each against 10,000 at 2x owed 6,000 and paid nothing. **The owner's decision** on how to share an account's debit between its instruments: in proportion to their long market value (shorts take none, since a short sale adds cash); equally if the account owes money while holding no longs; each share charged at that instrument's own currency's rate. `borrowed_portion(group_by=True)` on separate accounts now sums their borrowing instead of netting their cash.
* **Studio side.** Nothing to remove. The shipped shared template (shares at 1x, no rates) was not affected; FX or levered share strategies on shared cash with rates were.

### 17. Borrowing counted without a loan, and the replay path funds no leverage (fixed in `9fa427e`)

Found while fixing 16.

* **Borrowing without a margin loan.** A portfolio with no margin loan fell back to `|asset value| − free cash`, measured after the purchase, which counts money already spent as borrowed: 8,000 bought from 10,000 showed 6,000 borrowed. The front door's rebalancing (`weights=`) path runs without a loan, so **every fully invested share portfolio with `rates=` paid margin financing on its whole book** (check 17: 35.07 on a fully paid 60/40 book in 12 bars). Borrowing is now each account's overdraft, `max(0, −cash)`, everywhere, so a paid-for position and a short sale borrow nothing; interest on an overdraft is always a charge.
* **The replay path, `Portfolio.from_holding_costs`,** took `init_cash` as the engine's cash, so a margin account could never borrow: a 2x position sized beyond equity was part-filled at 1x. It now funds leverage as the front door does (`init_cash` is equity; the engine gets equity × leverage; the loan is recorded). Without `rates=` it no longer charges financing or carry at the markup alone.
* **Studio side.** Nothing to remove; the Studio uses only the front door. The shipped portfolio templates set no rates, so were not affected.

### 18. The borrow fee on a direct short (a feature; built in `9fa427e`)

vectorbtX's own audit (`docs/dev/AUDIT-G.md`, G6) had left the short borrow charge out as "document, do not build", since its rate is not published.

* **What was built.** **The owner's rule: the fee follows how the instrument is traded, not what it is.** Shorting something traded directly (`mode="equity"`: a share, ETF, fund or commodity product) means borrowing it, and the lender charges for that. A CFD or spread bet borrows nothing: its short side already pays funding at benchmark − markup, so it has no borrow fee (and an index, which can only be traded that way, never has one). Set `borrow_rate` on an equity instrument (an annual fraction, `0.005` is 0.5% a year) and a short pays it daily on its market value, accrued like financing. It needs no `rates=`. There is no default rate: a direct short held with none set is charged nothing and **warns**; `borrow_rate=0.0` states that the borrow is free. `validate_spec` rejects `borrow_rate` outside equity.
* **Studio side, done.** The Build page's instrument grid has a `borrow_rate` column (help text says equity only, blank warns, 0 is free) and warns beside a non-equity instrument that sets one; `spec.validate_spec` reports it before a run. Because vectorbtX reports what it could not price as Python warnings, which never reach a Streamlit page, `engine.run_backtest` now collects each run's warnings (`"warnings"` in its result), `analysis.extract_results` keeps them in the run's meta (saved runs keep them), and the Backtest page shows them in a warning box. `test_a_direct_short_pays_its_borrow_rate_and_the_run_says_when_it_has_none` pins both.

### 19. A per-instrument `financing` value was accepted and never read (fixed in `9fa427e`)

* **What happened.** `Instrument.financing` was documented as an annual rate or `"auto"`, and the Build page offered it as a grid column ("A number, or auto."), but nothing in vectorbtX read it. Financing was always the benchmark from `rates=` plus `financing_markup=`: with `financing=0.10`, a 10,000 CFD long paid 2.22 a day (5% + 3%) with rates and nothing without, never the 2.78 that 10% is.
* **What was done (the owner's call).** Financing stays one rule for every instrument, which is how retail brokers charge it. The field is kept, reserved for a possible future per-instrument cost structure (the kind professional desks negotiate), resolves to `"auto"`, and a number in it warns that it is not used.
* **Studio side, done.** The grid's financing column is removed; `engine.instrument_fields` drops a value an older spec may hold, so it is never passed on.

### 20. Orders that fill at the close, or expire (a feature request; not built)

Found while adding "good till the end of the day" to the Studio's Risk step (13 November 2026).

* **What is missing.** Every signal fills at the next bar's open. vectorbt 1.1.0's `from_signals` has no limit or stop entry orders and no time in force, and the front door offers neither. So two common intraday rules cannot be modelled exactly: a market-on-close exit (flat at the session's close), and an entry order that lapses at the end of its day.
* **What the Studio does meanwhile (honest approximations, labelled as such on the Risk step).** "Entries are good for the day only" drops an entry whose fill bar falls on the next day (UTC date). "Close before the end of the day" fires an exit on the bar before each day's last bar, so it fills at the last bar's open, not at the close; entries that would fill on that last bar are dropped. Both are signal masks in `compile_signals`, on the signal bar, so nothing is shifted.
* **Suggested API.** `exit_at="close"` on an exit signal array (fill at that bar's close, paying the spread), and `entry_expiry="day"` with a session-end time (`end_of_day`, already a vectorbtX setting) for pending entries. The Studio would then replace both masks with the real orders.

### 21. `min_trade` never skips a trade: it compares targets, not holdings (open)

Found on `9fa427e` while writing the Build page's tooltips (16 November 2026).

* **Where.** `construct/rebalance.py:195-212`, `_apply_min_trade`, called by `to_targets` (`:119-121`), which `Portfolio.from_weights` calls (`core/portfolio.py:410-417`) on the front door's weights route (`frontend/backtest.py:336`).
* **What happens.** On each rebalance bar the function compares the new target with the **previous target** (`change = (row - last).abs()`). Where that change is below `min_trade`, it writes the previous target back (`out.loc[ts, small] = last[small]`). The target is still emitted, not set to NaN, so the engine still trades from wherever the holding has drifted back to it. Check 21: a fixed 60/40 book of fractional shares, rebalanced every 5th bar while A rises 1% a bar, places the same 8 orders with `min_trade=0.5` as with none, including six rebalancing trades of at most 1.19% of the account. Each is far below the threshold, so none should be placed: only the 2 opening buys.
* **Why it is wrong.** All three descriptions promise skipped trades. The `to_targets` docstring (`:90`) says "suppress rebalance trades whose weight change is below this", `from_weights` (`core/portfolio.py:388`) and the front door (`frontend/backtest.py:125`) say much the same, and `docs/API.md` (`:735`) says it "suppresses trades costing more than the drift they correct", which is a third rule. Someone who sets `min_trade` to save costs gets no saving, and nothing says so.
* **The consequences.**
  * Fixed weights, the commonest case, never change, so `min_trade` does nothing at all: every small trade back to the target is made.
  * Weights that move (inverse volatility, a selection) only have a small target change frozen, and the trade back to the frozen target can be larger than the change. A target moving from 60.0% to 60.4% while the holding has drifted to 63% stays at 60.0%, and the engine trades 3% to reach it.
  * The drift report records the held-back target change as suppressed (`suppressed.loc[ts, small] = (row - last)[small]`), not the trade that was in fact made.
* **Fix, a suggestion.** Compare each target with the holding just before the rebalance, and emit NaN (no order for that column) where the gap is below `min_trade`, recording the gap as suppressed drift. The holding is only known exactly inside the simulation, because it depends on fills, costs and rounding, so two options:
  * **Exact:** do the comparison while simulating, for example in a pre-segment callback that reads each column's position and value and drops the small orders.
  * **Estimate:** use `drifted_weights(weights, close, schedule=...)` (`construct/schedule.py:99`), which already gives the weights a book drifts to between rebalances, to set the NaNs in `to_targets` ahead of the run. It ignores costs and rounding, so it can disagree with the real holding by a little; document that if you choose it.
  * Either way, settle which rule `min_trade` means: a weight gap (what the docstrings say) or a cost-benefit test (what `docs/API.md` says), and make the three descriptions agree.
* **Studio side, until then.** No workaround: it would need the Studio to simulate holdings itself. The Build page's tooltip for "Skip trades smaller than (share of value)" says what actually happens, and its label overstates it. When it is fixed:
  * reword that tooltip in `app/pages/2_Build.py` and drop its ⚠️ comment;
  * check `tests/test_portfolio.py::test_min_trade_suppression_shows_in_drift` still holds (it only checks that some drift is recorded);
  * expect fewer orders, and lower costs, in any portfolio run with `min_trade` set: the `sixty_forty` template sets 0.005. The golden numbers are a rule strategy, so they do not move.

### 22. The sizing warnings measure every bar, not the bars an order is placed on (open)

Found on `9fa427e` from the owner's manual test (17 November 2026): a spread bet on SPY, risk-sized with a 100-point stop, two trades in ten years, and a warning that the position "needs 303,845".

* **Where.** `frontend/prepare.py`: `warn_if_cash_constrained` (`:652-708`) and the minimum-size warning in `quantize_engine_size` (`:390-411`), both called from `prepare` (`:586-619`).
* **What happens.** `resolve_size` returns a size for **every bar** (a risk or stake size is the same on each bar; a notional one moves with the price), whether or not an order is placed there. `warn_if_cash_constrained` then takes `(size.abs() * fill_price).max(axis=0)`, the largest position value over the whole history, and warns when that exceeds `init_cash x leverage`. The minimum-size warning likewise counts "N of M sized bars" over every bar. Neither looks at the entries. Check 22: one entry, filled at 100, then the price rises to 1000 with no further entry. A 300-a-point stake (30,000 at the fill, inside 5x leverage on 10,000) fills in full, yet the run warns that "X needs 300,000 but its leverage funds only 50,000"; a 500 notional buys 5 shares at 100 in full, yet the run warns that "5 of 10 sized bars fall below the minimum tradeable size".
* **Why it is wrong.** A warning that fires when nothing was cut is noise, and one that names a figure no trade ever reached misleads: in the owner's run the two trades were too big as well, but the 303,845 named is the stake times the highest open in ten years, not either trade. With a rising price, an early, correctly sized trade is reported as cut down. The message also says "these orders will be partially filled", which is only true of orders that exist.
* **Fix, a suggestion.** Measure on the bars an order is placed: the entries (long and short), shifted to their fill bar as the front door already shifts them (`prepare` has the signals as `strategy`). For a weights strategy, the rebalance bars. Name the largest such trade and its date, and count only those bars in the minimum-size warning ("2 of 2 entries"). With no signals (a sized hold), the first bar.
* **Studio side, until then.** No workaround needed: the Build page's Risk tab now runs its own trade-size check (`engine.sizing_check`), which uses vectorbtX's `resolve_size` and `quantize_engine_size` but reads the size at each entry's fill bar, and says what to change before the run. The Backtest page shows vectorbtX's warning as it is, with the Studio's hint under it (`analysis.WARNING_HINTS`). When it is fixed, nothing to remove; the warning's figure should then match the Risk tab's.

### 23. Risk sizing from the balance or the equity (a required feature; open)

Asked for by the owner on 17 November 2026, after the manual test run: "sizing as a % of balance or of equity is the standard way to spread bet or trade CFDs".

* **What is missing.** `risk_pct` is a fraction of `init_cash`, worked out before the simulation runs (`frontend/prepare.py:246-336`, `resolve_size`; the ⚠️ in its docstring and at `frontend/backtest.py:186`). So 1% of a 10,000 account risks 100 on every trade for the whole backtest: after the account doubles it still risks 100, and after it halves it still risks 100, now 2% of what is left. No broker, and no spread-bet or CFD trader, sizes that way: the usual rule is a share of the **balance** (cash after closed trades) or of the **equity** (the balance plus the open trades' profit or loss). Over a long run with real drawdowns the two give very different curves, and `init_cash` is neither: it understates compounding when the strategy works and overstates the risk taken after it loses. Check 23: risk 1% over a 10-point stop; the first trade is stopped out for 100, so the second should be 9.9 units from either basis, and is 10 (and `risk_of=` is not accepted).
* **Why it needs vectorbtX.** The balance and the equity at an entry depend on every fill, cost and stop before it, so they are only known inside the simulation. Sizes cannot be worked out ahead, as `resolve_size` does today; the docstring already says "true compounding risk sizing needs level 3", and the level-3 callback (`holding/callback.py`) only charges holding costs.
* **Suggested API.** `risk_of="init_cash" | "balance" | "equity"` beside `risk_pct=` (and `risk_amount=` unchanged), the default kept at `"init_cash"` so existing results do not move. Settle in the docs:
  * **when** it is read: at the close of the signal bar, the last value known when the order is sent (the order fills at the next open), never at the fill;
  * **balance** = starting cash + closed trades' profit and loss − costs − holding costs charged so far; **equity** = balance + open positions marked at that close;
  * separate accounts read their own column; a shared account reads the account;
  * the same rounding, minimum size and buying-power rules as today, and the funded-buying-power warning (issue 22) measured on these sizes;
  * a `notional_pct=` (a position worth a share of the equity) would come almost for free, and is the other common rule.
* **Studio side, until then.** No workaround: it would need the Studio to simulate the account itself. The Risk tab says plainly that the risk is a fixed share of the starting cash, the same amount on every trade. When it is built: offer "of the starting cash / the balance / the equity" beside the risk box (`risk.sizing.risk_of` in the spec, through `call_plan` and the export), add it to `engine.sizing_check` (the check can then only estimate the size from the starting cash, and should say so), and add a variable for it on the Optimise page.

### 24. Bought outright and FX spot default to 2x leverage (open)

Asked for by the owner on 17 November 2026: "most portfolios and naked trading are non-margin; spread bets and CFDs standardise at 30x for FX, 10x for indices and commodities and 5x for single stocks." Naked trading means buying the instrument itself, not a spread bet or CFD on it: a share, a gram of gold, a currency. All of those default to no margin, and vectorbtX's own defaults should say so too, not only the Studio's.

* **Where.** `instruments/defaults.py:81-92`, `MAX_LEVERAGE`: `("equity", None): 2.0` and `("fx_spot", None): 2.0`.
* **What happens.** An instrument bought outright, or an FX spot pair, that names no leverage is funded at twice its cash: a margin account. Owning shares, or holding the currency, borrows nothing, so the default should be 1x; borrowing should be something a user asks for. The spread-bet and CFD rows already match the retail limits the owner gives (check 24 confirms them).
* **Why it matters.** A blank leverage on an equity strategy silently allows a position twice the account, and the buying-power warning (issue 22) then never fires where a cash account would have refused the trade.
* **Fix.** `("equity", None): 1.0` and `("fx_spot", None): 1.0`, with the docs saying so; a margin account is then `leverage=2`.
* **Studio side.** `engine.usual_leverage` already names 1x for both (and for every portfolio), and fills it into the grid and the plan, so Studio runs are unaffected either way. When it is fixed, nothing to remove; `usual_leverage` could then read vectorbtX's table for every mode except the portfolio rule, which stays the Studio's.

### 25. An index's point size defaults to 0.01, and a share priced in pence is not recognised (open)

Asked for by the owner on 17 November 2026 ("default filling for points is also very sensible"), with 24: the defaults should be the market's own conventions.

* **Where.** `instruments/defaults.py`: `MODE_DEFAULTS` gives every mode's `point_size` as 0.01 (0.0001 on FX spot), and `filled()` (`:274`) replaces it with `fx_pip(quote_currency)` for an FX underlying. That part is right, in every mode.
* **What happens.** An `asset_class="index"` spread bet or CFD that names no point size gets 0.01. Index spread bets and CFDs are quoted in index points: one point is 1.0 (the FTSE 100 at 7,500 moving to 7,501), so a 50-point stop given with the default is a 0.50 move, 100 times tighter than meant, and a stake per point is 100 times larger. A share priced in pence (Yahoo's `.L` quotes, quote currency `GBp`) has the same problem the other way: one point is a penny, 1.0 in a price written in pence, and the default makes it a hundredth of a penny. Check 25: `Instrument(mode="spreadbet", asset_class="index").filled().point_size` is 0.01, and a `GBp` share's is 0.01.
* **Why it matters.** Like the FX pip (`FX_PIP_BY_QUOTE`, whose own comment says getting it wrong "is silent and large"), the point is a market convention, not a broker setting, so it can be derived. Every stop, target, spread and stake is in points.
* **Fix, a suggestion.** In `filled()`, beside the FX rule: `asset_class="index"` gives 1.0; a quote currency of `GBp` or `GBX` (pence) gives 1.0 for a share. ⚠️ VERIFY how vectorbtX treats a pence quote elsewhere (FX conversion to GBP divides by 100); if it does not know `GBp` at all, that is part of this. Commodities stay 0.01, documented as broker-specific (spot gold is often quoted in points of 0.1).
* **Studio side.** `engine.usual_point_size` names these itself, fills them into the grid when the way of trading, an asset class, a quote currency or a symbol changes, and names a blank one in the plan, so Studio runs are unaffected either way. When it is fixed, `usual_point_size` can read `Instrument(...).filled().point_size` instead.

### 26. Stops and targets as a vector: per symbol and per bar, from the signal bar, with risk sizing from them (a required feature; open)

From the owner's manual test, 18 November 2026: "you can't pin a value to an existing structural feature. This is common usage, e.g. SL = price − 3 × current ATR, or the current EMA(200)." The intent was that such a value is one line of code (`3 * atr`), fed in as the stop. That needs the stop and the target to take a vector (one value per symbol, or a column per bar), which the point keywords never did. A stop in % of the price was the owner's other point, and needs nothing new: vectorbt's own `sl_stop`, `tp_stop` and `sl_trail` are fractions of the price, and they pass through the front door, anchored at the fill and checked on the entry bar like point stops. The Studio uses them now.

* **Where.** `frontend/prepare.py`: `resolve_stops` (`:442-475`) and the risk sizers in `resolve_size` (`:308-331`). `modes/helpers.py`: `points_to_price_fraction` (`:32`) and `size_for_risk` (`:84`). `frontend/backtest.py`: the signal shift (`:346-349`, `_next_bar` at `:579`), which the stops do not get.
* **What happens** (check 26, on `9fa427e`):
  * `stop_points`, `target_points` and `trail_points` take one number for the whole run. One value per symbol raises (a list: `ValueError`, a broadcast; a dict: `TypeError`). So does a per-bar frame, even on one column, because `points_to_price_fraction` multiplies the frame by one column's price. A book of SPY and EURUSD cannot have 60 cents on one and 30 pips on the other.
  * ⚠️ vectorbt's own `sl_stop` does take a per-bar frame, but reads it at the **fill** bar, and the front door does not shift it with the signals, unlike every other per-bar input it takes. So a stop worked out from the signal bar (3 × ATR at its close) and aligned with the signals, as the docs ask of signals, is read one bar late: the fill bar's ATR, which includes that bar's high, low and close. That is lookahead. The check gives the stop 2% at the signal bar and 5% at the fill bar, and the stop sits at 95.
  * The risk sizers read `stop_points` only. `risk_pct` or `risk_amount` with `sl_stop` (a stop in %) or with `trail_points` raises `BacktestError` ("Risk-based sizing needs `stop_points`"), though the distance is known at the fill in both cases: `sl_stop` × the fill price, and a trailing stop's first distance.
* **Why it needs vectorbtX.** The distance has to be read at the signal bar and turned into a price at the fill, which only the front door knows: it shifts the signals and resolves everything at `fill_price`. Risk sizing has to use the same distance. Done in the Studio instead, the Studio would have to shift stops itself, the opposite of the signals' rule, and two bars late the day the front door starts shifting them. It still could not risk-size from them.
* **Suggested API** (check 26 uses these names):
  1. `stop_points`, `target_points` and `trail_points` also take one value per column (a dict, or a list in column order) and a per-bar frame. The frame is **aligned with the signals and shifted with them**, like `entry_priority` scores: the value at the signal bar is used, converted against the fill price.
  2. The same for `sl_stop` and `tp_stop` given as frames: shifted with the signals. Or, if vectorbt's own reading should stay as it is, a front-door `stop_pct` / `target_pct` / `trail_pct` that is shifted, and a warning when a per-bar `sl_stop` frame is passed with signals.
  3. Risk sizing from whichever distance is given, per bar and per column: points × point size, a fraction × the fill price, and `trail_*` (its first distance).
  4. Optional: `stop_price` and `target_price`, a price level per bar aligned with the signals (the stop sits at the signal bar's EMA from the fill on). A per-bar distance computed at the signal bar's close (`close − ema`) misses the gap between that close and the next open, so the stop lands off the level by the gap. Also optional: `trail_price`, a level that moves the stop to each completed bar's value and never loosens it (a chandelier exit). Without it, the Studio can move the stop with an `adjust_sl_func_nb` template, as it does for breakeven and step trailing.
* **To settle in the docs:**
  * A NaN in a per-bar frame (an ATR still warming up). Suggested: no stop on that entry; with risk sizing, no entry, and a warning.
  * A level on the wrong side of the fill, if item 4 is built (vectorbt raises on a negative fraction, `portfolio/nb.py:1874`). Suggested: the entry is not placed, and a warning counts how many, as a broker rejects an order with invalid stops.
  * The split bars (`split_bars.py`) repeat per-bar values onto both steps, so the shifted frame must be made before the split.
* **Studio side, until then.** Stops and targets in points or in % of the price, one number per strategy (`risk.stop.unit`, `risk.target.unit`; in % they go to vectorbt as `sl_stop`, `sl_trail` and `tp_stop`). Risk sizing and the extra exits need a stop in points, and `validate_spec` says so. A stop from a feature waits on this, and the Build page's Stop tooltip says so.
* **Studio side, when it is built:**
  * A third way to measure a distance beside points and %: a multiple of a feature (3 × ATR), and a feature's level (an EMA) if item 4 is built. In the spec: `risk.stop.unit` with its number and the feature output.
  * `compile_signals` computes the stop and target series, unshifted like the signals, and passes them through `call_plan` and the export.
  * `engine.sizing_check` and the extra exits' starting distance (`exit_rules.runtime_values`, `point_sizes`) read the same distance, and risk sizing and the extra exits then work with a stop in % too.
  * `tests/test_lookahead.py` checks the stop series beside the signals.
  * The multiple becomes an Optimise variable. Distances per symbol could go in the instrument grid, if wanted.

### 27. A trailing stop in points widens as the price rises (open)

Found on `9fa427e` on 18 November 2026, while adding stops in % of the price: the Studio's tooltip said a trailing stop "keeps this distance behind the best price since entry", and the probe disagreed.

* **Where.** `frontend/prepare.py`: `resolve_stops` (`:442-475`) converts `trail_points` to a fraction of the fill price, `points × point_size / fill`, and hands it to vectorbt as `sl_stop` with `sl_trail=True`.
* **What happens.** vectorbt's trailing stop is that fraction of the best price since entry, so the distance in points grows in proportion to the price. Check 27: a long filled at 120 with `trail_points=2`. The high reaches 200.01 and the next bar dips to 190. The stop should be 2 points behind the peak, at 198.01; it fills at 196.68, 3.33 points behind (2 × 200.01 / 120). Falling, the trail narrows the same way.
* **Why it matters.** Wrong numbers: a trailing stop in points is a fixed distance behind the best price in every broker's terms (a spread bet's 20-point trailing stop stays 20 points behind). The error grows with the move since entry: small on FX (a 2% move widens the trail by 2%), large on a trend-following share strategy that doubles (the trail doubles), so it flatters exactly the trades that ran furthest. A trailing stop in % of the price is right as it is.
* **Fix, a suggestion.** Keep the distance in price, not as a fraction of the fill: each bar, the stop is the best price ∓ `trail_points × point_size`. Within vectorbt that is an `adjust_sl_func_nb` the front door installs for `trail_points`, which returns `distance / curr_price` as the trailing fraction each bar (vectorbt places a long's trailing stop at `curr_price × (1 − f)`), chained with any callback the caller passes. Or carry the stop in price terms in `split_bars`. Either way the stop is checked on the entry bar as now.
* **Studio side.** No workaround (it would need the Studio to install its own trailing callback, which is the library's job). The Build page's tooltip for a stop in points says what happens and that a trailing stop in % of the price does not drift. When it is fixed, change that tooltip back, and tighten `tests/test_engine.py::test_trailing_stop_follows_the_peak` to the exact price.

### 28. Risk sizing without a stop, from the leverage (a required feature; open)

Asked for by the owner on 19 November 2026: "Risk a share of starting cash requires a stop to size the position. In the absence of a stop, it should default to assuming a stop at the point of 0 balance instead, since that then effectively becomes your 'stop'. In the real world you would probably get stopped early due to a margin call, but we can keep it simple for now." The owner's guess was a mix of vectorbtX and the Studio; this entry is the vectorbtX half, with the Studio's below. **The rule is agreed:** the owner confirmed the same day that V = R × L (below) is what was meant, and that the Studio waits for vectorbtX rather than passing a notional in the meantime.

* **Where.** `frontend/prepare.py:308-314`, `resolve_size`: `risk_pct` or `risk_amount` with no `stop_points` raises `BacktestError` ("Risk-based sizing needs `stop_points`"). Check 28.
* **The rule, made exact.** Taken word for word, a stop at the point of zero balance cannot size a trade, because that point depends on the size: S units on a balance B are wiped out by a move of B / S, and sized to lose the risk R over that move, S × B / S = B. Every size loses the whole balance there, whatever R is. The reading that does size a trade takes the zero-balance point of a full-size position, which is the same as treating the risk as the trade's own margin: at leverage L a position worth V posts V / L, and loses all of it on a move of 1 / L of the price. So:
  * the implied stop is **1 / L of the fill price** from the fill: the price going to zero bought outright (1x), a 20% move at 5x, 10% at 10x, 3.33% at 30x;
  * the trade is sized to lose R over that move: **V = R × L**. Risking 1% of 10,000 (100) buys 100 of a share outright, a position worth 500 on a 5x stock spread bet, and 3,000 on a 30x FX pair. Check 28 runs 1x and 5x on one share priced at 100 and wants 100 and 500.
* **To settle in the docs.**
  * It sizes only: no stop is placed. A gap past 1 / L loses more than R, and from 1x a short can too (a 1x short of 100 loses 100 if the price doubles, more beyond). A real account would meet a margin call first, at whatever close-out level the broker sets; keep that out, as the owner asks, and say so. Warn once per run that sizes came from the leverage, naming the implied stop.
  * The leverage is each column's own (`filled()`: the instrument's, else the mode's default; the default matters while issue 24 is open, since a blank leverage bought outright would size at 2x).
  * `risk_amount` the same: V = `risk_amount` × L.
  * With V = R × L and R at most the cash, the trade always fits the account's buying power on its own (cash × L); in a shared account the cash left still caps it, as now.
  * With issue 23, R comes from the chosen basis (`risk_of`); with issue 26, decide whether a NaN in a per-bar stop falls back to this rule or skips the entry (26 suggests skipping).
* **Why it belongs in vectorbtX.** The Studio could pass `notional={symbol: R × L}` today (`_spread` takes one value per column) and the numbers would match. But what `risk_pct` means without a stop belongs beside what it means with one: done in the Studio, the exported script would say `notional=` while the strategy says risk, and the rule would have to be rebuilt once issue 23 sizes risk from the balance or the equity inside the simulation, which a notional worked out from `init_cash` cannot follow.
* **Suggested API.** No new keyword: `risk_pct` or `risk_amount` with no stop (no `stop_points`, `sl_stop` or `trail_points`) sizes each column at V = R × L, and warns. Or, if an error should stay the default, `risk_without_stop="leverage"`.
* **Studio side, until then.** No workaround: `validate_spec` still needs a fixed stop in points for risk sizing, and the Risk tab's Position size tooltip says that sizing a risk with no stop waits on vectorbtX.
* **Studio side, when it is built.**
  * `validate_spec` lets risk sizing run with no stop (a stop in % or a trailing stop still waits on issue 26).
  * The Risk tab says, under the risk box, what each symbol's trade is worth with no stop (R × L) and its implied stop (1 / L of the price), that no stop is placed, and that a real account would meet a margin call first.
  * `engine.sizing_check` shows the implied stop in its "stop in price" column, and its stop-narrower-than-a-bar warning ignores it.
  * `tunable_inputs` offers the stop's kind (none or fixed) beside risk sizing, which today it withholds.
  * `call_plan` and the export need nothing: they pass `risk_pct` and no stop, as the strategy says.

# ta_patterns issues found by OpenStrategy Studio

Found by the lookahead audit in `tests/test_indicators.py::test_no_entry_looks_ahead`, which computes every catalogue entry on full data and on truncated copies and requires the shared bars to agree. ta_patterns 1.2.1; the local checkout (`../ta_patterns`, `ce791c6`) is identical.

### P1. The flag and pennant detectors read the last close on their first bar

* **Where.** `chart_patterns/classic.py:872`, `_flagpole(c, i, min_move, pole_bars)`, called with `i - window` by `flag_bull` (909), `flag_bear` (952), `pennant_bull` (1030) and `pennant_bear` (1066).
* **What happens.** The guard is `if i < pole_bars: return -1, 0`, then `pole_end = i - 1` and `pole_start = pole_end - pole_bars`. When `i == pole_bars` (the first bar each detector evaluates, `pole_bars + window`), `pole_start` is `-1`, and `c[-1]` is **the last close in the whole array**. So that bar's signal depends on the final bar of the history: lookahead. Evidence: `flag_bear` on a 400-bar random walk gives a different signal at bar 40 when computed on the first 150 bars than on all 400.
* **Suggested fix.** `if i <= pole_bars: return -1, 0` (or `if i - 1 - pole_bars < 0`), plus a truncation test over every detector like the Studio's.
* **Studio side.** The four detectors are left out of the catalogue (`indicators.EXCLUDED`, with this reason) until ta_patterns is fixed. Then: delete the four lines from `EXCLUDED`, rerun `python -m openstrategy_studio.indicators`, and check the audit passes.
