# vectorbtX issues found by OpenStrategy Studio's phase 0

Found while running the Studio against vectorbtX `ace29e2` (identical to the local checkout apart from line endings) and vectorbt 1.1.0. Line numbers below are for that commit.

Every issue has a standalone reproduction that imports only vectorbtX, pandas and numpy:

```bash
uv run python -I docs/upstream/vectorbtx_issues.py     # prints PRESENT or FIXED per issue; exit code = number present
```

Run it in vectorbtX's own environment while patching, or copy the checks into vectorbtX's tests. At `ace29e2`, all 13 are present.

## Status (12 November 2026)

**All 13 are fixed in vectorbtX `16293a2`, and issue 14, found with those fixes, in `9f4cf54`**, which `uv.lock` now pins: the script reports 0 of 13 present, and vectorbtX carries the same checks in `tests/test_upstream_fixes.py`. Nothing here is open in vectorbtX. The Studio workarounds below are removed, except where an issue says it can stay:

| Issue | Studio side now |
|---|---|
| 1 | Removed: `compile_signals` and `export_script` shift nothing; `shift_shorts` is gone. |
| 2 | Kept by choice: an FX pair still goes through `PortfolioBacktest(instruments=...)`. The account currency is now passed as `account_currency=`, since vectorbtX's flat `base_currency=` names a pair's base on FX terms. |
| 3 | Studio to-do still open: require both currencies on an FX instrument. vectorbtX now skips carry for a missing leg, with a warning. |
| 4 | Carry now includes `carry_markup`, which the Settings page can set. Saved FX runs change by the markup. |
| 5, 6 | Removed: `engine.STOP_ENTRY_PRICE` / `STOP_EXIT_PRICE` and their `call_plan` lines. |
| 7, 8 | Nothing to remove; `validate_spec` still rejects fixed weights above 100%. |
| 9 | Removed: the Settings caption. The markup is charged, from the Settings page. |
| 10 | `pf[[column]]` stays in `run_vectorised` (harmless). |
| 11 | Removed: `analysis.extract_results` draws `pf.x.intrabar_drawdown_series()`. |
| 12 | Nothing to remove. |
| 13 | The API exists (`cash`, `entry_priority`, `when_cash_short`); the Studio side is still to build. |
| 14 | Fixed in `9f4cf54`; nothing to remove. The pinning test became `test_extra_exits_check_the_entry_bar_without_a_warning`, and the exit-rules gate gained entry-bar cases. |

## Where each phase 0 finding belongs

| Belongs in | Findings |
|---|---|
| **vectorbtX: bugs** | 1 to 12 below. Ten give wrong numbers without an error, one crashes, and one is housekeeping. |
| **vectorbtX: required feature** | 13, shared cash with an entry priority. Portfolios need it, and the Studio builds its account options on it. Check 13 in the script is its acceptance test. |
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
* **Studio side, once it exists** (no vectorbtX action):
  * an account choice between "each symbol trades its own copy of the account" and "one shared account";
  * for a shared account, a priority picker (symbol order, random with a seed, or by a feature such as the strongest momentum) and a skip-or-shrink choice;
  * results that label the replicated case as N accounts, with a per-symbol table. Today the stats tables silently average the columns.

### 14. Stops on the entry bar are not checked when an adjust callback is given (fixed in `9f4cf54`)

* **What was done.** Not the opt-in suggested below: vectorbtX wraps each `adjust_sl_func_nb` / `adjust_tp_func_nb` in a bar view, automatically. On the split index the wrapper calls the callback only on a bar's open step, with `c.i`, `c.init_i` and `c.curr_i` divided back to bars and its arguments untouched, and returns the stop unchanged on the rest of the bar. A callback therefore sees exactly the calls it saw on whole bars, so the Studio's templates needed no change, and the rest of the entry bar is checked. Only a custom `signal_func_nb`, or templates (`vbt.Rep`) in a callback's arguments, still fall back to whole bars with the warning; the Studio uses neither.
* **Found while fixing it.** Handed the split index unwrapped, a callback's `c.i` runs to twice the bar count, and because numba does not bounds-check, the templates would have written past the end of their `state_*` arrays and corrupted memory. The whole-bar fallback in `16293a2` was therefore necessary, not just conservative.
* **Studio side.** `test_extra_exits_check_the_entry_bar_without_a_warning` replaces the pinning test. The exit-rules gate gained four entry-bar cases (each template, long and short), checked against the reference simulator, which always checked the entry bar: they pass on `9f4cf54` and fail on `16293a2`.

The original report:

* **Where.** vectorbtX `16293a2`, `frontend/split_bars.py` (`can_split`) and the signals branch of `frontend/backtest.py`.
* **What happens.** vectorbtX now checks a stop on the bar a position is entered, by simulating each bar as its open plus the rest of the bar, and mapping the orders back. A custom `adjust_sl_func_nb`, `adjust_tp_func_nb` or `signal_func_nb` takes opaque arguments that cannot be re-indexed to the two-step index, so with one the front door falls back to whole bars and warns ("stops are NOT checked on the bar a position is entered"). The Studio's extra exits (`exit_rules.py`: time stop, breakeven, step trailing) are exactly such callbacks, so **every strategy with an extra exit keeps the entry-bar bias**: a stop the entry bar trades through is missed. It always flatters the result, most for tight stops on daily bars.
* **Fix, a suggestion.** Let a callback opt in to split bars, for example `split_bars=True` on the front door (or a marker attribute on the function), documented as: on the split index `c.i` and `c.init_i` are steps, the bar is `c.i // 2`, even steps are the open (O = H = L = C = open) and odd steps the rest of the bar; per-bar arrays in the callback's arguments stay per bar. The Studio's templates would then index by `c.i // 2` and count bars as `c.i // 2 - c.init_i // 2`, and the time stop would fire on the open step. The phase 3 gate (`tests/test_exit_rules.py::test_gate_matches_the_reference_simulator`) would need its reference simulator to check the entry bar too.
* **Studio side, until then.** `test_exit_rules.py::test_extra_exits_say_the_entry_bar_is_not_checked` pins the warning, so the gap stays visible; delete it when this is fixed.
