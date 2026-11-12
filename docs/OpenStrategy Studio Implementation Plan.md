# OpenStrategy Studio: Implementation Plan

Nov 12, 2026 · @Malhar Patel

Build a Streamlit app, working name **OpenStrategy Studio**, in which a person defines a trading strategy through forms, a rule builder and expressions, then backtests it, optimises it and reads a tear sheet, all without writing code. Everything runs on vectorbtX (`vbt.x.SingleBacktest` and `vbt.x.PortfolioBacktest`), vectorbt 1.1.0, ta\_patterns, BUTChC, Optuna, marsopt and plotly-themes.

## 1. Goal, scope and packaging

**In scope**

- Single-instrument signal strategies, including pattern-driven ones (300 detectors from ta\_patterns).
- Multi-instrument portfolios: cross-sectional selection, weighting, rebalance schedules, leverage budgets, multi-currency, holding costs.
- Backtest, optimisation (grid, random, Optuna, scipy, BUTChC, marsopt), walk-forward, tear sheet with selectable plotly themes.
- Local data management: import, store, validate, refresh.

**Packaging, in this order**

1. An installable Python package (`src/` layout, import name `openstrategy_studio`) with an `openstrategy-studio` command that launches the Streamlit app.
2. The same code later exported as a standalone application a non-developer can double-click. The standalone bundle targets Linux and Windows, works offline once data is stored locally, and is a late phase, but three habits from day one keep it possible: no reliance on the current working directory, bundled assets read through `importlib.resources`, and all user data under one configurable folder (default from `platformdirs`).

**Non-goals**

- Tick data, pending (limit/stop) entry orders, live trading, MT4/MT5 EA export.
- Margin calls and forced liquidation: vectorbtX defers them too, and enforces maximum leverage only.
- A paid hosted service: vectorbt is Apache 2.0 with Commons Clause, so selling hosting is not allowed. Personal and internal use is fine.

**Event model: on bar, not on tick**

fxDreema evaluates its blocks on every tick. Here every block is evaluated once per **bar close**, on whatever timeframe the data has, and any resulting order fills at the next bar's open. The same strategy spec therefore works on 5-minute, hourly or daily bars. Behaviour that depends on the order of ticks inside a bar is approximated from the bar's open, high, low and close, and the UI says where that matters (section 3, item 5).

## 2. Environment (uv) and code style

### Environment

The whole environment is managed with **uv**. Nothing is installed with bare `pip`, and `uv.lock` is committed.

```bash
uv init --package openstrategy-studio        # src/ layout, creates pyproject.toml
uv python pin 3.12                   # writes .python-version; vectorbt needs >=3.11,<3.15
uv add vectorbt==1.1.0 "plotly<7" "numpy>=2.4.6" "pandas>=3.0.3,<4"
uv add streamlit ta_patterns butchc optuna scipy plotly-themes yfinance pyarrow platformdirs
uv add --optional marsopt marsopt    # Cython extension: keep it an optional extra
uv add --dev pytest
uv sync
uv run openstrategy-studio                   # the console script that launches Streamlit
uv run pytest
```

- **vectorbtX** is installed from its GitHub repository until it is published to PyPI. Add it with uv add git+https://github.com/AdventuresInDataScience/vectorbtx, which writes an entry like `[tool.uv.sources] vectorbtx = { git = "https://github.com/AdventuresInDataScience/vectorbtx" }`; switch to the published distribution (`vectorbtX`, import name `vectorbtx`, always lowercase) when it is released. Install it with its excel and rates extras (vectorbtX\[excel,rates\]): openpyxl reads .xlsx universes, and requests is needed for FRED.
- **`plotly<7` is mandatory.** vectorbt 1.1.0 registers a template containing a `scattermapbox` key that plotly 7 removed, so `import vectorbt` fails. Keep the pin in the Studio's own `pyproject.toml`, with the same explanatory comment as vectorbtX.
- **numpy and pandas floors** come from vectorbt itself: `numpy>=2.4.6`, `pandas>=3.0.3,<4`, `numba>=0.66`. Do not loosen them.
- **marsopt** needs `numpy>=2.4.4` and builds a Cython extension from source if no wheel matches the platform. That is why it is an optional extra, imported lazily.
- Skip the optional `TA-Lib` and `vectorbt-rust` extras at first. Add them later behind extras if wanted.
- plotly-themes' own `pyproject.toml` sets `native-tls = true` for its author's network. Do not copy that into the Studio.

### Code style

These rules are the owner's, and they override the builder's defaults.

1. **Functional, no classes.** Use functions plus plain `dict` and `list`. No `@dataclass`. No pydantic models. A strategy spec is a dict, validated by one function. The only classes in the project are the ones vectorbt and vectorbtX already provide.
2. **Do not over-modularise.** One module per concept (about a dozen in total, see section 4). Do not create a function for one line of logic. A helper earns its place only when it is called from at least three places or the calling function would otherwise exceed roughly 100 lines. A reader should follow a feature top to bottom in one file.
3. **Explicit over clever.** Write the steps out. Registries are plain dict literals, not decorators or metaclass tricks. Prefer a visible `for` loop to a dense comprehension chain.
4. **Comment heavily.** Explain the why, and every vectorbtX or vectorbt gotcha, in the same style as the vectorbtX examples: banner comments between steps, and `# ⚠️` on anything that can silently give a wrong number.
5. **Few exception cases.** Validate once at the boundary (loading a spec, importing a file, the user pressing Run). Inside, let errors propagate. Do not wrap calls in `try/except` for conditions that cannot happen. The UI catches exceptions in one place around the run call and shows the message, which for vectorbtX errors already names the problem.
6. **Keyword arguments** at every vectorbt and vectorbtX call site, as vectorbtX itself insists.

```python
# Do this: one function, steps visible, comments say why.
def compile_entries(spec, features):
    # Start from an all-False frame; each rule ORs its own signal in.
    entries = features["close"] != features["close"]
    for rule in spec["rules"]["long_entry"]:
        entries = entries | evaluate_rule(rule, features)
    return entries

# Not this: a class, a dataclass config, and five one-line helpers
# (get_rule_list, make_empty_frame, combine_with_or, ...) that must be traced.
```

## 3. Verified facts that shape the design

These come from reading the uploaded source, not from memory. Items marked **TEST** were read from code but not executed (no vectorbt in the planning environment), so the builder must confirm each with a unit test before relying on it.

1. **The front door shifts only some inputs.** `PortfolioBacktest` applies `_next_bar` (shift by one bar) to `entries`, `exits` and `schedule`, then fills at the next open. `short_entries` and `short_exits` are not front-door arguments, so they travel through `**kwargs` straight to `Portfolio.from_signals` **unshifted**. A short signal computed at bar *t*'s close would then fill at bar *t*'s open, which is lookahead. The Studio must shift short signals itself (`.shift(1).fillna(False)`) before passing them. **TEST** with a truncation test.
2. **Direction semantics.** With `short_entries` or `short_exits` set, `entries` and `exits` act as long entries and long exits. With `direction="both"` and no short arrays, `entries` become long entries and `exits` become short entries, so nothing closes a position except the opposite signal or a stop. The Studio should always use the explicit four-array form, not `direction="both"`.
3. **Trailing stop replaces the fixed stop.** `trail_points` is converted to `sl_stop` plus `sl_trail=True` and overwrites any `stop_points`. The UI must present "fixed stop" and "trailing stop" as mutually exclusive. `target_points` (take profit) combines with either.
4. **No time stop in vectorbt 1.1.0.** `from_signals` offers `sl_stop`, `sl_trail`, `tp_stop`, plus numba callbacks `adjust_sl_func_nb`, `adjust_tp_func_nb` and `signal_func_nb`. Time-based exits, breakeven and step trailing therefore need those callbacks. They are a later phase, built as a small set of fixed numba templates, never generated from arbitrary user graphs.
5. **Gaps are handled; same-bar SL and TP order is unknown.** `get_stop_price_nb` fills a stop at the open when the open is already beyond the stop. Which of SL or TP wins when one bar's range contains both was not established from the source. Read `simulate_from_signals_nb` (around `portfolio/nb.py` line 2381 onward), document the rule in the UI, and add a test with a hand-built bar. **TEST**
6. **Sizing takes exactly one of** `risk_pct`, `risk_amount`, `stake`, `notional`, `size`. Two raises `BacktestError`. `risk_pct` needs `stop_points` and `init_cash`, and sizes off `init_cash`, not compounding equity.
7. **Rebalancing is funded at equity.** The source comment says weights summing above 1 express leverage via `leverage_type="portfolio_level"`, while `_warn_if_weights_exceed_cash` warns that the excess will not fill. The two statements conflict. Test what a 1.5-gross weight matrix actually does before exposing leverage on the portfolio path. **TEST**
8. **Documentation and code disagree in places.** Always call by keyword and check against source:
   - `inverse_vol_weight` is documented in `API.md` section 8 but exists nowhere in `vectorbtx/` and is not exported from `vbt.x`. The Studio writes its own now (a rolling-volatility inverse, normalised per row), and the owner will add it to vectorbtX later.
   - `price_weight(prices)` takes only `prices`; the docs show an `index` argument.
   - `to_base_currency(data, base, fx_table=None, quote_currency=None, instruments=None, fx_markup=0.005, fill_policy="ffill_with_warning", validate=True, fx=None)`. The examples pass `fx=<Series>` and `instruments=<Universe>`.
   - `calendar_schedule(index, freq="M", on="first")` has an `on` argument the docs omit.
9. **Data alignment traps (all from vectorbtX examples).** Yahoo indexes are tz-aware, FRED indexes are tz-naive, and their intersection is empty. LSE and US daily bars carry different local-midnight timestamps. A single-symbol `Data.get()` returns a Series named after the field, not the symbol. The data layer normalises all of this once, on import (section 8).
10. **Fill rule.** Orders fill at the next bar's open, always. There is no option to change it, and the Studio must not offer one. High and low are required, and a warning fires if they do not bracket the close.
11. **Conflicting costs raise.** Passing `fees=` beside `commission_pct` raises `BacktestError`. The Studio builds cost arguments only through instrument fields, never through raw passthrough.
12. **Patterns take parameters.** Every ta\_patterns detector has threshold keywords (`shadow_factor`, `require_trend`, `pivot_n`, `window`, `mode`, and so on). An earlier assumption that patterns are parameter-free was wrong, so pattern caching must key on the parameters (section 7), and pattern parameters can be optimised like any other.

## 4. Architecture and repo layout

&#91;embedded content: architecture · 4 layers, 1 engine boundary\]

Pages edit one spec dict. `compile.py` turns it into signals or weights, `engine.py` makes the single vectorbtX call, and `analysis.py` draws the result. Optimisation reuses compile and engine in a loop inside a worker process.

The layout is deliberately flat: about a dozen modules, each one a whole concept, and no sub-packages beyond the Streamlit pages.

```
openstrategy-studio/
  pyproject.toml   uv.lock   .python-version
  src/openstrategy_studio/
    spec.py        # default spec, validate_spec, apply_params, JSON load/save, spec hash
    blocks.py      # BLOCKS: one dict literal holding every block's parameters and compile function
    compile.py     # compile_signals, compile_weights, export_script
    engine.py      # run_backtest: the ONLY file that imports vectorbtx; also mode_rules()
    data.py        # normalise_ohlcv, validate, catalogue, importers, load_frames
    config.py      # settings.toml, apply_settings()
    analysis.py    # stats tables, fig_* functions, finish(fig, theme)
    optimise.py    # build_space, evaluate, METRICS, one function per backend, walk_forward
    jobs.py        # start_job, read progress, cancel
    worker.py      # entry point of a background job: python -m openstrategy_studio.worker <job_dir>
    store.py       # SQLite: catalogue, runs, trials index
    cli.py         # the `openstrategy-studio` command: launches Streamlit
    app/Home.py
    app/pages/1_Data.py ... 6_Settings.py
    templates/*.json          # starter strategies, shipped as package data
  tests/                      # conftest.py with synthetic data; one test file per area
```

## 5. Strategy spec

The spec is a plain dict, stored as JSON (one file per strategy, git-friendly). Every editor in the UI reads and writes it, and the compiler, optimiser and code export consume it. There is no other source of truth.

**Conventions**

- Any numeric or categorical parameter is either a literal (`50`) or an object `{"value": 50, "opt": {...}}`. The `opt` object uses **BUTChC's search-space dialect verbatim**: `{"min", "max"}`, plus `"int": true` or `"log": true`, or `{"values": [...]}` for categoricals. Building a search space is then a walk over the spec collecting every `opt`, and the same dict converts to Optuna, marsopt or scipy bounds.
- Each optimisable parameter's name is its path in the spec, such as `features.fast.window`. Paths are unique by construction, which satisfies BUTChC's unique-name rule.
- A trial is applied by `apply_params(spec, params)`, which returns a deep copy with each `{"value", "opt"}` replaced by the trial's literal. The compiler never sees `opt`.
- `validate_spec(spec)` returns a list of plain-English problems (empty means valid). It checks required keys, mutually exclusive sizing arguments, fixed-stop versus trailing-stop exclusivity, unknown block names and dangling feature references. It is called when a spec is loaded and when Run is pressed, and nowhere else.

**Rule grammar.** A rule is a tree. Inner nodes are `{"all": [...]}`, `{"any": [...]}` or `{"not": node}`. Leaves are one of:

- a comparison: `{"left": ref, "op": "crosses_above", "right": ref, "shift": 0}` with `op` in `>`, `<`, `>=`, `<=`, `crosses_above`, `crosses_below`, `between`, `rising`, `falling`;
- a pattern: `{"pattern": "hammer", "side": "bullish", "params": {"shadow_factor": 1.5}}` where `side` is `bullish`, `bearish` or `either`;
- a filter: `{"filter": "session", "from": "07:00", "to": "16:00"}` (also `weekday`, `month`, `higher_timeframe`).

A `ref` is `{"feature": name}`, `{"price": "close"}` (or open, high, low, volume) or `{"const": 30}`. `shift: n` means "the value n bars ago" and is for reading history only; it never moves a signal forward.

**Example 1: single-instrument signal strategy** (the vectorbtX `01_ma_cross_spreadbet_fx.py` strategy, used as the golden test in section 12).

```json
{
  "version": 1,
  "name": "EURUSD 50/200 MA cross",
  "data": {"symbols": ["EURUSD=X"], "timeframe": "1d", "start": "2019-01-01", "end": "2024-12-31"},
  "instruments": {
    "EURUSD=X": {"mode": "spreadbet", "asset_class": "fx", "point_size": 0.0001,
                 "base_currency": "EUR", "quote_currency": "USD",
                 "min_size": 1.0, "increment": 0.01, "rounding": "floor",
                 "spread_points": 0.9, "slippage": 0.0002, "leverage": 30}
  },
  "features": {
    "fast": {"block": "ma", "source": "close", "window": {"value": 50,  "opt": {"min": 10,  "max": 100, "int": true}}},
    "slow": {"block": "ma", "source": "close", "window": {"value": 200, "opt": {"min": 100, "max": 300, "int": true}}}
  },
  "rules": {
    "long_entry":  {"all": [{"left": {"feature": "fast"}, "op": "crosses_above", "right": {"feature": "slow"}}]},
    "long_exit":   {"all": [{"left": {"feature": "fast"}, "op": "crosses_below", "right": {"feature": "slow"}}]},
    "short_entry": null,
    "short_exit":  null
  },
  "risk": {
    "sizing": {"risk_pct": 0.01},
    "stop":   {"kind": "fixed", "points": 60},
    "target": {"points": 120}
  },
  "account": {"init_cash": 10000, "freq": "D", "rates": {"source": "fred"}},
  "engine_extra": {"upon_opposite_entry": "reverse", "seed": 42}
}
```

**Example 2: multi-asset momentum portfolio.** There are no entry or exit rules. Features produce a score, selection picks assets, weighting sizes them, and a schedule decides when to rebalance.

```json
{
  "version": 1,
  "name": "Top-3 momentum, monthly",
  "data": {"symbols": ["AAPL", "MSFT", "NVDA", "VOD.L", "BP.L"], "timeframe": "1d", "start": "2015-01-01", "end": "2024-12-31"},
  "instruments": {"from_file": "universe.csv"},
  "features": {
    "mom": {"block": "momentum", "source": "close", "window": {"value": 126, "opt": {"min": 20, "max": 252, "int": true}}}
  },
  "selection": {"score": {"feature": "mom"}, "top_n": {"value": 3, "opt": {"min": 1, "max": 5, "int": true}}, "min_score": 0.0},
  "weighting": {"scheme": {"value": "equal", "opt": {"values": ["equal", "inverse_vol"]}}, "vol_window": 20},
  "schedule": {"calendar": {"freq": "M", "on": "first"}, "threshold": null, "min_trade": 0.005},
  "risk": {"leverage_budget": true},
  "account": {"init_cash": 100000, "base_currency": "GBP", "freq": "D", "fx": {"source": "yahoo"}, "rates": {"source": "static", "values": {"GBP": 0.04, "USD": 0.05}}}
}
```

The compiler decides which `PortfolioBacktest` route to take from which sections exist: `rules` means `entries`/`exits`, `selection` plus `weighting` plus `schedule` means `weights`/`schedule`, and neither means buy-and-hold. This mirrors the front door's own dispatch, and having both is an error.

## 6. Stages and block registry

A strategy is built stage by stage. Each stage is a form by default (dropdowns, sliders, tables) with a slot for a rule, expression or custom block where free logic is needed. This is deliberately not one free-form flowchart: the stages follow the shape vectorbtX is built around, so every stage maps to a known call.

| Stage | Form controls | Free-logic slot | Target call |
| --- | --- | --- | --- |
| 1. Universe | Symbol picker from the data catalogue; per-symbol grid of `Instrument` fields (mode, asset\_class, point\_size, contract\_size, increment, min\_size, allow\_fractional, quote/base/cost currency, commission\_pct/flat/per\_unit/min, spread\_points, slippage as number or `"auto"`, financing as number or `"auto"`, leverage, leverage\_type, price\_field) | Per-symbol overrides | `vbt.x.load_universe`, `vbt.x.Instrument` |
| 2. Features | Pick a block, fill its parameters | Expression, Python block | vbt indicators, ta\_patterns |
| 3. Rules (signal strategies) | Presets (MA cross, RSI reversion, channel breakout) | Rule builder, expression | `entries`, `exits`, `short_entries`, `short_exits` |
| 4. Selection (portfolios) | Top-N, bottom-N, threshold, long/short split, score feature | Custom score expression | pandas `rank` on the score frame |
| 5. Weighting | Equal, price, cap, inverse-vol (window), fixed table | Custom weight frame | `vbt.x.equal_weight`, `price_weight`, `cap_weight`, Studio's own inverse-vol |
| 6. Schedule | Always, calendar (freq, first/last), drift threshold, and/or composition, `min_trade` | Custom mask | `calendar_schedule`, `threshold_schedule`, `drifted_weights`, `always_schedule` |
| 7. Risk and sizing | One of risk\_pct / risk\_amount / stake / notional; fixed or trailing stop; take profit; leverage budget | None | `risk_pct`, `stop_points`, `trail_points`, `target_points`, `leverage_budget` |
| 8. Account | init\_cash, base currency, bar frequency, FX source, rates source, holding-cost markups, holidays | None | `FXTable`, `rate_table`, `static_rate_table`, `FREDData.download_rates`, `base_currency`, `fx`, `rates` |

**How blocks are defined.** One dict literal, `BLOCKS`, in `blocks.py`. Each entry declares its parameters (type, bounds, default, label) and holds its compile function. The Build page generates forms from the parameter declarations, and the optimiser reads the bounds as defaults for `opt`. Adding a block is therefore one dict entry and one function, with no UI code.

```python
BLOCKS = {
    "ma": {
        "label": "Moving average",
        "kind": "feature",
        "params": {
            "source": {"type": "choice", "options": ["open", "high", "low", "close"], "default": "close"},
            "window": {"type": "int", "min": 2, "max": 500, "default": 20},
            "ewm":    {"type": "bool", "default": False},
        },
        # compile(params, ohlcv) -> DataFrame, one column per symbol
        "compile": compile_ma,
    },
    # ... one entry per block
}
```

### Feature blocks

Indicator names, inputs, parameters and outputs below are read from `vectorbt/indicators/basic.py`. Pass scalar parameters only; list parameters make vectorbt build a column MultiIndex that the compiler does not expect.

| Block | Call | Inputs | Parameters | Output used |
| --- | --- | --- | --- | --- |
| ma | `vbt.MA.run` | close | window, ewm | `.ma` |
| mstd | `vbt.MSTD.run` | close | window, ewm | `.mstd` |
| bbands | `vbt.BBANDS.run` | close | window, ewm, alpha | `.upper`, `.middle`, `.lower` |
| rsi | `vbt.RSI.run` | close | window, ewm | `.rsi` |
| stoch | `vbt.STOCH.run` | high, low, close | k\_window, d\_window, d\_ewm | `.percent_k`, `.percent_d` |
| macd | `vbt.MACD.run` | close | fast\_window, slow\_window, signal\_window, macd\_ewm, signal\_ewm | `.macd`, `.signal` |
| atr | `vbt.ATR.run` | high, low, close | window, ewm (default **True**) | `.atr`, `.tr` |
| obv | `vbt.OBV.run` | close, volume | none | `.obv` |
| momentum | pandas | close | window | `close / close.shift(window) - 1` |
| highest, lowest | pandas | high / low | window | `rolling(window).max()` / `.min()` |
| zscore | pandas | any series | window | `(x - mean) / std` |
| volatility | pandas | close | window | rolling std of returns |
| pattern | ta\_patterns | open, high, low, close (+ volume) | per-detector keywords | `int8` +1 / -1 / 0 |

A feature referring to another feature (an RSI of a moving average, say) is allowed. The compiler resolves features in dependency order and rejects cycles in `validate_spec`.

### Rule operators

| Operator | Compiled as |
| --- | --- |
| `>`, `<`, `>=`, `<=` | pandas comparison |
| `crosses_above`, `crosses_below` | `left.vbt.crossed_above(right)`, `left.vbt.crossed_below(right)` (vectorbt generic accessor) |
| `between` | `(x >= lo) & (x <= hi)` |
| `rising`, `falling` (n bars) | `x > x.shift(n)`, `x < x.shift(n)` |
| pattern leaf | `+1` for bullish, `-1` for bearish, `!= 0` for either, applied to the detector output |

### Filters

- **Session, weekday, month**: masks built from the DatetimeIndex. Time-of-day filters mean nothing on daily bars, so the UI hides them when the timeframe is daily.
- **Higher timeframe**: compute the feature on resampled bars, then shift and forward-fill back to the base index. The resample **must** use `label="left", closed="left"` followed by `.shift(1)`: pandas aliases such as `W` and `ME` default to right-labelled bins, and without both settings a weekly value can leak into the days before it completes.

### Expression editor and Python block

- **Expression** (phase 2): a text field such as `rsi(14) < 30 and close > sma(200)`, parsed with `ast.parse(mode="eval")` against a whitelist of node types, names and functions. It converts to and from the rule tree, so switching between the builder and the text keeps one spec. Never `eval` raw text.
- **Python block** (phase 3): a function body stored as a string in the spec, executed in-process on the user's own machine. It is labelled as trusted local code in the UI, and imported specs containing Python blocks show a warning before running.

## 7. Compiler and engine adapter

Two files do the work: `compile.py` turns a spec plus data into signals or weights, and `engine.py` turns those into one vectorbtX call. `engine.py` is the **only** file that imports `vectorbtx`, so any API drift is a one-file fix.

### Signal strategies, step by step

The whole path lives in one function, `compile_signals(spec, data)`, written top to bottom with a comment per step.

1. **Frames.** `data` is a dict of five DataFrames (`open`, `high`, `low`, `close`, `volume`), one column per symbol, on one shared index. A symbol with no volume gets a NaN column.
2. **Features, in dependency order.** Each feature compiles to a DataFrame aligned to `close`. A pattern feature calls the detector once per symbol on `.to_numpy()` arrays and rebuilds the frame, because ta\_patterns takes 1-D arrays.
3. **Rules.** Evaluate `long_entry`, `long_exit`, `short_entry` and `short_exit` into boolean DataFrames; a missing rule becomes all-False. `fillna(False)` on every comparison, since indicator warm-up periods are NaN.
4. **Filters** are ANDed into **entries only**. Gating exits would trap positions open, so exits are never filtered.
5. **Shift short signals by one bar** (section 3, item 1). Long signals are shifted by the front door and must **not** be shifted again here. A comment at this line says so.
6. Return `{"entries", "exits", "short_entries", "short_exits"}`.

### Portfolio strategies, step by step

`compile_weights(spec, data)` follows the same shape.

1. Compute features, then the **score** frame named in `selection.score`.
2. **Select.** Rank across columns per row (`score.rank(axis=1, ascending=False)`), keep `rank <= top_n` and `score >= min_score`. Rows where nothing qualifies select nothing, which means holding cash.
3. **Weight.** Equal weight is the selection mask divided by its row sum. Inverse-vol is `1 / rolling_std(pct_change)` masked to the selection, then divided by its row sum. It is one function in compile.py, inverse\_vol\_weight(prices, window=20), matching the signature in vectorbtX's API.md, so it can be swapped for vbt.x.inverse\_vol\_weight once the owner adds that. `price_weight` and `cap_weight` come from `vbt.x`, masked and re-normalised the same way. Rows summing to zero stay zero.
4. **Schedule.** Build the boolean Series from `calendar_schedule` and optionally `& threshold_schedule(drifted_weights(weights, close, schedule=calendar), weights, tolerance)`. Always pass `schedule=` to `drifted_weights`, or drift accumulates from the first bar and the threshold test becomes always-true.
5. Return `{"weights", "schedule"}`. Weights are a full-shape frame; the library turns them into NaN-off-schedule targets itself, and the Studio must never forward-fill them.

### Engine call

`run_backtest(spec, data, params=None)` applies trial parameters, compiles, then calls one of the two front doors. Single-symbol specs use `SingleBacktest` with the instrument fields flattened into keywords (it rejects `instruments=`). Multi-symbol specs use `PortfolioBacktest`.

| Spec section | Front-door argument |
| --- | --- |
| `data` frames | `close=`, `open=`, `high=`, `low=` (all four, by keyword) |
| `instruments` | Single: flat keywords (`mode=`, `point_size=`, `spread_points=` ...). Portfolio: `instruments=` as a dict of per-symbol field dicts, or a path to the universe file |
| `rules` output | `entries=`, `exits=`, and `short_entries=`, `short_exits=` (already shifted) |
| selection, weighting, schedule output | `weights=`, `schedule=`, `min_trade=`, `leverage_budget=` |
| `risk.sizing` | exactly one of `risk_pct`, `risk_amount`, `stake`, `notional` |
| `risk.stop` | `stop_points=` (fixed) or `trail_points=` (trailing), never both |
| `risk.target` | `target_points=` |
| `account` | `init_cash=`, `freq=`, `base_currency=`, `fx=`, `rates=` |
| `engine_extra` | passed through untouched (`upon_opposite_entry`, `seed`, `cash_sharing`, `group_by`) |

Pass `freq=` always. Annualised metrics are filtered out without it and fail confusingly.

⚠️ **Holding-cost path.** The `01_ma_cross` example comment says supplying `rates` opts into the level-3 `from_order_func` path. `frontend/backtest.py` actually applies holding costs *after* the simulation through `apply_holding_costs`. Trust the code, and note that cash feedback during the run is therefore absent.

### Caching

Optimisation reruns the compiler thousands of times, so feature results are memoised in a module-level dict keyed by `(block name, canonical JSON of its resolved params, data fingerprint)`. Pattern features include every detector keyword in the key. The cache has a size cap and is per process. Do not memoise whole backtests across trials; vectorbt is fast enough, and stale portfolio objects are memory-heavy.

### Code export

`export_script(spec)` returns Python source that reproduces the backtest with plain vectorbtX calls, in the style of the `examples/strategies/` files. Each block carries a short `export` snippet next to its `compile` function. The acceptance test (section 12) runs the exported script and the in-app run on the same fixture data and requires identical `pf.stats()`, which stops the two paths drifting apart.

## 8. Data layer and configuration

All data is bar data. Tick data is out of scope, and the UI says so wherever timeframe is chosen. `data.py` owns import, storage, validation and loading; `config.py` owns settings.

### Storage

```
<data_dir>/                      # default: platformdirs.user_data_dir("openstrategy-studio"); set in Settings
  catalogue.db                   # SQLite: one row per stored dataset
  prices/<symbol>/<timeframe>.parquet
  strategies/<name>.json         # saved specs
  runs/<run_id>/                 # saved results (see section 9)
  jobs/<job_id>/                 # background optimisations (see section 10)
```

- **Columns:** `open, high, low, close, volume`, plus `adj_close` when the source provides one. Lower case. The index is a tz-naive `DatetimeIndex`, sorted, duplicate-free, holding each bar's **start** time. Intraday data is stored in UTC with the original timezone recorded in the catalogue.
- **Catalogue columns:** symbol, timeframe, source, first\_bar, last\_bar, n\_bars, adjusted (bool), tz, imported\_at, notes.
- **Timeframes:** `1m 5m 15m 30m 1h 4h 1d 1w`, stored natively. Resampling **up** is allowed (open=first, high=max, low=min, close=last, volume=sum, with `label="left", closed="left"`). Resampling down is impossible and the UI never offers it. Pass the vectorbt `freq` string to every backtest.

### Importers

| Source | How | Notes |
| --- | --- | --- |
| Yahoo | `vbt.YFData.download(symbols, start=, end=, missing_index="drop")` | Daily history is deep. Intraday history is limited to recent windows (the limit varies by interval; treat as approximate and check at build time) |
| CSV or Parquet | Upload, then a column-mapping step (date, optional separate time column, o/h/l/c/v) | MT4/MT5 history exports typically split date and time into two columns; support combining them. Verify against a real export |

An import runs one function, `normalise_ohlcv(frame)`, that does what the vectorbtX examples do by hand: lower-case the columns, drop the timezone (`tz_localize(None)`), for daily bars `normalize()` to calendar dates, sort, keep the last of any duplicate timestamps. A single-symbol Yahoo download returns a Series named after the field, so rename it to the symbol before storing.

### Validation on import

Report every problem with counts and sample rows, let the user accept or reject, and never silently repair.

- `high >= max(open, close)` and `low <= min(open, close)` on every bar. vectorbtX warns, and stops then fire at prices that never traded.
- No zero or negative prices; NaN counts per column; duplicate or non-monotonic timestamps.
- Bars whose range or gap exceeds a configurable multiple of the median range (likely bad ticks).
- Calendar gaps longer than the market's normal closures.

Write this check explicitly in `data.py`. It is five lines of pandas, so there is no reason to import it from vectorbtX.

### Loading for a backtest

`load_frames(symbols, timeframe, start, end)` returns the dict of five frames from section 7. Rules:

- Align multiple symbols on the **common trading days** (`dropna(how="any")` on close, then reindex all fields to those dates), as vectorbtX example 03 does. LSE and US dailies would otherwise share no timestamps and produce zero rows with every column still present.
- **Adjusted prices.** Equity mode defaults to `adj_close`, and spread bets and CFDs default to `close`, because adjusting rescales the whole history and changes point counts. The front door receives only a `close` frame, so for equity instruments the Studio builds `close` from `adj_close` and scales `open`, `high` and `low` by the same per-bar factor so the bracket `high >= close >= low` still holds. **TEST** this against a stock with a dividend.
- **Foreign currencies.** Convert open, high, low **and** close with the same FX table, with `vbt.x.to_base_currency(..., fx=<Series>, instruments=<Universe>)`, before computing any signal. Convert any benchmark too. A multi-currency run needs the FX pair stored as an ordinary dataset (for example `GBPUSD=X`).
- **Rates.** `vbt.x.FREDData.download_rates([...])` needs a free FRED key (`FRED_API_KEY`) and caches under `$XDG_CACHE_HOME` or `~/.cache`. Without a key use `vbt.x.static_rate_table(currencies, index, {"GBP": 0.0525, ...})`. The UI warns that a flat rate misreads any period spanning a rate cycle. Carry on an FX instrument needs rates for **both** currencies and the instrument must name both, or carry resolves to zero.

### Settings

`settings.toml` in `platformdirs.user_config_dir("openstrategy-studio")`, read once at startup by `apply_settings()`.

| Setting | Notes |
| --- | --- |
| `data_dir`, `default_theme`, `workers` | Studio's own |
| FRED key | Prefer the `FRED_API_KEY` environment variable or a `.env` file. If stored in `settings.toml`, the Settings page warns that the file must stay out of version control |
| `rounding` (`floor`/`nearest`/`ceil`), `financing_markup`, `fx_markup`, `slippage_max`, `slippage_coef`, `vol_window`, `triple_day`, `end_of_day`, `day_count`, `fill_policy`, `static_rate`, `triangulation_tolerance` | Written to `vbt.settings["vectorbtx"][key]`. **Never touch `vbt.settings["portfolio"]`** |

Background optimisation workers are separate processes, so the settings dict is passed to each worker's start-up function and applied there before any run.

### Offline use

The app works without a network once the data is stored locally. Every download is written to the parquet store, so a symbol is fetched once and read from disk afterwards. An **Offline** switch in Settings turns off all downloaders: loading data that is not stored gives a clear message instead of a failed request, FRED rates fall back to the cache and then to the static table, and Yahoo is never called.

For Linux and Windows alike: use `pathlib` for every path, never shell commands, keep the numba cache directory writable, and run CI on both systems. Streamlit serves its own front-end files and plotly-themes bundles its fonts, so nothing else should be fetched at run time; verify this with the network disabled.

## 9. Runner and tear sheet

Pressing Run calls `run_backtest(spec, data)` once, inside a single `try/except` that shows the exception text to the user (vectorbtX's errors already name the offending argument). The result is then reduced to saved artefacts and drawn by `analysis.py`.

### What a run saves

Do not pickle the `Portfolio`: it is large and tied to library versions. Save derived data instead, under `runs/<run_id>/`, with a row in the SQLite `runs` table.

- The spec as JSON and its hash, the settings snapshot, and the installed versions of vectorbt, vectorbtX, ta\_patterns, butchc, optuna and plotly.
- A data fingerprint: symbols, timeframe, first and last bar, bar count, and a hash of each parquet file.
- Parquet files for the equity curve, returns, drawdown, trades (`pf.trades.records_readable`) and orders, plus the three stats tables as JSON.
- To redraw a past run, load these files. To get a live `Portfolio` again, re-run the spec on the same data.

### Stats tables

`pf.stats()`, `pf.trades.stats()` and `pf.returns_stats()` are shown as three tables, with a filter for `tags="mt5"` and `tags="vectorbtx"`. Three notes:

- Benchmark-dependent metrics (`m2`, `treynor_ratio`) need benchmark returns. Build a buy-and-hold benchmark with the same front door and no signals, converted to the same base currency, and confirm in testing how to hand it to `returns_stats`. **TEST**
- `absolute_drawdown` and `recovery_factor` need a resolved `init_cash`. Pass `pf.get_init_cash()`, not the raw argument.
- Z-score has an inverted reading: negative means outcomes cluster in streaks. Add that as a tooltip.

### Figures

Every figure is a function `fig_<name>(results, theme)` in `analysis.py` that returns a plotly `Figure`. Each is built only when its tab is selected, and cached on `(run_id, name, theme)`.

| Figure | Source | Notes |
| --- | --- | --- |
| Equity vs benchmark | `pf.value()` | Overlay `pf.x.intrabar_drawdown()` result in the hover |
| Underwater drawdown | `pf.drawdown()` and `pf.x.intrabar_drawdown()` | Show both; the intrabar one is routinely about a third larger |
| Monthly returns heatmap | `pf.returns()` compounded by month | Year by month grid, diverging colourscale from the theme |
| Rolling metrics | vectorbtX `rolling_*` methods (API.md section 11) | Window selectable |
| Trade scatter and PnL bars | `pf.trades.records_readable` | Colour by direction |
| Return distribution | trade returns | Histogram with mean and median |
| MAE and MFE scatter | Studio computes it | Worst and best price excursion between each trade's entry and exit, from `high` and `low`. vectorbt 1.1.0 has no native MAE/MFE (confirm) |
| Price chart with trades | `go.Candlestick` plus markers from trade records | Overlay any feature, and mark pattern hits; the main "why did it trade here" view |
| Exposure and leverage | `pf.x.user_size()`, `pf.x.margin_level()`, `pf.x.borrowed_portion()` | Hide when the mode has no such data |
| Holding costs | `pf.holding_costs` | Cumulative, only when `rates` were given |
| Portfolio weights over time | `pf.asset_value()` divided by total value | Stacked area; add rebalance markers |
| Per-asset contribution and correlation | per-column returns | Bars plus heatmap |
| Drift report | `pf.drift` and `pf.x.drift()` | What quantisation and `min_trade` suppression cost |
| Monte Carlo fan | Studio resamples trade returns | Shuffle with replacement, plot percentile bands |

The trade records' column names (entry and exit timestamps, PnL, return, direction) should be read from `records_readable` at build time rather than hard-coded from memory.

### Themes

- `import plotly_themes` once, at the top of `analysis.py`. It registers the themes in plotly's template registry on import. Read the list at runtime with `plotly_themes.registered_themes()`; at the time of reading it held 22: matrix, cyberpunk, star\_wars, star\_trek, synthwave, blueprint, financial\_times, lego, coke, barbie, vintage\_comic, xkcd, newsprint, minecraft, nasa, gameboy, tableau, powerbi, nike, amber\_terminal, nord, dracula.
- A single function `finish(fig, theme)` sets `template=theme` and the common layout; every figure ends with it. Nothing else in the code mentions a theme.
- **In Streamlit, always** `st.plotly_chart(fig, theme=None)`. The default `theme="streamlit"` overrides the template's colours and defeats the whole feature. Check the installed Streamlit version for the current width argument, because `use_container_width` has been deprecated in recent releases.
- **Fonts.** `plotly_themes.enable_fonts()` is for notebooks. For Streamlit, inject `plotly_themes.font_face_css()` into the page once, inside a `<style>` block via `st.markdown(..., unsafe_allow_html=True)`. Static image export (kaleido) needs the `.ttf` files installed on the operating system instead. Verify rendering in a browser.
- **Glow.** `plotly_themes.add_glow(fig)` post-processes line charts and suits `star_wars`, `cyberpunk` and `synthwave`. Offer it as a checkbox, applied only to line figures.
- Offer three groups in the selector: plain plotly (`plotly_white`, `plotly_dark`), the sober themes (financial\_times, tableau, nord, dracula, nasa, blueprint, powerbi), and the rest. Some themes, such as `nike`, set no background, so the Studio should not assume every theme styles every subplot type.

## 10. Optimisation

`optimise.py` holds the search-space builder, the objective, and one function per backend. `jobs.py` runs them in the background. Backends are interchangeable because they all receive the same three things: a search space, an objective that takes a params dict, and a budget.

### Search space and objective

- **Space.** `build_space(spec)` walks the spec and collects every `{"value", "opt"}`. The result is a dict `{param_path: opt_dict}` in BUTChC's format (section 5). Conditional choices use BUTChC's `next_level` and are available only to BUTChC and Optuna; for the other backends the UI disables structural choices and fixes them at their current value.
- **Objective.** `evaluate(params, spec_path, ...)` is one top-level function (it must be picklable for process pools). It runs `apply_params`, `run_backtest`, extracts the chosen metric, and applies constraints. Anything that raises `BacktestError`, or breaks a constraint such as minimum trade count, returns `float("nan")`.
- **Metrics.** Chosen from a dict `METRICS` of name to a function of the portfolio: Sharpe, Sortino, Calmar and total return from vectorbt; K-ratio (v3), Ulcer Performance Index, geometric Sharpe, recovery factor and GHPR from vectorbtX; profit factor and expectancy from the trade stats. SQN is not provided by either library, so compute `mean(pnl) / std(pnl) * sqrt(n_trades)`. Portfolios with several columns reduce to one number through `group_by=True`. Confirm each accessor name at build time.
- **Constraints**, each optional: minimum trades (default 30), maximum drawdown, minimum profit factor. Without a trade-count floor, optimisers reliably find parameters that trade twice and win both.

### Backends

| Backend | Call | Notes |
| --- | --- | --- |
| Grid | `itertools.product` over each parameter's values (`min`, `max` and a step, or `values`) | Phase 2, loop-based. A vectorised grid using vectorbt parameter broadcasting is phase 3 and only works for graphs where every swept parameter feeds an indicator or stop |
| Random / Sobol | `scipy.stats.qmc.Sobol(d=n, seed=s).random_base2(m)`, scaled into each range; round ints; `exp` for log | Baseline and sanity check |
| Optuna | `optuna.create_study(direction=..., sampler=..., storage="sqlite:///job/optuna.db")` then `study.optimize(fn, n_trials=...)` | Default backend. Samplers: TPE, CMA-ES (continuous only), NSGA-II (multi-objective), random, QMC. Define-by-run `trial.suggest_*` inside an `if` handles conditional spaces. `optuna.visualization` returns plotly figures that can go through `finish(fig, theme)`. Not in the uploaded sources: verify names and multi-process storage advice against Optuna's current docs |
| scipy | `scipy.optimize.differential_evolution(func, bounds, integrality=..., seed=..., workers=N, updating="deferred", polish=False)`; `dual_annealing(func, bounds, seed=...)` | They minimise, so negate. `dual_annealing` has no integrality, so round inside the objective. Categoricals are encoded as an integer index. `workers=N` needs a picklable top-level function |
| BUTChC 0.6.1 | `BUTChC_optimize(searchspace, objective, budget, direction="maximize", seed=..., batch=..., executor=..., n_warmup=..., verbose=False, start_prob_tree=...)` | Handles conditional spaces natively. Objective receives only the parameters on the sampled path. Non-finite returns are recorded but ignored, which suits invalid combinations. `batch` up to 8 costs almost nothing in quality; past 16 it does. Needs an order-preserving executor you create. Its own benchmarks are synthetic and written with the optimiser, so compare against Optuna on real strategies |
| marsopt 0.2.0 | `Study(direction="maximize", random_state=..., verbose=False)`, `study.optimize(objective, n_trials=...)`, `study.best_trial["variables"]` | The objective receives a `Trial`: `trial.suggest_float(name, low, high, log=)`, `suggest_int`, `suggest_categorical(name, [str, ...])`. **Categories must be strings**, so map numbers and booleans. Sequential only as far as the API shows. Flat spaces only. Return a large penalty instead of NaN, since NaN handling is not documented |

**BUTChC specifics worth honouring**

- `verbose=False` in the app. The default prints one line per trial.
- A rough budget is 10 times the number of parameters. Under about 50 trials on a smooth low-dimensional problem, prefer Optuna.
- It learns each parameter separately, so parameters that interact strongly inside one branch need reparameterising: optimise `fast` and `gap` and set `slow = fast + gap`, rather than two coupled windows with an invalid corner.
- Warm start: `start_prob_tree=previous["prob_tree"]` with the **same `direction`**, saved with `pickle`, not JSON.
- Narrowing: `prune(searchspace, result, narrow=2, return_tree=True)` returns a smaller space and a matching tree for a second round. The raw `prob_tree` no longer matches a pruned space.
- Pass only small values through its `**kwargs` (a job path, not a DataFrame). Each worker loads data once in its process initialiser into a module global.

### Robustness tools

- **Train/test split.** Optimise on the first part of the history and report the held-out part separately, always labelled.
- **Walk-forward.** `walk_forward(spec, window, step, mode)` with `mode` of anchored or rolling. For each window, optimise in-sample, freeze the best parameters, and run the next out-of-sample slice. **Compile signals on the full history and slice afterwards**, so indicators are warmed up and nothing looks ahead (a signal at bar *t* depends only on bars up to *t*). Stitch the out-of-sample returns into one equity curve and report the ratio of out-of-sample to in-sample metric.
- **Parameter stability.** After a run, re-evaluate a neighbourhood around the best point (about ±10% per parameter) and report the median metric and the share of neighbours that stay profitable. A sharp peak with poor neighbours is overfit.
- **Trial count and deflated Sharpe.** Always display how many configurations were tried. Compute the deflated Sharpe ratio (Bailey and López de Prado) from the trial count and the spread of trial Sharpes, with the formula in a comment.
- **Monte Carlo.** Resample trade returns with replacement to show the spread of outcomes (figure in section 9).

### Background jobs

Streamlit reruns the script on every interaction, so a long optimisation cannot live in the session.

1. `start_job(job_spec)` writes `jobs/<job_id>/job.json` (spec, backend, space, budget, objective, constraints, seed) and launches `subprocess.Popen([sys.executable, "-m", "openstrategy_studio.worker", job_dir])`.
2. The worker creates its own `ProcessPoolExecutor(max_workers=workers, initializer=load_data_and_settings)`, runs the backend, and after each trial appends to `trials.parquet` and rewrites `progress.json` (done, budget, best so far, timestamp).
3. The Optimise page polls `progress.json` on a timer (the Streamlit fragment API, or a Refresh button; check the installed version). Cancel writes a `cancel` file that the worker checks after each trial.
4. Results survive closing the page and restarting Streamlit, because everything is on disk.

### Results page

Trials table sorted by metric; best parameters with an **Apply to spec** button that writes the literals into the strategy; convergence curve; 2-D heatmap for any pair of parameters (binned mean metric for non-grid runs); parallel coordinates; scatter of metric against one parameter; parameter importance (Optuna runs only); a button to run any listed trial as a normal backtest.

## 11. Streamlit app and UX

The app is a thin layer: each page reads and writes the one spec dict and calls functions from the modules above. No business logic lives in page files.

### Pages

| Page | Purpose |
| --- | --- |
| Home | Open or create a strategy (start from a template or a saved file), recent runs |
| 1 Data | Catalogue table, import (Yahoo, CSV, Parquet), validation report, refresh, delete |
| 2 Build | Edit the spec, with a live preview |
| 3 Backtest | Run, tear sheet, save run |
| 4 Optimise | Choose parameters, backend, budget, constraints; start, watch and cancel jobs |
| 5 Results | Optimisation results; compare saved runs side by side |
| 6 Settings | Data folder, theme, workers, vectorbtX settings, FRED key |

Use Streamlit's multipage mechanism as recommended by the installed version (a `pages/` folder or `st.navigation`).

### State

- `st.session_state["spec"]` is the single source of truth. Every form field is seeded from the spec on each run and writes back through one function, `set_path(spec, "features.fast.window.value", n)`. Do not let widgets hold their own copies.
- Streamlit drops the state of widgets that were not rendered on a run, which is the classic way for a multi-tab form to lose input. Seeding every widget from the spec each time avoids it.
- Session state vanishes on a browser refresh, so every edit also writes `drafts/current.json`. Home offers to restore it.
- Explicit **Save** writes `strategies/<name>.json`.

### The Build page: several ways into one spec

Tabs, all editing the same dict:

1. **Instruments.** A grid of `Instrument` fields per symbol. Read `MODE_RULES` from `vectorbtx.frontend.prepare` (it is not re-exported on `vbt.x`; expose it through `engine.py`) so fields a mode rejects are disabled with the library's own reason, and required fields are marked. Exception: `base_currency` stays enabled on FX spread bets and CFDs, because carry needs it. Show `vbt.x.mode_defaults(mode)` as placeholder text so a blank cell reads as "default". Spread bets force commission to zero, so disable those fields.
2. **Features.** Add a block from a searchable picker (indicators, patterns grouped by family and direction, price transforms). Parameters are generated from the block's declaration.
3. **Rules.** A nested rule builder: groups of conditions joined by all/any/not, up to three levels deep. Rendered by one recursive function. Widget keys are built from the node's path in the tree so they stay unique and stable.
4. **Portfolio.** Selection, weighting and schedule forms (section 6). Hidden when the strategy is a single-symbol signal strategy.
5. **Risk.** Sizing choice (radio: exactly one), stop (none, fixed, trailing), take profit, leverage budget, with units shown (points, £ per point, shares).
6. **Account.** Cash, base currency, bar frequency, FX source, rates source.
7. **Spec and code.** The spec as editable JSON with live validation; the exported Python script, read-only.

Above the tabs: a **Start from** dropdown of templates (MA cross, RSI reversion, Donchian breakout, candlestick reversal, momentum top-N, 60/40 monthly rebalance). A template is a spec JSON file shipped as package data and read with `importlib.resources`.

Next to every numeric field is an **Optimise** toggle. Turning it on reveals min, max, step and log, and stores the `opt` object from section 5. The same control works for categorical fields with a multi-select of allowed values.

### Live preview

A panel beside the form draws the selected symbol's candlestick chart with every feature overlaid and entry and exit markers on it, and shows signal counts. It recomputes when the spec changes, cached on a hash of the data, features and rules sections. It must call out the failures that otherwise cost an afternoon: a rule that never fires (zero signals), entries that are always immediately followed by exits, and a pattern block that is neither bullish nor bearish. It also lists `validate_spec` problems in plain English.

### Performance and Streamlit gotchas

These are from general Streamlit knowledge, not the uploaded sources. Check each against the installed version.

- `st.tabs` still executes the code of every tab on each run, so wrap figure builders in `st.cache_data` or use a selector control to render one view at a time.
- Cache data loads with `st.cache_data` keyed on symbols, timeframe, dates and the parquet files' modification times.
- The first backtest in a fresh environment compiles numba code and can take many seconds. vectorbt marks many kernels `njit(cache=True)`, so later starts are faster. Show a spinner that says so.
- The default upload size limit is 200 MB; raise it in a packaged `.streamlit/config.toml` if large CSVs are expected.
- `st.plotly_chart(fig, theme=None)` always (section 9).

### Errors

One `try/except` around the run call, displaying `str(exception)` in an error box with the traceback in an expander. No other defensive wrapping.

## 12. Testing and acceptance

Tests are plain `pytest` functions in `tests/`, run with `uv run pytest`. They never touch the network. One `conftest.py` provides a seeded synthetic OHLCV generator (a random walk where `high >= max(open, close)` and `low <= min(open, close)` always hold) and a few small committed CSV fixtures. No class-based tests and no mocking framework unless a test truly needs one.

| # | Test | Pass criterion |
| --- | --- | --- |
| 1 | **Golden strategy.** Compile the MA-cross spec from section 5 and run it on fixture data with a static rate table (no FRED); also run the hand-written vectorbtX call for the same strategy | `pf.stats()`, `pf.trades.stats()` and `pf.returns_stats()` equal within 1e-9. The owner will supply real numbers after the build, from `examples/strategies/01_ma_cross_spreadbet_fx.py` on their machine, saved as text, so the same check also runs against Yahoo data once |
| 2 | **No lookahead.** For every block and rule operator, compile on the full data and on the data truncated at several bars *k* | Signals up to *k* are identical in both. Covers indicators, patterns, the higher-timeframe filter and weights |
| 3 | **Short-signal shift.** A single-bar short entry pulse at bar *t* | The simulated short fills at the open of bar *t+1*, never *t* (section 3, item 1) |
| 4 | **Stops.** Hand-built bars: a gap through the stop, a bar containing both stop and target, a trailing stop on a rising then falling series | Gap fills at the open; the same-bar rule matches what the UI documents (section 3, item 5) |
| 5 | **Spec validation.** Two sizing arguments, fixed plus trailing stop, a dangling feature reference, a feature cycle, rules plus weights together | Each returns the expected plain-English message; a good spec returns an empty list |
| 6 | **Spec round trip.** Load, save, load; apply parameters; hash | Identical dicts; stable hash; `apply_params` leaves no `opt` keys |
| 7 | **Every block compiles.** Loop over `BLOCKS` on synthetic data with default parameters | Output has the close frame's shape, no infinities, NaN only in warm-up |
| 8 | **Export equivalence.** Run `export_script(spec)` and the in-app path on the same fixture, for a signal spec and a portfolio spec | Identical `pf.stats()` |
| 9 | **Portfolio construction.** Equal-weight monthly rebalance; `calendar & threshold` composition with the drift anchor; weights rows sum to 1 or 0; `min_trade` suppression | Matches an independent pandas calculation; drift report non-empty when trades are suppressed |
| 10 | **Data normalisation.** Tz-aware, duplicated and unsorted frames; LSE-style and US-style daily indexes together | Clean, sorted, tz-naive output; combined load yields the common days, not zero rows; bracket violations are reported |
| 11 | **Pattern cache.** Same pattern with different keywords | Two cache entries; identical keywords hit the cache |
| 12 | **Optimiser plumbing.** A cheap synthetic objective with a known optimum over a mixed space with a conditional branch | Each backend gets within tolerance at a fixed budget; NaN returns do not crash or win; seeded BUTChC runs reproduce, including with an executor |
| 13 | **Job runner.** Start a job with budget 5; poll; cancel a second job | `progress.json` advances; cancel stops the worker; `trials.parquet` readable after the process exits |
| 14 | **App smoke.** Streamlit's `AppTest` (`streamlit.testing.v1`, verify the current API) loads each page with each template | No uncaught exceptions |

A test passes only when it passes after a clean `uv sync` from the lock file. Continuous integration is a single job, run on both Linux and Windows, that runs `uv sync` then `uv run pytest`.

## 13. Phases and milestones

&#91;embedded content: roadmap · 5 phases, 5 gates\]

Phases run in order, and the next one starts only when the gate before it passes. Test numbers refer to the table in section 12. No dates are given because the owner has not set any.

## 14. Risks

### Risks

| Risk | Mitigation |
| --- | --- |
| vectorbtX documentation and code disagree (section 3, item 8), and the library is young | All calls go through `engine.py`; golden and unit tests pin behaviour; report mismatches upstream as they are found |
| Lookahead through short signals, higher-timeframe filters or walk-forward slicing | Truncation tests (section 12, tests 2 and 3) run on every block |
| Optimisers find overfit parameters | Trade-count floor, parameter-stability check, trial count and deflated Sharpe always shown, walk-forward (section 10) |
| Compile time, not simulation, dominates optimisation | Per-process feature cache; BUTChC and Optuna have low per-trial overhead |
| Numba cold start in every worker process | vectorbt caches compiled kernels on disk; run one warm-up evaluation per worker at start |
| Streamlit's rerun model fights long jobs | Background subprocess with files on disk (section 10) |
| Standalone packaging of numba plus Streamlit is fragile | Defer to phase 4 and spike three options before choosing: a uv-based first-run installer script, a PyInstaller or similar bundle with a desktop webview wrapper, and Docker. Expect hidden-import and numba-cache trouble in the bundle option |
| Yahoo returns an empty frame instead of an error behind a TLS-intercepting proxy or antivirus (noted in the vectorbtX README) | Detect an empty download and show the explanation and fix |
| BUTChC is a young project with its own benchmarks, and marsopt is 0.2.0 with a Cython build | Both are optional backends behind a common interface, with a comparison harness against Optuna and random search |
| Time stops, breakeven and step trailing need numba callbacks | Fixed templates only, phase 3, each with hand-built-bar tests |
| Licence: vectorbt carries Commons Clause | Personal and internal use only; do not offer paid hosting |

## Annex: API and syntax reference

Snippets are taken from the uploaded sources unless marked **from memory**, which means the library was not in the uploads and the builder must check it against current documentation.

### A. vectorbtX call patterns

**A1. Single-instrument signal strategy.**

```python
import vectorbtx as vbt            # lower-case import name; distribution is "vectorbtX"

pf = vbt.x.SingleBacktest(
    close=close, open=open_, high=high, low=low,    # all four, keyword-only, no defaults
    entries=long_entries, exits=long_exits,         # shifted one bar by the front door
    short_entries=short_entries.shift(1, fill_value=False),   # NOT shifted by the front door
    short_exits=short_exits.shift(1, fill_value=False),       # (they pass through **kwargs)
    mode="spreadbet", asset_class="fx", point_size=0.0001,
    base_currency="EUR", quote_currency="USD",
    min_size=1.0, increment=0.01, rounding="floor",
    spread_points=0.9, slippage=0.0002, leverage=30,
    risk_pct=0.01, stop_points=60, target_points=120,   # exactly one sizing argument
    rates=rates,                                        # optional: enables holding costs
    init_cash=10_000, freq="D",
    upon_opposite_entry="reverse", seed=42,             # passthrough to vectorbt
)
```

**A2. Multi-asset rebalanced portfolio** (from example 03).

```python
calendar = vbt.x.calendar_schedule(price.index, freq="M")
drifted  = vbt.x.drifted_weights(weights, price, schedule=calendar)   # schedule= is essential
schedule = calendar & vbt.x.threshold_schedule(drifted, weights, tolerance=0.02)

pf = vbt.x.PortfolioBacktest(
    close=price, open=open_price, high=high, low=low,
    weights=weights,              # DataFrame, same shape as price; rows sum to 1 (or 0 for cash)
    schedule=schedule,            # boolean Series on the same index
    instruments=universe,         # Universe, dict of overrides, or a .csv/.xlsx path
    base_currency="GBP", fx=fx,   # fx is a price Series such as GBPUSD=X, or an FXTable
    min_trade=0.005,
    init_cash=250_000, freq="D",
)
```

**A3. The four ways to supply instruments.**

```python
# 1. flat keywords, applied to every column            (SingleBacktest and PortfolioBacktest)
vbt.x.PortfolioBacktest(..., mode="equity", point_size=0.01, commission_flat=10.0)
# 2. shared base plus per-symbol overrides             (PortfolioBacktest only)
vbt.x.PortfolioBacktest(..., mode="equity", point_size=0.01,
                        instruments={"VOD.L": dict(commission_flat=10.0),
                                     "NVDA":  dict(mode="cfd", contract_size=1.0)})
# 3. a path; mode is per row in the file
vbt.x.PortfolioBacktest(..., instruments="universe.csv")
# 4. objects
universe = vbt.x.load_universe({"MSFT": vbt.x.Instrument(mode="equity", point_size=0.01,
                                quote_currency="USD", cost_currency="USD",
                                commission_flat=10.0, commission_min=1.0)})
universe.to_frame()   # eyeball every resolved default before trusting it
```

Price column names and universe keys must match exactly, both ways. Any mismatch raises `UniverseError`.

**A4. Loading and normalising Yahoo data** (from example 01).

```python
data = vbt.YFData.download(SYMBOL, start=START, end=END)       # list of symbols: add missing_index="drop"

def by_date(series):
    series = series.rename(SYMBOL)                  # single-symbol get() is named after the FIELD
    series.index = series.index.tz_localize(None).normalize()   # Yahoo is tz-aware; FRED is naive
    return series

close, open_ = by_date(data.get("Close")), by_date(data.get("Open"))
high,  low   = by_date(data.get("High")),  by_date(data.get("Low"))
```

**A5. FX conversion**, always before any signal, on all four fields with the same table.

```python
close = vbt.x.to_base_currency(fields["Close"], base="GBP", fx=fx, instruments=universe, fx_markup=0.005)
# repeat for Open, High, Low, and for any benchmark
```

**A6. Rates.**

```python
rates = vbt.x.FREDData.download_rates(["GBP", "USD"], start=START, end=END)       # needs FRED_API_KEY
rates = vbt.x.static_rate_table(["GBP", "USD"], close.index, {"GBP": 0.0525, "USD": 0.0533})
```

**A7. Settings.**

```python
vbt.settings["vectorbtx"]["rounding"] = "nearest"        # never touch vbt.settings["portfolio"]
```

### B. Behaviours that differ from what you might expect

| You might expect | What actually happens | Where |
| --- | --- | --- |
| Positional price arguments | All four prices are required and keyword-only; none has a default | `frontend/backtest.py` |
| Entries fill on the signal bar's close | They fill at the **next bar's open**, always | `_next_bar`, `price=fill` |
| Every signal array is shifted for you | Only `entries`, `exits` and `schedule`. `short_entries` and `short_exits` are not | `frontend/backtest.py` |
| A fixed stop and a trailing stop together | `trail_points` overwrites `stop_points` | `resolve_stops` in `prepare.py` |
| Passing both sizing and risk arguments is allowed | Exactly one of `risk_pct`, `risk_amount`, `stake`, `notional`, `size` | `prepare.py` |
| `risk_pct` compounds with equity | It sizes off `init_cash` | `PortfolioBacktest` docstring |
| A spread bet can carry commission | Commission is forced to zero; the cost lives in `spread_points` | `_FORCED_ZERO` |
| A spread bet has a contract size | It is rejected: the stake is the size | `MODE_RULES` |
| A raw `fees=` plus a commission field adds up | It raises `BacktestError` | `collision_check` |
| Blank universe cells are NaN | They are `None` until emit, because NaN means something to vectorbt | `resolve/` |
| Rebalance targets can be forward-filled | Never: forward-fill rebalances every bar. Targets are NaN off-schedule | `to_targets` |
| `drifted_weights(weights, prices)` is enough | Without `schedule=` drift grows without bound and the threshold test is always true | `construct/schedule.py` |
| Equity prices should be adjusted everywhere | Equity defaults to `adj_close`; spread bets and CFDs default to `close` | `mode_defaults` |
| `min_size` can be anything | Must be a multiple of `increment`; sub-minimum sizes are rejected to zero, not rounded | `quantize` |
| Auto slippage works without volume | It raises; pass `volume=` | `resolve_costs` |
| A minimum commission is just a fee | It is precomputed per order and needs `notional` | `costs/` |
| Carry needs only rates | Rates for **both** currencies, and the instrument must name both | `docs/DATA.md` |
| `end_of_day` matters on daily bars | It is inert on daily bars | `day_multipliers` |
| Z-score high means good | Negative Z means outcomes cluster in streaks; it is a dependency test | `docs/API.md` section 11 |
| FX rates fetched in both directions are consistent | They are not; derive reciprocals from one canonical direction | `FXTable.from_canonical` |
| Rates with a gap are filled silently | `fill_policy="ffill_with_warning"` warns; mismatched tz-aware and naive indexes are rejected | `docs/DATA.md` |

### C. vectorbt 1.1.0: `from_signals` essentials

```python
pf = vbt.Portfolio.from_signals(
    close, entries, exits, short_entries, short_exits,   # four direction-aware boolean arrays
    size=..., price=open_, open=open_, high=high, low=low,
    sl_stop=0.02, sl_trail=True, tp_stop=0.04,           # FRACTIONS of price (0.02 = 2%), never negative
    accumulate=False, upon_opposite_entry="reverse",
    freq="D", init_cash=10_000,
)
```

- With `short_entries` or `short_exits` set, `entries` and `exits` are long-only signals. With neither, `direction` decides what `entries` and `exits` mean, and with `direction="both"` the `exits` array becomes short entries. The Studio always uses the explicit four-array form.
- Enum-valued arguments take lower-case strings, as the vectorbtX example does with `upon_opposite_entry="reverse"`; vectorbt maps them with `map_enum_fields`. Members (from `portfolio/enums.py`):
  - `OppositeEntryMode`: Ignore, Close, CloseReduce, Reverse, ReverseReduce
  - `StopExitMode`: Close, CloseReduce, Reverse, ReverseReduce
  - `StopExitPrice`: StopLimit, StopMarket, Price, Close
  - `StopEntryPrice`: ValPrice, Price, FillPrice, Close
  - `DirectionConflictMode`: Ignore, Long, Short, Adjacent, Opposite
- **Stop fill price** (`get_stop_price_nb`): for a long stop-loss, the stop price is the reference times `(1 - stop)`. If the bar opens at or below it, the fill is the **open** (gap through the stop). Otherwise, if the stop lies between low and high, the fill is the stop price. Otherwise no hit.
- **No time-stop argument.** Callbacks exist for custom logic: `signal_func_nb`, `adjust_sl_func_nb`, `adjust_tp_func_nb`.
- **Indicators.** `vbt.MA.run(close, window=50, ewm=False)` returns an object; read `.ma`. Crossing between two indicator outputs is `fast.ma_crossed_above(slow)`. Between any two series, use the generic accessor `a.vbt.crossed_above(b)`. Pass scalar parameters: a list creates a MultiIndex of columns.

### D. ta\_patterns 1.2.1

```python
import ta_patterns as tap
import ta_patterns.chart_patterns as cp

sig = tap.hammer(o, h, l, c, shadow_factor=1.5, require_trend=False)   # candlestick: int8 array
sig = cp.double_top(o, h, l, c, pivot_n=3, tol=0.06, mode="forming")   # chart pattern
names = tap.list_all_patterns(direction="bullish", module="chart")      # for the picker
all_sig = tap.scan_all_patterns(o, h, l, c, v=v, mode="confirmed", window=100, pivot_n=5, pivot_pct=None)
score = tap.net_score_all(o, h, l, c, v=v, mode="confirmed")
```

- Inputs are NumPy arrays or Series (the index is stripped). Output is `int8`, `+1` bullish, `-1` bearish, `0` none, always the input length.
- Call a single detector as `getattr(tap, name)` for candlesticks and `getattr(cp, name)` for chart patterns. Store both module and name in the spec, because `scan_all_patterns` prefixes colliding chart names with `cp_` and direct calls do not.
- Candlestick detectors take shape thresholds only (`penetration`, `shadow_factor`, `require_trend`, `tol`) and no `mode` or `window`. Chart detectors add `mode` (`confirmed` or `forming`), `window` (default 100), `pivot_n` (default 5), `pivot_pct`. Harmonics add `fib_tol`; busted patterns `reversal_bars`, `reversal_pct`; flags `pole_bars`, `min_pole`, `max_retrace`.
- Direction classes: bullish patterns emit only `+1`, bearish only `-1`, non-directional emit `+1` when the shape appears, bidirectional chart patterns emit either sign. Classification sets for the picker: `ta_patterns.scanner.BULLISH`, `BEARISH`, `DIRECTIONAL`, `NON_DIRECTIONAL`, and `ta_patterns.chart_patterns.BULLISH`, `BEARISH`, `DIRECTIONAL`, `BIDIRECTIONAL`, `NON_DIRECTIONAL`.
- Signals sit on the bar that **completes** the pattern and use only past bars, so the front door's next-open fill is the correct execution.
- Speed: about 170 ms for all 300 detectors on 1,254 bars. Eight detectors (`abc_correction`, `measured_move_up`, `measured_move_down`, `hs_top`, `hs_bottom`, `busted_hs_*`, `complex_hs_*`) scale roughly with the square of the pivot count, so cap `window` or exclude them on very long histories. Five volume patterns run only when `v` is given.
- Elliott Wave patterns are intentionally absent.

### E. BUTChC 0.6.1

```python
from butchc import BUTChC_optimize, prune

space = {
    "features.fast.window": {"min": 10, "max": 100, "int": True},
    "entry_family": {
        "values": ["ma_cross", "rsi"],
        "next_level": {                                  # the only nesting key; no implicit nesting
            "ma_cross": {"slow_gap": {"min": 10, "max": 200, "int": True}},
            "rsi":      {"rsi_window": {"min": 5, "max": 30, "int": True},
                         "rsi_level":  {"min": 20.0, "max": 40.0}},
        },
    },
}

def objective(config):                    # config holds only the parameters on the sampled path
    ...
    return float(metric)                  # float("nan") for a failed or constrained trial

result = BUTChC_optimize(space, objective, budget=150, direction="maximize",
                         seed=0, batch=4, executor=pool, verbose=False)
result["best_params"], result["best_value"], result["history"], result["prob_tree"]
```

- A categorical node accepts only `values`, `next_level`, `prior`, `prior_strength`. Anything else raises `SearchSpaceError` before the first trial.
- Continuous nodes: `min` and `max`, optional `int` or `log` (not both; `log` needs `min > 0`).
- Names may repeat across sibling branches but must be unique among parameters that can occur together.
- `executor` is anything with an order-preserving `map` or a `submit`; BUTChC never creates a pool. Process pools need a picklable top-level objective.
- Failed trials: return NaN. `best_value` is reported in the objective's own units even with `direction="minimize"`.

### F. marsopt 0.2.0

```python
from marsopt import Study, Trial

def objective(trial: Trial) -> float:
    window = trial.suggest_int("window", 10, 100)
    alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
    kind = trial.suggest_categorical("kind", ["sma", "ema"])      # strings only
    return metric

study = Study(direction="maximize", random_state=0, verbose=False)
study.optimize(objective, n_trials=200)
study.best_trial["variables"]; study.trials; study.objective_values
```

### G. Optuna and scipy (from memory)

```python
import optuna
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
def objective(trial):
    window = trial.suggest_int("window", 10, 100)
    family = trial.suggest_categorical("family", ["ma_cross", "rsi"])
    if family == "rsi":                                   # define-by-run conditional space
        level = trial.suggest_float("level", 20.0, 40.0)
    return metric
study.optimize(objective, n_trials=200)
study.best_params; study.trials_dataframe()

from scipy.optimize import differential_evolution          # minimises: negate the metric
res = differential_evolution(neg_objective, bounds=[(10, 100), (100, 300)],
                             integrality=[True, True], seed=0, polish=False)
```

### H. plotly-themes 0.1.0

```python
import plotly_themes                    # registers the themes on import; import name uses an underscore
plotly_themes.registered_themes()       # names available at runtime
fig.update_layout(template="nord")
plotly_themes.add_glow(fig, layers=4, width_growth=4.0, max_opacity=0.18)   # line figures only
css = plotly_themes.font_face_css()     # inject into the Streamlit page for bundled fonts
```

### I. Streamlit (from memory)

```python
st.plotly_chart(fig, theme=None, use_container_width=True)   # newer releases may prefer width="stretch"
st.session_state.setdefault("spec", load_template("ma_cross"))
```

### J. uv cheat sheet

```bash
uv sync                          # create .venv from uv.lock
uv add <package>                 # add a dependency and update the lock
uv run <command>                 # run inside the environment
uv python pin 3.12               # write .python-version
uv sync --native-tls             # behind a TLS-inspecting proxy or antivirus
```

The vectorbtX README notes that behind a TLS-inspecting proxy, `uv` fails every fetch with `UnknownIssuer` unless `--native-tls` is set. The data downloaders need separate certificate settings (`CURL_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`), described in that README. The commands in it (`setx`, `ssl.enum_certificates`) are Windows ones, which suggests the owner's machine; confirm the target operating system.

### K. Where to read in the sources

| Topic | File |
| --- | --- |
| The front door, shifts, routing | `vectorbtx/frontend/backtest.py`, `single.py` |
| Mode rules (line 45), stops (line 439), sizing, collisions | `vectorbtx/frontend/prepare.py` |
| Weights, schedules, targets | `vectorbtx/construct/` |
| Universe loading and the spreadsheet schema | `vectorbtx/resolve/`, `docs/API.md` section 2 |
| Complete strategies to copy | `examples/strategies/01_ma_cross_spreadbet_fx.py` to `03`; case matrix in `examples/cases/` |
| `from_signals` signature (line 2048) and docstring | `vectorbt/portfolio/base.py` |
| Stop logic and the signal simulator | `vectorbt/portfolio/nb.py` (`generate_stop_signal_nb` line 1580, `get_stop_price_nb` line 1861, `simulate_from_signals_nb` from about line 1930) |
| Built-in indicators | `vectorbt/indicators/basic.py`; crossing accessor `vectorbt/generic/accessors.py` line 1507 |
| Pattern catalogue and parameters | `ta_patterns/docs/api_reference.md`, `patterns_catalog.md`, `concepts.md` |
| Search-space rules, parallelism, pruning | `BUTChC/docs/api.md`, `examples.md`, `limitations.md` |
| Themes and fonts | `plotly-themes/README.md`, `src/plotly_themes/` |
