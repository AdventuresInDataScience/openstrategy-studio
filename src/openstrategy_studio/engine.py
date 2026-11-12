"""The engine adapter: the ONLY file that imports vectorbtx.

Everything that touches vectorbtX goes through here, so any API drift is a
one-file fix. That includes the construct helpers compile.py needs
(calendar_schedule, drifted_weights, ...) and the mode rules the Build page
shows.

The core is `call_plan(spec)`, a pure function that decides which front door to
call and with which keyword arguments. Each argument is either a literal from
the spec ("lit") or a named runtime value ("var": close, entries, rates ...).
`run_backtest` fills the runtime values and makes the call; `export_script`
renders the same plan as source code, so the two cannot disagree about
arguments.

Lines marked ⚠️ VERIFY were read from the vectorbtX docs or the plan but could
not be executed when this was written. docs/VERIFY.md lists them all.
"""

import importlib.metadata
from pathlib import Path

import pandas as pd

from .spec import SIZING_KEYS, apply_params, strategy_kind


# ⚠️ Which of stop-loss or take-profit wins when one bar's range contains both
# was NOT established from the source (plan section 3, item 5). Phase 0: read
# simulate_from_signals_nb (vectorbt/portfolio/nb.py, ~line 2381 on), run
# tests/test_engine.py::test_same_bar_stop_and_target, and set this to "stop" or
# "target". The Risk tab shows it to the user.
SAME_BAR_STOP_RULE = None


def vectorbtx():
    """The vectorbtx package (import name always lower case)."""
    import vectorbtx as vbt
    return vbt


def vectorbt():
    """Plain vectorbt: importing it registers the .vbt pandas accessors."""
    import vectorbt
    return vectorbt


def versions():
    """Installed versions recorded with every saved run."""
    found = {}
    for name in ("vectorbt", "vectorbtx", "ta_patterns", "butchc", "optuna", "plotly", "pandas", "numpy",
                 "streamlit", "plotly-themes", "marsopt", "openstrategy-studio"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


# ---------------------------------------------------------------------------
# Settings and mode rules
# ---------------------------------------------------------------------------

def apply_vectorbtx_settings(values):
    """Write Studio settings into vbt.settings["vectorbtx"].

    ⚠️ Never touch vbt.settings["portfolio"]: vectorbtX owns those defaults.
    """
    settings = vectorbtx().settings["vectorbtx"]
    for key, value in values.items():
        if value is not None:
            settings[key] = value


def mode_defaults(mode):
    """vbt.x.mode_defaults(mode): shown as placeholder text so a blank cell reads as 'default'."""
    return dict(vectorbtx().x.mode_defaults(mode))


def mode_rules(mode):
    """{"required": set of fields, "rejected": {field: reason}} for one instrument mode.

    MODE_RULES lives in vectorbtx.frontend.prepare (line 45) and is not
    re-exported on vbt.x. ⚠️ VERIFY: its exact shape was not executed; this reads
    any key containing "require" as required fields and any key containing
    "reject", "forbid" or "disallow" as rejected fields (a dict of reasons or a
    list). If the shape differs, every field stays enabled.
    """
    from vectorbtx.frontend.prepare import MODE_RULES

    rules = MODE_RULES.get(mode) or {}
    required = set()
    rejected = {}
    if isinstance(rules, dict):
        for key, value in rules.items():
            name = str(key).lower()
            if "require" in name:
                required |= set(value)
            elif any(word in name for word in ("reject", "forbid", "disallow")):
                if isinstance(value, dict):
                    rejected.update({field: str(reason) for field, reason in value.items()})
                else:
                    rejected.update({field: f"not used in {mode} mode" for field in value})
    return {"required": required, "rejected": rejected}


def known_modes():
    from vectorbtx.frontend.prepare import MODE_RULES
    return sorted(MODE_RULES)


# ---------------------------------------------------------------------------
# Construct helpers used by compile.compile_weights
# ---------------------------------------------------------------------------
# Thin on purpose: they exist so compile.py never imports vectorbtx itself.

def calendar_schedule(index, freq, on="first"):
    # `on` is a real argument even though the docs omit it.
    return vectorbtx().x.calendar_schedule(index, freq=freq, on=on)


def drifted_weights(weights, prices, schedule):
    # ⚠️ schedule is required here on purpose: without it drift grows without
    # bound from the first bar and the threshold test is always true.
    return vectorbtx().x.drifted_weights(weights, prices, schedule=schedule)


def threshold_schedule(drifted, weights, tolerance):
    return vectorbtx().x.threshold_schedule(drifted, weights, tolerance=tolerance)


def always_schedule(index):
    return vectorbtx().x.always_schedule(index)


def price_weight(prices):
    # price_weight takes only `prices`; the docs show an `index` argument that does not exist.
    return vectorbtx().x.price_weight(prices)


# ---------------------------------------------------------------------------
# Instruments, currencies, FX and rates
# ---------------------------------------------------------------------------

def instrument_fields(spec):
    """{symbol: Instrument fields} for dict-style instruments, minus Studio-only keys.

    `price_field` is consumed by the data loader (adjusted or raw close), because
    the front door receives only a close frame. ⚠️ VERIFY that the front door does
    not need it as well.
    """
    instruments = spec.get("instruments") or {}
    out = {}
    for symbol in spec["data"]["symbols"]:
        fields = dict(instruments.get(symbol) or {})
        fields.pop("price_field", None)
        # Blank cells mean "use the mode default". vectorbtX keeps them as None
        # until emit, because NaN means something to vectorbt, so drop them here.
        out[symbol] = {key: value for key, value in fields.items() if value is not None and value != ""}
    return out


def universe_path(spec, data_dir=None):
    """The universe file named by instruments.from_file, resolved against <data_dir>/universes."""
    path = Path(spec["instruments"]["from_file"])
    if not path.is_absolute() and data_dir is not None:
        path = Path(data_dir) / "universes" / path
    return str(path)


def load_universe(spec, data_dir=None):
    vbt = vectorbtx()
    if "from_file" in (spec.get("instruments") or {}):
        # ⚠️ VERIFY: load_universe accepting a .csv/.xlsx path (the front door does).
        return vbt.x.load_universe(universe_path(spec, data_dir))
    fields = instrument_fields(spec)
    return vbt.x.load_universe({symbol: vbt.x.Instrument(**values) for symbol, values in fields.items()})


def universe_code(spec, data_dir=None):
    """The same universe as source code, for export_script."""
    if "from_file" in (spec.get("instruments") or {}):
        return f"vbt.x.load_universe({universe_path(spec, data_dir)!r})"
    parts = [f"{symbol!r}: vbt.x.Instrument(**{values!r})" for symbol, values in instrument_fields(spec).items()]
    return "vbt.x.load_universe({" + ", ".join(parts) + "})"


def adjusted_symbols(spec, data_dir=None):
    """Symbols whose close is built from adj_close.

    Equity mode defaults to adj_close; spread bets and CFDs default to close,
    because adjusting rescales the whole history and changes point counts.
    """
    instruments = spec.get("instruments") or {}
    if "from_file" in instruments:
        # The resolved universe knows each row's mode and price_field.
        frame = load_universe(spec, data_dir).to_frame()
        if "price_field" in frame:
            return sorted(str(symbol) for symbol, field in frame["price_field"].items() if field == "adj_close")
        return sorted(str(symbol) for symbol, mode in frame["mode"].items() if mode == "equity")
    out = []
    for symbol in spec["data"]["symbols"]:
        fields = instruments.get(symbol) or {}
        default = "adj_close" if fields.get("mode") == "equity" else "close"
        if (fields.get("price_field") or default) == "adj_close":
            out.append(symbol)
    return out


def point_sizes(spec, columns, data_dir=None):
    """Each column's point size: the instrument's own, else its mode's default."""
    if "from_file" in (spec.get("instruments") or {}):
        frame = load_universe(spec, data_dir).to_frame()
        return [float(frame.loc[column, "point_size"]) for column in columns]
    fields = instrument_fields(spec)
    sizes = []
    for column in columns:
        values = fields[column]
        if values.get("point_size") is None:
            sizes.append(float(mode_defaults(values["mode"])["point_size"]))
        else:
            sizes.append(float(values["point_size"]))
    return sizes


def needed_currencies(spec, data_dir=None):
    """Every currency the run touches: instruments' base, quote and cost currencies plus the account's.

    ⚠️ Carry on an FX instrument needs rates for BOTH currencies and the
    instrument must name both, or carry silently resolves to zero.
    """
    found = set()
    account = spec.get("account") or {}
    if account.get("base_currency"):
        found.add(account["base_currency"])
    if "from_file" in (spec.get("instruments") or {}):
        frame = load_universe(spec, data_dir).to_frame()
        for column in ("base_currency", "quote_currency", "cost_currency"):
            if column in frame:
                found |= {str(value) for value in frame[column].dropna()}
    else:
        for fields in instrument_fields(spec).values():
            for key in ("base_currency", "quote_currency", "cost_currency"):
                if fields.get(key):
                    found.add(fields[key])
    rates = account.get("rates") or {}
    found |= set((rates.get("values") or {}).keys())
    return sorted(found)


def build_rates(spec, index, offline=False, static_rate=None, data_dir=None):
    """The rate table for holding costs, or None when the spec gives no rates.

    FRED needs a free key in FRED_API_KEY and caches under $XDG_CACHE_HOME or
    ~/.cache. Offline, the FRED call is still tried (it may be served from that
    cache) and falls back to a static table at `static_rate` for every currency.
    """
    rates = (spec.get("account") or {}).get("rates")
    if not rates:
        return None
    vbt = vectorbtx()
    currencies = needed_currencies(spec, data_dir)
    if rates["source"] == "static":
        # ⚠️ A flat rate misreads any period spanning a rate cycle; the UI says so.
        return vbt.x.static_rate_table(currencies, index, dict(rates["values"]))
    if offline:
        try:
            return vbt.x.FREDData.download_rates(currencies, start=index[0], end=index[-1])
        except Exception:   # noqa: BLE001  (network refused: use the static fallback)
            return vbt.x.static_rate_table(currencies, index, {code: static_rate or 0.0 for code in currencies})
    return vbt.x.FREDData.download_rates(currencies, start=index[0], end=index[-1])


def rates_code(spec, data_dir=None):
    rates = spec["account"]["rates"]
    currencies = needed_currencies(spec, data_dir)
    if rates["source"] == "static":
        return f"vbt.x.static_rate_table({currencies!r}, close.index, {dict(rates['values'])!r})"
    return f"vbt.x.FREDData.download_rates({currencies!r}, start=close.index[0], end=close.index[-1])"


def prepare_data(spec, data, data_dir=None):
    """Convert prices to the account's base currency, before any signal is computed.

    Open, high, low AND close are converted with the same FX table so the bracket
    high >= close >= low still holds. Volume is not a price and stays as it is.
    """
    account = spec.get("account") or {}
    if not (account.get("fx") and account.get("base_currency")):
        return data
    universe = load_universe(spec, data_dir)
    out = dict(data)
    for field in ("open", "high", "low", "close"):
        # Signature: to_base_currency(data, base, fx_table=None, quote_currency=None,
        # instruments=None, fx_markup=0.005, fill_policy="ffill_with_warning",
        # validate=True, fx=None). The examples pass fx=<Series>, instruments=<Universe>.
        out[field] = vectorbtx().x.to_base_currency(data[field], base=account["base_currency"],
                                                    fx=data["fx"], instruments=universe)
    return out


# ---------------------------------------------------------------------------
# The call plan
# ---------------------------------------------------------------------------

def call_plan(spec, data_dir=None):
    """Which front door to call and every keyword argument, from the spec alone.

    {"front_door", "single", "kind", "args": {keyword: ("lit", value) | ("var", name)}}
    `spec` must have its parameters applied.
    """
    kind = strategy_kind(spec)
    symbols = spec["data"]["symbols"]
    instruments = spec.get("instruments") or {}
    account = spec.get("account") or {}
    risk = spec.get("risk") or {}
    # SingleBacktest takes flat instrument keywords and rejects instruments=.
    # Its base_currency keyword is the INSTRUMENT's (EUR for EURUSD), so a
    # reporting-currency conversion goes through PortfolioBacktest instead.
    single = (kind != "portfolio" and len(symbols) == 1 and "from_file" not in instruments
              and not account.get("base_currency"))
    args = {}

    # All four prices, by keyword: they are keyword-only with no defaults.
    args["close"] = ("var", "close")
    args["open"] = ("var", "open_")
    args["high"] = ("var", "high")
    args["low"] = ("var", "low")

    # Instruments. ⚠️ Costs come ONLY from instrument fields, never a raw fees=
    # passthrough: fees= beside commission_pct raises BacktestError.
    if "from_file" in instruments:
        args["instruments"] = ("lit", universe_path(spec, data_dir))
    else:
        fields = instrument_fields(spec)
        if single:
            for key, value in fields[symbols[0]].items():
                args[key] = ("lit", value)
        else:
            args["instruments"] = ("lit", fields)
        # Auto slippage raises without volume.
        if any(values.get("slippage") == "auto" for values in fields.values()):
            args["volume"] = ("var", "volume")

    # The route, mirroring the front door's own dispatch.
    if kind == "signals":
        # Always the explicit four-array form, never direction="both" (which turns
        # exits into short entries, so nothing but a stop closes a position).
        args["entries"] = ("var", "entries")
        args["exits"] = ("var", "exits")
        args["short_entries"] = ("var", "short_entries")      # already shifted by compile
        args["short_exits"] = ("var", "short_exits")
    elif kind == "portfolio":
        args["weights"] = ("var", "weights")
        args["schedule"] = ("var", "schedule")
        args["min_trade"] = ("var", "min_trade")
    if risk.get("leverage_budget"):
        args["leverage_budget"] = ("lit", True)

    # Sizing: exactly one of risk_pct, risk_amount, stake, notional, size.
    # ⚠️ risk_pct sizes off init_cash, not compounding equity.
    sizing = risk.get("sizing") or {}
    for key in SIZING_KEYS:
        if key in sizing:
            args[key] = ("lit", sizing[key])
    # Stops: fixed or trailing, never both (trail_points overwrites stop_points).
    stop = risk.get("stop") or {}
    if stop.get("kind") == "fixed":
        args["stop_points"] = ("lit", stop["points"])
    elif stop.get("kind") == "trailing":
        args["trail_points"] = ("lit", stop["points"])
    if risk.get("target"):
        args["target_points"] = ("lit", risk["target"]["points"])
    # Extra exits: fixed numba templates that adjust the stop (and, for a time
    # stop, the target) on every bar. See exit_rules.py.
    from .exit_rules import FAR_TARGET_POINTS, exit_settings
    extra_exits = exit_settings(spec)
    if any(value == value for value in extra_exits.values()):          # any not NaN
        args["adjust_sl_func_nb"] = ("var", "adjust_sl_func_nb")
        args["adjust_sl_args"] = ("var", "adjust_sl_args")
        if extra_exits["time_bars"] == extra_exits["time_bars"]:
            args["adjust_tp_func_nb"] = ("var", "adjust_tp_func_nb")
            args["adjust_tp_args"] = ("var", "adjust_tp_args")
            if "target_points" not in args:
                # ⚠️ VERIFY: a far target is accepted, so that adjust_tp_nb is called.
                args["target_points"] = ("lit", FAR_TARGET_POINTS)

    # Account. ⚠️ Always pass freq=, or annualised metrics are filtered out.
    args["init_cash"] = ("lit", account["init_cash"])
    args["freq"] = ("lit", account["freq"])
    if account.get("base_currency"):
        args["base_currency"] = ("lit", account["base_currency"])
    if account.get("fx"):
        args["fx"] = ("var", "fx")
    if account.get("rates"):
        # ⚠️ The 01_ma_cross example comment says rates opt into the level-3
        # from_order_func path, but frontend/backtest.py applies holding costs
        # AFTER the simulation (apply_holding_costs). Trust the code: there is no
        # cash feedback from holding costs during the run.
        args["rates"] = ("var", "rates")

    # Passed through untouched: upon_opposite_entry, seed, cash_sharing, group_by ...
    for key, value in (spec.get("engine_extra") or {}).items():
        args[key] = ("lit", value)
    return {"front_door": "SingleBacktest" if single else "PortfolioBacktest",
            "single": single, "kind": kind, "args": args}


def benchmark_plan(plan):
    """The same front door with no signals, weights or risk: buy and hold.

    ⚠️ VERIFY (plan section 9, TEST): that a call with neither signals nor weights
    is buy-and-hold, and how its returns are best handed to returns_stats.
    """
    drop = {"entries", "exits", "short_entries", "short_exits", "weights", "schedule", "min_trade",
            "leverage_budget", "stop_points", "trail_points", "target_points", "upon_opposite_entry",
            "adjust_sl_func_nb", "adjust_sl_args", "adjust_tp_func_nb", "adjust_tp_args", *SIZING_KEYS}
    args = {key: value for key, value in plan["args"].items() if key not in drop}
    return dict(plan, kind="hold", args=args)


def _slice(value, start, end):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.loc[start:end]
    return value


def run_backtest(spec, data, params=None, window=None, data_dir=None, offline=False,
                 static_rate=None, benchmark=False):
    """Apply trial parameters, compile, and make the one vectorbtX call.

    window=(start, end) compiles on the FULL history and slices afterwards, so
    indicators are warmed up and nothing looks ahead (walk-forward and train/test
    splits rely on this). Returns {"pf", "spec", "data", "compiled", "plan"}.
    """
    from .compile import compile_signals, compile_weights   # compile imports this module

    spec = apply_params(spec, params)
    prepared = prepare_data(spec, data, data_dir)
    plan = call_plan(spec, data_dir)
    if benchmark:
        plan = benchmark_plan(plan)
        compiled = {}
    elif plan["kind"] == "signals":
        compiled = compile_signals(spec, prepared)
    elif plan["kind"] == "portfolio":
        compiled = compile_weights(spec, prepared)
    else:
        compiled = {}

    if window is not None:
        start, end = window
        prepared = {key: _slice(value, start, end) for key, value in prepared.items()}
        compiled = {key: _slice(value, start, end) for key, value in compiled.items()}

    runtime = dict(compiled)
    runtime.update(close=prepared["close"], open_=prepared["open"], high=prepared["high"],
                   low=prepared["low"], volume=prepared["volume"], fx=prepared.get("fx"))
    if "rates" in plan["args"]:
        runtime["rates"] = build_rates(spec, prepared["close"].index, offline=offline,
                                       static_rate=static_rate, data_dir=data_dir)
    if plan["single"]:
        # SingleBacktest takes Series: pick the one symbol's column everywhere.
        symbol = spec["data"]["symbols"][0]
        runtime = {key: value[symbol] if isinstance(value, pd.DataFrame) else value
                   for key, value in runtime.items()}

    if "adjust_sl_func_nb" in plan["args"]:
        from . import exit_rules
        columns = list(prepared["close"].columns)
        runtime.update(exit_rules.runtime_values(spec, runtime, point_sizes(spec, columns, data_dir)))

    kwargs = {}
    for keyword, (how, value) in plan["args"].items():
        kwargs[keyword] = runtime[value] if how == "var" else value
    front_door = getattr(vectorbtx().x, plan["front_door"])
    pf = front_door(**kwargs)
    return {"pf": pf, "spec": spec, "data": prepared, "compiled": compiled, "plan": plan}


# ---------------------------------------------------------------------------
# Vectorised grid: many trials in one front-door call (plan phase 3)
# ---------------------------------------------------------------------------

def vectorisable(spec, space):
    """(True, "") when every trial of `space` can share one call; otherwise (False, the reason).

    Each trial becomes its own column with its own copy of the instrument and its
    own starting cash, so only parameters that change the SIGNALS (features and
    rules) can vary between columns.
    """
    from .exit_rules import active
    from .optimise import is_conditional

    applied = apply_params(spec)
    if strategy_kind(applied) != "signals":
        return False, "Only entry and exit rule strategies can run many trials per call."
    if len(applied["data"]["symbols"]) != 1 or "from_file" in (applied.get("instruments") or {}):
        return False, "Only single-symbol strategies with their own instrument settings can run many trials per call."
    if (applied.get("account") or {}).get("base_currency"):
        return False, "Currency conversion is not supported with many trials per call."
    if active(applied):
        return False, "The extra exits keep state per trade; run them with the ordinary grid."
    if is_conditional(space):
        return False, "Alternatives are searched by BUTChC and Optuna only."
    outside = sorted(path for path in space if not path.startswith(("features.", "rules.")))
    if outside:
        return False, f"These parameters do not only change the signals: {', '.join(outside)}."
    return True, ""


def run_vectorised(spec, data, params_list, window=None, data_dir=None, offline=False, static_rate=None):
    """One PortfolioBacktest call for many trials. Returns (pf, trial columns).

    ⚠️ VERIFY (tests/test_vectorised.py): that PortfolioBacktest accepts
    cash_sharing=False and group_by=False, so each column is an independent
    account and pf[column] matches the same trial run alone.
    """
    from .compile import compile_signals

    base = apply_params(spec, params_list[0])
    symbol = base["data"]["symbols"][0]
    prepared = prepare_data(base, data, data_dir)
    columns = [f"trial_{number}" for number in range(len(params_list))]
    signals = {key: {} for key in ("entries", "exits", "short_entries", "short_exits")}
    for column, params in zip(columns, params_list):
        compiled = compile_signals(apply_params(spec, params), prepared)
        for key in signals:
            signals[key][column] = compiled[key][symbol]
    runtime = {key: pd.DataFrame(values) for key, values in signals.items()}
    for field, name in (("open", "open_"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume")):
        runtime[name] = pd.DataFrame({column: prepared[field][symbol] for column in columns})
    if window is not None:
        runtime = {key: value.loc[window[0]:window[1]] for key, value in runtime.items()}

    plan = call_plan(base, data_dir)
    instrument = instrument_fields(base)[symbol]
    args = {key: value for key, value in plan["args"].items() if key not in instrument}
    args["instruments"] = ("lit", {column: dict(instrument) for column in columns})
    args["cash_sharing"] = ("lit", False)
    args["group_by"] = ("lit", False)
    if "rates" in args:
        runtime["rates"] = build_rates(base, runtime["close"].index, offline=offline, static_rate=static_rate,
                                       data_dir=data_dir)
    kwargs = {keyword: runtime[value] if how == "var" else value for keyword, (how, value) in args.items()}
    return vectorbtx().x.PortfolioBacktest(**kwargs), columns


def backtest_error():
    """vectorbtX's BacktestError class, for the optimiser's 'invalid trial' check."""
    vbt = vectorbtx()
    return getattr(vbt.x, "BacktestError", None) or getattr(vbt, "BacktestError", ValueError)
