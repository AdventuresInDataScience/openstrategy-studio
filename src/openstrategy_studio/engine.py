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

Phase 0 has been run against vectorbt 1.1.0 and vectorbtX ace29e2, and re-run on
16293a2 (which fixes what it found): every call site here was executed, and the
findings (with what changed) are in docs/VERIFY.md. Anything still marked
⚠️ VERIFY is listed there as open.
"""

import importlib.metadata
import re
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import (FRACTION_KEYS, SIZING_KEYS, apply_params, cash_model, strategy_kind, trading_mode,
                   when_cash_short)

# A stop or target from a line: the vectorbtX argument each unit goes to, by whether the stop
# trails (or it is the target). compile_signals names its frames the same.
STOP_LINE_ARGS = {"multiple": {"fixed": "stop_points", "trailing": "trail_points", "target": "target_points"},
                  "level": {"fixed": "stop_price", "trailing": "trail_price", "target": "target_price"}}
# Every stop and target argument a plan may send.
STOP_ARGS = ("stop_points", "target_points", "trail_points", "sl_stop", "tp_stop", "sl_trail",
             "stop_price", "target_price", "trail_price")

# ⚠️ warnings.catch_warnings swaps process-wide state, so two runs at once (a second
# browser tab, or the start-up warm-up below beside a user's run) would swap each other's
# warning lists: one run's notes could vanish, or show on the other. Every front-door call
# that records warnings holds this lock. Numba holds its own compile lock, so a run that
# arrives while the warm-up is compiling waits for that code either way.
RUN_LOCK = threading.Lock()


# ⚠️ Which of stop-loss or take-profit wins when one bar's range contains both.
# Phase 0 answer: the STOP. vectorbt 1.1.0 computes the stop-loss price first and
# only looks at the take-profit when no stop-loss price was hit (get_stop_price_nb
# for the SL, then `if np.isnan(stop_price)` for the TP, in both
# simulate_from_signals_nb and simulate_from_signal_func_nb). That is the
# pessimistic reading of a bar whose order of high and low is unknown.
# tests/test_engine.py::test_same_bar_stop_and_target pins it. The Risk tab shows it.
SAME_BAR_STOP_RULE = "stop"

# Stops are measured from the fill and pay the spread: vectorbtX's front door sets
# stop_entry_price="price" and stop_exit_price="stopmarket" itself (vectorbtX 16293a2,
# docs/UPSTREAM.md issues 5 and 6). vectorbt's own defaults anchor a stop at the entry
# bar's close and fill it with no slippage, so nothing here may pass them.
#
# It also checks a stop on the bar the position is entered, by simulating each bar as
# its open plus the rest of the bar. That includes the extra exits (exit_rules.py): since
# vectorbtX 9f4cf54 each adjust callback is wrapped so it sees bars and is called once per
# bar at the open, exactly as on whole bars (docs/UPSTREAM.md, issue 14, fixed).


def vectorbtx():
    """The vectorbtx package (import name always lower case)."""
    import vectorbtx as vbt
    return vbt


def vectorbt():
    """Plain vectorbt: importing it registers the .vbt pandas accessors."""
    import vectorbt
    return vectorbt


def versions():
    """Installed versions recorded with every saved run, and part of the range check's cache key.

    ⚠️ A library installed from git keeps its version number from commit to commit (vectorbtX is 0.1.0 at
    every one), so the commit it was installed from is added, as in "0.1.0+47afc3e". Without it a bump
    that changes results (vectorbtX issue 41) left the range check showing the trade counts from before
    it, and a saved run could not say which vectorbtX it was made with.
    """
    import json

    found = {}
    for name in ("vectorbt", "vectorbtx", "ta_patterns", "butchc", "optuna", "plotly", "pandas", "numpy",
                 "streamlit", "plotly-themes", "marsopt", "nevergrad", "scipy", "openstrategy-studio"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
            continue
        # pip and uv write where a package came from (PEP 610); a git install names its commit there.
        origin = distribution.read_text("direct_url.json")
        commit = (json.loads(origin).get("vcs_info") or {}).get("commit_id") if origin else None
        found[name] = f"{distribution.version}+{commit[:7]}" if commit else distribution.version
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

    MODE_RULES lives in vectorbtx.frontend.prepare and is not re-exported on
    vbt.x. Verified shape (phase 0): {mode: {"required": (fields,), "rejected":
    (fields,)}}, tuples, with the reasons kept separately. This reads any key
    containing "require" as required fields and any key containing "reject",
    "forbid" or "disallow" as rejected fields (a dict of reasons or a list), so a
    renamed key degrades to "every field stays enabled" rather than an error.
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
    the front door receives only a close frame. Verified: vectorbtX uses the field
    only for its defaults and validation, never in the simulation.
    """
    instruments = spec.get("instruments") or {}
    out = {}
    for symbol in spec["data"]["symbols"]:
        fields = dict(instruments.get(symbol) or {})
        # One trading mode per strategy, set on the Setup step: every instrument takes it.
        fields["mode"] = trading_mode(spec)
        fields.pop("price_field", None)
        # `financing` is reserved in vectorbtX (9fa427e): financing is the rate table plus
        # the Settings page's markup for every instrument, and a per-instrument value was
        # never read. The Build page no longer offers it; an older spec may still hold one.
        fields.pop("financing", None)
        # Blank cells mean "use the mode default". vectorbtX keeps them as None
        # until emit, because NaN means something to vectorbt, so drop them here.
        fields = {key: value for key, value in fields.items() if value is not None and value != ""}
        # ⚠️ Except leverage and the point size: a blank one is the Studio's usual one, named here
        # so the plan, the export and point_sizes (the extra exits' stop distances) all say the same.
        # Leverage differs from vectorbtX's own blank bought outright, on FX spot and in a portfolio
        # (usual_leverage); the point size for an index or a share priced in pence (usual_point_size).
        if "leverage" not in fields:
            fields["leverage"] = usual_leverage(fields["mode"], fields.get("asset_class"), strategy_kind(spec))
        if "point_size" not in fields:
            fields["point_size"] = usual_point_size(fields["mode"], fields.get("asset_class"),
                                                    fields.get("quote_currency"))
        # ⚠️ And slippage: vectorbtX's blank is none at all, the Studio's is usual_slippage. Named
        # here so a blank cell and a filled-in one run the same, and the export says it.
        if "slippage" not in fields:
            fields["slippage"] = usual_slippage(fields["mode"], fields.get("asset_class"))
        out[symbol] = fields
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
        # load_universe accepts a .csv, .xlsx or .xls path (verified in phase 0).
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
    default = "adj_close" if trading_mode(spec) == "equity" else "close"
    for symbol in spec["data"]["symbols"]:
        fields = instruments.get(symbol) or {}
        if (fields.get("price_field") or default) == "adj_close":
            out.append(symbol)
    return out


def point_sizes(spec, columns, data_dir=None):
    """Each column's point size, as the run uses it.

    ⚠️ instrument_fields always names one (the instrument's own, else usual_point_size), so this
    agrees with the plan. It used to fall back to the mode's plain default, 0.01 on a spread
    bet even for an FX pair, while vectorbtX filled in the pip: the extra exits then measured
    their stop in points 100 times too coarse on an FX instrument with no point size.
    """
    if "from_file" in (spec.get("instruments") or {}):
        frame = load_universe(spec, data_dir).to_frame()
        return [float(frame.loc[column, "point_size"]) for column in columns]
    fields = instrument_fields(spec)
    return [float(fields[column]["point_size"]) for column in columns]


def usual_point_size(mode, asset_class=None, quote_currency=None):
    """The point size the Studio fills in: the market's own convention for one point.

    FX: the pip for the quote currency (vectorbtX's fx_pip: 0.0001, or 0.01 for yen and a few
    others), in every way of trading. An index: 1.0, an index point. A share priced in pence
    (Yahoo's .L quotes, GBp): 1.0, a penny. Anything else: 0.01, a cent (or a penny). vectorbtX
    fills in the same since 1de5744 (docs/UPSTREAM.md issue 25); the Studio still names it
    itself, because an instrument still being typed into the grid may not build yet.
    ⚠️ VERIFY per broker for commodities: gold and oil spread bets quote points of 0.1 or 0.01.
    """
    from vectorbtx.instruments.defaults import fx_pip

    asset_class = asset_class or mode_defaults(mode).get("asset_class")
    if asset_class == "fx" or mode == "fx_spot":
        return float(fx_pip(quote_currency))
    if asset_class == "index" or (asset_class == "stocks" and quote_currency in ("GBp", "GBX")):
        return 1.0
    return 0.01


# The slippage the Studio fills in, as a FRACTION of the price on each fill (the grid shows it in
# %), paid on top of the spread: 0.0001 is 0.01%, 1.1 pips on EURUSD at 1.10 or half a point on an
# index at 5,000. A starting point for a liquid market filled at the next bar's open, the owner's
# call (20 November 2026: a blank of none flattered every run, and a cell with no usual value
# invited points), not a measurement. ⚠️ VERIFY against your own fills: a thin share, a gap or a
# fill at FX's daily rollover can cost far more.
USUAL_SLIPPAGE = {"fx": 0.0001, "index": 0.0001, "stocks": 0.0002, "commodities": 0.0002}


def usual_slippage(mode, asset_class=None):
    """The slippage the Studio fills in for a way of trading and an asset class (USUAL_SLIPPAGE)."""
    asset_class = "fx" if mode == "fx_spot" else (asset_class or mode_defaults(mode).get("asset_class"))
    return USUAL_SLIPPAGE.get(asset_class, USUAL_SLIPPAGE["stocks"])


def usual_leverage(mode, asset_class=None, kind="signals"):
    """The leverage the Studio fills in for a way of trading, an asset class and a kind of strategy.

    The owner's rule (17 November 2026): unlevered unless it is a spread bet or a CFD.
    Spread bets and CFDs trade at the usual retail limits (30x FX, 10x indices and
    commodities, 5x single stocks). Everything else is 1x, no borrowing: shares bought
    outright and FX spot ("most portfolios and naked trading are non-margin"). That is
    vectorbtX's own table since 1de5744 (docs/UPSTREAM.md issue 24), so it is read from
    there. Every portfolio is 1x too, which is the Studio's rule, not vectorbtX's.
    """
    if kind == "portfolio":
        return 1.0
    # Read from vectorbtX's table, not by building an Instrument: one that is still incomplete (a
    # CFD with no contract size, which the grid already flags) would refuse to build.
    from vectorbtx.instruments.defaults import max_leverage

    return float(max_leverage(mode, asset_class or mode_defaults(mode).get("asset_class")))


# The currency a ticker's market quotes in, by Yahoo's exchange suffix: the Studio's usual
# reading when a symbol is added, which the grid shows and the user can change. ⚠️ A guess
# from the name, not from the data: a London ETF quoted in dollars ends in .L too.
TICKER_SUFFIX_CURRENCY = {
    ".L": "GBp", ".IL": "USD", ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR",
    ".BR": "EUR", ".LS": "EUR", ".VI": "EUR", ".HE": "EUR", ".IR": "EUR", ".SW": "CHF", ".TO": "CAD", ".V": "CAD",
    ".AX": "AUD", ".NZ": "NZD", ".T": "JPY", ".HK": "HKD", ".SI": "SGD", ".KS": "KRW", ".NS": "INR", ".BO": "INR",
    ".ST": "SEK", ".OL": "NOK", ".CO": "DKK", ".SA": "BRL", ".MX": "MXN",
}
# Yahoo's index tickers (^...) outside the US, by the currency their points are worth.
INDEX_CURRENCY = {"^FTSE": "GBP", "^FTMC": "GBP", "^GDAXI": "EUR", "^FCHI": "EUR", "^STOXX50E": "EUR",
                  "^AEX": "EUR", "^IBEX": "EUR", "^N225": "JPY", "^HSI": "HKD", "^AXJO": "AUD", "^GSPTSE": "CAD",
                  "^SSMI": "CHF"}


def usual_market(ticker):
    """{"asset_class", "quote_currency", "base_currency"} read from a ticker by Yahoo's conventions.

    EURUSD=X is an FX pair (EUR against USD); JPY=X is Yahoo's dollar against the yen; ^FTSE an
    index in pounds; GC=F a future (a commodity), in dollars; VOD.L a share in pence. Anything
    else is a share in dollars. Only what the ticker says: the asset class is left out when it
    does not say (a plain share is the mode's default, stocks).
    """
    ticker = str(ticker or "").upper()
    if ticker.endswith("=X"):
        pair = ticker[:-2]
        if len(pair) == 6 and pair.isalpha():
            return {"asset_class": "fx", "base_currency": pair[:3], "quote_currency": pair[3:]}
        if len(pair) == 3 and pair.isalpha():
            return {"asset_class": "fx", "base_currency": "USD", "quote_currency": pair}
    if ticker.startswith("^"):
        return {"asset_class": "index", "quote_currency": INDEX_CURRENCY.get(ticker, "USD")}
    if ticker.endswith("=F"):
        return {"asset_class": "commodities", "quote_currency": "USD"}
    if "-" in ticker and ticker.rsplit("-", 1)[1] in ("USD", "EUR", "GBP"):
        return {"quote_currency": ticker.rsplit("-", 1)[1]}           # a crypto pair such as BTC-USD
    for suffix, currency in TICKER_SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return {"quote_currency": currency}
    return {"quote_currency": "USD"}


def usual_instrument(mode, kind="signals", ticker=None, given=None):
    """Every instrument field the Build page's grid shows for `mode`, at its usual value.

    What a new symbol starts with, and what a cell follows when the way of trading changes (the
    owner's call, 20 November 2026: no blank cells, even where the value is 0 or false). The
    market (asset class and currencies) comes from `given` (the instrument's own fields) where
    it names them, else from the ticker (usual_market); the rest from the market and the mode:
    the usual leverage and point size (usual_leverage, usual_point_size), a contract of 100,000
    units (a standard lot) on FX CFDs and FX spot and 1 on other CFDs, vectorbtX's mode defaults
    for sizes, the price field, commission and the spread, the usual slippage (usual_slippage), no
    other costs, and floor rounding. Each value is what a blank runs as: vectorbtX's own, or, for
    leverage, the point size and slippage, the Studio's, which instrument_fields names for a blank.
    The market is the exception: a blank leaves it unknown.

    ⚠️ No borrow_rate: no rate is published, and a blank one makes the run say the short was
    borrowed for free, which is the point (CLAUDE.md, "The borrow fee follows how a thing is
    traded").
    """
    given = given or {}
    market = usual_market(ticker)
    defaults = mode_defaults(mode)
    rejected = set(mode_rules(mode)["rejected"])
    asset_class = given.get("asset_class") or market.get("asset_class") or defaults.get("asset_class")
    if mode == "fx_spot":
        asset_class = "fx"
    quote = given.get("quote_currency") or market.get("quote_currency")
    out = {"asset_class": asset_class, "quote_currency": quote,
           "leverage": usual_leverage(mode, asset_class, kind),
           "point_size": usual_point_size(mode, asset_class, quote),
           "increment": defaults.get("increment"), "min_size": defaults.get("min_size"),
           "allow_fractional": bool(defaults.get("allow_fractional")),
           "cost_currency": given.get("cost_currency") or quote,
           "commission_pct": defaults.get("commission_pct") or 0.0,
           "commission_flat": defaults.get("commission_flat") or 0.0,
           "commission_per_unit": 0.0, "commission_min": 0.0,
           "spread_points": defaults.get("spread_points"), "slippage": usual_slippage(mode, asset_class),
           "rounding": "floor", "leverage_type": "instrument_level",
           "price_field": defaults.get("price_field")}
    if asset_class == "fx":
        out["base_currency"] = given.get("base_currency") or market.get("base_currency")
    if mode in ("cfd", "fx_spot"):
        out["contract_size"] = 100000.0 if asset_class == "fx" else 1.0
    # Only the fields this way of trading takes (base_currency stays on an FX pair: carry needs it).
    return {name: value for name, value in out.items()
            if value is not None and (name not in rejected or (name == "base_currency" and asset_class == "fx"))}


def point_meaning(spec, data_dir=None):
    """{symbol: what one point is for it, in its market's words}: "1 cent (0.01)", "1 pip (0.0001)".

    Stops, targets, spreads and a spread bet's stake are all counted in points: the point
    size of price. On a share that is a cent (or a penny), so a 1.00 move is 100 points; on
    an FX pair the market counts pips (0.0001, or 0.01 when the quote is yen), and a point
    is named a pip only when the point size is the pip (vectorbtX's fx_pip, by quote
    currency); on an index it is usually 1.0, an index point.
    """
    from vectorbtx.instruments.defaults import fx_pip

    symbols = list(spec["data"]["symbols"])
    if "from_file" in (spec.get("instruments") or {}):
        return {symbol: f"{size:g} of the price" for symbol, size in zip(symbols, point_sizes(spec, symbols, data_dir))}
    out = {}
    for (symbol, fields), size in zip(instrument_fields(spec).items(), point_sizes(spec, symbols, data_dir)):
        mode = fields["mode"]
        asset_class = fields.get("asset_class") or mode_defaults(mode).get("asset_class")
        quote = fields.get("quote_currency")
        if asset_class == "fx" or mode == "fx_spot":
            pip = float(fx_pip(quote))
            out[symbol] = (f"1 pip ({size:g})" if abs(size - pip) < 1e-12 else
                           f"{size:g} of the price, {size / pip:g} of a pip ({pip:g})")
        elif asset_class == "stocks" and quote in ("GBp", "GBX"):
            # A share priced in pence (Yahoo's .L quotes): a point of 1.0 is a penny.
            out[symbol] = "1p (1.0: the price is in pence)" if abs(size - 1.0) < 1e-12 else \
                f"{size:g}p (the price is in pence)"
        elif asset_class == "stocks" and abs(size - 0.01) < 1e-12:
            out[symbol] = "1p (0.01)" if quote == "GBP" else "1 cent (0.01)"
        elif asset_class == "index" and abs(size - 1.0) < 1e-12:
            out[symbol] = "1 index point (1.0)"
        else:
            out[symbol] = f"{size:g} of the price"
    return out


def default_leverage(spec):
    """{symbol: the leverage each instrument trades at}: its own, else the usual one (usual_leverage)."""
    out = {}
    for symbol, fields in instrument_fields(spec).items():
        out[symbol] = float(fields["leverage"])
    return out


def sizing_check(spec, data, entries=None, data_dir=None, compiled=None):
    """One trade of a rule strategy against what its account can fund, per symbol, before any run.

    The sizes are vectorbtX's own (its stop_plan, resolve_size and quantize_engine_size, at the
    open the order fills at), so this says what the run will do: a trade worth more than cash x
    leverage is cut down (or skipped), and one below the minimum size is not placed at all.
    Every stop is read as the run reads it: points, % of the price, a trailing stop's first
    distance, a multiple of a line or a line's level (from `compiled`, compile_signals' output;
    computed here when not given). With risk sizing and no stop, a trade is sized from its
    leverage, and the implied stop (1 / leverage of the price) is shown, marked as implied.
    `entries`: the signal-bar entry masks (long or short); each fills at the next bar's open, and
    the largest of those trades is shown. Without them, the entries in `compiled`, else the last
    bar's.

    ⚠️ A share of the balance or the equity (risk_of) is estimated from the starting cash here:
    the real one is only known from the run, and the Risk tab says so.

    [{"symbol", "price", "bar", "typical_range" (the median high-to-low of the 20 bars before),
      "stop_price", "stop_pct", "implied_stop", "size" (as traded), "asked" (before rounding to
      the increment), "min_size", "unit", "value", "times_cash", "leverage", "funded", "fits",
      "below_minimum"}], or [] when nothing is sized. Sizes are in the instrument's own unit:
      shares, lots, or a stake per point.
    """
    from vectorbtx.frontend.prepare import quantize_engine_size, resolve_size, stop_plan
    from vectorbtx.fx.convert import fx_rate_of
    from vectorbtx.modes.base import get_adapter

    spec = apply_params(spec)
    risk = spec.get("risk") or {}
    sizing = {key: value for key, value in (risk.get("sizing") or {}).items() if key in SIZING_KEYS}
    if strategy_kind(spec) != "signals" or not sizing:
        return []
    init_cash = float(spec["account"]["init_cash"])
    prepared = prepare_data(spec, data, data_dir)
    if compiled is None:
        from .compile import compile_signals   # compile imports this module

        compiled = compile_signals(spec, prepared, data_dir=data_dir)
    # The conversion rate rides on the converted frames, so it is read before they are coerced.
    point_fx = fx_rate_of(prepared["open"], index=pd.DataFrame(prepared["open"]).index,
                          columns=pd.DataFrame(prepared["open"]).columns)
    fill = pd.DataFrame(prepared["open"])
    columns = list(fill.columns)
    universe = load_universe(spec, data_dir)

    # The stops exactly as the run passes them, and the bars entries fill on (the next open).
    plan = call_plan(spec, data_dir)
    stop_args = {}
    for name in STOP_ARGS:
        if name in plan["args"]:
            how, value = plan["args"][name]
            stop_args[name] = compiled[value] if how == "var" else value

    def filled(mask):
        frame = pd.DataFrame(mask).reindex(index=fill.index, columns=columns).fillna(False).astype(bool)
        return frame.shift(1, fill_value=False)

    long_open, short_open = filled(compiled["entries"]), filled(compiled["short_entries"])
    stops = stop_plan(universe, fill, columns, point_fx=point_fx, long_open=long_open, short_open=short_open,
                      **stop_args)
    placed = long_open | short_open
    if stops["rejected"] is not None:
        placed = placed & ~stops["rejected"]
    with RUN_LOCK, warnings.catch_warnings():
        # Sizes below the minimum, entries with no stop distance and sizing from the leverage are
        # reported in the rows and on the Risk tab; vectorbtX's own warnings would go to the
        # console. (The lock: catch_warnings is process-wide, see RUN_LOCK.)
        warnings.simplefilter("ignore")
        wanted = resolve_size(universe, fill, columns, init_cash=init_cash, stop_distance=stops["risk_distance"],
                              stop_columns=stops["stop_columns"], order_bars=placed, **sizing)
        traded = quantize_engine_size(universe, wanted, columns)
    risk_sized = any(key in sizing for key in ("risk_pct", "risk_amount"))
    if entries is not None:
        chosen = filled(entries)
    else:
        chosen = placed
    rows = []
    for column in columns:
        instrument = universe[column].filled()
        adapter = get_adapter(instrument.mode)
        fills = chosen[column].to_numpy() if column in chosen else np.zeros(len(fill), dtype=bool)
        if not fills.any():
            fills = np.zeros(len(fill), dtype=bool)
            fills[-1] = True
        values = (traded[column].abs() * fill[column]).to_numpy()
        candidates = np.where(fills & np.isfinite(wanted[column].to_numpy()))[0]
        if not len(candidates):
            continue
        # The largest trade. Latest first, so on a tie (a stake the same on every bar) argmax,
        # which takes the first maximum, picks the latest.
        latest_first = candidates[::-1]
        scores = np.where(np.isfinite(values[latest_first]), values[latest_first], -1.0)
        bar = int(latest_first[int(np.argmax(scores))])
        price = float(fill[column].iloc[bar])
        size = float(adapter.from_engine_units(np.asarray([traded[column].iloc[bar]], dtype=float), instrument)[0])
        asked = float(adapter.from_engine_units(np.asarray([wanted[column].iloc[bar]], dtype=float), instrument)[0])
        value = abs(float(traded[column].iloc[bar])) * price
        leverage = float(instrument.leverage or 1.0)
        implied = False
        distance = None
        if column in stops["stop_columns"] and stops["risk_distance"] is not None:
            found = float(pd.DataFrame(stops["risk_distance"])[column].iloc[bar])
            distance = found if np.isfinite(found) and found > 0 else None
        elif risk_sized:
            # No stop: the trade is sized as if its margin were the risk, which a move of
            # 1 / leverage of the price takes (vectorbtX 1de5744, docs/UPSTREAM.md issue 28).
            distance, implied = price / leverage, True
        # How far price usually travels in one bar here: the median high-to-low range of the 20
        # bars before the fill. A stop inside it is likely hit within a bar or two of the entry.
        ranges = (pd.DataFrame(prepared["high"])[column] - pd.DataFrame(prepared["low"])[column]).iloc[max(0, bar - 20):bar]
        typical_range = float(ranges.median()) if len(ranges.dropna()) else None
        rows.append({"symbol": column, "price": price, "bar": fill.index[bar], "typical_range": typical_range,
                     "stop_price": distance, "stop_pct": distance / price if distance else None,
                     "implied_stop": implied,
                     "size": size, "asked": asked, "min_size": float(instrument.min_size or 0.0),
                     "unit": adapter.unit_name(instrument), "value": value,
                     "times_cash": value / init_cash, "leverage": leverage, "funded": init_cash * leverage,
                     "fits": value <= init_cash * leverage * (1 + 1e-9),
                     "below_minimum": size == 0 and abs(asked) > 0})
    return rows


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
    quotes = conversion_quotes(spec, data_dir)
    if quotes is None:
        return data
    out = dict(data)
    for field in ("open", "high", "low", "close"):
        # Signature (vectorbtX 16293a2): to_base_currency(data, base, fx_table=None,
        # quote_currency=None, instruments=None, fx_markup=None, fill_policy=
        # "ffill_with_warning", validate=True, fx=None). fx_markup=None reads
        # vbt.settings["vectorbtx"]["fx_markup"], which the Settings page writes.
        # The markup rides on the result's .fx_markup and the front door charges it as
        # slippage on every fill of a converted column (docs/UPSTREAM.md issue 9, fixed).
        # ⚠️ So these frames must reach the front door as ConvertedFrames: slicing them
        # with .loc keeps the attribute (tests/test_engine.py checks the fill price).
        # ⚠️ Points on a converted price are priced at point_size x the conversion rate, which
        # vectorbtX reads off these ConvertedFrames (docs/UPSTREAM.md issue 29, fixed in 1de5744).
        out[field] = vectorbtx().x.to_base_currency(data[field], base=account["base_currency"],
                                                    fx=data["fx"], quote_currency=quotes)
    return out


def conversion_quotes(spec, data_dir=None):
    """{symbol: the currency its price is converted from}, or None when nothing is converted.

    ⚠️ A spread bet keeps its own price. Its stake is money per point in the account's
    currency, so its profit is stake x points whatever currency the market is quoted in;
    converting the price would multiply that profit by the exchange rate. vectorbtX's own
    design converts bought-outright and FX spot prices only (its docs/dev/PLAN.md, D4); a CFD
    quoted in a foreign currency is converted too, one quoted in the account's is not. A
    spread bet is given the account currency here, so its rate is 1. A universe file may
    mix ways of trading, so this goes by each instrument's own mode.
    """
    base = (spec.get("account") or {})["base_currency"]
    universe = load_universe(spec, data_dir)
    quotes = {}
    for symbol in spec["data"]["symbols"]:
        instrument = universe[symbol]
        quotes[symbol] = base if instrument.mode == "spreadbet" else instrument.quote_currency
    if all(quote == base for quote in quotes.values()):
        return None
    return quotes


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
    fields = {} if "from_file" in instruments else instrument_fields(spec)
    # SingleBacktest takes flat instrument keywords and rejects instruments=, so it
    # is used only when every instrument field survives the flat form.
    # An instrument that names its base currency (any FX pair), or a reporting
    # currency, goes through PortfolioBacktest's instruments= form, which keeps every
    # field on the instrument unambiguously. Same implementation, one column.
    # (Phase 0, docs/UPSTREAM.md issue 2: vectorbtX ace29e2 read a flat base_currency=
    # as the ACCOUNT currency and dropped the pair. 16293a2 fixed that; the route stays
    # because it never depends on how a flat keyword is read.)
    single = (kind != "portfolio" and len(symbols) == 1 and "from_file" not in instruments
              and not account.get("base_currency") and not fields[symbols[0]].get("base_currency"))
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
        args["short_entries"] = ("var", "short_entries")      # shifted by the front door, like entries
        args["short_exits"] = ("var", "short_exits")
    elif kind == "portfolio":
        args["weights"] = ("var", "weights")
        args["schedule"] = ("var", "schedule")
        args["min_trade"] = ("var", "min_trade")
        # ⚠️ Always named, True or False: the front door defaults leverage_budget to True,
        # so leaving it out when the box is unticked would still apply the budget. It
        # exists on the weights route only (the signal route ignores it), so a rule
        # strategy never passes it, and validate_spec rejects one.
        args["leverage_budget"] = ("lit", bool(risk.get("leverage_budget")))

    # Sizing: exactly one of risk_pct, risk_amount, stake, notional, size, notional_pct. A share
    # of the account (risk_pct, notional_pct) is of the starting cash unless risk_of says the
    # balance or the equity at the signal bar (vectorbtX 1de5744 and 2b2d090, docs/UPSTREAM.md
    # issues 23 and 30). With no stop, a risk is sized from the leverage (issue 28).
    sizing = risk.get("sizing") or {}
    for key in SIZING_KEYS:
        if key in sizing:
            args[key] = ("lit", sizing[key])
    if sizing.get("risk_of") and any(key in sizing for key in FRACTION_KEYS):
        args["risk_of"] = ("lit", sizing["risk_of"])
    # Stops: fixed or trailing, never both (trail_points overwrites stop_points). A stop or target
    # that is off, or measured the other way, keeps its other distance in the spec (the optimiser
    # can switch it): only its kind and unit say what is sent.
    # In points: vectorbtX's own keywords, converted to fractions of the fill price. A trailing
    # stop in points stays that many points behind the best price (vectorbtX 1de5744 keeps the
    # distance in price; docs/UPSTREAM.md issue 27, fixed).
    # In % of the price: vectorbt's own sl_stop / tp_stop (0.02 = 2%), with sl_trail for a
    # trailing stop. The front door anchors them at the fill (stop_entry_price="price"), pays the
    # spread on the exit and checks them on the entry bar, exactly as it does point stops.
    # From a line (a feature or a price, read on the signal bar): a multiple of it is a per-bar
    # distance in points, and its level is where the stop sits (vectorbtX's stop_price, and
    # trailing, trail_price, which moves to each completed bar's level and never loosens). Both
    # are frames from compile_signals on the signal bar, which the front door shifts with the
    # signals (docs/UPSTREAM.md issues 26 and 31). A level on the wrong side of the fill (a long's
    # stop at or above it) is an order a broker rejects: vectorbtX does not place that entry.
    stop = risk.get("stop") or {}
    stop_unit = stop.get("unit", "points")
    if stop.get("kind") in ("fixed", "trailing"):
        trailing = stop["kind"] == "trailing"
        if stop_unit == "pct":
            args["sl_stop"] = ("lit", stop["pct"])
            if trailing:
                args["sl_trail"] = ("lit", True)
        elif stop_unit == "points":
            args["trail_points" if trailing else "stop_points"] = ("lit", stop["points"])
        else:
            name = STOP_LINE_ARGS[stop_unit]["trailing" if trailing else "fixed"]
            args[name] = ("var", name)
    target = risk.get("target") or {}
    if target.get("kind") == "fixed":
        target_unit = target.get("unit", "points")
        if target_unit == "pct":
            args["tp_stop"] = ("lit", target["pct"])
        elif target_unit == "points":
            args["target_points"] = ("lit", target["points"])
        else:
            name = STOP_LINE_ARGS[target_unit]["target"]
            args[name] = ("var", name)
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
            if not any(name in args for name in ("target_points", "tp_stop", "target_price")):
                # A far target is accepted, so adjust_tp_nb is called on every bar
                # (verified by the phase 3 gate in tests/test_exit_rules.py).
                args["target_points"] = ("lit", FAR_TARGET_POINTS)
    # No stop_entry_price / stop_exit_price: the front door sets both (see the top of this file).

    # Account. ⚠️ Always pass freq=, or annualised metrics are filtered out.
    args["init_cash"] = ("lit", account["init_cash"])
    args["freq"] = ("lit", account["freq"])
    if account.get("base_currency"):
        # The spec's "base_currency" is the REPORTING currency. vectorbtX's own
        # `base_currency=` names a pair's base when the flat terms describe FX, so the
        # account goes by the name that has only one meaning.
        args["account_currency"] = ("lit", account["base_currency"])
    if account.get("fx"):
        args["fx"] = ("var", "fx")
    if account.get("rates"):
        # ⚠️ The 01_ma_cross example comment says rates opt into the level-3
        # from_order_func path, but frontend/backtest.py applies holding costs
        # AFTER the simulation (apply_holding_costs). Trust the code: there is no
        # cash feedback from holding costs during the run.
        args["rates"] = ("var", "rates")

    # Cash, for a rule strategy on several symbols (docs/UPSTREAM.md issue 13).
    # ⚠️ Left unsaid, vectorbt gives each symbol its own copy of init_cash, so three
    # symbols on 10,000 trade 30,000. The plan always names the model, so the export
    # says which one it is too.
    model = cash_model(spec)
    if model is not None:
        args["cash"] = ("lit", model)
    if model == "shared":
        # Priority decides only between entries: vectorbtX places the orders that close a
        # position first, so the cash they free is there for the entries.
        priority = account.get("entry_priority") or {"by": "symbols"}
        if priority["by"] == "symbols":
            # Listed symbols first, in that order; vectorbtX puts any left out after them.
            args["entry_priority"] = ("lit", list(priority.get("order") or symbols))
        elif priority["by"] == "random":
            # Seeded by engine_extra's seed=, which validate_spec requires.
            args["entry_priority"] = ("lit", "random")
        else:
            # A per-bar score frame from compile_signals, on the signals' bar: the front
            # door shifts it with them.
            args["entry_priority"] = ("var", "entry_priority")
    if kind == "signals":
        # ⚠️ What an entry bigger than the account can fund does (cash x leverage, so a
        # risk-sized trade with a tight stop can outgrow it on one symbol too): "partial"
        # cuts it down to what is funded, "skip" places nothing. Named on every rule
        # strategy, so the export says which; the default per account is spec.when_cash_short's.
        args["when_cash_short"] = ("lit", when_cash_short(spec))

    # Passed through untouched: upon_opposite_entry, seed ... (validate_spec keeps the
    # cash passthroughs, cash_sharing, group_by and call_seq, out of a rule strategy).
    for key, value in (spec.get("engine_extra") or {}).items():
        args[key] = ("lit", value)
    return {"front_door": "SingleBacktest" if single else "PortfolioBacktest",
            "single": single, "kind": kind, "args": args}


# There is deliberately no "benchmark" front-door call. A call with neither signals
# nor weights is a hold funded at BUYING POWER: an FX spread bet "held" at 30x lost
# 247% while the market fell 6.5% (phase 0, VERIFY 2.7; vectorbtX issue 7), and handing those returns to
# returns_stats turned beta, M2 and the Treynor ratio into nonsense. The benchmark is
# vectorbt's own, pf.benchmark_value(): the market, held unlevered from the first bar.
# analysis.extract_results draws it.


def _slice(value, start, end):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.loc[start:end]
    return value


def run_backtest(spec, data, params=None, window=None, data_dir=None, offline=False, static_rate=None):
    """Apply trial parameters, compile, and make the one vectorbtX call.

    window=(start, end) compiles on the FULL history and slices afterwards, so
    indicators are warmed up and nothing looks ahead (walk-forward and train/test
    splits rely on this). Returns {"pf", "spec", "data", "compiled", "plan",
    "warnings"}: "warnings" is vectorbtX's notes about this run, in plain text.
    """
    from .compile import compile_signals, compile_weights   # compile imports this module

    spec = apply_params(spec, params)
    prepared = prepare_data(spec, data, data_dir)
    plan = call_plan(spec, data_dir)
    if plan["kind"] == "signals":
        compiled = compile_signals(spec, prepared, data_dir=data_dir)
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
    if plan["single"]:
        # SingleBacktest takes Series: pick the one symbol's column everywhere.
        symbol = spec["data"]["symbols"][0]
        runtime = {key: value[symbol] if isinstance(value, pd.DataFrame) else value
                   for key, value in runtime.items()}
    if "rates" in plan["args"]:
        # Added after the column pick: the rate table is keyed by CURRENCY, not symbol.
        runtime["rates"] = build_rates(spec, prepared["close"].index, offline=offline,
                                       static_rate=static_rate, data_dir=data_dir)

    if "adjust_sl_func_nb" in plan["args"]:
        from . import exit_rules
        columns = list(prepared["close"].columns)
        runtime.update(exit_rules.runtime_values(spec, runtime, point_sizes(spec, columns, data_dir)))

    kwargs = {}
    for keyword, (how, value) in plan["args"].items():
        kwargs[keyword] = runtime[value] if how == "var" else value
    front_door = getattr(vectorbtx().x, plan["front_door"])
    # ⚠️ vectorbtX reports what it could not price as WARNINGS, not errors: a short held
    # with no borrow_rate (borrowed for free), a carry leg with no currency, high/low that
    # do not bracket the close. Python sends warnings to the console, which a Studio user
    # never sees, so each run collects its own and returns them for the page to show.
    # "always", because Python otherwise shows a warning once per process and a long-lived
    # Streamlit server would swallow it on every run after the first.
    with RUN_LOCK, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        pf = front_door(**kwargs)
    notes = []
    for warning in caught:
        message = str(warning.message)
        if issubclass(warning.category, UserWarning) and message not in notes:
            notes.append(message)
    return {"pf": pf, "spec": spec, "data": prepared, "compiled": compiled, "plan": plan, "warnings": notes}


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
    if cash_model(applied) == "shared":
        # Checked before the symbol count so the reason names the real obstacle.
        return False, ("A shared account cannot run many trials per call: the vectorised grid gives each trial "
                       "its own column and its own cash. Run it with the ordinary grid.")
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
    """One PortfolioBacktest call for many trials. Returns [{"pf", "warnings"}], one per trial, in order:
    each trial's portfolio and vectorbtX's warnings about it, as run_backtest gives them.

    PortfolioBacktest accepts cash_sharing=False and group_by=False, so each column
    is an independent account; tests/test_vectorised.py checks that every trial
    equals the same trial run alone (the phase 3 gate).

    Each trial is selected as pf[[column]], a one-column portfolio like the one a
    single run returns. (vectorbtX ace29e2 also crashed every metric of pf[column] on
    a leveraged instrument; 16293a2 fixed that, docs/UPSTREAM.md issue 10.)
    """
    from .compile import compile_signals

    base = apply_params(spec, params_list[0])
    symbol = base["data"]["symbols"][0]
    prepared = prepare_data(base, data, data_dir)
    columns = [f"trial_{number}" for number in range(len(params_list))]
    signals = {key: {} for key in ("entries", "exits", "short_entries", "short_exits")}
    for column, params in zip(columns, params_list):
        compiled = compile_signals(apply_params(spec, params), prepared, data_dir=data_dir)
        for key, frame in compiled.items():
            # The signals, and a stop or target from a line, which a trial's features change too.
            if key in signals or key in STOP_ARGS:
                signals.setdefault(key, {})[column] = frame[symbol]
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
    # vectorbtX's warnings, collected as run_backtest collects them (see there, and RUN_LOCK).
    with RUN_LOCK, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        pf = vectorbtx().x.PortfolioBacktest(**kwargs)
    notes = []
    for warning in caught:
        message = str(warning.message)
        if issubclass(warning.category, UserWarning) and message not in notes:
            notes.append(message)
    out = []
    for column in columns:
        # A warning that names this trial's column is about it, said with the symbol's name; one that
        # names no trial is about every one. ⚠️ A warning listing several trials keeps the others'
        # labels (trial_5) beside this one's: vectorbtX joins them into one sentence.
        own = [note.replace(f"{column}:", f"{symbol}:") for note in notes
               if f"{column}:" in note or not re.search(r"\btrial_\d+:", note)]
        out.append({"pf": pf[[column]], "warnings": own})
    return out


def warm_up():
    """Run every template once on synthetic prices, so numba's code is compiled before the first real run.

    The first backtest in a process compiles vectorbt's and vectorbtX's numba kernels (and
    the extra exits' templates), which takes up to a minute in a fresh environment. ui.boot
    calls this once per Studio start, in a background thread, so the wait comes while the
    user is still building. It reads no stored data and makes no network call: currency
    conversion is taken out, and FRED rates become a static table (static ones are kept,
    so the holding-cost path compiles too).
    """
    from . import analysis, data as data_module
    from .spec import list_templates, load_template

    specs = [load_template(name) for name in list_templates()]
    # The extra exits' numba templates (exit_rules.py), which no template uses.
    with_exits = load_template("ma_cross")
    with_exits["risk"]["exits"] = {"time_bars": 20, "breakeven_points": 30.0, "step_points": 20.0}
    specs.append(with_exits)
    for spec in specs:
        account = spec.setdefault("account", {})
        account.pop("fx", None)
        account.pop("base_currency", None)
        if (account.get("rates") or {}).get("source") == "fred":
            account["rates"] = {"source": "static", "values": {"USD": 0.0}}
        frames = data_module.synthetic_ohlcv(spec["data"]["symbols"], n=300, seed=1)
        run = run_backtest(spec, frames, offline=True)
        analysis.extract_results(run, benchmark=True)


def backtest_error():
    """vectorbtX's BacktestError class, for the optimiser's 'invalid trial' check."""
    vbt = vectorbtx()
    return getattr(vbt.x, "BacktestError", None) or getattr(vbt, "BacktestError", ValueError)
