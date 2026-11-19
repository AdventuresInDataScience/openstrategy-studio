"""Build: set the strategy up step by step, with the whole of it at a glance beside the form.

Every step edits the same dict through ui.set_value; no widget keeps its own copy.
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, indicators, store, style, ui
from openstrategy_studio.compile import compile_features, compile_signals, compile_weights, export_script
from openstrategy_studio.spec import (CALENDAR_FREQS, DISTANCE_DEFAULTS, FREQ, SELECTION_MODES, SIZING_KEYS, STOP_UNITS,
                                      TIMEFRAMES, TRADING, WEIGHT_SCHEMES, apply_params, blank_spec, is_choice,
                                      is_intraday, list_templates, load_template, spec_hash, strategy_kind,
                                      trading_mode, validate_spec, when_cash_short)

settings, data_dir = ui.boot("Build")
ui.page_help("build")
spec = ui.current()
st.title("Build")

# Every Instrument field the grid can show; each mode shows the ones vectorbtX accepts for it.
# Leverage is per instrument (an index, a stock and an FX pair in one book have different
# limits), second in the grid so it is in sight, and filled in with the usual one when the
# way of trading, an asset class or a symbol changes (follow_usual). The Risk tab
# summarises it and says where it is set.
INSTRUMENT_FIELDS = {
    "asset_class": "text", "leverage": "number", "point_size": "number", "contract_size": "number", "increment": "number",
    "min_size": "number", "allow_fractional": "flag", "quote_currency": "text", "base_currency": "text",
    "cost_currency": "text", "commission_pct": "number", "commission_flat": "number", "commission_per_unit": "number",
    "commission_min": "number", "spread_points": "number", "slippage": "text", "borrow_rate": "number",
    "rounding": "text", "leverage_type": "text", "price_field": "text",
}
# The ⓘ on each grid column: what vectorbtX does with the field (its docs/API.md, sections 0, 1, 3 and 6).
INSTRUMENT_HELP = {
    "asset_class": "stocks, commodities, fx or index. On CFDs and spread bets it sets the default leverage (5x "
                   "stocks, 10x commodities and indices, 30x FX), and fx makes the overnight cost carry rather "
                   "than financing.",
    "point_size": "The price move that counts as one point. On a share, 0.01: a point is 1 cent (or 1p), so a "
                  "1.00 move is 100 points; on a share priced in pence (GBp), 1.0. On an FX pair, the pip: 0.0001, "
                  "or 0.01 when the quote is yen. On an index, 1.0. Picking how you trade, an asset class, a quote "
                  "currency or a new symbol fills in the usual one; type your own to change it (a commodity's point "
                  "varies by broker). Stops, targets, the spread and a spread bet's stake are all counted in points.",
    "contract_size": "Units of the underlying in one contract or lot (100000 for a standard FX lot). CFDs and FX "
                     "spot only: a spread bet's stake is its size.",
    "increment": "The step a position size moves in (1 share, 0.01 lots). Sizes are rounded to it.",
    "min_size": "The smallest position allowed, a multiple of the increment. An order that rounds below it is "
                "not placed.",
    "allow_fractional": "Buying outright only: allow fractions of a share. The size step drops to 0.000001 unless "
                        "an increment is given.",
    "quote_currency": "The currency the price is quoted in: USD for AAPL, and for EURUSD.",
    "base_currency": "FX only: the pair's first currency (EUR for EURUSD). Carry needs it.",
    "cost_currency": "The currency of fixed fees (flat, per-unit and minimum commission). Kept apart from the quote "
                     "currency so a $1 minimum stays $1 when prices are converted.",
    "commission_pct": "Commission as a fraction of each trade's value (0.001 = 0.1%).",
    "commission_flat": "A fixed fee on every order, in the cost currency.",
    "commission_per_unit": "A fee for each share, contract or lot traded, in the cost currency.",
    "commission_min": "The least commission an order pays, in the cost currency.",
    "spread_points": "The full gap between bid and ask, in points. Half is paid on each fill, so a round trip "
                     "pays the whole spread.",
    "slippage": "Extra cost on each fill, as a fraction of price (0.0005 = 0.05%), or auto: estimated from "
                "volatility and trade size against volume (needs volume data).",
    "borrow_rate": "The yearly fee for borrowing what you short (0.005 = 0.5% a year), charged daily on the short's "
                   "value. Blank: unknown, so a short is charged nothing and the run says so. 0: the borrow is "
                   "free.",
    "rounding": "How sizes are rounded to the increment: floor (down, the default), nearest or ceil.",
    "leverage": "The most this instrument's position may be worth, as a multiple of the cash: 5 lets 10,000 hold "
                "50,000. Picking how you trade, an asset class, a new symbol or a portfolio fills in the usual limit: "
                "on spread bets and CFDs traded on signals, 30 for FX, 10 for indices and commodities, 5 for single "
                "stocks; 1 (no borrowing) bought outright, on FX spot and in every portfolio. Type your own to change "
                "it. A trade worth more is cut down or skipped (Risk tab).",
    "leverage_type": "instrument_level: a cap on each instrument on its own. portfolio_level: one cap on the "
                     "whole book, applied by scaling a portfolio's target weights.",
    "price_field": "close, or adj_close: prices adjusted for dividends and splits (the default when buying "
                   "outright). Needs adj_close in the stored data.",
}
KIND_LABELS = {"signals": "Trade on signals", "portfolio": "Hold a portfolio"}
KIND_HELP = {"signals": "Entry and exit rules decide when to buy and when to sell, long or short.",
             "portfolio": "Target weights across the symbols, rebalanced on a schedule or never. One symbol held "
                          "from the start is buy and hold."}
TRADING_LABELS = {"equity": "Buy outright", "cfd": "CFDs", "spreadbet": "Spread bets", "fx_spot": "FX spot"}
TRADING_HELP = {
    "equity": "You own the shares or units. Costs: commission and the spread. A short borrows what it sells, "
              "so it pays a borrow fee.",
    "cfd": "A contract for difference, on margin. Costs: commission, the spread and overnight funding. A short "
           "pays funding, not a borrow fee.",
    "spreadbet": "A stake per point, on margin. The cost is in the spread (no commission), plus overnight funding.",
    "fx_spot": "Currency pairs in lots, on margin. Costs: the spread, commission, and carry on both currencies.",
}
SIZING_LABELS = {
    "equity": {"risk_pct": "Risk a share of starting cash per trade (0.01 = 1%)",
               "risk_amount": "Risk a cash amount per trade", "stake": "Shares per trade",
               "notional": "A cash amount per trade"},
    "cfd": {"risk_pct": "Risk a share of starting cash per trade (0.01 = 1%)",
            "risk_amount": "Risk a cash amount per trade", "stake": "Contracts per trade",
            "notional": "Position value per trade"},
    "spreadbet": {"risk_pct": "Risk a share of starting cash per trade (0.01 = 1%)",
                  "risk_amount": "Risk a cash amount per trade", "stake": "Stake per point",
                  "notional": "Position value per trade"},
    "fx_spot": {"risk_pct": "Risk a share of starting cash per trade (0.01 = 1%)",
                "risk_amount": "Risk a cash amount per trade", "stake": "Lots per trade",
                "notional": "Position value per trade"},
}
SIZING_DEFAULTS = {"risk_pct": 0.01, "risk_amount": 100.0, "stake": 1.0, "notional": 1000.0, "size": 1.0}


def feature(indicator_id, **params):
    """A catalogue feature with some parameters set, for the presets."""
    made = indicators.new_feature(indicator_id)
    made["params"].update(params)
    return made


def compare(left, op, right):
    return {"left": left, "op": op, "right": right}


PRESETS = {
    "ma_cross": ("Moving-average cross",
                 {"fast": feature("talib:SMA", timeperiod=50), "slow": feature("talib:SMA", timeperiod=200)},
                 {"long_entry": {"conditions": {"A": compare({"feature": "fast"}, "crosses_above", {"feature": "slow"})},
                                 "logic": "A"},
                  "long_exit": {"conditions": {"A": compare({"feature": "fast"}, "crosses_below", {"feature": "slow"})},
                                "logic": "A"}}),
    "rsi": ("RSI reversion", {"rsi": feature("talib:RSI", timeperiod=14)},
            {"long_entry": {"conditions": {"A": compare({"feature": "rsi"}, "crosses_above", {"const": 30.0})},
                            "logic": "A"},
             "long_exit": {"conditions": {"A": compare({"feature": "rsi"}, "crosses_above", {"const": 70.0})},
                           "logic": "A"}}),
    "channel": ("Channel breakout",
                {"upper": dict(feature("talib:MAX", timeperiod=20), inputs={"close": "high"}),
                 "lower": dict(feature("talib:MIN", timeperiod=20), inputs={"close": "low"})},
                {"long_entry": {"conditions": {"A": compare({"price": "close"}, ">", {"feature": "upper", "shift": 1})},
                                "logic": "A"},
                 "long_exit": {"conditions": {"A": compare({"price": "close"}, "<", {"feature": "lower", "shift": 1})},
                               "logic": "A"}}),
}


@st.cache_data(max_entries=8, show_spinner=False)
def compute_preview(section_json, data_dir_text, mtimes):
    from pathlib import Path
    from openstrategy_studio import data as data_module

    section = json.loads(section_json)
    frames = data_module.load_for_spec(section, Path(data_dir_text))
    applied = apply_params(section)
    computed = compile_features(applied.get("features") or {}, frames)
    if strategy_kind(applied) == "signals":
        # The compiler's signals are on the bar they fire, which is what the preview marks.
        return frames, computed, compile_signals(applied, frames)
    return frames, computed, compile_weights(applied, frames)


def preview():
    """(frames, features, signals or weights) for the spec as it stands, cached on it and the data files.

    The Preview tab draws it; the Risk tab's trade-size check reads the entries from it.
    """
    section = {name_: spec.get(name_) for name_ in ("version", "name", "setup", "data", "instruments", "features",
                                                    "rules", "selection", "weighting", "schedule", "risk",
                                                    "account", "optimise") if name_ in spec}
    return compute_preview(json.dumps(section, sort_keys=True), str(data_dir), ui.data_mtimes(spec, data_dir))


MODE_WORDS = {"equity": "bought outright", "cfd": "on CFDs", "spreadbet": "on spread bets", "fx_spot": "on FX spot"}


def usual_leverage_for(fields, mode=None, kind_=None):
    """The leverage the Studio fills in for one instrument's fields (engine.usual_leverage)."""
    return engine.usual_leverage(mode or trading, fields.get("asset_class"), kind_ or kind)


def usual_terms(fields, mode=None, kind_=None):
    """{"leverage", "point_size"}: what the Studio fills in for one instrument's fields.

    Leverage by the way of trading, the asset class and the kind of strategy; the point size
    by the market's convention (engine.usual_point_size: a pip on FX, 1.0 on an index, a cent
    on a share).
    """
    return {"leverage": usual_leverage_for(fields, mode, kind_),
            "point_size": engine.usual_point_size(mode or trading, fields.get("asset_class"),
                                                  fields.get("quote_currency"))}


def follow_usual(fields, before, after):
    """Give `fields` the usual leverage and point size of `after` where they held the usual ones of `before`.

    `before` and `after` are the usual terms (usual_terms) for the instrument as it was and as
    it is now. A value that was blank, or the usual one, follows the change; one typed in by
    hand stays. Called when the way of trading or the kind of strategy changes.
    """
    for name in ("leverage", "point_size"):
        if fields.get(name) is None or fields[name] == before[name]:
            fields[name] = after[name]


def points_in_price(points):
    """What a distance in points is in each symbol's price, for the caption under a stop or target box."""
    meaning = ui.run_guarded("Reading the point sizes", lambda: engine.point_meaning(spec, data_dir)) or {}
    sizes = ui.run_guarded("Reading the point sizes", lambda: engine.point_sizes(spec, symbols, data_dir)) or []
    parts = [f"on {symbol}, {points:g} points is a price move of {points * size:.6g}: a point is "
             f"{meaning.get(symbol, size)}" for symbol, size in zip(symbols, sizes)]
    return "; ".join(parts) + "." if parts else "Pick a symbol on the Setup tab to see it in price."


def distance_inputs(part, help_points, help_pct, trailing=False):
    """On the Risk tab: how the stop or the target (`part`) is measured, and its distance.

    In points, or as a fraction of the price. Both distances stay in the spec whichever is in
    use, so the optimiser can switch the unit (risk.<part>.unit); only the one in use is shown.
    """
    node = spec["risk"][part]
    unit = node.get("unit", "points")
    units = list(STOP_UNITS)
    chosen = st.radio("Measured", units, index=units.index(unit) if unit in units else 0,
                      format_func={"points": "In points", "pct": "In % of the price"}.get, horizontal=True,
                      key=ui.key(f"risk.{part}", ":unit"),
                      help="In points (a fixed price distance: cents on a share, pips on FX), or in % of the price "
                           "the trade fills at. How it is measured can be searched on the Optimise page too.")
    if chosen != unit:
        def switch_unit():
            node["unit"] = chosen
            node.setdefault(chosen, DISTANCE_DEFAULTS[part][chosen])
            ui.prune_variables()
        ui.rerun_after(switch_unit)
    if chosen == "points":
        value = ui.param_input(f"risk.{part}.points", {"type": "float", "default": DISTANCE_DEFAULTS[part]["points"]},
                               label="Distance (points)", help=help_points)
        if isinstance(value, (int, float)):
            st.caption(points_in_price(value))
    else:
        value = ui.param_input(f"risk.{part}.pct", {"type": "float", "default": DISTANCE_DEFAULTS[part]["pct"]},
                               label="Distance (a fraction of the price: 0.02 = 2%)", help=help_pct)
        if isinstance(value, (int, float)):
            # 0.02 is 2%, not 0.02%: the same reading as the risk box above.
            st.caption(f"{value:g} is {value:.2%} of " + ("the best price since the entry (at first, the fill "
                                                          "price): the stop trails that far behind it."
                                                          if trailing else "the price the trade fills at."))


def leverage_summary():
    """On the Risk tab: each symbol's leverage and what it funds, read-only, and where to change it.

    Set per instrument in the Setup tab's grid, the one place it is written, because one book
    can mix limits (an index at 10x, a stock at 5x, an FX pair at 30x on spread bets). A
    second widget here would fight the grid over the same value.
    """
    st.markdown("**Leverage**", help=INSTRUMENT_HELP["leverage"])
    found = spec.get("instruments") or {}
    if "from_file" in found:
        st.caption("Each instrument's leverage comes from the universe file.")
        return
    st.caption("Set for each instrument in the leverage column of the instrument grid (Setup tab), since one book "
               "can mix limits. Picking how you trade or an asset class fills in the usual one.")
    if not symbols:
        st.caption("Pick a symbol on the Setup tab to see its leverage.")
        return
    used = ui.run_guarded("Reading the leverage", lambda: engine.default_leverage(spec)) or {}
    cash = float((spec.get("account") or {}).get("init_cash") or 0.0)
    rows = []
    for symbol in symbols:
        fields = found.get(symbol) or {}
        usual = ui.run_guarded("Reading the usual leverage", lambda fields=fields: usual_leverage_for(fields))
        lever = used.get(symbol)
        rows.append({"symbol": symbol, "asset class": fields.get("asset_class") or "", "leverage": lever,
                     "funds": None if lever is None else lever * cash,
                     "set to": "the usual" if lever == usual else
                               f"your own (the usual is {usual:g}x)" if usual is not None else "your own"})
        # vectorbtX allows more than the usual limit, with a warning that only reaches the console.
        if lever is not None and usual is not None and lever > usual:
            st.caption(f"{symbol}: {lever:g}x is above the usual {usual:g}x for this way of trading. It is allowed; "
                       "check that your broker offers it.")
    ui.table(pd.DataFrame(rows).set_index("symbol"), hide_index=False, column_config={
        "leverage": st.column_config.NumberColumn(format="%gx"),
        "funds": st.column_config.NumberColumn(format="localized")})


def trade_size_check():
    """The largest trade the rules place, against what the account funds, before any run.

    vectorbtX's own sizes (engine.sizing_check), so a warning here is the warning the run
    would give, with what to change in the Studio's words.
    """
    st.markdown("**Trade size check**",
                help="The largest trade the entry rules place over the backtest period, sized as vectorbtX sizes it "
                     "at the open it fills at, against the cash times the leverage. When the rules never fire, the "
                     "last bar's trade is shown.")
    if validate_spec(spec):
        st.caption("Fix the problems in the At a glance panel to check the trade size.")
        return
    if ui.missing_data(spec, data_dir):
        st.caption("Import the data on the Data page to check the trade size.")
        return
    result = ui.run_guarded("Computing signals", preview)
    if result is None:
        return
    frames, _, compiled = result
    entries = compiled["entries"] | compiled["short_entries"]
    rows = ui.run_guarded("Checking the trade size", lambda: engine.sizing_check(spec, frames, entries, data_dir))
    if not rows:
        return
    table = pd.DataFrame([{"symbol": row["symbol"], "fills on": str(row["bar"])[:16], "price": row["price"],
                           "stop in price": row["stop_price"], "stop % of price": row["stop_pct"],
                           "size": f"{row['size']:g} {row['unit']}", "trade value": row["value"],
                           "× cash": row["times_cash"], "funded": row["funded"]} for row in rows]).set_index("symbol")
    ui.table(table, hide_index=False, column_config={
        "price": st.column_config.NumberColumn(format="%.4g"),
        "stop in price": st.column_config.NumberColumn(format="%.4g"),
        "stop % of price": st.column_config.NumberColumn(format="percent"),
        "trade value": st.column_config.NumberColumn(format="localized"),
        "× cash": st.column_config.NumberColumn(format="%.1f"),
        "funded": st.column_config.NumberColumn(format="localized")})
    sizing_key = next((name_ for name_ in SIZING_KEYS if name_ in ((spec.get("risk") or {}).get("sizing") or {})), "")
    lower = "lower the risk or widen the stop" if sizing_key.startswith("risk") else "lower the amount"
    for row in rows:
        if row["below_minimum"]:
            st.warning(f"{row['symbol']}: the size works out at {row['asked']:.4g} {row['unit']}, below the smallest "
                       f"allowed ({row['min_size']:g}), so no order is placed and the run has no trades. Raise the "
                       "risk or the amount, tighten the stop, or set min_size in the instrument grid (Setup) to "
                       "your broker's.")
        elif not row["fits"]:
            cut = (f"cut it down to {row['funded']:,.0f}" + (", so it risks less than you asked"
                                                              if sizing_key.startswith("risk") else "")
                   if when_cash_short(spec) == "partial" else "skip it")
            st.warning(f"{row['symbol']}: the largest trade is worth {row['value']:,.0f}, {row['times_cash']:.1f} "
                       f"times the starting cash, but {row['leverage']:g}x leverage funds {row['funded']:,.0f}. The "
                       f"run will {cut}. To trade it in full, {lower}, or raise the leverage in the instrument grid "
                       "(Setup tab).")
        if row["stop_price"] and row["typical_range"] and row["stop_price"] < row["typical_range"]:
            st.warning(f"{row['symbol']}: the stop is {row['stop_price']:.4g} from the entry ({row['stop_pct']:.2%} of "
                       f"the price), but a bar here usually spans {row['typical_range']:.4g} from high to low, so most "
                       "trades will be stopped out within a bar or two. Points are price moves divided by the point "
                       "size: widen the stop, or check the instrument's point_size (Setup tab).")


# --- start blank or from a template ------------------------------------------------
templates = list_templates()
starts = {"blank": "A blank strategy", **templates}
left, right = st.columns([4, 1])
chosen = left.selectbox("Start from", [None] + list(starts),
                        format_func=lambda name: "Keep the current strategy" if name is None else starts[name],
                        help="A blank strategy (no symbols, features or rules), or a ready-made one to edit. Either "
                             "replaces the whole strategy, so save the current one first (Home page) if you want to "
                             "keep it.")
right.write("")
if right.button("Start from this", disabled=chosen is None):
    ui.replace_spec(blank_spec() if chosen == "blank" else load_template(chosen))
    st.rerun()

form, side = st.columns([5, 2], gap="large")
features = spec.setdefault("features", {})
kind = strategy_kind(spec)
trading = trading_mode(spec)
symbols = spec["data"]["symbols"]
timeframe = spec["data"]["timeframe"]
stored = ui.stored_symbols(data_dir)

with form:
    third = "Rules" if kind == "signals" else "Portfolio"
    tab_names = ["Setup", "Features", third, "Risk", "Account", "Preview", "Pipeline", "Strategy file"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    # --- setup ----------------------------------------------------------------------
    with tabs["Setup"]:
        row = st.columns([2, 3])
        name = row[0].text_input("Name", value=spec.get("name", ""), key=ui.key("name"),
                                 help="Names the strategy in saved runs, jobs and downloaded files.")
        if name != spec.get("name"):
            ui.set_value("name", name)
        options = sorted(set(stored) | set(symbols))
        # Each stored name with what it holds (SPY: 1d, 1w from yahoo), so two sources stored under
        # different names (SPY, SPY_csv) can be told apart here.
        held = {}
        listing = store.list_datasets(data_dir)
        for name_, rows_ in (listing.groupby("symbol") if not listing.empty else []):
            held[name_] = f"{name_} ({', '.join(rows_['timeframe'])} · {', '.join(sorted(set(rows_['source'])))})"
        picked_symbols = row[1].multiselect("Symbols", options, default=symbols, key=ui.key("data.symbols"),
                                            format_func=lambda name_: held.get(name_, f"{name_} (not stored)"),
                                            help="What the strategy trades: the names datasets are stored under on "
                                                 "the Data page, with the timeframes and source each holds. A longer "
                                                 "timeframe is resampled from the finest stored one. With several, "
                                                 "only the dates they all share are used.")
        row = st.columns(3)
        picked_timeframe = row[0].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(timeframe),
                                            key=ui.key("data.timeframe"),
                                            help="The bar size. Every rule is evaluated once per bar. Data stored "
                                                 "at a finer timeframe is resampled up.")
        # Streamlit limits date pickers to ten years either side unless told otherwise.
        span = {"min_value": date(1970, 1, 1), "max_value": date(2100, 12, 31)}
        start = row[1].date_input("From", value=date.fromisoformat(str(spec["data"]["start"])[:10]),
                                  key=ui.key("data.start"), **span,
                                  help="The first date of the backtest. Indicators warm up from here, so their "
                                       "first bars are blank.")
        end = row[2].date_input("To", value=date.fromisoformat(str(spec["data"]["end"])[:10]),
                                key=ui.key("data.end"), **span, help="The last date of the backtest.")
        if picked_symbols != symbols:
            def switch_symbols():
                instruments = spec.setdefault("instruments", {})
                if "from_file" not in instruments:
                    template = dict(next(iter(instruments.values()), {}))
                    for symbol in picked_symbols:
                        if symbol not in instruments:
                            # A new symbol copies the first one's settings, and gets the usual leverage
                            # and point size for its asset class where that one names none.
                            instruments[symbol] = dict(template)
                            for name_, value in usual_terms(template).items():
                                instruments[symbol].setdefault(name_, value)
                    for symbol in list(instruments):
                        if symbol not in picked_symbols:
                            del instruments[symbol]
                # A removed symbol leaves the shared account's priority order too.
                priority = (spec.get("account") or {}).get("entry_priority") or {}
                if priority.get("order"):
                    priority["order"] = [symbol for symbol in priority["order"] if symbol in picked_symbols]
                    if not priority["order"]:
                        del priority["order"]
                spec["data"]["symbols"] = picked_symbols
            ui.rerun_after(switch_symbols)
        if picked_timeframe != timeframe:
            ui.set_value("data.timeframe", picked_timeframe)
            ui.set_value("account.freq", FREQ[picked_timeframe])
        for field, value in (("start", start.isoformat()), ("end", end.isoformat())):
            if value != spec["data"][field]:
                ui.set_value(f"data.{field}", value)

        left, right = st.columns(2, gap="large")
        picked_kind = left.radio("What are you building?", list(KIND_LABELS), index=list(KIND_LABELS).index(kind),
                                 format_func=KIND_LABELS.get, captions=[KIND_HELP[name] for name in KIND_LABELS],
                                 key=ui.key("setup.kind"),
                                 help="It decides the third tab: Rules or Portfolio. Switching starts that tab, "
                                      "the Risk tab and the shared-cash settings afresh; features are kept.")
        picked_trading = right.radio("How are you trading it?", TRADING, index=TRADING.index(trading),
                                     format_func=TRADING_LABELS.get, captions=[TRADING_HELP[name] for name in TRADING],
                                     key=ui.key("setup.trading"),
                                     help="One way of trading for every instrument. It decides which costs apply, "
                                          "which instrument fields are shown below and how sizes are counted. "
                                          "Switching drops the fields the new way does not use.")
        if picked_kind != kind:
            def switch_kind():
                spec["setup"]["kind"] = picked_kind
                for section in ("rules", "selection", "weighting", "schedule"):
                    spec.pop(section, None)
                account = spec.setdefault("account", {})
                for name_ in ("cash", "entry_priority", "when_cash_short"):
                    account.pop(name_, None)
                if picked_kind == "signals":
                    spec["rules"] = {"long_entry": ui.new_rule(features), "long_exit": None, "short_entry": None,
                                     "short_exit": None}
                    # A blank strategy's: a stake of 1, the stop and target off but with the distances
                    # the Risk tab would start from, so the optimiser can switch either on.
                    spec["risk"] = blank_spec()["risk"]
                else:
                    spec["weighting"] = {"scheme": "equal", "vol_window": 20}
                    spec["schedule"] = {"rebalance": "never", "calendar": None, "threshold": None, "combine": "and",
                                        "min_trade": 0.005}
                    spec["risk"] = {}
                # The usual leverage follows the kind (a stock spread bet is 5x traded on signals, 1x
                # held in a portfolio); one typed in by hand stays.
                for fields in (spec.get("instruments") or {}).values():
                    if isinstance(fields, dict):
                        follow_usual(fields, usual_terms(fields, trading, kind), usual_terms(fields, trading, picked_kind))
                ui.prune_variables()
            ui.rerun_after(switch_kind)
        if picked_trading != trading:
            def switch_trading():
                spec["setup"]["trading"] = picked_trading
                read = ui.run_guarded("Reading instrument modes", lambda: engine.mode_rules(picked_trading))
                rejected = set((read or {"rejected": {}})["rejected"])
                for fields in (spec.get("instruments") or {}).values():
                    if isinstance(fields, dict):
                        # The usual leverage and point size for the old way of trading become the usual
                        # ones for the new (1x bought outright, 5x for a stock spread bet ...); values
                        # typed in by hand stay.
                        before = usual_terms(fields, trading)
                        if trading == "fx_spot" and not fields.get("asset_class"):
                            # FX spot is FX without saying so; another way of trading must be told.
                            fields["asset_class"] = "fx"
                        for field in list(fields):
                            # base_currency stays on an FX pair whatever the mode: carry needs it.
                            if (field in rejected and not (field == "base_currency" and fields.get("asset_class") == "fx")) \
                                    or (field == "borrow_rate" and picked_trading != "equity"):
                                del fields[field]
                        follow_usual(fields, before, usual_terms(fields, picked_trading))
            ui.rerun_after(switch_trading)

        st.markdown("**Instruments**")
        instruments = spec.setdefault("instruments", {})
        universes = sorted(path.name for path in (data_dir / "universes").glob("*"))
        use_file = st.toggle("Read instruments from a universe file", value="from_file" in instruments,
                             disabled=not universes and "from_file" not in instruments,
                             help="Store universe files on the Data page. A file names each row's own mode.",
                             key=ui.key("instruments.use_file"))
        if not use_file and "from_file" in instruments:
            # Back from a universe file to the grid: each symbol starts from the usual terms.
            spec["instruments"] = {symbol: usual_terms({}) for symbol in symbols}
            instruments = spec["instruments"]
        if use_file:
            current_file = instruments.get("from_file")
            file = st.selectbox("Universe file", universes,
                                index=universes.index(current_file) if current_file in universes else 0,
                                key=ui.key("instruments.from_file"),
                                help="A stored file with one instrument per row, each with its own mode and costs. "
                                     "It replaces the grid.")
            if instruments != {"from_file": file}:
                ui.set_value("instruments", {"from_file": file})
        elif not symbols:
            st.caption("Pick the symbols to trade above: each gets a row here, filled in with the usual leverage "
                       "and point size for how you trade it.")
        else:
            rules = ui.run_guarded("Reading instrument modes", lambda: engine.mode_rules(trading)) or \
                {"required": set(), "rejected": {}}
            shown = {field: kind_name for field, kind_name in INSTRUMENT_FIELDS.items()
                     if (field not in rules["rejected"] or field == "base_currency")
                     and (field != "borrow_rate" or trading == "equity")}
            grid = pd.DataFrame([{field: instruments.get(symbol, {}).get(field) for field in shown}
                                 for symbol in symbols], index=symbols)
            for field, kind_name in shown.items():
                if kind_name == "number":
                    grid[field] = pd.to_numeric(grid[field], errors="coerce")
                else:
                    grid[field] = grid[field].map(lambda value: "" if value is None else str(value).lower()
                                                  if isinstance(value, bool) else str(value))
            # Every column gets its ⓘ (INSTRUMENT_HELP); a few also get a list of choices.
            choices = {"rounding": ["", "floor", "nearest", "ceil"], "allow_fractional": ["", "true", "false"],
                       "price_field": ["", "close", "adj_close"],
                       "asset_class": ["", "stocks", "commodities", "fx", "index"],
                       "leverage_type": ["", "instrument_level", "portfolio_level"]}
            config = {}
            for field, kind_name in shown.items():
                if field in choices:
                    config[field] = st.column_config.SelectboxColumn(field, options=choices[field],
                                                                     help=INSTRUMENT_HELP[field])
                elif field == "borrow_rate":
                    config[field] = st.column_config.NumberColumn(field, min_value=0.0, format="%.4f",
                                                                  help=INSTRUMENT_HELP[field])
                elif kind_name == "number":
                    config[field] = st.column_config.NumberColumn(field, help=INSTRUMENT_HELP[field])
                else:
                    config[field] = st.column_config.TextColumn(field, help=INSTRUMENT_HELP[field])
            st.caption(f"{TRADING_LABELS[trading]}: blank cells use the mode's default, and only the fields this mode "
                       "uses are shown. Leverage fills in with the usual limit: on spread bets and CFDs 30x for FX, "
                       "10x for indices and commodities and 5x for single stocks; 1x (no borrowing) for everything "
                       "else and in every portfolio. Type your own to change it. Costs come only from these fields. "
                       "Financing on held positions comes from the account's rates and the markup on the Settings "
                       "page.")
            # What a point is, per symbol, in its market's own words: stops, targets, spreads and
            # stakes are all counted in points, and the conventions differ (cents, pips, index points).
            meaning = ui.run_guarded("Reading the point sizes", lambda: engine.point_meaning(spec, data_dir)) or {}
            st.caption("A point: " + "; ".join(f"on {symbol}, {text_}" for symbol, text_ in meaning.items())
                       + ". A share counts cents (a 1.00 move is 100 points), FX counts pips, an index counts index "
                         "points.")
            edited = st.data_editor(grid, column_config=config, key=ui.key(f"instruments.grid:{trading}"),
                                    width="stretch")
            updated = {}
            for symbol, values in edited.iterrows():
                # Fields the grid does not show are kept as they are.
                old = instruments.get(symbol) or {}
                fields = {field: value for field, value in old.items() if field not in shown}
                for field, kind_name in shown.items():
                    value = values[field]
                    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
                        continue
                    if kind_name == "number":
                        fields[field] = float(value)
                    elif kind_name == "flag":
                        fields[field] = value == "true"
                    elif field == "slippage" and value != "auto":
                        fields[field] = float(value)
                    else:
                        fields[field] = str(value)
                # A new asset class or quote currency brings its usual leverage and point size (a pip
                # on USDJPY is 0.01), unless that cell was typed in by hand, or is being typed in this
                # very edit (it then differs from the spec's value).
                if (fields.get("asset_class"), fields.get("quote_currency")) != (old.get("asset_class"),
                                                                                  old.get("quote_currency")):
                    before, after = usual_terms(old), usual_terms(fields)
                    for name_ in ("leverage", "point_size"):
                        if fields.get(name_) == old.get(name_) and old.get(name_) in (None, before[name_]):
                            fields[name_] = after[name_]
                updated[symbol] = fields
            if updated != instruments:
                ui.set_value("instruments", updated)
            for symbol, fields in updated.items():
                for field in sorted(rules["required"] - set(fields)):
                    st.warning(f"{symbol}: {TRADING_LABELS[trading].lower()} need {field}.")
                if (fields.get("asset_class") == "fx" or trading == "fx_spot") and \
                        not (fields.get("base_currency") and fields.get("quote_currency")):
                    st.warning(f"{symbol}: an FX instrument needs both base_currency and quote_currency (EUR and USD "
                               "for EURUSD), or its carry cannot be priced.")
            with st.expander(f"Defaults for {TRADING_LABELS[trading].lower()}"):
                st.json(ui.run_guarded("Reading defaults", lambda: engine.mode_defaults(trading)) or {},
                        expanded=False)

    # --- features -------------------------------------------------------------------
    with tabs["Features"]:
        st.caption("Features are the lines and signals the rules compare: indicators, patterns, levels and "
                   "transforms, computed on every bar. Each output is a line you can use, such as bb.upperband.")
        for feature_name, item in list(features.items()):
            entry = indicators.catalogue().get(item.get("indicator"))
            described = indicators.describe(item)
            title = f"{feature_name}: {entry['label'] if entry else item.get('indicator')} · {described}"
            # Keyed by name, so the panel stays open while its title changes (a setting edited,
            # the type swapped); a renamed one opens under its new name (ui.feature_editor).
            with st.expander(title, key=f"feature-panel:{feature_name}",
                             expanded=feature_name == st.session_state.get("renamed_feature")):
                ui.feature_editor(feature_name, item, features)
            # The title is drawn before the inputs and settings inside write their new values
            # back, so after an edit it would show the old ones until the next click. When an
            # edit changed what it says, draw the page again: once, as the second pass finds
            # them equal. (Renames, swaps and deletes already redraw through ui.rerun_after.)
            if feature_name in features and indicators.describe(features[feature_name]) != described:
                st.rerun()
        st.markdown("**Add a feature**")
        ui.feature_picker(features)

    # --- rules or portfolio ---------------------------------------------------------
    if kind == "signals":
        with tabs["Rules"]:
            rules = spec.setdefault("rules", {})
            row = st.columns([3, 1])
            preset = row[0].selectbox("Preset", [None] + list(PRESETS),
                                      format_func=lambda p: "Choose a preset to replace the rules" if p is None
                                      else PRESETS[p][0], key=ui.key("rules", ":preset"),
                                      help="A common pair of entry and exit rules, with the features they need. "
                                           "Applying one replaces all four rules and adds its features (one with "
                                           "the same name is overwritten).")
            row[1].write("")
            if row[1].button("Apply preset", disabled=preset is None):
                new = ui.spec_copy()
                new.setdefault("features", {}).update(json.loads(json.dumps(PRESETS[preset][1])))
                new["rules"] = {"long_entry": None, "long_exit": None, "short_entry": None, "short_exit": None}
                new["rules"].update(json.loads(json.dumps(PRESETS[preset][2])))
                ui.replace_spec(new)
                st.rerun()
            slot_help = {
                "long_entry": "When to buy. Fires on a bar's close; the order fills at the next bar's open.",
                "long_exit": "When to close a long. Without one, a long is closed only by a stop, a target, an "
                             "extra exit or an opposite entry.",
                "short_entry": "When to sell short. Fires on a bar's close; the order fills at the next bar's open.",
                "short_exit": "When to close a short. Without one, a short is closed only by a stop, a target, an "
                              "extra exit or an opposite entry.",
            }

            def slot_title(label, node):
                """A rule panel's title: the rule's logic line, so a closed panel still says what it does."""
                if node is None:
                    return f"{label} · not used"
                if is_choice(node):
                    return f"{label} · version {node['choose']}: {node['options'][node['choose']].get('logic') or '?'}"
                return f"{label} · {node.get('logic') or '(no logic line)'}"

            for slot, label in (("long_entry", "Long entry"), ("long_exit", "Long exit"),
                                ("short_entry", "Short entry"), ("short_exit", "Short exit")):
                node = rules.get(slot)
                title = slot_title(label, node)
                # Keyed by slot, not by revision: the panel stays open or closed while its title changes.
                # A rule is added and deleted with a button each (the owner's call, 19 November 2026:
                # a "use this rule" switch did not read as a way to delete one).
                with st.expander(title, expanded=True, key=f"rule-panel:{slot}"):
                    if node is None:
                        st.caption(slot_help[slot])
                        if st.button(f"Add a {label.lower()} rule", key=ui.key(f"rules.{slot}", ":add_rule"),
                                     help="Starts the rule with one condition, A, comparing two lines. Add more "
                                          "conditions and join them in its logic line."):
                            def start(slot=slot):
                                rules[slot] = ui.new_rule(features)
                                ui.open_panel(f"rules.{slot}.conditions.A")
                            ui.rerun_after(start)
                    else:
                        ui.rule_editor(f"rules.{slot}", features, timeframe)
                        if st.button(f"Delete the {label.lower()} rule", key=ui.key(f"rules.{slot}", ":delete_rule"),
                                     help="Removes the whole rule: its conditions, its logic line and any versions "
                                          "of it, and takes their values off the Optimise page."):
                            def clear(slot=slot):
                                rules[slot] = None
                                ui.prune_variables()
                            ui.rerun_after(clear)
                # The title is drawn before the logic line inside writes back: draw again once if it changed.
                if slot_title(label, rules.get(slot)) != title:
                    st.rerun()
            st.caption("Every signal fills at the next bar's open. Each rule takes every kind of condition: a "
                       "weekday or higher-timeframe condition in an exit rule is a time exit (A or Friday) or narrows "
                       "the exit (A and the week is down). A stop, a target and the extra exits close a position "
                       "whatever the exit rule says.")
    else:
        with tabs["Portfolio"]:
            st.markdown("**Selection**")
            use_selection = st.toggle("Pick assets by a score", value=spec.get("selection") is not None,
                                      key=ui.key("selection", ":on"), help="Without a selection every asset is held.")
            if use_selection and spec.get("selection") is None:
                first = next(iter(features), None)
                ui.rerun_after(lambda: spec.__setitem__("selection", {
                    "score": {"feature": first} if first else {"price": "close"}, "mode": "top", "top_n": 1,
                    "min_score": None}))
            if not use_selection and spec.get("selection") is not None:
                def drop_selection():
                    spec.pop("selection")
                    ui.prune_variables()
                ui.rerun_after(drop_selection)
            if use_selection:
                ui.ref_input("selection.score", features, "Rank assets by",
                             help="The score each asset is ranked by on every bar, such as a momentum feature.")
                ui.param_input("selection.mode", {"type": "choice", "options": SELECTION_MODES, "default": "top"},
                               label="Keep", help="Which assets are held: the highest scores, the lowest, or "
                                                  "every asset above the minimum score.")
                if spec["selection"].get("mode", "top") in ("top", "bottom"):
                    ui.param_input("selection.top_n", {"type": "int", "default": 1}, label="How many",
                                   help="How many assets to hold at once.")
                use_floor = st.checkbox("Require a minimum score", value=spec["selection"].get("min_score") is not None,
                                        key=ui.key("selection.min_score", ":on"),
                                        help="An asset scoring below the minimum is not held, even if it ranks "
                                             "high enough. On a bar where none qualifies, the portfolio holds "
                                             "cash.")
                if use_floor and spec["selection"].get("min_score") is None:
                    ui.set_value("selection.min_score", 0.0)
                    st.rerun()
                if use_floor:
                    ui.param_input("selection.min_score", {"type": "float", "default": 0.0}, label="Minimum score",
                                   help="An asset is held only while its score is at or above this.")
                elif spec["selection"].get("min_score") is not None:
                    ui.set_value("selection.min_score", None)
            st.markdown("**Weighting**")
            ui.param_input("weighting.scheme", {"type": "choice", "options": WEIGHT_SCHEMES, "default": "equal"},
                           label="Scheme",
                           help="How the held assets share the money. equal: the same share each. inverse_vol: "
                                "more in the calmer assets, by 1 / volatility. price: in proportion to price. "
                                "fixed: the table below, as given.")
            scheme = spec["weighting"]["scheme"]
            if scheme == "inverse_vol":
                ui.param_input("weighting.vol_window", {"type": "int", "default": 20}, label="Volatility window (bars)",
                               help="Bars of returns used to measure each asset's volatility.")
            if scheme == "fixed":
                table = spec["weighting"].get("table") or {symbol: 1 / len(symbols) for symbol in symbols}
                frame = pd.DataFrame({"weight": [table.get(symbol, 0.0) for symbol in symbols]}, index=symbols)
                edited = st.data_editor(frame, key=ui.key("weighting.table"), width="stretch", column_config={
                    "weight": st.column_config.NumberColumn("weight", help="Each asset's share of the account "
                                                                           "(0.6 = 60%).")})
                new_table = {symbol: float(weight) for symbol, weight in edited["weight"].items()}
                if new_table != spec["weighting"].get("table"):
                    ui.set_value("weighting.table", new_table)
                st.caption("Fixed weights are used as given: below 100% the rest stays in cash.")
            st.markdown("**Rebalance**")
            schedule = spec["schedule"]
            names = {"never": "Never: buy once and hold", "every_bar": "On every bar", "calendar": "On a calendar"}
            rebalance = st.radio("Go back to the target weights", list(names), index=list(names).index(
                schedule.get("rebalance", "never")), format_func=names.get, horizontal=True,
                key=ui.key("schedule.rebalance"),
                help="How often the holdings are traded back to their target weights. Between those dates "
                     "they drift with prices. Every bar trades the most and pays the most in costs.")
            if rebalance != schedule.get("rebalance"):
                def switch_rebalance():
                    schedule["rebalance"] = rebalance
                    schedule["calendar"] = {"freq": "M", "on": "first"} if rebalance == "calendar" else None
                    if rebalance != "calendar":
                        schedule["threshold"] = None
                    ui.prune_variables()
                ui.rerun_after(switch_rebalance)
            if rebalance == "never":
                st.caption("The weights are bought on the first bar that has them, then left to drift: buy and hold.")
            if rebalance == "calendar":
                row = st.columns(2)
                with row[0]:
                    ui.param_input("schedule.calendar.freq", {"type": "choice", "options": CALENDAR_FREQS,
                                                              "default": "M"}, label="Every",
                                   help="W week, M month, Q quarter, Y year.")
                with row[1]:
                    ui.param_input("schedule.calendar.on", {"type": "choice", "options": ["first", "last"],
                                                            "default": "first"}, label="On the",
                                   help="The first or the last bar of each period. The trades fill at the next "
                                        "bar's open.")
                use_threshold = st.checkbox("Only when the weights have drifted", value=schedule.get("threshold") is not None,
                                            key=ui.key("schedule.threshold", ":on"),
                                            help="Drift is measured from the last calendar rebalance.")
                if use_threshold and schedule.get("threshold") is None:
                    ui.set_value("schedule.threshold", 0.05)
                    st.rerun()
                if use_threshold:
                    ui.param_input("schedule.threshold", {"type": "float", "default": 0.05},
                                   label="Tolerance (0.05 = 5%)",
                                   help="How far an asset's weight may drift, relative to its target, before it "
                                        "counts as drifted: 0.05 lets a 40% target move between 38% and 42%.")
                    ui.param_input("schedule.combine", {"type": "choice", "options": ["and", "or"], "default": "and"},
                                   label="Rebalance when",
                                   help="and: calendar dates where drift is too large. or: calendar dates, plus any "
                                        "bar where drift is too large.")
                elif schedule.get("threshold") is not None:
                    ui.set_value("schedule.threshold", None)
            ui.param_input("schedule.min_trade", {"type": "float", "default": 0.005},
                           label="Skip trades smaller than (share of value)",
                           # ⚠️ vectorbtX's to_targets compares each new target with the last one, not with
                           # the drifted holding, so a small trade back to an unchanged target is still made
                           # (docs/UPSTREAM.md, issue 21). Reword this when it is fixed.
                           help="When an asset's target weight changes by less than this (0.005 = 0.5% of the "
                                "account), it keeps its previous target. ⚠️ The new target is compared with the "
                                "last one, not with the drifted holding, so a small trade back to an unchanged "
                                "target is still made.")

    # --- risk ------------------------------------------------------------------------
    with tabs["Risk"]:
        risk = spec.setdefault("risk", {})
        if kind != "signals":
            st.write("A portfolio is sized by its weights, so there is no sizing, stop or target here.")
        else:
            labels = dict(SIZING_LABELS[trading])
            sizing = risk.get("sizing") or {"stake": 1.0}
            used = next((name_ for name_ in SIZING_KEYS if name_ in sizing), "stake")
            if used not in labels:
                labels[used] = "Units (raw size)"
            chosen_sizing = st.radio("Position size", list(labels), index=list(labels).index(used),
                                     format_func=labels.get, key=ui.key("risk.sizing", ":kind"),
                                     # ⚠️ Risk sizing with no stop (from the leverage) waits on vectorbtX
                                     # (docs/UPSTREAM.md issue 28); reword this when it is built.
                                     help="How big each new position is. The two risk choices size the trade so "
                                          "that hitting the stop loses that much, so they need a fixed stop in "
                                          "points. Sizing a risk with no stop (from the leverage: the move that "
                                          "would take the trade's whole margin) waits on vectorbtX.")
            if chosen_sizing != used:
                def switch_sizing():
                    risk["sizing"] = {chosen_sizing: SIZING_DEFAULTS[chosen_sizing]}
                    ui.prune_variables()
                ui.rerun_after(switch_sizing)
            amount_help = {"risk_pct": "The share of the starting cash lost if the stop is hit (0.01 = 1%): a fixed "
                                       "amount on every trade, not a share of the equity or balance at the time.",
                           "risk_amount": "The cash lost if the stop is hit, in the account's currency.",
                           "stake": f"{labels.get('stake', 'Units')}: the instrument's own unit, the same on every "
                                    "trade." + (" A spread bet's stake is money per point: per cent of price on a "
                                                "share, per pip on FX, per index point on an index."
                                                if trading == "spreadbet" else ""),
                           "notional": "The position's full value when it opens, the same on every trade.",
                           "size": "A raw size in the engine's own units."}
            amount = ui.param_input(f"risk.sizing.{used}", {"type": "float", "default": 1.0},
                                    label="Amount (a fraction: 0.01 = 1%)" if used == "risk_pct" else "Amount",
                                    help=amount_help.get(used))
            cash = float((spec.get("account") or {}).get("init_cash") or 0.0)
            # What the number means in money, under the box: 0.05 is 5%, not 0.05%.
            # ⚠️ vectorbtX sizes risk_pct off init_cash, before the run (frontend/prepare.py resolve_size):
            # the same money on every trade, whatever the account has made or lost since.
            if used == "risk_pct" and isinstance(amount, (int, float)):
                st.caption(f"{amount:g} is {amount:.2%} of the starting cash of {cash:,.0f}: {amount * cash:,.2f} lost "
                           "on each trade that hits its stop. That amount is fixed for the whole backtest: it is not "
                           "a share of the account's equity or balance at the time, so it does not grow with profits "
                           "or shrink after losses.")
            elif used == "risk_amount" and isinstance(amount, (int, float)) and cash:
                st.caption(f"{amount:,.2f} lost on each trade that hits its stop, the same on every trade: "
                           f"{amount / cash:.2%} of the starting cash of {cash:,.0f}.")
            stop = risk.get("stop") or {"kind": "none"}
            stop_kind = st.radio("Stop", ["none", "fixed", "trailing"],
                                 index=["none", "fixed", "trailing"].index(stop.get("kind", "none")),
                                 format_func={"none": "No stop", "fixed": "Fixed stop", "trailing": "Trailing stop"}.get,
                                 horizontal=True, key=ui.key("risk.stop", ":kind"),
                                 help="Fixed and trailing stops cannot be combined: a trailing stop replaces the fixed "
                                      "one. Whether there is a stop, which kind, how it is measured and its distance "
                                      "can all be searched on the Optimise page (risk sizing and the extra exits need "
                                      "a fixed stop in points, so with either only the distance is). A stop from a "
                                      "feature, such as 3 x ATR or an EMA, waits on vectorbtX taking a stop per bar.")
            if stop_kind != stop.get("kind", "none"):
                def switch_stop():
                    # The unit and both distances stay while the stop is off, so the optimiser can switch it on.
                    risk["stop"] = dict(stop, kind=stop_kind)
                    risk["stop"].setdefault("points", DISTANCE_DEFAULTS["stop"]["points"])
                    ui.prune_variables()
                ui.rerun_after(switch_stop)
            if stop_kind != "none":
                distance_inputs(
                    "stop", trailing=stop_kind == "trailing",
                    help_points="How far from the fill price the stop sits, in points: the price distance divided by "
                                "the point size. On a share a point is 1 cent, so a 1.00 move is 100 points; on FX it "
                                "is a pip (0.0001, or 0.01 on a yen pair). ⚠️ A trailing stop in points starts this "
                                "far behind the fill, but vectorbtX turns the distance into a share of the fill price, "
                                "so it widens as the price rises: at twice the entry price it trails twice as far. A "
                                "trailing stop in % of the price does not drift.",
                    help_pct="How far from the fill price the stop sits, as a fraction of that price: 0.02 puts it 2% "
                             "below a long's fill (above a short's). A trailing stop keeps that share of the best "
                             "price since the entry behind it. vectorbt's own stop, measured from the fill like a "
                             "stop in points.")
                if (risk["stop"].get("unit") == "pct" and
                        any(key in (risk.get("sizing") or {}) for key in ("risk_pct", "risk_amount"))):
                    st.warning("Risk sizing needs the stop in points: vectorbtX cannot size a trade from a stop in % "
                               "of the price yet. Measure the stop in points, or size by stake or notional.")
            target = risk.get("target") or {"kind": "none"}
            had_target = target.get("kind") == "fixed"
            use_target = st.checkbox("Take profit", value=had_target, key=ui.key("risk.target", ":on"),
                                     help="Close the trade once it has moved a set distance in your favour. Whether "
                                          "there is one, how it is measured, and its distance can be searched on the "
                                          "Optimise page.")
            if use_target != had_target:
                def switch_target():
                    # The unit and both distances stay while the target is off, so the optimiser can switch it on.
                    risk["target"] = dict(target, kind="fixed" if use_target else "none")
                    risk["target"].setdefault("points", DISTANCE_DEFAULTS["target"]["points"])
                    ui.prune_variables()
                ui.rerun_after(switch_target)
            if use_target:
                distance_inputs(
                    "target",
                    help_points="How far from the fill price the target sits, in points (cents on a share, pips on "
                                "FX). It fills like any exit, paying the spread.",
                    help_pct="How far from the fill price the target sits, as a fraction of that price: 0.04 puts it "
                             "4% above a long's fill (below a short's). It fills like any exit, paying the spread.")
            # No leverage budget here: vectorbtX applies one to a portfolio's weights only.
            # Each instrument's own leverage (just below) caps a rule strategy's trades.
            leverage_summary()
            shorts = {"partial": "Fill as much as the account funds", "skip": "Skip the trade"}
            current_short = when_cash_short(spec)
            chosen_short = st.radio(
                "When a trade is bigger than the account can fund", list(shorts), index=list(shorts).index(current_short),
                format_func=shorts.get, horizontal=True, key=ui.key("account.when_cash_short"),
                help="An account funds its cash times the leverage (in a shared account, the cash left). Filling as "
                     "much as it funds cuts the trade down, so a risk-sized trade then risks less than you asked. "
                     "Skipping places nothing, so the run has fewer trades than signals. The default is to fill "
                     "with an account per symbol and to skip in a shared account.")
            if chosen_short != current_short:
                ui.set_value("account.when_cash_short", chosen_short)
            trade_size_check()

            st.markdown("**Extra exits**")
            exits = risk.get("exits") or {}
            extra = [("time_bars", "Close after a number of bars", {"type": "int", "default": 20}, "Bars after entry"),
                     ("breakeven_points", "Move the stop to the entry price once in profit",
                      {"type": "float", "default": 50.0}, "Points in favour"),
                     ("step_points", "Step the stop up as the trade advances", {"type": "float", "default": 25.0},
                      "Step (points)")]
            # The ⓘ for each: (on the checkbox, on its amount). Favourable moves are measured on
            # completed bars only (exit_rules.py), so none of them peeks at the current bar.
            extra_help = {
                "time_bars": ("Closes a trade that is still open after a set number of bars, at that bar's open.",
                              "How many bars after the entry bar the trade is closed."),
                "breakeven_points": ("Once the trade has been this far in your favour, the stop moves to the entry "
                                     "price, so the trade can no longer lose (before costs and gaps).",
                                     "How far in your favour, in points, the best price must have gone."),
                "step_points": ("Each time the best price advances another step from the entry, the stop moves up "
                                "(down for a short) by one step. It never moves back.",
                                "The step, in points."),
            }
            for exit_key, label, declaration, amount in extra:
                on = st.checkbox(label, value=exits.get(exit_key) is not None, key=ui.key(f"risk.exits.{exit_key}", ":on"),
                                 help=extra_help[exit_key][0])
                if on and exits.get(exit_key) is None:
                    ui.rerun_after(lambda exit_key=exit_key, declaration=declaration:
                                   risk.setdefault("exits", {}).__setitem__(exit_key, declaration["default"]))
                if not on and exits.get(exit_key) is not None:
                    def drop_exit(exit_key=exit_key):
                        risk["exits"].pop(exit_key)
                        ui.prune_variables()
                    ui.rerun_after(drop_exit)
                if on:
                    ui.param_input(f"risk.exits.{exit_key}", declaration, label=amount, help=extra_help[exit_key][1])
            if any(exits.get(name_) is not None for name_, *_ in extra) and \
                    (stop_kind != "fixed" or (risk.get("stop") or {}).get("unit") == "pct"):
                st.warning("The extra exits adjust a stop from its distance in points, so they need a fixed stop "
                           "measured in points.")

            st.markdown("**Order timing**")
            orders = risk.get("orders") or {}
            intraday = is_intraday(spec)
            for order_key, label, help_text in (
                    ("good_for_day", "Entries are good for the day only",
                     "An entry that would fill on the next day lapses instead: a signal on a day's last bar is dropped."),
                    ("flat_by_day_end", "Close positions before the end of the day",
                     "Exits on the bar before each day's last bar, so it fills at the last bar's open, and takes no new "
                     "entries that would fill on that last bar.")):
                on = st.checkbox(label, value=bool(orders.get(order_key)), key=ui.key(f"risk.orders.{order_key}"),
                                 disabled=not intraday and not orders.get(order_key), help=help_text)
                if on != bool(orders.get(order_key)):
                    ui.set_value(f"risk.orders.{order_key}", on)
            st.caption("For intraday bars. A day is the bars' calendar date (UTC). ⚠️ Every order fills at a bar's "
                       "open, so closing before the end of the day fills at the last bar's open, not the close: "
                       "vectorbtX has no market-on-close order yet." if intraday else
                       "Order timing needs intraday bars (under one day).")
            rule = engine.SAME_BAR_STOP_RULE
            st.caption("Orders fill at the next bar's open. Stops and targets are measured from that open, and a "
                       "stopped-out trade pays the spread on exit like any other. A bar that opens beyond the stop "
                       "fills at its open. "
                       + (f"When one bar's range holds both the stop and the target, the {rule} is taken."
                          if rule else "Which of stop and target wins inside one bar has not been confirmed yet.")
                       + " Points are price distances divided by the instrument's point size.")
        if kind != "signals":
            leverage_summary()
            # call_plan passes the box's value either way (vectorbtX's own default is on).
            budget = st.checkbox("Use a leverage budget", value=bool(risk.get("leverage_budget")),
                                 key=ui.key("risk.leverage_budget"),
                                 help="Scales the target weights down on any bar where the book's gross exposure "
                                      "is above the leverage allowed. Only instruments whose leverage_type is "
                                      "portfolio_level set that limit, so without one this changes nothing.")
            if budget != bool(risk.get("leverage_budget")):
                ui.set_value("risk.leverage_budget", budget)

    # --- account ----------------------------------------------------------------------
    with tabs["Account"]:
        account = spec.setdefault("account", {})
        row = st.columns(2)
        with row[0]:
            ui.param_input("account.init_cash", {"type": "float", "default": 10000.0}, label="Starting cash",
                           help="The account's opening balance, in the reporting currency below (or the "
                                "instruments' own when none is set). With separate accounts, each symbol starts "
                                "with this much.")
        with row[1]:
            freqs = sorted(set(FREQ.values()) | {account.get("freq", "D")})
            ui.param_input("account.freq", {"type": "choice", "options": freqs, "default": FREQ[timeframe]},
                           label="Bar frequency",
                           help="How long one bar is, so annualised figures (Sharpe, yearly return) scale "
                                "correctly. It follows the timeframe; change it only for unusual data.")
        # Read, not created: rendering must not change the spec. set_value makes it on the first write.
        extra = spec.get("engine_extra") or {}

        # --- cash across symbols: one account per symbol, or one shared account ---
        random_priority = False
        if kind == "signals" and len(symbols) > 1:
            # The radio below has no label of its own, so the heading carries its ⓘ.
            st.markdown("**Cash across symbols**",
                        help="Separate accounts: each symbol trades its own copy of the starting cash, and the "
                             "results add them up. One shared account: every symbol draws on the same cash, so "
                             "entries can compete for it.")
            cash_each = float(account.get("init_cash", 10000.0) or 0.0)
            models = {"per_instrument": f"Each symbol trades its own copy of the account ({len(symbols)} × "
                                        f"{cash_each:,.0f} = {len(symbols) * cash_each:,.0f} in all)",
                      "shared": f"One account of {cash_each:,.0f}, shared by every symbol"}
            current_cash = account.get("cash") or "per_instrument"
            chosen_cash = st.radio("Cash across symbols", list(models), index=list(models).index(current_cash)
                                   if current_cash in models else 0, format_func=models.get,
                                   key=ui.key("account.cash"), label_visibility="collapsed")
            if chosen_cash != current_cash:
                def switch_cash():
                    account["cash"] = chosen_cash
                    if chosen_cash != "shared":
                        # The priority only means something in a shared account (validate_spec says so).
                        # What a trade too big to fund does (when_cash_short) stays as chosen on the
                        # Risk tab; left unset, its default follows the account (spec.when_cash_short).
                        account.pop("entry_priority", None)
                ui.rerun_after(switch_cash)
            if chosen_cash == "shared":
                orders_by = {"symbols": "In symbol order", "random": "In a random order (seeded, so runs repeat)",
                             "feature": "By a feature's value on the signal bar"}
                priority = account.get("entry_priority") or {"by": "symbols"}
                chosen_by = st.radio("When entries compete for the cash, fill them", list(orders_by),
                                     index=list(orders_by).index(priority.get("by")) if priority.get("by") in orders_by
                                     else 0, format_func=orders_by.get, key=ui.key("account.entry_priority", ":by"),
                                     help="Exits always go first on a bar, so the cash they free is there for entries.")
                if chosen_by != priority.get("by", "symbols"):
                    def switch_priority():
                        if chosen_by == "feature":
                            first = next(iter(features), None)
                            account["entry_priority"] = {"by": "feature", "highest_first": True,
                                                         "score": {"feature": first} if first else {"price": "close"}}
                        else:
                            account["entry_priority"] = {"by": chosen_by}
                    ui.rerun_after(switch_priority)
                if chosen_by == "symbols":
                    order = priority.get("order") or []
                    picked_order = st.multiselect("Fill first, in this order (optional)", symbols,
                                                  default=[symbol for symbol in order if symbol in symbols],
                                                  key=ui.key("account.entry_priority.order"),
                                                  help="Symbols left out follow, in the order of the symbol list.")
                    if picked_order != order:
                        if picked_order:
                            ui.set_value("account.entry_priority", {"by": "symbols", "order": picked_order})
                        else:
                            ui.set_value("account.entry_priority", {"by": "symbols"})
                elif chosen_by == "random":
                    random_priority = True
                    st.caption("Each bar's competing entries are shuffled with the random seed just below.")
                else:
                    ui.ref_input("account.entry_priority.score", features, "Rank by",
                                 help="The value that decides which competing entry is filled first, such as a "
                                      "momentum or volatility feature.")
                    highest = st.radio("Which goes first", [True, False],
                                       index=0 if priority.get("highest_first", True) else 1,
                                       format_func={True: "The highest value", False: "The lowest value"}.get,
                                       horizontal=True, key=ui.key("account.entry_priority.highest_first"),
                                       help="Whether the entry with the highest or the lowest value is filled "
                                            "first.")
                    if highest != priority.get("highest_first", True):
                        ui.set_value("account.entry_priority.highest_first", highest)
                    st.caption("The value on the bar the entry signal fires, before the fill at the next open.")
                st.caption("What an entry does when it needs more cash than is left (skip it, or fill part of it) is "
                           f"set on the Risk tab: now {'skip it' if when_cash_short(spec) == 'skip' else 'fill part'}.")
            st.caption("The optimiser's vectorised grid needs separate accounts; a shared account uses the ordinary grid.")
        # The one random seed (vectorbt's seed=), right under the priority that may need it.
        seed_text = st.text_input("Random seed (needed for a random entry order)" if random_priority else
                                  "Random seed (optional)",
                                  value="" if extra.get("seed") is None else str(extra["seed"]),
                                  key=ui.key("engine_extra.seed"),
                                  help="A whole number that makes random choices repeat from run to run, such as a "
                                       "random entry order. Blank: none set.")
        seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
        if seed != extra.get("seed"):
            if seed is None:
                ui.delete_key("engine_extra.seed")
            else:
                ui.set_value("engine_extra.seed", seed)
        base = st.text_input("Report in currency (optional)", value=account.get("base_currency") or "",
                             key=ui.key("account.base_currency"),
                             help="A three-letter code, such as GBP. Prices are converted into it with the FX "
                                  "dataset below, paying the broker's conversion markup (Settings). Leave blank to "
                                  "report in each instrument's currency.")
        if (base or None) != account.get("base_currency"):
            if base:
                ui.set_value("account.base_currency", base.upper())
            else:
                ui.delete_key("account.base_currency")
                ui.delete_key("account.fx")
        if base:
            fx = account.get("fx") or {}
            fx_symbol = st.selectbox("FX dataset to convert with", [None] + stored,
                                     index=([None] + stored).index(fx.get("symbol")) if fx.get("symbol") in stored else 0,
                                     key=ui.key("account.fx"), help="Store the pair (for example GBPUSD=X) on the Data page.")
            if fx_symbol and fx.get("symbol") != fx_symbol:
                ui.set_value("account.fx", {"symbol": fx_symbol})
        rates = account.get("rates")
        source = st.radio("Holding costs from interest rates", ["none", "static", "fred"],
                          index=["none", "static", "fred"].index(rates["source"] if rates else "none"),
                          format_func={"none": "None", "static": "Fixed rates", "fred": "FRED (needs a free key)"}.get,
                          horizontal=True, key=ui.key("account.rates", ":source"),
                          help="The overnight cost of holding a position: financing on borrowed money, and carry "
                               "on FX. None charges neither. Fixed rates use one yearly rate per currency for the "
                               "whole period. FRED uses the published overnight rates day by day.")
        if source != (rates["source"] if rates else "none"):
            ui.rerun_after(lambda: account.__setitem__("rates", None if source == "none" else
                                                       {"source": source, "values": {"USD": 0.05}} if source == "static"
                                                       else {"source": "fred"}))
        if source == "static":
            st.warning("A flat rate misreads any period that spans a rate cycle.")
            values = rates.get("values") or {}
            frame = pd.DataFrame({"currency": list(values.keys()), "rate": list(values.values())})
            edited = st.data_editor(frame, num_rows="dynamic", key=ui.key("account.rates.values"), width="stretch",
                                    hide_index=True, column_config={
                                        "currency": st.column_config.TextColumn(
                                            "currency", help="A three-letter code, such as USD."),
                                        "rate": st.column_config.NumberColumn(
                                            "rate", help="The yearly interest rate as a fraction (0.05 = 5%).")})
            new_values = {str(code).strip().upper(): float(rate) for code, rate in zip(edited["currency"], edited["rate"])
                          if isinstance(code, str) and code.strip() and pd.notna(rate)}
            if new_values != values:
                ui.set_value("account.rates.values", new_values)
        if source == "fred":
            st.caption("FRED key found." if os.environ.get("FRED_API_KEY") else
                       "No FRED key yet: set FRED_API_KEY or add it on the Settings page.")
        st.caption("Carry on an FX instrument needs rates for both of its currencies, and the instrument must name both.")
        # Nothing is written here unless changed: a blank choice keeps vectorbt's default.
        modes = [None, "ignore", "close", "closereduce", "reverse", "reversereduce"]
        current_mode = extra.get("upon_opposite_entry")
        opposite = st.selectbox("When an opposite entry arrives", modes,
                                index=modes.index(current_mode) if current_mode in modes else 0,
                                format_func=lambda mode: "Library default" if mode is None else mode,
                                key=ui.key("engine_extra.upon_opposite_entry"),
                                help="A long entry while short, or a short entry while long. ignore: keep the "
                                     "position. close: close it. reverse: close it and open the other way. The "
                                     "reduce forms differ only when positions add up, which the Studio does not "
                                     "do. Library default: reversereduce.")
        if opposite != current_mode:
            if opposite is None:
                ui.delete_key("engine_extra.upon_opposite_entry")
            else:
                ui.set_value("engine_extra.upon_opposite_entry", opposite)

    # --- preview ------------------------------------------------------------------------
    problems = validate_spec(spec)
    with tabs["Preview"]:
        if problems:
            st.write("Fix the problems in the At a glance panel to see the preview.")
        elif ui.missing_data(spec, data_dir):
            st.info(f"Import {', '.join(ui.missing_data(spec, data_dir))} at {timeframe} (or finer) on the Data page "
                    "to see the preview.")
        else:
            result = ui.run_guarded("Computing signals", preview)
            if result is not None:
                frames, computed, compiled = result
                theme = settings.get("default_theme", analysis.PAGE_THEME)
                if kind == "signals":
                    counts = {name_: int(compiled[name_].to_numpy().sum())
                              for name_ in ("entries", "exits", "short_entries", "short_exits")}
                    labels = {"entries": "Long entries", "exits": "Long exits", "short_entries": "Short entries",
                              "short_exits": "Short exits"}
                    columns = st.columns(4)
                    for column, (name_, count) in zip(columns, counts.items()):
                        column.metric(labels[name_], count)
                    rules = spec.get("rules") or {}
                    for slot, name_ in (("long_entry", "entries"), ("long_exit", "exits"),
                                        ("short_entry", "short_entries"), ("short_exit", "short_exits")):
                        if rules.get(slot) and counts[name_] == 0:
                            st.warning(f"The {slot.replace('_', ' ')} rule never fires on this data.")
                    for entry_name, exit_name in (("entries", "exits"), ("short_entries", "short_exits")):
                        entries = compiled[entry_name].to_numpy()
                        if entries.sum():
                            soon = compiled[exit_name].to_numpy() | np.roll(compiled[exit_name].to_numpy(), -1, axis=0)
                            if (entries & soon).sum() / entries.sum() > 0.5:
                                st.warning(f"Most {labels[entry_name].lower()} are followed by an exit on the same or "
                                           "the next bar.")
                    for feature_name, outputs in computed.items():
                        if "signal" in outputs and not outputs["signal"].to_numpy().any():
                            st.warning(f"Pattern {feature_name} never appears on this data.")
                    row = st.columns([1, 2])
                    symbol = row[0].selectbox("Symbol", list(frames["close"].columns), key="preview:symbol",
                                              help="Which symbol's chart to show.")
                    overlays = row[1].multiselect("Overlay", [f"{name_}.{output}" for name_, outputs in computed.items()
                                                              for output in outputs], key="preview:overlays",
                                                  help="Feature lines to draw over the price, to check they look "
                                                       "right before backtesting.")
                    ui.chart(analysis.fig_preview(frames, computed, compiled, symbol, theme, overlays))
                else:
                    ui.chart(analysis.fig_weight_preview(compiled["weights"], compiled["schedule"], theme))
                    st.caption(f"{int(compiled['schedule'].sum())} rebalance dates.")

    # --- pipeline --------------------------------------------------------------------------
    with tabs["Pipeline"]:
        st.caption("How the strategy flows from data to the backtest. Versions of a rule are drawn as currently chosen.")
        if problems:
            st.write("Fix the problems in the At a glance panel to draw the pipeline.")
        else:
            palette = style.pipeline_palette(style.active_theme(settings)[0])
            dot = ui.run_guarded("Drawing the pipeline", lambda: analysis.pipeline_dot(spec, palette))
            if dot:
                st.graphviz_chart(dot)

    # --- the strategy file and the code --------------------------------------------------
    with tabs["Strategy file"]:
        # Keyed on the spec's hash: the box re-seeds whenever another step changes the
        # spec, so it can never hold a stale copy that would undo those edits.
        text = st.text_area("Strategy file (JSON)", value=json.dumps(spec, indent=2), height=420,
                            key=ui.key("json", ":" + spec_hash(spec)),
                            help="The whole strategy as one file: what the other tabs edit. You can edit it here "
                                 "too, then press Use the edited file.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            st.error(f"That is not valid JSON: {error.msg} at line {error.lineno}.")
            parsed = None
        if parsed is not None and parsed != spec:
            ui.show_problems(validate_spec(parsed), "The edited file has problems")
            if st.button("Use the edited file"):
                ui.replace_spec(parsed)
                st.rerun()
        st.download_button("Download strategy file", json.dumps(spec, indent=2), file_name=f"{spec['name']}.json",
                           mime="application/json")
        if not problems:
            script = ui.run_guarded("Writing the script", lambda: export_script(spec, data_dir=data_dir))
            if script:
                st.markdown("**The same backtest as a plain vectorbtX script** (read-only)")
                st.code(script, language="python")
                st.download_button("Download script", script, file_name=f"{spec['name']}.py", mime="text/x-python")

# Drawn last, so it shows what this run's edits made of the strategy.
with side:
    with st.container(key="glance"):
        ui.glance(spec, validate_spec(spec))
