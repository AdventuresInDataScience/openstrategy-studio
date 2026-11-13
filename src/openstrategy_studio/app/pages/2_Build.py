"""Build: set the strategy up step by step, with the whole of it at a glance beside the form.

Every step edits the same dict through ui.set_value; no widget keeps its own copy.
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, indicators, style, ui
from openstrategy_studio.compile import compile_features, compile_signals, compile_weights, export_script
from openstrategy_studio.spec import (CALENDAR_FREQS, FREQ, SELECTION_MODES, SIZING_KEYS, TIMEFRAMES, TRADING,
                                      WEIGHT_SCHEMES, apply_params, is_intraday, list_templates, load_template,
                                      spec_hash, strategy_kind, trading_mode, tunable_inputs, validate_spec)

settings, data_dir = ui.boot("Build")
ui.page_help("build")
spec = ui.current()
st.title("Build")

# Every Instrument field the grid can show; each mode shows the ones vectorbtX accepts for it.
INSTRUMENT_FIELDS = {
    "asset_class": "text", "point_size": "number", "contract_size": "number", "increment": "number",
    "min_size": "number", "allow_fractional": "flag", "quote_currency": "text", "base_currency": "text",
    "cost_currency": "text", "commission_pct": "number", "commission_flat": "number", "commission_per_unit": "number",
    "commission_min": "number", "spread_points": "number", "slippage": "text", "borrow_rate": "number",
    "rounding": "text", "leverage": "number", "leverage_type": "text", "price_field": "text",
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


def prune_variables():
    """Drop optimise variables whose value the strategy no longer has (after a switch that removes it)."""
    known = {row["path"] for row in tunable_inputs(spec)}
    found = (spec.get("optimise") or {}).get("variables") or {}
    for path in [path for path in found if path not in known]:
        del found[path]


# --- start from a template -----------------------------------------------------------
templates = list_templates()
left, right = st.columns([4, 1])
chosen = left.selectbox("Start from", [None] + list(templates),
                        format_func=lambda name: "Keep the current strategy" if name is None else templates[name])
right.write("")
if right.button("Start from this", disabled=chosen is None):
    ui.replace_spec(load_template(chosen))
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
        name = row[0].text_input("Name", value=spec.get("name", ""), key=ui.key("name"))
        if name != spec.get("name"):
            ui.set_value("name", name)
        options = sorted(set(stored) | set(symbols))
        picked_symbols = row[1].multiselect("Symbols", options, default=symbols, key=ui.key("data.symbols"),
                                            help="Import symbols on the Data page first.")
        row = st.columns(3)
        picked_timeframe = row[0].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(timeframe),
                                            key=ui.key("data.timeframe"), help="Every rule is evaluated once per bar.")
        # Streamlit limits date pickers to ten years either side unless told otherwise.
        span = {"min_value": date(1970, 1, 1), "max_value": date(2100, 12, 31)}
        start = row[1].date_input("From", value=date.fromisoformat(str(spec["data"]["start"])[:10]),
                                  key=ui.key("data.start"), **span)
        end = row[2].date_input("To", value=date.fromisoformat(str(spec["data"]["end"])[:10]),
                                key=ui.key("data.end"), **span)
        if picked_symbols != symbols:
            def switch_symbols():
                instruments = spec.setdefault("instruments", {})
                if "from_file" not in instruments:
                    template = dict(next(iter(instruments.values()), {"point_size": 0.01}))
                    for symbol in picked_symbols:
                        instruments.setdefault(symbol, dict(template))
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
                                 key=ui.key("setup.kind"))
        picked_trading = right.radio("How are you trading it?", TRADING, index=TRADING.index(trading),
                                     format_func=TRADING_LABELS.get, captions=[TRADING_HELP[name] for name in TRADING],
                                     key=ui.key("setup.trading"))
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
                    spec["risk"] = {"sizing": {"stake": 1.0}, "stop": {"kind": "none"}, "target": None}
                else:
                    spec["weighting"] = {"scheme": "equal", "vol_window": 20}
                    spec["schedule"] = {"rebalance": "never", "calendar": None, "threshold": None, "combine": "and",
                                        "min_trade": 0.005}
                    spec["risk"] = {}
                prune_variables()
            ui.rerun_after(switch_kind)
        if picked_trading != trading:
            def switch_trading():
                spec["setup"]["trading"] = picked_trading
                read = ui.run_guarded("Reading instrument modes", lambda: engine.mode_rules(picked_trading))
                rejected = set((read or {"rejected": {}})["rejected"])
                for fields in (spec.get("instruments") or {}).values():
                    if isinstance(fields, dict):
                        for field in list(fields):
                            # base_currency stays on an FX pair whatever the mode: carry needs it.
                            if (field in rejected and not (field == "base_currency" and fields.get("asset_class") == "fx")) \
                                    or (field == "borrow_rate" and picked_trading != "equity"):
                                del fields[field]
            ui.rerun_after(switch_trading)

        st.markdown("**Instruments**")
        instruments = spec.setdefault("instruments", {})
        universes = sorted(path.name for path in (data_dir / "universes").glob("*"))
        use_file = st.toggle("Read instruments from a universe file", value="from_file" in instruments,
                             disabled=not universes and "from_file" not in instruments,
                             help="Store universe files on the Data page. A file names each row's own mode.",
                             key=ui.key("instruments.use_file"))
        if use_file:
            current_file = instruments.get("from_file")
            file = st.selectbox("Universe file", universes,
                                index=universes.index(current_file) if current_file in universes else 0,
                                key=ui.key("instruments.from_file"))
            if instruments != {"from_file": file}:
                ui.set_value("instruments", {"from_file": file})
        else:
            if "from_file" in instruments:
                spec["instruments"] = {symbol: {"point_size": 0.01} for symbol in symbols}
                instruments = spec["instruments"]
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
            config = {"rounding": st.column_config.SelectboxColumn("rounding", options=["", "floor", "nearest", "ceil"]),
                      "allow_fractional": st.column_config.SelectboxColumn("allow_fractional", options=["", "true", "false"]),
                      "price_field": st.column_config.SelectboxColumn("price_field", options=["", "close", "adj_close"]),
                      "slippage": st.column_config.TextColumn("slippage", help="A number, or auto (needs volume)."),
                      "borrow_rate": st.column_config.NumberColumn(
                          "borrow_rate", min_value=0.0, format="%.4f",
                          help="The yearly fee for borrowing what you short (0.005 = 0.5% a year), charged daily on "
                               "the short's value. Blank: unknown, so a short is charged nothing and the run says "
                               "so. 0: the borrow is free.")}
            st.caption(f"{TRADING_LABELS[trading]}: blank cells use the mode's default, and only the fields this mode "
                       "uses are shown. Costs come only from these fields. Financing on held positions comes from the "
                       "account's rates and the markup on the Settings page.")
            edited = st.data_editor(grid, column_config=config, key=ui.key(f"instruments.grid:{trading}"),
                                    width="stretch")
            updated = {}
            for symbol, values in edited.iterrows():
                fields = {}
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
            title = f"{feature_name}: {entry['label'] if entry else item.get('indicator')} · {indicators.describe(item)}"
            with st.expander(title):
                ui.feature_editor(feature_name, item, features)
        st.markdown("**Add a feature**")
        ui.feature_picker(features)

    # --- rules or portfolio ---------------------------------------------------------
    if kind == "signals":
        with tabs["Rules"]:
            rules = spec.setdefault("rules", {})
            row = st.columns([3, 1])
            preset = row[0].selectbox("Preset", [None] + list(PRESETS),
                                      format_func=lambda p: "Choose a preset to replace the rules" if p is None
                                      else PRESETS[p][0], key=ui.key("rules", ":preset"))
            row[1].write("")
            if row[1].button("Apply preset", disabled=preset is None):
                new = ui.spec_copy()
                new.setdefault("features", {}).update(json.loads(json.dumps(PRESETS[preset][1])))
                new["rules"] = {"long_entry": None, "long_exit": None, "short_entry": None, "short_exit": None}
                new["rules"].update(json.loads(json.dumps(PRESETS[preset][2])))
                ui.replace_spec(new)
                st.rerun()
            for slot, label in (("long_entry", "Long entry"), ("long_exit", "Long exit"),
                                ("short_entry", "Short entry"), ("short_exit", "Short exit")):
                node = rules.get(slot)
                st.subheader(label)
                enabled = st.toggle(f"Use a {label.lower()} rule", value=node is not None,
                                    key=ui.key(f"rules.{slot}", ":on"))
                if enabled and node is None:
                    ui.rerun_after(lambda slot=slot: rules.__setitem__(slot, ui.new_rule(features)))
                if not enabled and node is not None:
                    def clear(slot=slot):
                        rules[slot] = None
                        prune_variables()
                    ui.rerun_after(clear)
                if enabled:
                    ui.rule_editor(f"rules.{slot}", features, timeframe, entry=slot.endswith("entry"))
            st.caption("Filters only gate entries. Exits are never filtered, so a position is never trapped open. "
                       "Every signal fills at the next bar's open.")
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
                    prune_variables()
                ui.rerun_after(drop_selection)
            if use_selection:
                ui.ref_input("selection.score", features, "Rank assets by")
                ui.param_input("selection.mode", {"type": "choice", "options": SELECTION_MODES, "default": "top"},
                               label="Keep")
                if spec["selection"].get("mode", "top") in ("top", "bottom"):
                    ui.param_input("selection.top_n", {"type": "int", "default": 1}, label="How many")
                use_floor = st.checkbox("Require a minimum score", value=spec["selection"].get("min_score") is not None,
                                        key=ui.key("selection.min_score", ":on"))
                if use_floor and spec["selection"].get("min_score") is None:
                    ui.set_value("selection.min_score", 0.0)
                    st.rerun()
                if use_floor:
                    ui.param_input("selection.min_score", {"type": "float", "default": 0.0}, label="Minimum score")
                elif spec["selection"].get("min_score") is not None:
                    ui.set_value("selection.min_score", None)
            st.markdown("**Weighting**")
            ui.param_input("weighting.scheme", {"type": "choice", "options": WEIGHT_SCHEMES, "default": "equal"},
                           label="Scheme")
            scheme = spec["weighting"]["scheme"]
            if scheme == "inverse_vol":
                ui.param_input("weighting.vol_window", {"type": "int", "default": 20}, label="Volatility window (bars)")
            if scheme == "fixed":
                table = spec["weighting"].get("table") or {symbol: 1 / len(symbols) for symbol in symbols}
                frame = pd.DataFrame({"weight": [table.get(symbol, 0.0) for symbol in symbols]}, index=symbols)
                edited = st.data_editor(frame, key=ui.key("weighting.table"), width="stretch")
                new_table = {symbol: float(weight) for symbol, weight in edited["weight"].items()}
                if new_table != spec["weighting"].get("table"):
                    ui.set_value("weighting.table", new_table)
                st.caption("Fixed weights are used as given: below 100% the rest stays in cash.")
            st.markdown("**Rebalance**")
            schedule = spec["schedule"]
            names = {"never": "Never: buy once and hold", "every_bar": "On every bar", "calendar": "On a calendar"}
            rebalance = st.radio("Go back to the target weights", list(names), index=list(names).index(
                schedule.get("rebalance", "never")), format_func=names.get, horizontal=True,
                key=ui.key("schedule.rebalance"))
            if rebalance != schedule.get("rebalance"):
                def switch_rebalance():
                    schedule["rebalance"] = rebalance
                    schedule["calendar"] = {"freq": "M", "on": "first"} if rebalance == "calendar" else None
                    if rebalance != "calendar":
                        schedule["threshold"] = None
                    prune_variables()
                ui.rerun_after(switch_rebalance)
            if rebalance == "never":
                st.caption("The weights are bought on the first bar that has them, then left to drift: buy and hold.")
            if rebalance == "calendar":
                row = st.columns(2)
                with row[0]:
                    ui.param_input("schedule.calendar.freq", {"type": "choice", "options": CALENDAR_FREQS,
                                                              "default": "M"}, label="Every")
                with row[1]:
                    ui.param_input("schedule.calendar.on", {"type": "choice", "options": ["first", "last"],
                                                            "default": "first"}, label="On the")
                use_threshold = st.checkbox("Only when the weights have drifted", value=schedule.get("threshold") is not None,
                                            key=ui.key("schedule.threshold", ":on"),
                                            help="Drift is measured from the last calendar rebalance.")
                if use_threshold and schedule.get("threshold") is None:
                    ui.set_value("schedule.threshold", 0.05)
                    st.rerun()
                if use_threshold:
                    ui.param_input("schedule.threshold", {"type": "float", "default": 0.05},
                                   label="Tolerance (0.05 = 5%)")
                    ui.param_input("schedule.combine", {"type": "choice", "options": ["and", "or"], "default": "and"},
                                   label="Rebalance when",
                                   help="and: calendar dates where drift is too large. or: calendar dates, plus any "
                                        "bar where drift is too large.")
                elif schedule.get("threshold") is not None:
                    ui.set_value("schedule.threshold", None)
            ui.param_input("schedule.min_trade", {"type": "float", "default": 0.005},
                           label="Skip trades smaller than (share of value)")

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
                                     format_func=labels.get, key=ui.key("risk.sizing", ":kind"))
            if chosen_sizing != used:
                def switch_sizing():
                    risk["sizing"] = {chosen_sizing: SIZING_DEFAULTS[chosen_sizing]}
                    prune_variables()
                ui.rerun_after(switch_sizing)
            ui.param_input(f"risk.sizing.{used}", {"type": "float", "default": 1.0}, label="Amount")
            if used == "risk_pct":
                st.caption("Sized from starting cash, not compounding equity, and from the stop's distance.")
            stop = risk.get("stop") or {"kind": "none"}
            stop_kind = st.radio("Stop", ["none", "fixed", "trailing"],
                                 index=["none", "fixed", "trailing"].index(stop.get("kind", "none")),
                                 format_func={"none": "No stop", "fixed": "Fixed stop", "trailing": "Trailing stop"}.get,
                                 horizontal=True, key=ui.key("risk.stop", ":kind"),
                                 help="Fixed and trailing stops cannot be combined: a trailing stop replaces the fixed one.")
            if stop_kind != stop.get("kind", "none"):
                def switch_stop():
                    risk["stop"] = {"kind": stop_kind} if stop_kind == "none" else \
                        {"kind": stop_kind, "points": stop.get("points", 50.0)}
                    prune_variables()
                ui.rerun_after(switch_stop)
            if stop_kind != "none":
                ui.param_input("risk.stop.points", {"type": "float", "default": 50.0}, label="Distance (points)")
            use_target = st.checkbox("Take profit", value=bool(risk.get("target")), key=ui.key("risk.target", ":on"))
            if use_target and not risk.get("target"):
                ui.rerun_after(lambda: risk.__setitem__("target", {"points": 100.0}))
            if not use_target and risk.get("target"):
                def drop_target():
                    risk["target"] = None
                    prune_variables()
                ui.rerun_after(drop_target)
            if use_target:
                ui.param_input("risk.target.points", {"type": "float", "default": 100.0}, label="Distance (points)")
            budget = st.checkbox("Use a leverage budget", value=bool(risk.get("leverage_budget")),
                                 key=ui.key("risk.leverage_budget"))
            if budget != bool(risk.get("leverage_budget")):
                ui.set_value("risk.leverage_budget", budget)

            st.markdown("**Extra exits**")
            exits = risk.get("exits") or {}
            extra = [("time_bars", "Close after a number of bars", {"type": "int", "default": 20}, "Bars after entry"),
                     ("breakeven_points", "Move the stop to the entry price once in profit",
                      {"type": "float", "default": 50.0}, "Points in favour"),
                     ("step_points", "Step the stop up as the trade advances", {"type": "float", "default": 25.0},
                      "Step (points)")]
            for exit_key, label, declaration, amount in extra:
                on = st.checkbox(label, value=exits.get(exit_key) is not None, key=ui.key(f"risk.exits.{exit_key}", ":on"))
                if on and exits.get(exit_key) is None:
                    ui.rerun_after(lambda exit_key=exit_key, declaration=declaration:
                                   risk.setdefault("exits", {}).__setitem__(exit_key, declaration["default"]))
                if not on and exits.get(exit_key) is not None:
                    def drop_exit(exit_key=exit_key):
                        risk["exits"].pop(exit_key)
                        prune_variables()
                    ui.rerun_after(drop_exit)
                if on:
                    ui.param_input(f"risk.exits.{exit_key}", declaration, label=amount)
            if any(exits.get(name_) is not None for name_, *_ in extra) and stop_kind != "fixed":
                st.warning("The extra exits adjust a stop, so they need a fixed stop.")

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
            budget = st.checkbox("Use a leverage budget", value=bool(risk.get("leverage_budget")),
                                 key=ui.key("risk.leverage_budget"))
            if budget != bool(risk.get("leverage_budget")):
                ui.set_value("risk.leverage_budget", budget)

    # --- account ----------------------------------------------------------------------
    with tabs["Account"]:
        account = spec.setdefault("account", {})
        row = st.columns(2)
        with row[0]:
            ui.param_input("account.init_cash", {"type": "float", "default": 10000.0}, label="Starting cash")
        with row[1]:
            freqs = sorted(set(FREQ.values()) | {account.get("freq", "D")})
            ui.param_input("account.freq", {"type": "choice", "options": freqs, "default": FREQ[timeframe]},
                           label="Bar frequency", help="Annualised metrics need this.")
        extra = spec.setdefault("engine_extra", {})

        # --- cash across symbols: one account per symbol, or one shared account ---
        random_priority = False
        if kind == "signals" and len(symbols) > 1:
            st.markdown("**Cash across symbols**")
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
                    if chosen_cash == "shared":
                        account["when_cash_short"] = "skip"
                    else:
                        # Both only mean something in a shared account (validate_spec says so).
                        account.pop("entry_priority", None)
                        account.pop("when_cash_short", None)
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
                    ui.ref_input("account.entry_priority.score", features, "Rank by")
                    highest = st.radio("Which goes first", [True, False],
                                       index=0 if priority.get("highest_first", True) else 1,
                                       format_func={True: "The highest value", False: "The lowest value"}.get,
                                       horizontal=True, key=ui.key("account.entry_priority.highest_first"))
                    if highest != priority.get("highest_first", True):
                        ui.set_value("account.entry_priority.highest_first", highest)
                    st.caption("The value on the bar the entry signal fires, before the fill at the next open.")
                shorts = {"skip": "Skip it", "partial": "Fill as much as the cash allows"}
                current_short = account.get("when_cash_short") or "skip"
                chosen_short = st.radio("When an entry needs more cash than is left", list(shorts),
                                        index=list(shorts).index(current_short) if current_short in shorts else 0,
                                        format_func=shorts.get, horizontal=True, key=ui.key("account.when_cash_short"),
                                        help="Skipping keeps every position at the size the sizing rule chose. "
                                             "Filling part of it makes a risk-sized trade risk less than intended.")
                if chosen_short != current_short:
                    ui.set_value("account.when_cash_short", chosen_short)
            st.caption("The optimiser's vectorised grid needs separate accounts; a shared account uses the ordinary grid.")
        # The one random seed (vectorbt's seed=), right under the priority that may need it.
        seed_text = st.text_input("Random seed (needed for a random entry order)" if random_priority else
                                  "Random seed (optional)",
                                  value="" if extra.get("seed") is None else str(extra["seed"]),
                                  key=ui.key("engine_extra.seed"))
        seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
        if seed != extra.get("seed"):
            if seed is None:
                ui.delete_key("engine_extra.seed")
            else:
                ui.set_value("engine_extra.seed", seed)
        base = st.text_input("Report in currency (optional)", value=account.get("base_currency") or "",
                             key=ui.key("account.base_currency"), help="Leave blank to report in each instrument's currency.")
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
                          horizontal=True, key=ui.key("account.rates", ":source"))
        if source != (rates["source"] if rates else "none"):
            ui.rerun_after(lambda: account.__setitem__("rates", None if source == "none" else
                                                       {"source": source, "values": {"USD": 0.05}} if source == "static"
                                                       else {"source": "fred"}))
        if source == "static":
            st.warning("A flat rate misreads any period that spans a rate cycle.")
            values = rates.get("values") or {}
            frame = pd.DataFrame({"currency": list(values.keys()), "rate": list(values.values())})
            edited = st.data_editor(frame, num_rows="dynamic", key=ui.key("account.rates.values"), width="stretch",
                                    hide_index=True)
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
                                key=ui.key("engine_extra.upon_opposite_entry"))
        if opposite != current_mode:
            if opposite is None:
                ui.delete_key("engine_extra.upon_opposite_entry")
            else:
                ui.set_value("engine_extra.upon_opposite_entry", opposite)

    # --- preview ------------------------------------------------------------------------
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

    problems = validate_spec(spec)
    with tabs["Preview"]:
        if problems:
            st.write("Fix the problems in the At a glance panel to see the preview.")
        elif ui.missing_data(spec, data_dir):
            st.info(f"Import {', '.join(ui.missing_data(spec, data_dir))} at {timeframe} (or finer) on the Data page "
                    "to see the preview.")
        else:
            section = {key_: spec.get(key_) for key_ in ("version", "name", "setup", "data", "instruments", "features",
                                                         "rules", "selection", "weighting", "schedule", "risk",
                                                         "account", "optimise") if key_ in spec}
            result = ui.run_guarded("Computing signals", lambda: compute_preview(
                json.dumps(section, sort_keys=True), str(data_dir), ui.data_mtimes(spec, data_dir)))
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
                    symbol = row[0].selectbox("Symbol", list(frames["close"].columns), key="preview:symbol")
                    overlays = row[1].multiselect("Overlay", [f"{name_}.{output}" for name_, outputs in computed.items()
                                                              for output in outputs], key="preview:overlays")
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
                            key=ui.key("json", ":" + spec_hash(spec)))
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
