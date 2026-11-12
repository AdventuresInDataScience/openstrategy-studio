"""Build: edit the one spec dict, stage by stage, with a live preview beside the form.

Every tab edits the same dict through ui.set_value; no widget keeps its own copy.
"""

import json
import re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, engine, expressions, ui
from openstrategy_studio.blocks import BLOCKS, block_defaults
from openstrategy_studio.compile import compile_features, compile_signals, compile_weights, export_script
from openstrategy_studio.spec import (FREQ, PRICE_FIELDS, SELECTION_MODES, SIZING_KEYS, TIMEFRAMES, WEIGHT_SCHEMES,
                                      apply_params, is_choice, list_templates, load_template, spec_hash,
                                      strategy_kind, validate_spec)

settings, data_dir = ui.boot("Build")
spec = ui.current()
st.title("Build")

INSTRUMENT_FIELDS = {
    "mode": "text", "asset_class": "text", "point_size": "number", "contract_size": "number", "increment": "number",
    "min_size": "number", "allow_fractional": "flag", "quote_currency": "text", "base_currency": "text",
    "cost_currency": "text", "commission_pct": "number", "commission_flat": "number", "commission_per_unit": "number",
    "commission_min": "number", "spread_points": "number", "slippage": "text", "financing": "text",
    "rounding": "text", "leverage": "number", "leverage_type": "text", "price_field": "text",
}
SIZING_LABELS = {"risk_pct": "Risk a share of starting cash per trade (0.01 = 1%)",
                 "risk_amount": "Risk a cash amount per trade", "stake": "Stake per point (spread bets)",
                 "notional": "Position value in cash", "size": "Units (shares, contracts, lots)"}


def rerun_after(change):
    change()
    ui.save_draft()
    st.rerun()


# --- start from ------------------------------------------------------------------
templates = list_templates()
left, right = st.columns([4, 1])
chosen = left.selectbox("Start from", [None] + list(templates),
                        format_func=lambda name: "Keep the current strategy" if name is None else templates[name])
right.write("")
if right.button("Start from this", disabled=chosen is None):
    ui.replace_spec(load_template(chosen))
    st.rerun()

# --- basics ----------------------------------------------------------------------
stored = ui.stored_symbols(data_dir)
row = st.columns([3, 2, 1, 1, 1])
name = row[0].text_input("Name", value=spec.get("name", ""), key=ui.key("name"))
if name != spec.get("name"):
    ui.set_value("name", name)
options = sorted(set(stored) | set(spec["data"]["symbols"]))
symbols = row[1].multiselect("Symbols", options, default=spec["data"]["symbols"], key=ui.key("data.symbols"),
                             help="Import symbols on the Data page first.")
timeframe = row[2].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(spec["data"]["timeframe"]),
                             key=ui.key("data.timeframe"), help="Every rule is evaluated once per bar close.")
# Streamlit limits date pickers to ten years either side unless told otherwise.
span = {"min_value": date(1970, 1, 1), "max_value": date(2100, 12, 31)}
start = row[3].date_input("From", value=date.fromisoformat(str(spec["data"]["start"])[:10]), key=ui.key("data.start"), **span)
end = row[4].date_input("To", value=date.fromisoformat(str(spec["data"]["end"])[:10]), key=ui.key("data.end"), **span)
if symbols != spec["data"]["symbols"]:
    instruments = spec.setdefault("instruments", {})
    if "from_file" not in instruments:
        template = dict(next(iter(instruments.values()), {"mode": "equity", "point_size": 0.01}))
        for symbol in symbols:
            instruments.setdefault(symbol, dict(template))
        for symbol in list(instruments):
            if symbol not in symbols:
                del instruments[symbol]
    ui.set_value("data.symbols", symbols)
    st.rerun()
if timeframe != spec["data"]["timeframe"]:
    ui.set_value("data.timeframe", timeframe)
    ui.set_value("account.freq", FREQ[timeframe])
for field, value in (("start", start.isoformat()), ("end", end.isoformat())):
    if value != spec["data"][field]:
        ui.set_value(f"data.{field}", value)

kind = strategy_kind(spec)
kinds = {"signals": "Entry and exit rules", "portfolio": "Portfolio (weights and a rebalance schedule)",
         "hold": "Buy and hold"}
picked = st.radio("Strategy type", list(kinds), index=list(kinds).index(kind) if kind in kinds else 0,
                  format_func=kinds.get, horizontal=True, key=ui.key("kind"))
if kind == "conflict":
    st.warning("This strategy has both rules and a portfolio. Pick one type above; the other is removed.")
if picked != kind:
    def switch():
        for section in ("rules", "selection", "weighting", "schedule"):
            spec.pop(section, None)
        if picked == "signals":
            spec["rules"] = {"long_entry": {"all": [ui.new_leaf("comparison", spec.get("features") or {})]},
                             "long_exit": None, "short_entry": None, "short_exit": None}
            spec["risk"] = {"sizing": {"size": 1}, "stop": {"kind": "none"}, "target": None}
        elif picked == "portfolio":
            spec["weighting"] = {"scheme": "equal", "vol_window": 20}
            spec["schedule"] = {"calendar": {"freq": "M", "on": "first"}, "threshold": None, "combine": "and",
                                "min_trade": 0.005}
            spec["risk"] = {}
    rerun_after(switch)

form, preview = st.columns([3, 2], gap="large")
features = spec.setdefault("features", {})

with form:
    tab_names = ["Instruments", "Features", "Rules" if picked == "signals" else "Portfolio", "Risk", "Account",
                 "Pipeline", "Spec and code"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    # --- instruments -------------------------------------------------------------
    with tabs["Instruments"]:
        instruments = spec.setdefault("instruments", {})
        universes = sorted(path.name for path in (data_dir / "universes").glob("*"))
        use_file = st.toggle("Read instruments from a universe file", value="from_file" in instruments,
                             disabled=not universes and "from_file" not in instruments,
                             help="Store universe files on the Data page.", key=ui.key("instruments.use_file"))
        if use_file:
            current = instruments.get("from_file")
            file = st.selectbox("Universe file", universes, index=universes.index(current) if current in universes else 0,
                                key=ui.key("instruments.from_file"))
            if instruments != {"from_file": file}:
                ui.set_value("instruments", {"from_file": file})
        else:
            if "from_file" in instruments:
                spec["instruments"] = {symbol: {"mode": "equity", "point_size": 0.01} for symbol in spec["data"]["symbols"]}
                instruments = spec["instruments"]
            modes = ui.run_guarded("Reading instrument modes", engine.known_modes) or ["equity", "cfd", "spreadbet"]
            grid = pd.DataFrame([{field: instruments.get(symbol, {}).get(field) for field in INSTRUMENT_FIELDS}
                                 for symbol in spec["data"]["symbols"]], index=spec["data"]["symbols"])
            for field, kind_name in INSTRUMENT_FIELDS.items():
                if kind_name == "number":
                    grid[field] = pd.to_numeric(grid[field], errors="coerce")
                else:
                    grid[field] = grid[field].map(lambda value: "" if value is None else str(value).lower()
                                                  if isinstance(value, bool) else str(value))
            config = {"mode": st.column_config.SelectboxColumn("mode", options=modes, required=True),
                      "rounding": st.column_config.SelectboxColumn("rounding", options=["", "floor", "nearest", "ceil"]),
                      "allow_fractional": st.column_config.SelectboxColumn("allow_fractional", options=["", "true", "false"]),
                      "price_field": st.column_config.SelectboxColumn("price_field", options=["", "close", "adj_close"]),
                      "slippage": st.column_config.TextColumn("slippage", help="A number, or auto (needs volume)."),
                      "financing": st.column_config.TextColumn("financing", help="A number, or auto.")}
            st.caption("Blank cells use the mode's default. Costs come only from these fields.")
            edited = st.data_editor(grid, column_config=config, key=ui.key("instruments.grid"), width="stretch")
            updated = {}
            for symbol, values in edited.iterrows():
                fields = {}
                for field, kind_name in INSTRUMENT_FIELDS.items():
                    value = values[field]
                    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
                        continue
                    if kind_name == "number":
                        fields[field] = float(value)
                    elif kind_name == "flag":
                        fields[field] = value == "true"
                    elif field in ("slippage", "financing") and value != "auto":
                        fields[field] = float(value)
                    else:
                        fields[field] = str(value)
                updated[symbol] = fields
            if updated != instruments:
                ui.set_value("instruments", updated)

            def explain_modes():
                for symbol, fields in updated.items():
                    mode = fields.get("mode")
                    if not mode:
                        continue
                    rules = engine.mode_rules(mode)
                    rejected = dict(rules["rejected"])
                    # base_currency stays usable on FX spread bets and CFDs: carry needs it.
                    if fields.get("asset_class") == "fx":
                        rejected.pop("base_currency", None)
                    for field in sorted(set(fields) & set(rejected)):
                        st.warning(f"{symbol}: {field} is not used in {mode} mode ({rejected[field]}).")
                    for field in sorted(rules["required"] - set(fields)):
                        st.warning(f"{symbol}: {mode} mode needs {field}.")
                    if mode == "spreadbet":
                        st.caption(f"{symbol}: spread bets force commission to zero; the cost is in spread_points, "
                                   "and the stake is the size, so there is no contract size.")
                with st.expander("Defaults for each mode in use"):
                    for mode in sorted({fields.get("mode") for fields in updated.values() if fields.get("mode")}):
                        st.write(f"**{mode}**")
                        st.json(engine.mode_defaults(mode), expanded=False)

            ui.run_guarded("Checking instrument modes", explain_modes)

    # --- features ----------------------------------------------------------------
    with tabs["Features"]:
        st.caption("Features are indicators and patterns computed on every bar. Rules compare them.")
        for feature_name, feature in list(features.items()):
            if is_choice(feature):
                st.info(f"{feature_name} has alternatives ({', '.join(feature['options'])}); edit them in the "
                        "Spec and code tab.")
                continue
            block = BLOCKS.get(feature.get("block"))
            if block is None:
                st.error(f"{feature_name} uses an unknown block ({feature.get('block')}).")
                continue
            with st.expander(f"{feature_name}: {block['label']}"):
                if feature["block"] == "python":
                    st.warning("Python blocks run as code on your machine. Only use code you trust.")
                for param, declaration in block["params"].items():
                    path = f"features.{feature_name}.{param}"
                    if declaration["type"] == "source":
                        ui.source_input(path, features, feature_name)
                    elif feature["block"] == "pattern" and param == "name":
                        catalogue = [row for row in ui.pattern_catalogue() if row["module"] == feature.get("module", "candle")]
                        names = [row["name"] for row in catalogue]
                        value = st.selectbox("Pattern", names, index=names.index(feature["name"]) if feature.get("name") in names else 0,
                                             key=ui.key(path))
                        if value != feature.get("name"):
                            ui.set_value(path, value)
                            ui.set_value(f"features.{feature_name}.params", {})
                            st.rerun()
                    elif feature["block"] == "pattern" and param == "params":
                        row = next((item for item in ui.pattern_catalogue() if item["name"] == feature.get("name")
                                    and item["module"] == feature.get("module", "candle")), None)
                        for keyword, default in (row["params"] if row else {}).items():
                            kind_name = ("bool" if isinstance(default, bool) else "int" if isinstance(default, int)
                                         else "float" if isinstance(default, float) else "str")
                            ui.param_input(f"{path}.{keyword}", {"type": kind_name, "default": default},
                                           label=keyword.replace("_", " "))
                    elif declaration["type"] != "dict":
                        ui.param_input(path, declaration)
                if st.button("Delete feature", key=ui.key(f"features.{feature_name}", ":delete")):
                    rerun_after(lambda feature_name=feature_name: features.pop(feature_name))
        st.markdown("**Add a feature**")
        groups = sorted(BLOCKS, key=lambda block: (BLOCKS[block]["group"], BLOCKS[block]["label"]))
        row = st.columns([3, 2, 1])
        block_name = row[0].selectbox("Block", groups, format_func=lambda b: f"{BLOCKS[b]['group']}: {BLOCKS[b]['label']}",
                                      key=ui.key("features", ":new_block"))
        suggestion = f"{block_name}_{sum(1 for f in features.values() if f.get('block') == block_name) + 1}"
        new_name = row[1].text_input("Name", value=suggestion, key=ui.key("features", f":new_name:{block_name}"))
        row[2].write("")
        if row[2].button("Add", key=ui.key("features", ":add")):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new_name) or new_name in PRICE_FIELDS:
                st.error("Use letters, digits and underscores for the name, starting with a letter, "
                         "and not a price name such as close.")
            elif new_name in features:
                st.error(f"There is already a feature called {new_name}.")
            else:
                rerun_after(lambda: features.__setitem__(new_name, block_defaults(block_name)))

    # --- rules or portfolio ---------------------------------------------------------
    if picked == "signals":
        with tabs["Rules"]:
            rules = spec.setdefault("rules", {})
            presets = {
                "ma_cross": ("Moving-average cross", {"fast": dict(block_defaults("ma"), window=50),
                                                     "slow": dict(block_defaults("ma"), window=200)},
                             {"long_entry": "crosses_above(fast, slow)", "long_exit": "crosses_below(fast, slow)"}),
                "rsi": ("RSI reversion", {"rsi_14": dict(block_defaults("rsi"), window=14)},
                        {"long_entry": "crosses_above(rsi_14, 30)", "long_exit": "crosses_above(rsi_14, 70)"}),
                "channel": ("Channel breakout", {"upper": dict(block_defaults("highest"), window=20),
                                                 "lower": dict(block_defaults("lowest"), window=20)},
                            {"long_entry": "close > upper[1]", "long_exit": "close < lower[1]"}),
            }
            row = st.columns([3, 1])
            preset = row[0].selectbox("Preset", [None] + list(presets),
                                      format_func=lambda p: "Choose a preset to replace the rules" if p is None else presets[p][0],
                                      key=ui.key("rules", ":preset"))
            row[1].write("")
            if row[1].button("Apply preset", disabled=preset is None):
                new = ui.spec_copy()
                new.setdefault("features", {}).update(presets[preset][1])
                new["rules"] = {"long_entry": None, "long_exit": None, "short_entry": None, "short_exit": None}
                for slot, text in presets[preset][2].items():
                    new["rules"][slot], _ = expressions.parse(text, new["features"])
                ui.replace_spec(new)
                st.rerun()
            for slot, label in (("long_entry", "Long entry"), ("long_exit", "Long exit"),
                                ("short_entry", "Short entry"), ("short_exit", "Short exit")):
                node = rules.get(slot)
                st.markdown(f"**{label}**")
                enabled = st.toggle(f"Use a {label.lower()} rule", value=node is not None, key=ui.key(f"rules.{slot}", ":on"))
                if enabled and node is None:
                    rerun_after(lambda slot=slot: rules.__setitem__(slot, {"all": [ui.new_leaf("comparison", features)]}))
                if not enabled and node is not None:
                    rerun_after(lambda slot=slot: rules.__setitem__(slot, None))
                if not enabled:
                    continue
                how = st.radio("Edit as", ["Builder", "Expression"], horizontal=True, key=ui.key(f"rules.{slot}", ":how"),
                               label_visibility="collapsed")
                if how == "Builder":
                    ui.rule_editor(f"rules.{slot}", features, timeframe, entry=slot.endswith("entry"))
                else:
                    try:
                        text = expressions.to_text(node, features)
                    except ValueError as error:          # a construct with no text form
                        st.info(str(error))
                        text = None
                    if text is not None:
                        typed = st.text_area("Expression", value=text, key=ui.key(f"rules.{slot}", ":text"),
                                             help="For example: rsi(14) < 30 and close > ma(200). Constants typed here "
                                                  "drop any optimisation range set in the builder.")
                        if st.button("Apply expression", key=ui.key(f"rules.{slot}", ":apply")):
                            try:
                                rule, new_features = expressions.parse(typed, features)
                            except ValueError as error:
                                st.error(str(error))
                            else:
                                new = ui.spec_copy()
                                new["features"].update(new_features)
                                new["rules"][slot] = rule
                                ui.replace_spec(new)
                                st.rerun()
            st.caption("Filters only gate entries. Exits are never filtered, so a position is never trapped open. "
                       "Every signal fills at the next bar's open.")
    else:
        with tabs["Portfolio"]:
            if picked == "hold":
                st.write("Buy and hold has no rules or portfolio settings.")
            else:
                st.markdown("**Selection**")
                use_selection = st.toggle("Pick assets by a score", value=spec.get("selection") is not None,
                                          key=ui.key("selection", ":on"),
                                          help="Without a selection every asset is held.")
                if use_selection and spec.get("selection") is None:
                    first = next(iter(features), None)
                    rerun_after(lambda: spec.__setitem__("selection", {"score": {"feature": first} if first else {"price": "close"},
                                                                       "mode": "top", "top_n": 1, "min_score": None}))
                if not use_selection and spec.get("selection") is not None:
                    rerun_after(lambda: spec.pop("selection"))
                if use_selection:
                    ui.ref_input("selection.score", features, "Rank assets by")
                    ui.param_input("selection.mode", {"type": "choice", "options": SELECTION_MODES, "default": "top"},
                                   label="Keep", optimisable=False)
                    if spec["selection"].get("mode", "top") in ("top", "bottom"):
                        ui.param_input("selection.top_n", {"type": "int", "default": 1, "min": 1, "max": len(symbols)},
                                       label="How many")
                    use_floor = st.checkbox("Require a minimum score", value=spec["selection"].get("min_score") is not None,
                                            key=ui.key("selection.min_score", ":on"))
                    if use_floor:
                        ui.param_input("selection.min_score", {"type": "float", "default": 0.0}, label="Minimum score")
                    elif spec["selection"].get("min_score") is not None:
                        ui.set_value("selection.min_score", None)
                st.markdown("**Weighting**")
                ui.param_input("weighting.scheme", {"type": "choice", "options": WEIGHT_SCHEMES, "default": "equal"},
                               label="Scheme")
                scheme = spec["weighting"]["scheme"]
                scheme = scheme["value"] if isinstance(scheme, dict) else scheme
                if scheme == "inverse_vol":
                    ui.param_input("weighting.vol_window", {"type": "int", "default": 20, "min": 2, "max": 250},
                                   label="Volatility window (bars)")
                if scheme == "fixed":
                    table = spec["weighting"].get("table") or {symbol: 1 / len(symbols) for symbol in symbols}
                    frame = pd.DataFrame({"weight": [table.get(symbol, 0.0) for symbol in symbols]}, index=symbols)
                    edited = st.data_editor(frame, key=ui.key("weighting.table"), width="stretch")
                    new_table = {symbol: float(weight) for symbol, weight in edited["weight"].items()}
                    if new_table != spec["weighting"].get("table"):
                        ui.set_value("weighting.table", new_table)
                    st.caption("Fixed weights are used as given: below 100% the rest stays in cash.")
                st.markdown("**Schedule**")
                schedule = spec["schedule"]
                use_calendar = st.toggle("Rebalance on a calendar", value=schedule.get("calendar") is not None,
                                         key=ui.key("schedule.calendar", ":on"))
                if use_calendar and schedule.get("calendar") is None:
                    ui.set_value("schedule.calendar", {"freq": "M", "on": "first"})
                    st.rerun()
                if not use_calendar and schedule.get("calendar") is not None:
                    ui.set_value("schedule.calendar", None)
                    ui.set_value("schedule.threshold", None)
                    st.rerun()
                if use_calendar:
                    row = st.columns(2)
                    with row[0]:
                        ui.param_input("schedule.calendar.freq", {"type": "choice", "options": ["W", "M", "Q", "Y"],
                                                                  "default": "M"}, label="Every", optimisable=False)
                    with row[1]:
                        ui.param_input("schedule.calendar.on", {"type": "choice", "options": ["first", "last"],
                                                                "default": "first"}, label="On the", optimisable=False)
                    use_threshold = st.checkbox("Also use a drift threshold", value=schedule.get("threshold") is not None,
                                                key=ui.key("schedule.threshold", ":on"),
                                                help="Drift is measured from the last calendar rebalance.")
                    if use_threshold:
                        ui.param_input("schedule.threshold", {"type": "float", "default": 0.05}, label="Tolerance (0.05 = 5%)")
                        ui.param_input("schedule.combine", {"type": "choice", "options": ["and", "or"], "default": "and"},
                                       label="Rebalance when", optimisable=False,
                                       help="and: calendar dates where drift is too large. or: calendar dates, plus any "
                                            "bar where drift is too large.")
                    elif schedule.get("threshold") is not None:
                        ui.set_value("schedule.threshold", None)
                else:
                    st.caption("Without a calendar the portfolio rebalances on every bar.")
                ui.param_input("schedule.min_trade", {"type": "float", "default": 0.005},
                               label="Skip trades smaller than (share of value)", optimisable=False)

    # --- risk ----------------------------------------------------------------------
    with tabs["Risk"]:
        if picked != "signals":
            st.write("Portfolios are sized by their weights, so there is no sizing, stop or target here.")
            risk = spec.setdefault("risk", {})
            budget = st.checkbox("Use a leverage budget", value=bool(risk.get("leverage_budget")),
                                 key=ui.key("risk.leverage_budget"))
            if budget != bool(risk.get("leverage_budget")):
                ui.set_value("risk.leverage_budget", budget)
        else:
            risk = spec.setdefault("risk", {})
            sizing = risk.get("sizing") or {"size": 1}
            used = next((name for name in SIZING_KEYS if name in sizing), "size")
            chosen_sizing = st.radio("Position size", SIZING_KEYS, index=SIZING_KEYS.index(used),
                                     format_func=SIZING_LABELS.get, key=ui.key("risk.sizing", ":kind"))
            if chosen_sizing != used:
                defaults = {"risk_pct": 0.01, "risk_amount": 100.0, "stake": 1.0, "notional": 1000.0, "size": 1.0}
                rerun_after(lambda: risk.__setitem__("sizing", {chosen_sizing: defaults[chosen_sizing]}))
            ui.param_input(f"risk.sizing.{used}", {"type": "float", "default": 1.0}, label="Amount")
            if used == "risk_pct":
                st.caption("Sized from starting cash, not compounding equity.")
            stop = risk.get("stop") or {"kind": "none"}
            stop_kind = st.radio("Stop", ["none", "fixed", "trailing"], index=["none", "fixed", "trailing"].index(stop.get("kind", "none")),
                                 format_func={"none": "No stop", "fixed": "Fixed stop", "trailing": "Trailing stop"}.get,
                                 horizontal=True, key=ui.key("risk.stop", ":kind"),
                                 help="Fixed and trailing stops cannot be combined: a trailing stop replaces the fixed one.")
            if stop_kind != stop.get("kind", "none"):
                rerun_after(lambda: risk.__setitem__("stop", {"kind": stop_kind} if stop_kind == "none"
                                                     else {"kind": stop_kind, "points": stop.get("points", 50)}))
            if stop_kind != "none":
                ui.param_input("risk.stop.points", {"type": "float", "default": 50.0, "min": 1.0, "max": 500.0},
                               label="Distance (points)")
            use_target = st.checkbox("Take profit", value=bool(risk.get("target")), key=ui.key("risk.target", ":on"))
            if use_target and not risk.get("target"):
                rerun_after(lambda: risk.__setitem__("target", {"points": 100}))
            if not use_target and risk.get("target"):
                rerun_after(lambda: risk.__setitem__("target", None))
            if use_target:
                ui.param_input("risk.target.points", {"type": "float", "default": 100.0, "min": 1.0, "max": 1000.0},
                               label="Distance (points)")
            budget = st.checkbox("Use a leverage budget", value=bool(risk.get("leverage_budget")),
                                 key=ui.key("risk.leverage_budget"))
            if budget != bool(risk.get("leverage_budget")):
                ui.set_value("risk.leverage_budget", budget)
            st.markdown("**Extra exits**")
            exits = risk.get("exits") or {}
            extra = [("time_bars", "Close after a number of bars", {"type": "int", "default": 20, "min": 1, "max": 500},
                      "Bars after entry"),
                     ("breakeven_points", "Move the stop to the entry price once in profit",
                      {"type": "float", "default": 50.0, "min": 1.0, "max": 1000.0}, "Points in favour"),
                     ("step_points", "Step the stop up as the trade advances",
                      {"type": "float", "default": 25.0, "min": 1.0, "max": 1000.0}, "Step (points)")]
            for exit_key, label, declaration, amount in extra:
                on = st.checkbox(label, value=exits.get(exit_key) is not None, key=ui.key(f"risk.exits.{exit_key}", ":on"))
                if on and exits.get(exit_key) is None:
                    rerun_after(lambda exit_key=exit_key, declaration=declaration:
                                risk.setdefault("exits", {}).__setitem__(exit_key, declaration["default"]))
                if not on and exits.get(exit_key) is not None:
                    rerun_after(lambda exit_key=exit_key: risk["exits"].pop(exit_key))
                if on:
                    ui.param_input(f"risk.exits.{exit_key}", declaration, label=amount)
            if any(exits.get(name) is not None for name, *_ in extra) and stop_kind != "fixed":
                st.warning("The extra exits adjust a stop, so they need a fixed stop.")
            rule = engine.SAME_BAR_STOP_RULE
            st.caption("Orders fill at the next bar's open. Stops and targets are measured from that open, and a "
                       "stopped-out trade pays the spread on exit like any other. A bar that opens beyond the "
                       "stop fills at its open. "
                       + (f"When one bar's range holds both the stop and the target, the {rule} is taken."
                          if rule else "Which of stop and target wins inside one bar has not been confirmed yet."))
            st.caption("Points are price distances divided by the instrument's point size. Time stops and breakeven "
                       "are not available yet.")

    # --- account -------------------------------------------------------------------
    with tabs["Account"]:
        account = spec.setdefault("account", {})
        ui.param_input("account.init_cash", {"type": "float", "default": 10000.0}, label="Starting cash", optimisable=False)
        freqs = sorted(set(FREQ.values()) | {account.get("freq", "D")})
        ui.param_input("account.freq", {"type": "choice", "options": freqs, "default": FREQ[timeframe]},
                       label="Bar frequency", optimisable=False, help="Annualised metrics need this.")
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
            rerun_after(lambda: account.__setitem__("rates", None if source == "none" else
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
            import os
            st.caption("FRED key found." if os.environ.get("FRED_API_KEY") else
                       "No FRED key yet: set FRED_API_KEY or add it on the Settings page.")
        st.caption("Carry on an FX instrument needs rates for both of its currencies, and the instrument must name both.")
        # Nothing is written here unless changed: a blank choice keeps vectorbt's default.
        extra = spec.setdefault("engine_extra", {})
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
        seed_text = st.text_input("Random seed (optional)", value="" if extra.get("seed") is None else str(extra["seed"]),
                                  key=ui.key("engine_extra.seed"))
        seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
        if seed != extra.get("seed"):
            if seed is None:
                ui.delete_key("engine_extra.seed")
            else:
                ui.set_value("engine_extra.seed", seed)

    # --- pipeline ------------------------------------------------------------------
    with tabs["Pipeline"]:
        st.caption("How the strategy flows from data to the backtest. Alternatives are drawn as currently chosen.")
        if validate_spec(spec):
            st.write("Fix the problems listed beside the form to draw the pipeline.")
        else:
            dot = ui.run_guarded("Drawing the pipeline", lambda: analysis.pipeline_dot(spec))
            if dot:
                st.graphviz_chart(dot)

    # --- spec and code -------------------------------------------------------------
    with tabs["Spec and code"]:
        # Keyed on the spec's hash: the box re-seeds whenever another tab changes the
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
        if not validate_spec(spec):
            script = export_script(spec, data_dir)
            st.markdown("**The same backtest as a plain vectorbtX script** (read-only)")
            st.code(script, language="python")
            st.download_button("Download script", script, file_name="strategy.py", mime="text/x-python")


# --- live preview --------------------------------------------------------------------
@st.cache_data(max_entries=8, show_spinner=False)
def compute_preview(spec_json, data_dir_text, mtimes):
    preview_spec = json.loads(spec_json)
    frames = ui.load_frames_for(preview_spec, data_dir_text)
    applied = apply_params(preview_spec)
    computed = compile_features(applied.get("features") or {}, frames)
    if strategy_kind(applied) == "signals":
        # The compiler's signals are on the bar they fire, which is what the preview marks.
        return frames, computed, compile_signals(applied, frames)
    return frames, computed, compile_weights(applied, frames)


with preview:
    st.subheader("Live preview")
    problems = validate_spec(spec)
    if problems:
        ui.show_problems(problems)
    elif picked == "hold":
        st.write("Buy and hold: nothing to preview. Run it on the Backtest page.")
    elif ui.missing_data(spec, data_dir):
        st.info(f"Import {', '.join(ui.missing_data(spec, data_dir))} at {timeframe} (or finer) on the Data page "
                "to see the preview.")
    else:
        section = {key: spec.get(key) for key in ("version", "name", "data", "instruments", "features", "rules",
                                                  "selection", "weighting", "schedule", "account") if key in spec}
        result = ui.run_guarded("Computing signals", lambda: compute_preview(
            json.dumps(section, sort_keys=True), str(data_dir), ui.data_mtimes(spec, data_dir)))
        if result is not None:
            frames, computed, compiled = result
            theme = settings.get("default_theme", "plotly_white")
            if picked == "signals":
                counts = {name: int(compiled[name].to_numpy().sum()) for name in ("entries", "exits", "short_entries", "short_exits")}
                labels = {"entries": "Long entries", "exits": "Long exits", "short_entries": "Short entries",
                          "short_exits": "Short exits"}
                columns = st.columns(4)
                for column, (name, count) in zip(columns, counts.items()):
                    column.metric(labels[name], count)
                rules = spec.get("rules") or {}
                for slot, name in (("long_entry", "entries"), ("long_exit", "exits"), ("short_entry", "short_entries"),
                                   ("short_exit", "short_exits")):
                    if rules.get(slot) and counts[name] == 0:
                        st.warning(f"The {slot.replace('_', ' ')} rule never fires on this data.")
                for entry, exit_ in (("entries", "exits"), ("short_entries", "short_exits")):
                    entries = compiled[entry].to_numpy()
                    if entries.sum():
                        soon = compiled[exit_].to_numpy() | np.roll(compiled[exit_].to_numpy(), -1, axis=0)
                        if (entries & soon).sum() / entries.sum() > 0.5:
                            st.warning(f"Most {labels[entry].lower()} are followed by an exit on the same or the next bar.")
                for feature_name, outputs in computed.items():
                    if "signal" in outputs and not outputs["signal"].to_numpy().any():
                        st.warning(f"Pattern {feature_name} never appears on this data.")
                symbol = st.selectbox("Symbol", list(frames["close"].columns), key="preview:symbol")
                overlays = st.multiselect("Overlay", [f"{name}.{output}" for name, outputs in computed.items() for output in outputs],
                                          key="preview:overlays")
                ui.chart(analysis.fig_preview(frames, computed, compiled, symbol, theme, overlays))
            else:
                ui.chart(analysis.fig_weight_preview(compiled["weights"], compiled["schedule"], theme))
                st.caption(f"{int(compiled['schedule'].sum())} rebalance dates.")
