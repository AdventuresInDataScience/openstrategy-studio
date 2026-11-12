"""Patterns: which of the 300 detectors were followed by a move, on your data?

Each hit is judged as a trade from the next bar's open (where a signal fills),
bearish signals as shorts, against the same trades taken on every bar.
"""

from datetime import date

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, data as data_module, research, ui
from openstrategy_studio.spec import TIMEFRAMES, strategy_kind

settings, data_dir = ui.boot("Patterns")
ui.font_css()
st.title("Pattern research")
st.write("See how each candlestick and chart pattern was followed on your data, before building a rule on one.")

stored = ui.stored_symbols(data_dir)
if not stored:
    st.info("Import some data on the Data page first.")
    st.stop()

spec = ui.current()
row = st.columns([3, 1, 1, 1])
symbols = row[0].multiselect("Symbols", stored, default=[s for s in spec["data"]["symbols"] if s in stored][:5] or stored[:1])
timeframe = row[1].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(spec["data"]["timeframe"]))
span = {"min_value": date(1970, 1, 1), "max_value": date(2100, 12, 31)}
start = row[2].date_input("From", value=date.fromisoformat(str(spec["data"]["start"])[:10]), **span)
end = row[3].date_input("To", value=date.fromisoformat(str(spec["data"]["end"])[:10]), **span)

row = st.columns(4)
families = row[0].multiselect("Families", ["candle", "chart"], default=["candle", "chart"],
                              format_func={"candle": "Candlesticks", "chart": "Chart patterns"}.get)
directions = row[1].multiselect("Directions", ["bullish", "bearish", "directional", "bidirectional", "non_directional"],
                                default=["bullish", "bearish", "directional", "bidirectional"],
                                format_func=lambda d: d.replace("_", "-"))
horizons = row[2].multiselect("Horizons (bars)", [1, 3, 5, 10, 20, 40], default=[1, 5, 10, 20])
min_hits = row[3].number_input("Ignore patterns with fewer hits than", min_value=2, value=20, step=5)
skip_slow = st.checkbox("Skip the slow head-and-shoulders family", value=True,
                        help="Eight chart detectors scale with the square of the pivot count; they are slow on long histories.")

if st.button("Run the study", type="primary", disabled=not symbols or not horizons):
    def run_study():
        frames = data_module.load_frames(data_dir, symbols, timeframe, start.isoformat(), end.isoformat())
        catalogue = [item for item in ui.pattern_catalogue() if item["module"] in families
                     and item["direction"] in directions and not (skip_slow and research.is_slow(item["name"]))]
        return frames, research.study(frames, catalogue, horizons=tuple(sorted(horizons)), min_hits=int(min_hits))

    outcome = ui.run_guarded("Scanning every pattern (all 300 take well under a second per symbol)", run_study)
    if outcome is not None:
        st.session_state["pattern_study"] = {"frames": outcome[0], "table": outcome[1],
                                             "horizons": sorted(horizons), "symbols": symbols}

found = st.session_state.get("pattern_study")
if not found:
    st.stop()

table = found["table"]
horizon = st.selectbox("Rank by horizon", found["horizons"], index=min(2, len(found["horizons"]) - 1))
column = f"excess {horizon}"
if column not in table:
    st.write("No pattern had enough hits. Lower the minimum, widen the dates or add symbols.")
    st.stop()
ranked = table.dropna(subset=[column]).sort_values(column, ascending=False)
st.caption(f"{len(ranked)} patterns with at least {int(min_hits)} hits on {', '.join(found['symbols'])}. "
           "Excess is the mean return after the pattern minus the same trades taken on every bar. Hits closer "
           "together than the horizon overlap, so t is optimistic; with this many patterns some will look good by "
           "chance. Confirm a candidate with the optimiser's held-back test.")
shown = ranked.copy()
for name in shown.columns:
    if name.startswith(("mean", "excess")):
        shown[name] = shown[name].map(lambda value: f"{value:.3%}" if pd.notna(value) else "")
    elif name.startswith("hit rate"):
        shown[name] = shown[name].map(lambda value: f"{value:.0%}" if pd.notna(value) else "")
    elif name.startswith("t "):
        shown[name] = shown[name].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
ui.table(shown, height=360)

labels = [f"{row.pattern} ({row.module})" for row in ranked.itertuples()]
if labels:
    picked = st.selectbox("Look closer at", range(len(labels)), format_func=labels.__getitem__)
    chosen = ranked.iloc[picked]
    theme, glow = ui.theme_picker(settings, "patterns")
    path = research.event_path(found["frames"], chosen["module"], chosen["pattern"], horizon=max(found["horizons"]))
    ui.chart(analysis.fig_event_path(path, chosen["pattern"].replace("_", " "), theme, glow))

    st.markdown("**Use it in the current strategy**")
    row = st.columns([2, 2, 1])
    slot = row[0].selectbox("Add to", ["long_entry", "short_entry", "long_exit", "short_exit"],
                            format_func=lambda s: s.replace("_", " ").capitalize())
    side = row[1].selectbox("Side", ["bullish", "bearish", "either"],
                            index=0 if chosen["bullish"] >= chosen["bearish"] else 1)
    row[2].write("")
    portfolio = strategy_kind(spec) == "portfolio"
    if portfolio:
        st.caption("The current strategy is a portfolio; pattern rules belong to entry and exit rule strategies.")
    if row[2].button("Add rule", disabled=portfolio):
        updated = research.add_pattern_to_rule(ui.spec_copy(), slot, chosen["module"], chosen["pattern"], side)
        ui.replace_spec(updated)
        st.success(f"Added {chosen['pattern']} ({side}) to the {slot.replace('_', ' ')} rule. Open the Build page "
                   "to see it in the preview.")
