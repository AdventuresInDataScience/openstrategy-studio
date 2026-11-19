"""Backtest results: run the current strategy, read the tear sheet, save the run, reopen and compare past runs.

The chart comes first and every metric under it (the owner's call, 19 November 2026): the
equity curve is the first sign of whether the rest is worth reading.
"""

import uuid

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, data as data_module, engine, store, ui
from openstrategy_studio.analysis import FIGURES
from openstrategy_studio.spec import validate_spec

settings, data_dir = ui.boot("Backtest results")
ui.page_help("backtest")
ui.font_css()
spec = ui.current()
st.title("Backtest results")
symbols_text = ", ".join(spec["data"]["symbols"]) or "no symbols yet"
st.write(f"**{spec['name']}** on {symbols_text}, {spec['data']['timeframe']} bars, "
         f"{spec['data']['start']} to {spec['data']['end']}.")

problems = validate_spec(spec)
missing = ui.missing_data(spec, data_dir)
if missing:
    problems = problems + [f"Import {', '.join(missing)} at {spec['data']['timeframe']} (or finer) on the Data page."]
ui.show_problems(problems)
trusted = True
if ui.has_python_block(spec) and st.session_state.get("untrusted_python"):
    trusted = st.checkbox("This strategy came from a file and contains Python code. I have read it on the "
                          "Build page and trust them to run on this machine.",
                          help="Python features run with your user's full access to this machine. Read the code on "
                               "the Build page's Features tab before ticking this.")

row = st.columns([1, 2, 2])
compare = row[1].checkbox("Compare with buy and hold", value=True,
                          help="The market: the same instruments held from the first bar without leverage, "
                               "equally weighted, before costs, in the same currency. Beta, alpha and the other "
                               "relative statistics are measured against it either way.")
if row[0].button("Run backtest", type="primary", disabled=bool(problems) or not trusted):
    def run_all():
        frames = ui.load_frames_for(spec, data_dir)
        keywords = {"data_dir": data_dir, "offline": bool(settings.get("offline")),
                    "static_rate": settings["vectorbtx"].get("static_rate")}
        run = engine.run_backtest(spec, frames, **keywords)
        results = analysis.extract_results(run, benchmark=compare)
        return {"results": results, "spec": run["spec"], "fingerprint": data_module.fingerprint(data_dir, spec, frames),
                "saved": None, "id": uuid.uuid4().hex}

    # The start-up warm-up (ui.boot) compiles numba's code in the background; a run that comes
    # before it has finished waits for it, and says so.
    label = ("Running the backtest. The Studio is still preparing its backtest engine in the background (once per "
             "start), so this first run waits for that." if ui.warming_up() else "Running the backtest.")
    outcome = ui.run_guarded(label, run_all)
    if ui.WARM_UP.get("error"):
        st.caption(f"The start-up warm-up stopped ({ui.WARM_UP['error']}), so this run compiled its own code. "
                   "Nothing else is affected.")
    if outcome is not None:
        st.session_state["last_run"] = outcome
        st.session_state["figures"] = {}

runs = store.list_runs(data_dir)
labels = {row.run_id: f"{row.name} ({row.created_at})" for row in runs.itertuples()} if not runs.empty else {}
with st.expander("Open a saved run"):
    if runs.empty:
        st.write("No saved runs yet.")
    else:
        run_id = st.selectbox("Run", list(labels), format_func=labels.get,
                              help="A saved run, newest first. Opening it shows its results as they were saved, "
                                   "without running again.")
        open_column, delete_column = st.columns(2)
        if open_column.button("Open run"):
            path = runs.set_index("run_id").loc[run_id, "path"]
            loaded = analysis.load_results(path)
            st.session_state["last_run"] = {"results": loaded, "spec": loaded.get("spec"),
                                            "fingerprint": loaded.get("fingerprint"), "saved": path, "id": run_id}
            st.session_state["figures"] = {}
            st.rerun()
        if delete_column.button("Remove from the list"):
            store.delete_run(data_dir, run_id)
            st.rerun()

# Saved backtests side by side (on the old Results page until 19 November 2026, which now holds
# the optimiser's results only).
with st.expander("Compare saved runs"):
    if runs.empty:
        st.write("No saved runs yet. Save a run below to compare it with others.")
    else:
        # None picked at first: a collapsed expander still runs, and two saved runs would be read
        # from disk on every click on this page.
        chosen = st.multiselect("Runs", list(labels), format_func=labels.get, placeholder="Pick two or more runs",
                                help="Saved runs to draw on one chart and list side by side.")
        if chosen:
            compare_theme, compare_glow = ui.theme_picker(settings, "compare")
            curves = []
            table = {}
            for run_id in chosen:
                saved = analysis.load_results(runs.set_index("run_id").loc[run_id, "path"])
                curves.append((labels[run_id], saved["series"]["value"]))
                table[labels[run_id]] = saved["stats"]["portfolio"]["all"]
            ui.chart(analysis.fig_compare(curves, compare_theme, compare_glow))
            ui.table(pd.DataFrame(table).astype(str), hide_index=False)

last = st.session_state.get("last_run")
if last is None:
    st.stop()

results = last["results"]
st.divider()
title, save = st.columns([4, 1])
title.subheader(results["meta"]["name"])
if last["saved"]:
    save.caption("Saved")
elif save.button("Save run"):
    last["saved"] = analysis.save_run(data_dir, results, last["spec"], settings, last["fingerprint"])
    st.success("Saved. It is listed on Home and can be reopened or compared here.")

# How the run was funded. Runs saved before the meta recorded it say nothing.
accounts = results["meta"].get("accounts") or {}
if accounts.get("model") == "separate":
    st.info(f"{accounts['count']} separate accounts, one per symbol, each starting with {accounts['each']:,.0f} "
            f"({accounts['count'] * accounts['each']:,.0f} in all). The tables and charts treat them as one book; "
            "the table under them shows each account on its own."
            + (" To trade every symbol from one account, make the account shared on the Build page's Account tab."
               if results["meta"]["kind"] == "signals" else ""))
elif accounts.get("model") == "shared" and results["meta"]["kind"] == "signals":
    st.caption(f"One account of {accounts['each']:,.0f}, shared by every symbol.")

# What vectorbtX could not price, or priced on an assumption. Runs saved before the meta
# recorded these say nothing.
notes = results["meta"].get("warnings") or []
if notes:
    lines = []
    for note, hint in analysis.warning_hints(notes):
        lines.append(f"- {note}" + (f"\n\n  **{hint}**" if hint else ""))
    st.warning("vectorbtX flagged this run. The run finished, but its numbers may leave something out:\n\n"
               + "\n".join(lines))

# --- the chart first --------------------------------------------------------------------
names = analysis.available_figures(results)
row = st.columns([2, 3])
name = row[0].selectbox("View", names, format_func=lambda key: FIGURES[key]["label"],
                        help="Which chart to draw from this run. It opens on the equity curve against buy and hold.")
with row[1]:
    theme, glow = ui.theme_picker(settings, "backtest")
options = {}
if name == "price":
    symbols = results["meta"]["symbols"]
    choices = sorted(results["features"]["feature"].unique()) if "features" in results else []
    left, right = st.columns(2)
    options["symbol"] = left.selectbox("Symbol", symbols, help="Which symbol's prices and trades to show.")
    options["overlays"] = tuple(right.multiselect("Overlay features (pattern hits show as diamonds)", choices,
                                                  help="Feature lines to draw over the price."))
elif name == "rolling":
    options["window"] = st.slider("Window (bars)", 10, 504, 63,
                                  help="How many bars each rolling figure is measured over (63 is about a "
                                       "quarter of daily bars).")
elif name == "monte_carlo":
    options["paths"] = st.select_slider("Paths", [100, 250, 500, 1000, 2500], value=500,
                                        help="How many times the trades are put in a random order to draw the "
                                             "bands. More gives steadier bands and takes longer.")

# Built only when selected, and cached on (run, view, theme, options).
cache = st.session_state.setdefault("figures", {})
cache_key = (last["id"], name, theme, glow, tuple(sorted(options.items())))
if cache_key not in cache:
    cache[cache_key] = FIGURES[name]["fn"](results, theme, glow and FIGURES[name]["lines"], **options)
figure = cache[cache_key]
for item in figure if isinstance(figure, tuple) else (figure,):
    ui.chart(item)

# --- then every metric --------------------------------------------------------------------
st.divider()
columns = st.columns(3)
for column, (table_name, label) in zip(columns, (("portfolio", "Portfolio"), ("trades", "Trades"), ("returns", "Returns"))):
    with column:
        st.markdown(f"**{label}**")
        ui.table(analysis.stats_frame(results, table_name), height=420)
st.caption("Z-score reads the other way round from most metrics: a negative value means wins and losses cluster "
           "in streaks. It is a dependency test, not a quality score.")
if "per_symbol" in results:
    st.markdown("**Each symbol**")
    ui.table(results["per_symbol"], hide_index=False)
    if accounts.get("model") != "separate":
        st.caption("A symbol in a shared account has no equity of its own, so only what it traded is shown.")

with st.expander("Trades"):
    ui.table(results["trades"])
with st.expander("Orders"):
    ui.table(results["orders"])
