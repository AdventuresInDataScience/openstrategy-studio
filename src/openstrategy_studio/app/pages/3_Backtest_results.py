"""Backtest results: run the current strategy, read the tear sheet, save the run, reopen and compare past runs.

The chart comes first and every metric under it (the owner's call, 19 November 2026): the
equity curve is the first sign of whether the rest is worth reading.
"""

import uuid

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, data as data_module, engine, store, ui
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
            log = st.checkbox("Log scale", key=ui.kept("compare:log", False), help=ui.LOG_SCALE_HELP)
            curves = []
            table = {}
            for run_id in chosen:
                saved = analysis.load_results(runs.set_index("run_id").loc[run_id, "path"])
                curves.append((labels[run_id], saved["series"]["value"]))
                table[labels[run_id]] = saved["stats"]["portfolio"]["all"]
            figure = analysis.fig_compare(curves, compare_theme, compare_glow)
            ui.chart(analysis.log_scale(figure) if log else figure)
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

# The tear sheet: the chart first, then every metric (ui.tear_sheet, which the Optimiser results page
# draws for each part of a best trial too).
ui.tear_sheet(results, "backtest", last["id"], settings=settings)
