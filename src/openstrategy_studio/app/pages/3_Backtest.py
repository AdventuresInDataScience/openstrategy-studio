"""Backtest: run the current strategy, read the tear sheet, save the run, reopen past runs."""

import uuid

import streamlit as st

from openstrategy_studio import analysis, data as data_module, engine, store, ui
from openstrategy_studio.analysis import FIGURES
from openstrategy_studio.spec import validate_spec

settings, data_dir = ui.boot("Backtest")
ui.font_css()
spec = ui.current()
st.title("Backtest")
st.write(f"**{spec['name']}** on {', '.join(spec['data']['symbols'])}, {spec['data']['timeframe']} bars, "
         f"{spec['data']['start']} to {spec['data']['end']}.")

problems = validate_spec(spec)
missing = ui.missing_data(spec, data_dir)
if missing:
    problems = problems + [f"Import {', '.join(missing)} at {spec['data']['timeframe']} (or finer) on the Data page."]
ui.show_problems(problems)
trusted = True
if ui.has_python_block(spec) and st.session_state.get("untrusted_python"):
    trusted = st.checkbox("This strategy came from a file and contains Python blocks. I have read them on the "
                          "Build page and trust them to run on this machine.")

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

    outcome = ui.run_guarded("Running the backtest. The first run in a fresh environment compiles numba code, "
                             "which can take a minute; later runs are fast.", run_all)
    if outcome is not None:
        st.session_state["last_run"] = outcome
        st.session_state["figures"] = {}

with st.expander("Open a saved run"):
    runs = store.list_runs(data_dir)
    if runs.empty:
        st.write("No saved runs yet.")
    else:
        labels = {row.run_id: f"{row.name} ({row.created_at})" for row in runs.itertuples()}
        run_id = st.selectbox("Run", list(labels), format_func=labels.get)
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
    st.success("Saved. It is listed on Home and can be reopened here.")

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

# --- stats ----------------------------------------------------------------------------
tags = {"all": "All metrics", "mt5": "MT5 report", "vectorbtx": "vectorbtX extras"}
tag = st.radio("Metrics", list(tags), format_func=tags.get, horizontal=True)
columns = st.columns(3)
for column, (table_name, label) in zip(columns, (("portfolio", "Portfolio"), ("trades", "Trades"), ("returns", "Returns"))):
    with column:
        st.markdown(f"**{label}**")
        ui.table(analysis.stats_frame(results, table_name, tag), height=420)
st.caption("Z-score reads the other way round from most metrics: a negative value means wins and losses cluster "
           "in streaks. It is a dependency test, not a quality score.")
if "per_symbol" in results:
    st.markdown("**Each symbol**")
    ui.table(results["per_symbol"], hide_index=False)
    if accounts.get("model") != "separate":
        st.caption("A symbol in a shared account has no equity of its own, so only what it traded is shown.")

# --- figures --------------------------------------------------------------------------
st.divider()
names = analysis.available_figures(results)
row = st.columns([2, 3])
name = row[0].selectbox("View", names, format_func=lambda key: FIGURES[key]["label"])
with row[1]:
    theme, glow = ui.theme_picker(settings, "backtest")
options = {}
if name == "price":
    symbols = results["meta"]["symbols"]
    choices = sorted(results["features"]["feature"].unique()) if "features" in results else []
    left, right = st.columns(2)
    options["symbol"] = left.selectbox("Symbol", symbols)
    options["overlays"] = tuple(right.multiselect("Overlay features (pattern hits show as diamonds)", choices))
elif name == "rolling":
    options["window"] = st.slider("Window (bars)", 10, 504, 63)
elif name == "monte_carlo":
    options["paths"] = st.select_slider("Paths", [100, 250, 500, 1000, 2500], value=500)

# Built only when selected, and cached on (run, view, theme, options).
cache = st.session_state.setdefault("figures", {})
cache_key = (last["id"], name, theme, glow, tuple(sorted(options.items())))
if cache_key not in cache:
    cache[cache_key] = FIGURES[name]["fn"](results, theme, glow and FIGURES[name]["lines"], **options)
figure = cache[cache_key]
for item in figure if isinstance(figure, tuple) else (figure,):
    ui.chart(item)

with st.expander("Trades"):
    ui.table(results["trades"])
with st.expander("Orders"):
    ui.table(results["orders"])
