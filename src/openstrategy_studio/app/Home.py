"""Home: open or create a strategy, restore the last draft, see recent runs and jobs."""

import json
import re

import streamlit as st

from openstrategy_studio import jobs, store, ui
from openstrategy_studio.spec import (list_templates, load_spec, load_template, strategy_kind,
                                      validate_spec)

settings, data_dir = ui.boot("Home")

st.title("OpenStrategy Studio")
st.write("Build a trading strategy from forms and rules, backtest it, optimise it and read the tear sheet. "
         "Everything runs on your machine with vectorbtX.")

# Session state vanishes on a browser refresh; every edit is also written to a draft.
draft_path = data_dir / "drafts" / "current.json"
if draft_path.exists() and not st.session_state.get("draft_offered"):
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft != ui.current():
        with st.container(border=True):
            st.write(f"You have unsaved work from last time: **{draft.get('name', 'untitled')}**.")
            restore, discard = st.columns(2)
            if restore.button("Restore it", type="primary"):
                ui.replace_spec(draft)
                st.session_state["draft_offered"] = True
                st.rerun()
            if discard.button("Start fresh"):
                st.session_state["draft_offered"] = True
                st.rerun()

left, right = st.columns(2)
with left:
    st.subheader("Start a strategy")
    templates = list_templates()
    chosen = st.selectbox("From a template", list(templates), format_func=templates.get)
    if st.button("Use this template"):
        ui.replace_spec(load_template(chosen))
        st.success(f"Loaded “{templates[chosen]}”. Edit it on the Build page.")

    saved = sorted((data_dir / "strategies").glob("*.json"))
    if saved:
        picked = st.selectbox("A saved strategy", saved, format_func=lambda path: path.stem)
        if st.button("Open saved strategy"):
            spec = load_spec(picked)
            ui.replace_spec(spec)
            ui.show_problems(validate_spec(spec), "This file has problems")
            st.success(f"Opened {picked.stem}.")

    upload = st.file_uploader("Or open a strategy file (.json)", type=["json"])
    if upload is not None and st.button("Open uploaded file"):
        spec = json.loads(upload.getvalue().decode("utf-8"))
        problems = validate_spec(spec)
        ui.replace_spec(spec)
        if ui.has_python_block(spec):
            # Imported specs with Python blocks warn before running.
            st.session_state["untrusted_python"] = True
            st.warning("This file contains Python blocks, which run as code on your machine. Read them on the "
                       "Build page before running; the Backtest page will ask you to confirm.")
        ui.show_problems(problems, "This file has problems")
        if not problems:
            st.success(f"Opened {spec['name']}.")

with right:
    st.subheader("Current strategy")
    spec = ui.current()
    kinds = {"signals": "Entry and exit rules", "portfolio": "Portfolio (weights and schedule)",
             "hold": "Buy and hold", "conflict": "Rules and a portfolio (pick one)"}
    st.write(f"**{spec.get('name', 'Untitled')}**")
    st.write(f"{kinds[strategy_kind(spec)]}, {', '.join(spec['data']['symbols'])} on "
             f"{spec['data']['timeframe']} bars, {spec['data']['start']} to {spec['data']['end']}.")
    problems = validate_spec(spec)
    if problems:
        ui.show_problems(problems)
    else:
        st.success("Ready to backtest.")
    name = st.text_input("Save as", value=re.sub(r"[^A-Za-z0-9_-]+", "_", spec.get("name", "strategy")).strip("_"))
    if st.button("Save strategy"):
        from openstrategy_studio.spec import save_spec
        save_spec(spec, data_dir / "strategies" / f"{name}.json")
        st.success(f"Saved {name}.json.")

st.divider()
runs_column, jobs_column = st.columns(2)
with runs_column:
    st.subheader("Recent runs")
    runs = store.list_runs(data_dir)
    if runs.empty:
        st.write("No saved runs yet. Run a backtest and save it to see it here.")
    else:
        rows = []
        for _, run in runs.head(10).iterrows():
            summary = json.loads(run["summary"] or "{}")
            rows.append({"Run": run["name"], "When": run["created_at"], **summary})
        ui.table(rows)
with jobs_column:
    st.subheader("Recent optimisations")
    listing = jobs.list_jobs(data_dir)
    if listing.empty:
        st.write("No optimisations yet.")
    else:
        ui.table(listing.head(10)[["name", "backend", "status", "done", "budget", "best_value"]])
