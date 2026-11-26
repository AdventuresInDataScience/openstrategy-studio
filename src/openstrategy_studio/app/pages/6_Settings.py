"""Settings: the look, data folder, workers, offline mode, vectorbtX settings and the FRED key."""

import copy
import os

import pandas as pd
import streamlit as st

from openstrategy_studio import config, engine, style, ui

settings, data_dir = ui.boot("Settings")
ui.page_help("settings")
st.title("Settings")
st.caption(f"Stored in {config.settings_path()}")

draft = copy.deepcopy(settings)

# --- the look ------------------------------------------------------------------------
st.subheader("Look")
looks, broken = style.catalogue()
showing = style.active_theme(settings)[0]["name"]
names = list(looks)
chosen = settings.get("page_theme") if settings.get("page_theme") in looks else style.DEFAULT_THEME
picker, preview = st.columns([2, 3], gap="large")
with picker:
    draft["page_theme"] = st.selectbox("Page theme", names, index=names.index(chosen),
                                       format_func=lambda name: looks[name]["studio"]["label"],
                                       help="The colours, fonts and corners of every page. It applies when the "
                                            "Studio next starts.")
    chart_themes, chart_labels = ui.chart_theme_options()
    draft["default_theme"] = st.selectbox("Default chart theme", chart_themes, format_func=chart_labels.get,
                                          index=chart_themes.index(settings["default_theme"])
                                          if settings["default_theme"] in chart_themes else 0,
                                          help="Match the page theme follows the page's colours. The Backtest, "
                                               "Results and Patterns pages can switch per chart.")
with preview:
    st.html(style.preview_html(looks[draft["page_theme"]]))
if draft["page_theme"] != showing:
    st.info("Save, then restart the Studio to see this page theme: close its window, or stop it with Ctrl+C and "
            "start it again.")
for problem in broken:
    st.warning(f"Skipped a theme file. {problem}")
copy_column, where_column = st.columns([1, 3])
if copy_column.button("Make an editable copy",
                      help="Copies the theme above into your themes folder under a new name, to edit."):
    copied = style.copy_theme(draft["page_theme"])
    st.success(f"Copied to {copied}. Edit its colours, then pick it here and restart.")
where_column.caption(f"Your own themes live in {style.user_themes_dir()}. Each is one .toml file; the comments in "
                     "a copy of Daylight explain every setting.")

st.subheader("Studio")
draft["data_dir"] = st.text_input("Data folder", value=settings.get("data_dir") or "", placeholder=str(data_dir),
                                  help="Every price file, strategy, run and job lives here. Blank uses the default.")
row = st.columns(3)
draft["workers"] = int(row[0].number_input("Worker processes for optimisation", min_value=1,
                                           value=int(settings.get("workers", 1)), step=1,
                                           help="The number a new optimisation starts with; the Optimise page can "
                                                "change it per job. Up to your CPU's core count is faster."))
draft["spike_multiple"] = float(row[1].number_input("Flag moves this many times the usual", min_value=2.0,
                                                    value=float(settings.get("spike_multiple", 10.0)), step=1.0,
                                                    help="The data check on import: a bar whose move from the "
                                                         "previous close, or whose high-low range, is more than this "
                                                         "many times the usual one over the 100 bars before it is "
                                                         "listed for review, as a likely bad price. Nothing is "
                                                         "changed."))
draft["offline"] = row[2].toggle("Offline", value=bool(settings.get("offline")),
                                 help="Turns off every download. Stored data still works; FRED falls back to its cache, "
                                      "then to the static rate below.")
draft["warm_up"] = st.toggle("Prepare the backtest engine at start", value=bool(settings.get("warm_up", True)),
                             help="The first backtest after a start compiles numba code, which can take a minute. "
                                  "With this on, the Studio does that in the background as soon as it starts, by "
                                  "running every template once on made-up prices, so your first run does not wait. "
                                  "It touches none of your data. Applies at the next start.")

st.subheader("FRED rates")
if os.environ.get("FRED_API_KEY") and not settings.get("fred_api_key"):
    st.write("Using the FRED_API_KEY environment variable.")
draft["fred_api_key"] = st.text_input("FRED API key", value=settings.get("fred_api_key") or "", type="password",
                                      help="Interest-rate history for holding costs, when a strategy's Account tab "
                                           "picks FRED. The key is free from the St. Louis Fed "
                                           "(fred.stlouisfed.org/docs/api/api_key.html). A key here is used instead "
                                           "of the FRED_API_KEY environment variable or a .env file, from the moment "
                                           "you save; clearing it takes effect at the next start.")
if draft["fred_api_key"]:
    st.warning("The key will be stored in settings.toml. Keep that file out of version control.")

st.subheader("vectorbtX settings")
st.caption("Written to vbt.settings[\"vectorbtx\"]. Blank keeps vectorbtX's own default.")
vectorbtx_values = dict(settings["vectorbtx"])
roundings = [None, "floor", "nearest", "ceil"]
rounding = st.selectbox("Size rounding", roundings,
                        index=roundings.index(vectorbtx_values.get("rounding")) if vectorbtx_values.get("rounding") in roundings else 0,
                        format_func=lambda value: "Default" if value is None else value,
                        help="How position sizes are rounded to an instrument's size step, where the instrument "
                             "sets no rounding of its own: floor (down, vectorbtX's default), nearest or ceil.")
vectorbtx_values["rounding"] = rounding
numbers = ["financing_markup", "carry_markup", "fx_markup", "slippage_max", "slippage_coef", "vol_window",
           "static_rate", "triangulation_tolerance"]
texts = ["triple_day", "end_of_day", "day_count", "fill_policy"]
# What each does, shown in the table (a cell cannot carry an ⓘ of its own). From vectorbtX's
# docs/API.md (the settings table) and its code; defaults are vectorbtX's own.
meanings = {
    "financing_markup": "The broker's yearly markup over the benchmark rate on financing, always against you "
                        "(0.03 = 3%). Default 0.03.",
    "carry_markup": "The broker's yearly fee on FX carry (the tom-next rate), always against you. Default 0.01.",
    "fx_markup": "The broker's worse-than-market currency conversion, charged as slippage on each fill of a "
                 "converted instrument (0.005 = 0.5%). Default 0.005.",
    "slippage_max": "The ceiling on automatic slippage, as a fraction of price. Default 0.005.",
    "slippage_coef": "The multiplier in automatic slippage: coef x volatility x square root of (size / average "
                     "volume). Default 1.",
    "vol_window": "Bars of returns behind the volatility in automatic slippage. Default 20.",
    "static_rate": "One yearly rate for every currency when FRED cannot be reached (0.05 = 5%). Default: none.",
    "triangulation_tolerance": "How far an FX rate may stray from the one implied through other pairs before the "
                               "FX table is rejected. Default 0.0001.",
    "triple_day": "The weekday charged three nights of holding costs, for the weekend: WED or FRI, say. "
                  "Default WED.",
    "end_of_day": "The broker's daily cut-off for holding costs, HH:MM. No effect on daily bars. Default 22:00.",
    "day_count": "Days in the interest year, 360 or 365. Default 360.",
    "fill_policy": "Where FX rates have gaps against the prices: ffill_with_warning, ffill, strict (stop with an "
                   "error) or nan. Default ffill_with_warning.",
}
frame = pd.DataFrame({"setting": numbers + texts,
                      "value": [None if vectorbtx_values.get(key) is None else str(vectorbtx_values[key])
                                for key in numbers + texts],
                      "what it does": [meanings[key] for key in numbers + texts]})
edited = st.data_editor(frame, hide_index=True, disabled=["setting", "what it does"], width="stretch",
                        column_config={"setting": st.column_config.TextColumn("setting", help="vectorbtX's name."),
                                       "value": st.column_config.TextColumn(
                                           "value", help="Your value. Blank keeps vectorbtX's own default."),
                                       "what it does": st.column_config.TextColumn("what it does", width="large")})
st.caption("carry_markup is the broker's fee on FX carry and fx_markup its worse-than-market currency "
           "conversion rate; both are charged on every backtest that uses them.")
for key, value in zip(edited["setting"], edited["value"]):
    if value is None or str(value).strip() == "":
        vectorbtx_values[key] = None
    elif key in numbers:
        vectorbtx_values[key] = float(value) if key != "vol_window" else int(float(value))
    else:
        vectorbtx_values[key] = str(value).strip()
draft["vectorbtx"] = vectorbtx_values

if st.button("Save settings", type="primary"):
    config.save_settings(draft)
    st.session_state["settings"] = config.load_settings()
    if draft["data_dir"] != settings.get("data_dir"):
        st.info("The data folder changed. Restart the Studio so background jobs use it too.")
    st.success("Saved and applied.")
    st.rerun()

st.subheader("Installed versions")
ui.table(pd.DataFrame([{"package": name, "version": version or "not installed"}
                       for name, version in engine.versions().items()]))
