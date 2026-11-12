"""Settings: data folder, theme, workers, offline mode, vectorbtX settings and the FRED key."""

import copy
import os

import pandas as pd
import streamlit as st

from openstrategy_studio import analysis, config, engine, ui

settings, data_dir = ui.boot("Settings")
st.title("Settings")
st.caption(f"Stored in {config.settings_path()}")

draft = copy.deepcopy(settings)
st.subheader("Studio")
draft["data_dir"] = st.text_input("Data folder", value=settings.get("data_dir") or "", placeholder=str(data_dir),
                                  help="Every price file, strategy, run and job lives here. Blank uses the default.")
themes = [theme for names in analysis.theme_groups().values() for theme in names]
draft["default_theme"] = st.selectbox("Default chart theme", themes,
                                      index=themes.index(settings["default_theme"]) if settings["default_theme"] in themes else 0)
row = st.columns(3)
draft["workers"] = int(row[0].number_input("Worker processes for optimisation", min_value=1,
                                           value=int(settings.get("workers", 1)), step=1))
draft["spike_multiple"] = float(row[1].number_input("Flag bars wider than this many median ranges", min_value=2.0,
                                                    value=float(settings.get("spike_multiple", 10.0)), step=1.0))
draft["offline"] = row[2].toggle("Offline", value=bool(settings.get("offline")),
                                 help="Turns off every download. Stored data still works; FRED falls back to its cache, "
                                      "then to the static rate below.")

st.subheader("FRED rates")
if os.environ.get("FRED_API_KEY") and not settings.get("fred_api_key"):
    st.write("Using the FRED_API_KEY environment variable.")
draft["fred_api_key"] = st.text_input("FRED API key", value=settings.get("fred_api_key") or "", type="password",
                                      help="Free from the St. Louis Fed. Prefer the FRED_API_KEY environment variable "
                                           "or a .env file in the data folder.")
if draft["fred_api_key"]:
    st.warning("The key will be stored in settings.toml. Keep that file out of version control.")

st.subheader("vectorbtX settings")
st.caption("Written to vbt.settings[\"vectorbtx\"]. Blank keeps vectorbtX's own default.")
vectorbtx_values = dict(settings["vectorbtx"])
roundings = [None, "floor", "nearest", "ceil"]
rounding = st.selectbox("Size rounding", roundings,
                        index=roundings.index(vectorbtx_values.get("rounding")) if vectorbtx_values.get("rounding") in roundings else 0,
                        format_func=lambda value: "Default" if value is None else value)
vectorbtx_values["rounding"] = rounding
numbers = ["financing_markup", "carry_markup", "fx_markup", "slippage_max", "slippage_coef", "vol_window",
           "static_rate", "triangulation_tolerance"]
texts = ["triple_day", "end_of_day", "day_count", "fill_policy"]
frame = pd.DataFrame({"setting": numbers + texts,
                      "value": [None if vectorbtx_values.get(key) is None else str(vectorbtx_values[key])
                                for key in numbers + texts]})
edited = st.data_editor(frame, hide_index=True, disabled=["setting"], width="stretch")
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
