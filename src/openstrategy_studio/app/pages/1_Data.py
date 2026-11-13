"""Data: the catalogue, imports (Yahoo, CSV, Parquet), the validation report, refresh and delete.

Every import is normalised, then checked. The report lists every problem with
counts and sample rows, and nothing is stored until you accept it.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from openstrategy_studio import data as D, store, ui
from openstrategy_studio.spec import TIMEFRAMES

settings, data_dir = ui.boot("Data")
ui.page_help("data")
st.title("Data")
st.caption("All data is bar data (1 minute to weekly). Tick data is not supported.")

pending = st.session_state.setdefault("pending_imports", [])


def queue(symbol, timeframe, frame, source):
    """Normalise and check an import, then hold it for review."""
    clean, info = D.normalise_ohlcv(frame, timeframe)
    problems = D.validate_ohlcv(clean, timeframe, spike_multiple=float(settings.get("spike_multiple", 10.0)))
    pending.append({"symbol": symbol, "timeframe": timeframe, "frame": clean, "info": info, "problems": problems,
                    "source": source})


# --- catalogue ---------------------------------------------------------------
catalogue = store.list_datasets(data_dir)
st.subheader("Stored data")
if catalogue.empty:
    st.write("Nothing stored yet. Import a symbol below.")
else:
    ui.table(catalogue.drop(columns=["path"]))

yahoo_tab, file_tab, manage_tab, universe_tab = st.tabs(["Download from Yahoo", "Import a CSV or Parquet file",
                                                         "Refresh or delete", "Universe files"])

with yahoo_tab:
    if settings.get("offline"):
        st.info("Offline mode is on (Settings), so downloads are switched off.")
    with st.form("yahoo"):
        symbols = st.text_input("Symbols, separated by commas", placeholder="EURUSD=X, SPY, VOD.L")
        columns = st.columns(3)
        timeframe = columns[0].selectbox("Timeframe", list(D.YAHOO_INTERVALS), index=list(D.YAHOO_INTERVALS).index("1d"),
                                         help="Yahoo has no 4h bars: store 1h and the Studio resamples up.")
        start = columns[1].date_input("From", value=date(2010, 1, 1), min_value=date(1970, 1, 1))
        end = columns[2].date_input("To", value=date.today(), min_value=date(1970, 1, 1))
        st.caption("Intraday history on Yahoo only reaches back a limited window, which varies by interval.")
        submitted = st.form_submit_button("Download", disabled=bool(settings.get("offline")))
    if submitted:
        for symbol in [item.strip() for item in symbols.split(",") if item.strip()]:
            frame = ui.run_guarded(f"Downloading {symbol}",
                                   lambda symbol=symbol: D.download_yahoo(symbol, timeframe, start, end))
            if frame is not None:
                queue(symbol, timeframe, frame, "yahoo")

with file_tab:
    upload = st.file_uploader("CSV or Parquet file", type=["csv", "txt", "parquet"])
    if upload is not None:
        columns = st.columns(3)
        header = columns[0].checkbox("First row holds column names", value=True,
                                     help="MT4/MT5 history exports have no header row.")
        separator = columns[1].text_input("Separator", value=",")
        raw = D.read_upload(upload.getvalue(), upload.name, header=header, separator=separator or ",")
        st.write("First rows of the file:")
        ui.table(raw.head(), hide_index=False)
        names = list(raw.columns)
        optional = [None] + names

        def guess(*words):
            for column in names:
                if str(column).strip().lower() in words:
                    return names.index(column)
            return 0

        def guess_optional(*words):
            for column in names:
                if str(column).strip().lower() in words:
                    return optional.index(column)
            return 0

        with st.form("mapping"):
            st.write("Which column holds what?")
            row = st.columns(4)
            date_column = row[0].selectbox("Date (or date and time)", names, index=guess("date", "datetime", "time", "timestamp"))
            time_column = row[1].selectbox("Separate time column", optional, index=0,
                                           help="MT4/MT5 exports put the time in its own column.")
            date_format = row[2].text_input("Date format (optional)", placeholder="%Y.%m.%d %H:%M")
            symbol = row[3].text_input("Store as symbol", value=upload.name.rsplit(".", 1)[0])
            row = st.columns(4)
            mapping = {"date": date_column, "time": time_column, "date_format": date_format or None}
            for column, field in zip(row, ("open", "high", "low", "close")):
                mapping[field] = column.selectbox(field.capitalize(), names, index=guess(field, field[0]))
            row = st.columns(3)
            mapping["volume"] = row[0].selectbox("Volume (optional)", optional, index=guess_optional("volume", "vol", "v"))
            mapping["adj_close"] = row[1].selectbox("Adjusted close (optional)", optional,
                                                    index=guess_optional("adj close", "adj_close", "adjusted close"))
            file_timeframe = row[2].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1d"))
            read = st.form_submit_button("Check the file")
        if read:
            frame = ui.run_guarded("Reading the file", lambda: D.apply_mapping(raw, mapping, file_timeframe))
            if frame is not None:
                queue(symbol.strip(), file_timeframe, frame, f"file:{upload.name}")

with manage_tab:
    if catalogue.empty:
        st.write("Nothing stored yet.")
    else:
        labels = [f"{row.symbol} {row.timeframe}" for row in catalogue.itertuples()]
        picked = st.selectbox("Dataset", range(len(labels)), format_func=labels.__getitem__)
        row = catalogue.iloc[picked]
        st.write(f"{row['n_bars']} bars from {row['first_bar']} to {row['last_bar']}, source {row['source']}.")
        refresh, delete = st.columns(2)
        if row["source"] == "yahoo" and refresh.button("Refresh from Yahoo", disabled=bool(settings.get("offline"))):
            old = D.load_dataset(data_dir, row["symbol"], row["timeframe"])
            begin = pd.Timestamp(row["last_bar"]) - timedelta(days=7)
            new = ui.run_guarded("Downloading new bars", lambda: D.download_yahoo(row["symbol"], row["timeframe"], begin, None))
            if new is not None:
                fresh, _ = D.normalise_ohlcv(new, row["timeframe"])
                queue(row["symbol"], row["timeframe"], D.merge_refresh(old, fresh), "yahoo")
        confirm = delete.checkbox("I want to delete this dataset")
        if delete.button("Delete", disabled=not confirm):
            D.delete_dataset(data_dir, row["symbol"], row["timeframe"])
            st.rerun()

with universe_tab:
    st.write("A universe file lists instruments, one per row, for portfolios with many symbols "
             "(the spreadsheet schema is in vectorbtX's docs/API.md, section 2).")
    universe = st.file_uploader("Universe file", type=["csv", "xlsx"], key="universe")
    if universe is not None and st.button("Store universe file"):
        (data_dir / "universes" / universe.name).write_bytes(universe.getvalue())
        st.success(f"Stored {universe.name}. Pick it on the Build page, Instruments tab.")
    stored = sorted(path.name for path in (data_dir / "universes").glob("*"))
    if stored:
        st.write("Stored: " + ", ".join(stored))

# --- review --------------------------------------------------------------------
if pending:
    st.divider()
    st.subheader("Review before storing")
    for number, item in enumerate(list(pending)):
        frame = item["frame"]
        with st.container(border=True):
            st.write(f"**{item['symbol']}** ({item['timeframe']}): {len(frame)} bars from {frame.index[0]} "
                     f"to {frame.index[-1]}.")
            if item["info"]["tz"]:
                st.caption(f"Original timezone {item['info']['tz']}; intraday bars are stored in UTC.")
            for repair in item["info"]["repairs"]:
                st.info(repair)
            if not item["problems"]:
                st.success("No problems found.")
            for problem in item["problems"]:
                with st.expander(f"{problem['check']}: {problem['count']} bars"):
                    ui.table(problem["sample"], hide_index=False)
            accept, reject = st.columns(2)
            if accept.button("Store it", key=f"accept{number}", type="primary"):
                D.store_dataset(data_dir, item["symbol"], item["timeframe"], frame, item["source"], tz=item["info"]["tz"],
                                notes=f"{len(item['problems'])} problems accepted" if item["problems"] else "")
                pending.remove(item)
                st.rerun()
            if reject.button("Discard", key=f"reject{number}"):
                pending.remove(item)
                st.rerun()
