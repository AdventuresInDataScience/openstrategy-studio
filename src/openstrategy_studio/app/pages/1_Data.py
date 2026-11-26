"""Data: the catalogue, imports (Yahoo, CSV, Parquet), the check, refresh, rename and delete.

Every import is normalised, then checked. The review lists what is wrong (and what is worth
knowing) with sample rows, lets you name the dataset, and nothing is stored until you accept it.
"""

import uuid
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


def queue(name, timeframe, frame, source, ticker):
    """Normalise and check an import, then hold it for review under a suggested name."""
    clean, info = D.normalise_ohlcv(frame, timeframe)
    found = D.validate_ohlcv(clean, timeframe, spike_multiple=float(settings.get("spike_multiple", 10.0)))
    pending.append({"id": uuid.uuid4().hex, "name": name, "ticker": ticker, "timeframe": timeframe, "frame": clean,
                    "info": info, "found": found, "source": source})


# --- catalogue ---------------------------------------------------------------
catalogue = store.list_datasets(data_dir)
st.subheader("Stored data")
if catalogue.empty:
    st.write("Nothing stored yet. Import a symbol below.")
else:
    shown = catalogue.drop(columns=["path"]).rename(columns={
        "symbol": "name", "first_bar": "from", "last_bar": "to", "n_bars": "bars", "tz": "timezone",
        "imported_at": "imported"})
    ui.table(shown[["name", "ticker", "timeframe", "source", "from", "to", "bars", "adjusted", "timezone", "imported",
                    "notes"]])
    st.caption("One row per name and timeframe. Strategies pick a name on the Build page; a strategy on a longer "
               "timeframe reads the name's finest stored bars and resamples them up. Keep one source per name, and "
               "store a second source under a name of its own (SPY_csv beside SPY).")
stored_names = sorted(catalogue["symbol"].unique()) if not catalogue.empty else []

yahoo_tab, file_tab, manage_tab, universe_tab = st.tabs(["Download from Yahoo", "Import a CSV or Parquet file",
                                                         "Refresh, rename or delete", "Universe files"])

with yahoo_tab:
    if settings.get("offline"):
        st.info("Offline mode is on (Settings), so downloads are switched off.")
    with st.form("yahoo"):
        symbols = st.text_input("Symbols, separated by commas", placeholder="EURUSD=X, SPY, VOD.L",
                                help="Yahoo Finance's own symbols: =X for FX pairs (EURUSD=X), a suffix for other "
                                     "exchanges (.L for London), ^ for indices (^GSPC). Each is stored under its "
                                     "symbol unless you name it otherwise before storing.")
        columns = st.columns(3)
        timeframe = columns[0].selectbox("Timeframe", list(D.YAHOO_INTERVALS), index=list(D.YAHOO_INTERVALS).index("1d"),
                                         help="Yahoo has no 4h bars: store 1h and the Studio resamples up.")
        start = columns[1].date_input("From", value=date(2010, 1, 1), min_value=date(1970, 1, 1),
                                      help="The first date to download. Yahoo starts later if it has less history.")
        end = columns[2].date_input("To", value=date.today(), min_value=date(1970, 1, 1),
                                    help="The last date to download.")
        st.caption("Intraday history on Yahoo only reaches back a limited window, which varies by interval.")
        submitted = st.form_submit_button("Download", disabled=bool(settings.get("offline")))
    if submitted:
        for symbol in [item.strip() for item in symbols.split(",") if item.strip()]:
            frame = ui.run_guarded(f"Downloading {symbol}",
                                   lambda symbol=symbol: D.download_yahoo(symbol, timeframe, start, end))
            if frame is not None:
                queue(symbol, timeframe, frame, "yahoo", ticker=symbol)

with file_tab:
    upload = st.file_uploader("CSV or Parquet file", type=["csv", "txt", "parquet"],
                              help="One symbol's bars, one row per bar, such as an MT4 or MT5 history export. You "
                                   "say which column holds what below.")
    if upload is not None:
        columns = st.columns(3)
        header = columns[0].checkbox("First row holds column names", value=True,
                                     help="MT4/MT5 history exports have no header row.")
        separator = columns[1].text_input("Separator", value=",",
                                          help="The character between columns, such as a comma or a semicolon.")
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
            date_column = row[0].selectbox("Date (or date and time)", names, index=guess("date", "datetime", "time", "timestamp"),
                                           help="The column holding each bar's date, or its date and time together.")
            time_column = row[1].selectbox("Separate time column", optional, index=0,
                                           help="MT4/MT5 exports put the time in its own column.")
            date_format = row[2].text_input("Date format (optional)", placeholder="%Y.%m.%d %H:%M",
                                            help="Only if the dates are not read correctly: the pattern they follow, "
                                                 "in Python's strftime codes (%Y year, %m month, %d day, %H hour, "
                                                 "%M minute).")
            symbol = row[3].text_input("Symbol", value=upload.name.rsplit(".", 1)[0],
                                       help="What the file holds, such as EURUSD. It is the dataset's name unless "
                                            "you change the name before storing.")
            row = st.columns(4)
            mapping = {"date": date_column, "time": time_column, "date_format": date_format or None}
            for column, field in zip(row, ("open", "high", "low", "close")):
                mapping[field] = column.selectbox(field.capitalize(), names, index=guess(field, field[0]),
                                                  help=f"The column holding each bar's {field} price.")
            row = st.columns(3)
            mapping["volume"] = row[0].selectbox("Volume (optional)", optional, index=guess_optional("volume", "vol", "v"),
                                                 help="Needed for volume indicators and automatic slippage. MT4/MT5 "
                                                      "tick volume counts price changes, not traded amounts.")
            mapping["adj_close"] = row[1].selectbox("Adjusted close (optional)", optional,
                                                    index=guess_optional("adj close", "adj_close", "adjusted close"),
                                                    help="The close adjusted for dividends and splits, if the file "
                                                         "has one. Buying outright uses it by default.")
            file_timeframe = row[2].selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1d"),
                                              help="The file's bar size. Strategies on a longer timeframe resample "
                                                   "it up.")
            read = st.form_submit_button("Check the file")
        if read:
            frame = ui.run_guarded("Reading the file", lambda: D.apply_mapping(raw, mapping, file_timeframe))
            if frame is not None:
                queue(symbol.strip(), file_timeframe, frame, f"file:{upload.name}", ticker=symbol.strip())

with manage_tab:
    if catalogue.empty:
        st.write("Nothing stored yet.")
    else:
        labels = [f"{row.symbol} {row.timeframe} ({row.source})" for row in catalogue.itertuples()]
        picked = st.selectbox("Dataset", range(len(labels)), format_func=labels.__getitem__,
                              help="A stored name at one timeframe, with the source it came from.")
        row = catalogue.iloc[picked]
        stored_frame = D.load_dataset(data_dir, row["symbol"], row["timeframe"])
        st.write(D.data_summary(stored_frame, row["timeframe"]) + f" Source {row['source']}, ticker {row['ticker']}.")
        refresh, rename, delete = st.columns(3)
        if row["source"] == "yahoo" and refresh.button("Refresh from Yahoo", disabled=bool(settings.get("offline")),
                                                       help="Downloads the bars since the last stored one (and the "
                                                            "week before, which Yahoo may have revised), for review."):
            begin = pd.Timestamp(row["last_bar"]) - timedelta(days=7)
            new = ui.run_guarded("Downloading new bars", lambda: D.download_yahoo(row["ticker"], row["timeframe"],
                                                                                 begin, None))
            if new is not None:
                fresh, _ = D.normalise_ohlcv(new, row["timeframe"])
                queue(row["symbol"], row["timeframe"], D.merge_refresh(stored_frame, fresh), "yahoo",
                      ticker=row["ticker"])
        new_name = rename.text_input("New name", value=row["symbol"], key=f"rename:{row['symbol']}:{row['timeframe']}",
                                     help="Store this dataset under another name, such as SPY_csv. A strategy that "
                                          "picks the old name will need the new one picked on the Build page.")
        problem = D.dataset_name_problem(new_name, stored_names)
        if new_name != row["symbol"]:
            if problem:
                rename.error(problem)
            elif rename.button("Rename"):
                if ui.run_guarded("Renaming", lambda: D.rename_dataset(data_dir, row["symbol"], row["timeframe"],
                                                                       new_name)) is not None:
                    st.rerun()
        confirm = delete.checkbox("I want to delete this dataset",
                                  help="Unlocks the Delete button. Deleting removes the stored bars; it cannot be "
                                       "undone.")
        if delete.button("Delete", disabled=not confirm):
            D.delete_dataset(data_dir, row["symbol"], row["timeframe"])
            st.rerun()
        with st.expander("Check this dataset again"):
            again = D.validate_ohlcv(stored_frame, row["timeframe"],
                                     spike_multiple=float(settings.get("spike_multiple", 10.0)))
            if not again:
                st.write("Nothing to report.")
            for finding in again:
                st.markdown(f"**{finding['check']}** ({finding['count']:,} bars, "
                            f"{'a problem' if finding['level'] == 'problem' else 'a note'}): {finding['why']}")

with universe_tab:
    st.write("A universe file lists instruments, one per row, for portfolios with many symbols "
             "(the spreadsheet schema is in vectorbtX's docs/API.md, section 2).")
    universe = st.file_uploader("Universe file", type=["csv", "xlsx"], key="universe",
                                help="A spreadsheet with one instrument per row: its mode, point size, costs and "
                                     "so on, in vectorbtX's column names.")
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
    for item in list(pending):
        frame = item["frame"]
        with st.container(border=True):
            st.write(f"**{item['ticker']}** ({item['timeframe']}, {item['source']}): "
                     + D.data_summary(frame, item["timeframe"]))
            if item["info"]["tz"]:
                st.caption(f"Original timezone {item['info']['tz']}; intraday bars are stored in UTC.")
            for repair in item["info"]["repairs"]:
                st.info(repair)
            problems = [finding for finding in item["found"] if finding["level"] == "problem"]
            notes = [finding for finding in item["found"] if finding["level"] == "note"]
            if not item["found"]:
                st.success("No problems found.")
            elif not problems:
                st.success("No problems found; a few things worth knowing below.")
            for finding in problems + notes:
                icon = ":material/error:" if finding["level"] == "problem" else ":material/info:"
                with st.expander(f"{finding['check']}: {finding['count']:,} bars", icon=icon):
                    st.write(finding["why"])
                    if len(finding["sample"]):
                        ui.table(finding["sample"], hide_index=False)

            # The name: what strategies pick it by. Storing under a name and timeframe that is
            # already stored replaces those bars, so say what would be replaced.
            name = st.text_input("Store as", value=item["name"], key=f"pending:{item['id']}:name",
                                 help="The dataset's name: what strategies pick on the Build page, such as SPY or "
                                      "SPY_csv. A name holds one series at several timeframes; give a second source "
                                      "of the same symbol a name of its own.")
            problem = D.dataset_name_problem(name, stored_names)
            if problem:
                st.error(problem)
            same = catalogue[(catalogue["symbol"] == name) & (catalogue["timeframe"] == item["timeframe"])] \
                if not catalogue.empty else catalogue
            replacing_other = False
            if len(same):
                old = same.iloc[0]
                if old["source"] == item["source"]:
                    st.caption(f"Replaces the stored {name} {item['timeframe']} ({old['n_bars']:,} bars from "
                               f"{old['source']}, imported {str(old['imported_at'])[:10]}).")
                else:
                    replacing_other = True
                    st.warning(f"{name} is already stored at {item['timeframe']} from another source ({old['source']}, "
                               f"{old['n_bars']:,} bars). Storing replaces it. To keep both, give this one another "
                               f"name, such as {name}_{item['source'].split(':')[0]}.")
            elif name in stored_names:
                others = catalogue[catalogue["symbol"] == name]
                sources = sorted(set(others["source"]) - {item["source"]})
                if sources:
                    st.caption(f"{name} is stored at {', '.join(others['timeframe'])} from {', '.join(sources)}. A "
                               "strategy reads whichever timeframe fits, so keep one source per name.")
            sure = st.checkbox("Replace it", key=f"pending:{item['id']}:replace",
                               help="Store these bars in place of the ones from the other source.") \
                if replacing_other else True
            accept, reject = st.columns(2)
            if accept.button("Store it", key=f"pending:{item['id']}:store", type="primary",
                             disabled=bool(problem) or not sure):
                D.store_dataset(data_dir, name, item["timeframe"], frame, item["source"], tz=item["info"]["tz"],
                                notes=f"{len(problems)} problems accepted" if problems else "", ticker=item["ticker"])
                pending.remove(item)
                st.rerun()
            if reject.button("Discard", key=f"pending:{item['id']}:discard"):
                pending.remove(item)
                st.rerun()
