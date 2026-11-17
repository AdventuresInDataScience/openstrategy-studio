"""The data layer: import, store, validate and load bar data.

All data is bar data; tick data is out of scope. Storage (plan section 8):

    <data_dir>/catalogue.db                      SQLite index (store.py)
    <data_dir>/prices/<symbol>/<timeframe>.parquet

Stored frames have lower-case columns open, high, low, close, volume (plus
adj_close when the source provides one) and a tz-naive, sorted, duplicate-free
DatetimeIndex holding each bar's START time. Intraday bars are stored in UTC
with the original timezone recorded in the catalogue; daily bars keep their
local calendar date.
"""

import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from .spec import RESAMPLE_RULE, TIMEFRAME_MINUTES, TIMEFRAMES

OHLC = ["open", "high", "low", "close"]
OHLCV = OHLC + ["volume"]
COLUMN_ALIASES = {"adj close": "adj_close", "adjclose": "adj_close", "adjusted close": "adj_close",
                  "vol": "volume", "tick volume": "volume", "tickvol": "volume", "o": "open", "h": "high",
                  "l": "low", "c": "close", "v": "volume"}
# Yahoo intervals. There is no 4h interval: store 1h and the loader resamples up.
YAHOO_INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "1d": "1d", "1w": "1wk"}


def safe_name(symbol):
    """A folder name valid on Windows and Linux (EURUSD=X and ^GSPC are fine as they are)."""
    return "".join("_" if char in '\\/:*?"<>|' else char for char in symbol)


def price_path(data_dir, symbol, timeframe):
    return Path(data_dir) / "prices" / safe_name(symbol) / f"{timeframe}.parquet"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Resampling (up only)
# ---------------------------------------------------------------------------

def resample_frames(frames, rule):
    """Resample the five frames UP to a coarser rule.

    open=first, high=max, low=min, close=last, volume=sum, always with
    label="left", closed="left" so each bar keeps its start time. Empty bins
    (weekends, holidays) are dropped. ⚠️ Anything computed on the result and
    used on the finer index must also be shifted by one coarse bar.
    """
    def bins(frame):
        return frame.resample(rule, label="left", closed="left")

    out = {"open": bins(frames["open"]).first(), "high": bins(frames["high"]).max(),
           "low": bins(frames["low"]).min(), "close": bins(frames["close"]).last(),
           "volume": bins(frames["volume"]).sum(min_count=1)}
    keep = out["close"].notna().any(axis=1)
    return {field: frame[keep] for field, frame in out.items()}


def resample_ohlcv(frame, timeframe):
    """One stored symbol's frame resampled up to `timeframe`."""
    wide = {field: frame[[field]] for field in OHLCV}
    out = resample_frames(wide, RESAMPLE_RULE[timeframe])
    result = pd.concat([out[field][field] for field in OHLCV], axis=1)
    if "adj_close" in frame:
        adjusted = frame["adj_close"].resample(RESAMPLE_RULE[timeframe], label="left", closed="left").last()
        result["adj_close"] = adjusted.reindex(result.index)
    return result


# ---------------------------------------------------------------------------
# Normalisation and validation
# ---------------------------------------------------------------------------

def normalise_ohlcv(frame, timeframe):
    """Clean one symbol's frame the way the vectorbtX examples do by hand.

    Lower-case the columns, drop the timezone (intraday: convert to UTC first;
    daily: keep the local calendar date and normalize() to midnight), sort, and
    keep the last of any duplicate timestamps. Returns (frame, info) where info
    records the original timezone and every repair made, so the import report
    can show them: nothing is repaired silently.
    """
    frame = frame.copy()
    columns = []
    for column in frame.columns:
        name = str(column).strip().lower()
        columns.append(COLUMN_ALIASES.get(name, name.replace(" ", "_")))
    frame.columns = columns
    missing = [field for field in OHLC if field not in frame.columns]
    if missing:
        raise ValueError(f"The data has no {', '.join(missing)} column. Map the columns before importing.")
    keep = [column for column in OHLCV + ["adj_close"] if column in frame.columns]
    frame = frame[keep].astype(float)

    index = pd.DatetimeIndex(frame.index)
    original_tz = str(index.tz) if index.tz is not None else None
    daily = TIMEFRAME_MINUTES[timeframe] >= 1440
    if index.tz is not None:
        # ⚠️ Yahoo indexes are tz-aware and FRED's are naive: their intersection is
        # empty. Everything stored is naive.
        index = index.tz_localize(None) if daily else index.tz_convert("UTC").tz_localize(None)
    if daily:
        # LSE and US daily bars carry different local-midnight timestamps; dates
        # only, so they share rows when aligned.
        index = index.normalize()
    frame.index = index
    frame.index.name = "time"

    repairs = []
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index(kind="stable")
        repairs.append("Timestamps were out of order and have been sorted.")
    duplicated = frame.index.duplicated(keep="last")
    if duplicated.any():
        frame = frame[~duplicated]
        repairs.append(f"{int(duplicated.sum())} duplicate timestamps were dropped (the last of each was kept).")
    if "volume" not in frame:
        frame["volume"] = np.nan
    return frame, {"tz": original_tz, "repairs": repairs}


# Price ratios a stock split or consolidation leaves in an unadjusted close (2-for-1 halves it).
SPLIT_RATIOS = [1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3, 4, 5, 10]
# How many earlier bars a bar's move and range are judged against: long enough to be steady,
# short enough that a calm year and a wild one are each judged by their own standard.
USUAL_WINDOW = 100


def validate_ohlcv(frame, timeframe, spike_multiple=10.0, now=None):
    """What is wrong with the bars, or worth knowing, each with what it means. Never repairs.

    [{"check", "level", "why", "count", "sample"}]. "problem": prices that would give a backtest
    wrong numbers (a high below the close, a bad tick, an unadjusted split). "note": worth
    knowing, often fine (a gap over a holiday, no volume). Empty means nothing to report.

    A bad tick is judged against the bars before it, not one average for the whole file: a
    move counts as unusual when it is more than `spike_multiple` times the median move of the
    USUAL_WINDOW bars before it, so the same move is ordinary in 2008 and unusual in 2017.
    """
    found = []

    def report(check, level, why, mask, sample=None):
        mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
        count = int(mask.sum())
        if count:
            found.append({"check": check, "level": level, "why": why, "count": count,
                          "sample": frame[mask].head(5) if sample is None else sample.head(5)})

    open_, high, low, close = (frame[column] for column in OHLC)
    k = float(spike_multiple)

    # --- prices that cannot be right ----------------------------------------------------
    # ⚠️ vectorbtX only warns about the first two, and stops then fire at prices that never traded.
    report("High below the open or the close", "problem",
           "A bar's high must be at or above its open and its close. Stops and targets are checked against the high "
           "and low, so they would fire at prices that never traded.", high < frame[["open", "close"]].max(axis=1))
    report("Low above the open or the close", "problem",
           "A bar's low must be at or below its open and its close, for the same reason.",
           low > frame[["open", "close"]].min(axis=1))
    report("Zero or negative prices", "problem",
           "No traded price is zero or below. Returns, sizes and indicators through them are meaningless.",
           (frame[OHLC] <= 0).any(axis=1))
    report("Missing prices", "problem",
           "A bar with no open, high, low or close. Indicators go blank around it, and an order due there cannot "
           "fill.", frame[OHLC].isna().any(axis=1))
    later = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    report("Bars dated in the future", "problem",
           "Bars after today's date: the dates are probably misread (a day and month swapped, or the wrong date "
           "format).", frame.index > later + pd.Timedelta(days=1))

    # --- bad ticks and splits, each bar against the bars before it -----------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.log(close.where(close > 0)).diff()                      # close to close, as a log return
        bar_range = np.log(high.where(high > 0) / low.where(low > 0))      # high to low, the same way
    usual_move = move.abs().rolling(USUAL_WINDOW, min_periods=20).median().shift(1)
    usual_range = bar_range.rolling(USUAL_WINDOW, min_periods=20).median().shift(1)
    jump = (move.abs() > k * usual_move) & (move.abs() > 1e-9)
    # A spike that comes back: the next bar undoes at least half of it, in the other direction.
    back = jump & (move.shift(-1) * move < 0) & (move.shift(-1).abs() >= 0.5 * move.abs())
    report("Spikes that come straight back", "problem",
           f"The close jumps more than {k:g} times its usual move (the median of the {USUAL_WINDOW} bars before) "
           "and the next bar undoes most of it: usually one bad price, not a real trade. A strategy would trade on "
           "it as if it were real.", back)
    report("Bars far wider than usual", "problem",
           f"The high-to-low range is more than {k:g} times the usual range of the {USUAL_WINDOW} bars before: "
           "often a bad high or low, which would set off stops that a real market never reached.",
           (bar_range > k * usual_range) & ~back)
    ratio = np.exp(move)
    split_like = pd.Series(False, index=frame.index)
    for split in SPLIT_RATIOS:
        split_like |= (ratio / split - 1).abs() < 0.03
    splits = jump & ~back & split_like
    adjusted = "adj_close" in frame
    report("Jumps that look like a stock split", "note" if adjusted else "problem",
           "The close moves by a split's ratio (a half, a third, double ...) in one bar and stays there. "
           + ("The adjusted close allows for it, and buying outright reads that; CFDs and spread bets read the raw "
              "close, where this shows as a real gain or loss." if adjusted else
              "With no adjusted close, every mode reads it as a real gain or loss. Download it again from a source "
              "with adjusted prices, or adjust the file first."), splits)
    report("Large moves that hold", "note",
           f"The close moves more than {k:g} times its usual move and stays there: news, an earnings gap or a "
           "market shock, most likely real. Worth a look on a chart.", jump & ~back & ~split_like)

    # --- the bar timing ------------------------------------------------------------------
    minutes = TIMEFRAME_MINUTES[timeframe]
    bar = pd.Timedelta(minutes=minutes)
    step = frame.index.to_series().diff()
    usual_step = step.mode().iloc[0] if step.notna().any() else bar
    if minutes >= 1440:
        # Daily bars have no times, and weekly ones are 7 days apart; a month's are not regular.
        bar = pd.Timedelta(days=7) if minutes >= 10080 else pd.Timedelta(days=1)
    report("Bars closer together than the timeframe", "problem",
           f"These bars come less than {timeframe} after the bar before, so the file holds finer bars than {timeframe}. "
           "Pick the timeframe the bars really are: the Studio resamples finer bars up, never down.",
           step < bar * 0.99)
    if usual_step > bar * 1.5:
        report("Bars further apart than the timeframe", "problem",
               f"Most bars are {usual_step} apart, but the timeframe says {timeframe}. Pick the timeframe the bars "
               "really are, or annualised figures and every time-based setting will be wrong.",
               pd.Series(True, index=frame.index))
    long_gap = pd.Timedelta(days=14) if minutes >= 10080 else pd.Timedelta(days=4)
    gaps = step > long_gap
    where = pd.DataFrame({"from": frame.index.to_series().shift(1)[gaps], "to": frame.index[gaps],
                          "missing": step[gaps]})
    report("Stretches with no bars", "note",
           f"More than {long_gap.days} days without a bar: a holiday, a market closure, or bars missing from the "
           "source. Indicators read straight across it, as if no time had passed.", gaps,
           sample=where.sort_values("missing", ascending=False))
    if minutes >= 1440 and minutes < 10080:
        weekend = frame.index.day_of_week >= 5
        if weekend.mean() < 0.1:
            report("Bars on a Saturday or Sunday", "note",
                   "A few bars dated at the weekend: some feeds put Sunday evening's FX trading on a bar of its own. "
                   "Each counts as a bar for every indicator and every bar-based setting.", weekend)

    # --- prices that did not move, and volume ---------------------------------------------
    flat = (open_ == high) & (high == low) & (low == close)
    report("Bars where the price never moved", "note",
           "Open, high, low and close are all the same: no trading was recorded, or the bar was filled in. Common in "
           "quiet intraday hours; on daily bars, a sign of stale data.", flat)
    same = close.diff().eq(0)
    runs = same.groupby((~same).cumsum()).transform("sum")
    report("The same close for 5 bars or more in a row", "problem" if minutes >= 1440 else "note",
           "The price looks stuck: the source may have repeated its last value instead of reporting a gap. Returns "
           "there read as zero, which flatters volatility and drawdown. On quiet intraday hours it can be real.",
           same & (runs >= 4))
    if "volume" in frame:
        volume = frame["volume"]
        if volume.fillna(0).eq(0).all():
            # Yahoo's FX pairs report a volume of 0 on every bar: one note, not thousands of rows.
            found.append({"check": "No volume", "level": "note", "count": len(frame), "sample": frame.head(0),
                          "why": "The source gives no traded volume (FX sources often give none). Volume "
                                 "indicators and automatic slippage need it; everything else works."})
        else:
            report("Negative volume", "problem", "Volume cannot be below zero: the column is probably misread.",
                   volume < 0)
            report("Bars with zero volume", "note",
                   "Nothing was traded on these bars (or the source left volume out), so their prices may be "
                   "stale.", volume.eq(0))
    if adjusted:
        adjusted_close = frame["adj_close"]
        report("Adjusted close missing or not positive", "problem",
               "Buying outright reads the adjusted close, so these bars have no usable price in that mode.",
               (adjusted_close.isna() & close.notna()) | (adjusted_close <= 0))
    return found


def data_summary(frame, timeframe):
    """One line about a dataset: its span, how many bars and how far apart, volume and adjustment."""
    step = frame.index.to_series().diff()
    years = max((frame.index[-1] - frame.index[0]).days / 365.25, 1e-9)
    usual = step.mode().iloc[0] if step.notna().any() else None
    parts = [f"{len(frame):,} bars from {frame.index[0]:%Y-%m-%d} to {frame.index[-1]:%Y-%m-%d}",
             f"about {len(frame) / years:,.0f} a year" if years >= 0.5 else None,
             f"usually {usual} apart" if usual is not None and TIMEFRAME_MINUTES[timeframe] < 1440 else None,
             "with volume" if "volume" in frame and frame["volume"].notna().any() else "no volume",
             "with an adjusted close" if "adj_close" in frame else "no adjusted close"]
    return ", ".join(part for part in parts if part) + "."


# ---------------------------------------------------------------------------
# Importers
# ---------------------------------------------------------------------------

def download_yahoo(symbol, timeframe, start=None, end=None):
    """One symbol from Yahoo via vectorbt's YFData, as a raw frame (not yet normalised).

    start and end are dates (or anything pd.Timestamp reads); both are included.
    Daily history is deep. Intraday history is limited to recent windows (the
    limit varies by interval). auto_adjust=False keeps the raw close beside
    "Adj Close", so the loader can choose per instrument mode.
    Verified in phase 0: vectorbt 1.1.0's YFData.download_symbol passes interval= and
    auto_adjust= on to yfinance's Ticker.history, a single symbol's get() is the field
    frame (Open ... Adj Close, Volume, Dividends, Stock Splits), and the adjusted series
    the loader builds matches Yahoo's own (AAPL 2024, four dividends, within 3.3e-7).
    """
    import vectorbt as vbt

    # st.date_input gives a datetime.date, which vectorbt 1.1.0's to_tzaware_datetime does
    # not accept: it handles str, numbers, pd.Timestamp and datetime, and a date falls
    # through to `.tzinfo`, which a date lacks (AttributeError). A naive Timestamp is read
    # in local time, as yfinance expects.
    if start is not None:
        start = pd.Timestamp(start)
    if end is not None:
        # ⚠️ yfinance's end is exclusive (its history() docstring, 1.8.0): without the
        # extra day the "To" date the page asks for would be silently left out.
        end = pd.Timestamp(end) + pd.Timedelta(days=1)
    data = vbt.YFData.download(symbol, start=start, end=end, interval=YAHOO_INTERVALS[timeframe],
                               auto_adjust=False)
    frame = data.get()
    if isinstance(frame, pd.Series):
        # ⚠️ A single-symbol get() of one field is a Series named after the FIELD.
        frame = frame.to_frame()
    if frame is None or frame.empty:
        # Behind a TLS-intercepting proxy or antivirus Yahoo returns an empty frame
        # instead of an error (noted in the vectorbtX README).
        raise ValueError(
            f"Yahoo returned no data for {symbol}. Check the symbol and the dates. If both are right, a "
            "proxy or antivirus that inspects TLS traffic may be blocking it: set CURL_CA_BUNDLE, "
            "SSL_CERT_FILE and REQUESTS_CA_BUNDLE to your company's certificate bundle (see the vectorbtX README).")
    return frame


def read_upload(content, filename, header=True, separator=","):
    """A raw frame from an uploaded CSV or Parquet file, before column mapping."""
    if filename.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), sep=separator, header=0 if header else None)


def apply_mapping(raw, mapping, timeframe="1d"):
    """Build an indexed OHLCV frame from an upload and the user's column mapping.

    mapping: {"date": column, "time": column or None, "open", "high", "low",
    "close", "volume": column or None, "adj_close": column or None,
    "date_format": strptime format or None}. MT4/MT5 history exports split the
    date and the time into two columns ("2024.01.02", "13:00"); both are joined.
    ⚠️ VERIFY against a real MT4/MT5 export.

    Timestamps with UTC offsets (2024-03-25T00:00:00+01:00) often mix offsets
    across a daylight-saving change. Daily bars keep their LOCAL date (the
    offset is dropped); intraday bars are converted to UTC.
    """
    stamps = raw[mapping["date"]].astype(str).str.strip()
    if mapping.get("time") is not None:
        stamps = stamps + " " + raw[mapping["time"]].astype(str).str.strip()
    has_offset = stamps.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True)
    if has_offset.any() and TIMEFRAME_MINUTES[timeframe] >= 1440:
        stamps = stamps.str.replace(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
        parsed = pd.to_datetime(stamps, format=mapping.get("date_format") or None, errors="coerce")
    elif has_offset.any():
        parsed = pd.to_datetime(stamps, format=mapping.get("date_format") or None, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(stamps, format=mapping.get("date_format") or None, errors="coerce")
    if parsed.isna().any():
        bad = stamps[parsed.isna()].iloc[0]
        raise ValueError(f"{int(parsed.isna().sum())} timestamps could not be read, for example '{bad}'. "
                         "Set the date format (such as %Y.%m.%d %H:%M).")
    frame = pd.DataFrame(index=pd.DatetimeIndex(parsed))
    for field in OHLC + ["volume", "adj_close"]:
        column = mapping.get(field)
        if column is not None:
            frame[field] = pd.to_numeric(raw[column], errors="coerce").to_numpy()
    return frame


def dataset_name_problem(name, stored_names):
    """Why `name` cannot name a stored dataset, or None. `stored_names`: the names already stored.

    The name is what strategies pick and the folder the bars sit in. Characters a folder
    cannot hold are refused (so two names never share a folder), and so is a name that
    differs from a stored one only in capitals: Windows folders ignore case, so spy and
    SPY would overwrite each other. The same name exactly is fine: it adds a timeframe to it.
    """
    if not name or name != name.strip():
        return "Give the dataset a name, with no spaces at either end."
    if any(char in name for char in '\\/:*?"<>|,'):
        return f"'{name}' cannot name a dataset: leave out \\ / : * ? \" < > | and commas."
    for other in stored_names:
        if other != name and other.lower() == name.lower():
            return f"'{name}' differs from the stored {other} only in capitals, and they would share a folder."
    return None


def store_dataset(data_dir, symbol, timeframe, frame, source, tz=None, notes="", ticker=None):
    """Write the parquet file and its catalogue row. `symbol` is the dataset's name; `ticker` the source's symbol."""
    path = price_path(data_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    store.upsert_dataset(data_dir, {
        "symbol": symbol, "timeframe": timeframe, "source": source,
        "first_bar": str(frame.index[0]), "last_bar": str(frame.index[-1]), "n_bars": int(len(frame)),
        "adjusted": int("adj_close" in frame), "tz": tz, "notes": notes,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(path.relative_to(data_dir)), "ticker": ticker or symbol,
    })
    return path


def rename_dataset(data_dir, symbol, timeframe, new):
    """Store one dataset under a new name: its file moves to the new name's folder.

    ⚠️ A strategy that picks the old name no longer finds it: the Data page says so.
    """
    old_path, new_path = price_path(data_dir, symbol, timeframe), price_path(data_dir, new, timeframe)
    if new_path.exists():
        raise ValueError(f"{new} is already stored at {timeframe}; delete or rename that one first.")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.replace(new_path)
    if old_path.parent.exists() and not any(old_path.parent.iterdir()):
        old_path.parent.rmdir()
    store.rename_dataset(data_dir, symbol, timeframe, new, new_path.relative_to(data_dir))
    return new_path


def merge_refresh(old, new):
    """Old rows plus newly downloaded ones; where both have a bar, the new one wins."""
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def delete_dataset(data_dir, symbol, timeframe):
    price_path(data_dir, symbol, timeframe).unlink(missing_ok=True)
    store.delete_dataset(data_dir, symbol, timeframe)


# ---------------------------------------------------------------------------
# Loading for a backtest
# ---------------------------------------------------------------------------

def load_dataset(data_dir, symbol, timeframe):
    """One stored symbol at `timeframe`, resampled up from a finer stored timeframe if needed."""
    path = price_path(data_dir, symbol, timeframe)
    if path.exists():
        return pd.read_parquet(path)
    finer = [tf for tf in TIMEFRAMES if TIMEFRAME_MINUTES[tf] < TIMEFRAME_MINUTES[timeframe]
             and price_path(data_dir, symbol, tf).exists()]
    if not finer:
        raise ValueError(f"{symbol} is not stored at {timeframe} (or any finer timeframe). "
                         "Import it on the Data page first.")
    # The coarsest finer timeframe is the cheapest to resample. Resampling down is impossible.
    return resample_ohlcv(pd.read_parquet(price_path(data_dir, symbol, finer[-1])), timeframe)


def load_frames(data_dir, symbols, timeframe, start, end, adjusted=()):
    """The dict of five frames for a backtest: open, high, low, close, volume.

    Symbols in `adjusted` read adj_close as close, with open, high and low scaled
    by the same per-bar factor so the bracket high >= close >= low still holds.
    """
    per_symbol = {}
    for symbol in symbols:
        raw = load_dataset(data_dir, symbol, timeframe).loc[start:end]
        if symbol in adjusted and "adj_close" in raw:
            factor = raw["adj_close"] / raw["close"]
            raw = raw.assign(open=raw["open"] * factor, high=raw["high"] * factor,
                             low=raw["low"] * factor, close=raw["adj_close"])
        per_symbol[symbol] = raw
    # Align on the common trading days, as vectorbtX example 03 does. Without the
    # dropna, LSE and US dailies would share no timestamps and give zero rows with
    # every column still present.
    close = pd.concat({symbol: frame["close"] for symbol, frame in per_symbol.items()}, axis=1)
    close = close.dropna(how="any")
    if close.empty:
        raise ValueError(f"{', '.join(symbols)} have no bars in common between {start} and {end}.")
    frames = {}
    for field in OHLCV:
        columns = {}
        for symbol, frame in per_symbol.items():
            columns[symbol] = frame[field] if field in frame else pd.Series(np.nan, index=frame.index)
        frames[field] = pd.concat(columns, axis=1).reindex(close.index)
        frames[field].columns.name = None
    return frames


def load_fx(data_dir, symbol, timeframe, start, end):
    """An FX pair's close, stored as an ordinary dataset (for example GBPUSD=X).

    A little history before `start` is kept so the first bars can be filled.
    """
    frame = load_dataset(data_dir, symbol, timeframe)
    begin = pd.Timestamp(start) - pd.Timedelta(days=10)
    return frame["close"].loc[begin:end].rename(symbol)


def load_for_spec(spec, data_dir):
    """Everything a run of `spec` needs: the five frames, plus "fx" when it converts currency."""
    from . import engine   # engine reads the instrument modes to decide which symbols are adjusted

    data = spec["data"]
    frames = load_frames(data_dir, data["symbols"], data["timeframe"], data["start"], data["end"],
                         adjusted=engine.adjusted_symbols(spec, data_dir))
    fx = (spec.get("account") or {}).get("fx")
    if fx:
        frames["fx"] = load_fx(data_dir, fx["symbol"], data["timeframe"], data["start"], data["end"])
    return frames


def fingerprint(data_dir, spec, frames):
    """What a run was computed on: symbols, timeframe, bars and a hash of each parquet file."""
    data = spec["data"]
    files = {}
    for symbol in data["symbols"]:
        path = price_path(data_dir, symbol, data["timeframe"])
        files[symbol] = file_hash(path) if path.exists() else None
    close = frames["close"]
    return {"symbols": list(data["symbols"]), "timeframe": data["timeframe"],
            "first_bar": str(close.index[0]), "last_bar": str(close.index[-1]),
            "n_bars": int(len(close)), "files": files}


def synthetic_ohlcv(symbols, n=800, seed=0, freq="B", start="2018-01-01"):
    """A seeded random walk per symbol, in the shape load_frames returns.

    Used by `openstrategy-studio selfcheck`, so a fresh machine can be checked
    without any network access. High and low always bracket open and close.
    FX-style symbols (ending in =X) are priced near 1.1, everything else near 100.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq)
    fields = {field: {} for field in OHLCV}
    for symbol in symbols:
        base = 1.1 if symbol.endswith("=X") else 100.0
        close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n)))
        open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.001, n - 1))])
        fields["open"][symbol] = open_
        fields["high"][symbol] = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
        fields["low"][symbol] = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
        fields["close"][symbol] = close
        fields["volume"][symbol] = rng.integers(1_000, 5_000, n).astype(float)
    return {field: pd.DataFrame(values, index=index) for field, values in fields.items()}
