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


def validate_ohlcv(frame, timeframe, spike_multiple=10.0, gap_days=None):
    """Every data problem, with counts and up to five sample rows. Never repairs.

    Returns a list of {"check", "count", "sample"} dicts; empty means clean.
    """
    problems = []

    def report(check, mask):
        count = int(mask.sum())
        if count:
            problems.append({"check": check, "count": count, "sample": frame[mask].head(5)})

    top = frame[["open", "close"]].max(axis=1)
    bottom = frame[["open", "close"]].min(axis=1)
    # ⚠️ vectorbtX only warns about these, and stops then fire at prices that never traded.
    report("High is below the open or close", frame["high"] < top)
    report("Low is above the open or close", frame["low"] > bottom)
    report("Zero or negative prices", (frame[OHLC] <= 0).any(axis=1))
    for column in OHLC:
        report(f"Missing values in {column}", frame[column].isna())

    # Likely bad ticks: a range or a gap far larger than the typical bar's range.
    bar_range = frame["high"] - frame["low"]
    typical = bar_range.median()
    if typical and np.isfinite(typical):
        report(f"Bar range above {spike_multiple:g}x the median range", bar_range > spike_multiple * typical)
        gap = (frame["open"] - frame["close"].shift(1)).abs()
        report(f"Opening gap above {spike_multiple:g}x the median range", gap > spike_multiple * typical)

    # Calendar gaps longer than normal market closures (weekend plus a holiday).
    if gap_days is None:
        minutes = TIMEFRAME_MINUTES[timeframe]
        gap_days = 15 if minutes >= 10080 else 5 if minutes >= 1440 else 4.5
    step = frame.index.to_series().diff()
    report(f"Gaps longer than {gap_days:g} days", step > pd.Timedelta(days=gap_days))
    return problems


# ---------------------------------------------------------------------------
# Importers
# ---------------------------------------------------------------------------

def download_yahoo(symbol, timeframe, start=None, end=None):
    """One symbol from Yahoo via vectorbt's YFData, as a raw frame (not yet normalised).

    Daily history is deep. Intraday history is limited to recent windows (the
    limit varies by interval). auto_adjust=False keeps the raw close beside
    "Adj Close", so the loader can choose per instrument mode.
    ⚠️ VERIFY: vectorbt 1.1.0 passes interval= and auto_adjust= on to yfinance.
    """
    import vectorbt as vbt

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


def store_dataset(data_dir, symbol, timeframe, frame, source, tz=None, notes=""):
    """Write the parquet file and its catalogue row."""
    path = price_path(data_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    store.upsert_dataset(data_dir, {
        "symbol": symbol, "timeframe": timeframe, "source": source,
        "first_bar": str(frame.index[0]), "last_bar": str(frame.index[-1]), "n_bars": int(len(frame)),
        "adjusted": int("adj_close" in frame), "tz": tz, "notes": notes,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(path.relative_to(data_dir)),
    })
    return path


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
