"""Test 10: data normalisation, validation and loading."""

import types

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import data as D

from conftest import FIXTURES, make_ohlcv


def raw_frame(n=10, tz="America/New_York"):
    index = pd.date_range("2024-01-01 09:30", periods=n, freq="1h", tz=tz)
    close = np.linspace(10, 11, n)
    return pd.DataFrame({"Open": close, "High": close + 0.1, "Low": close - 0.1, "Close": close,
                         "Volume": 100.0}, index=index)


def test_normalise_tz_duplicates_and_order():
    raw = raw_frame()
    shuffled = pd.concat([raw.iloc[5:], raw.iloc[:5], raw.iloc[[2]] * 1.01])     # unsorted + a duplicate
    clean, info = D.normalise_ohlcv(shuffled, "1h")
    assert clean.index.tz is None
    assert clean.index.is_monotonic_increasing and not clean.index.duplicated().any()
    assert list(clean.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert info["tz"] == "America/New_York"
    assert len(info["repairs"]) == 2                     # sorted, and one duplicate dropped
    # Intraday is stored in UTC: 09:30 New York in January is 14:30 UTC.
    assert clean.index[0] == pd.Timestamp("2024-01-01 14:30")
    # The LAST copy of a duplicated timestamp is kept.
    assert clean.loc[raw.index[2].tz_convert("UTC").tz_localize(None), "close"] == pytest.approx(raw["Close"].iloc[2] * 1.01)


YAHOO_CSV = {"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
             "adj_close": "Adj Close", "volume": "Volume"}


def load_fixture(name, timeframe="1d"):
    raw = D.read_upload((FIXTURES / name).read_bytes(), name)
    return D.normalise_ohlcv(D.apply_mapping(raw, YAHOO_CSV, timeframe), timeframe)


def test_daily_keeps_local_dates():
    # The LSE file crosses the March clock change, so its offsets are mixed.
    for name in ("lse_daily.csv", "us_daily.csv"):
        clean, _ = load_fixture(name)
        assert clean.index.tz is None
        assert (clean.index == clean.index.normalize()).all()
        assert clean.index[0] == pd.Timestamp("2024-03-25")
        assert "adj_close" in clean
        assert len(clean) == 12


def test_intraday_offsets_convert_to_utc():
    raw = pd.DataFrame({"t": ["2024-03-29T09:00:00+00:00", "2024-04-02T09:00:00+01:00"],
                        "o": [1, 1], "h": [2, 2], "l": [0.5, 0.5], "c": [1, 1]})
    frame = D.apply_mapping(raw, {"date": "t", "open": "o", "high": "h", "low": "l", "close": "c"}, "1h")
    clean, _ = D.normalise_ohlcv(frame, "1h")
    assert list(clean.index) == [pd.Timestamp("2024-03-29 09:00"), pd.Timestamp("2024-04-02 08:00")]


def test_validation_reports_without_repairing():
    frame, _ = D.normalise_ohlcv(raw_frame(30, tz=None), "1h")
    frame.iloc[3, frame.columns.get_loc("high")] = frame["close"].iloc[3] - 1     # high below close
    frame.iloc[7, frame.columns.get_loc("low")] = -1.0                           # negative price
    before = frame.copy()
    problems = {p["check"]: p for p in D.validate_ohlcv(frame, "1h")}
    assert problems["High below the open or the close"]["count"] == 1
    assert problems["Zero or negative prices"]["count"] == 1
    assert len(problems["High below the open or the close"]["sample"]) == 1
    assert all(problem["why"] and problem["level"] in ("problem", "note") for problem in problems.values())
    pd.testing.assert_frame_equal(frame, before)                                   # never repairs


def daily(n=300, seed=1, base=100.0):
    bars = make_ohlcv(n, ("AAA",), seed=seed, base=base)
    return pd.DataFrame({field: bars[field]["AAA"] for field in ("open", "high", "low", "close", "volume")})


def found(frame, timeframe="1d"):
    return {item["check"]: item for item in D.validate_ohlcv(frame, timeframe)}


def test_a_bad_tick_is_judged_against_the_bars_before_it():
    """One bad close that comes straight back is a problem; a big move that holds is only a note."""
    frame = daily()
    frame.iloc[150, frame.columns.get_loc("close")] *= 1.2                  # a bad print, undone at once
    frame.iloc[150, frame.columns.get_loc("high")] = frame["close"].iloc[150]
    frame.iloc[220:, :4] *= 1.15                                           # a gap that stays (news, say)
    checks = found(frame)
    assert checks["Spikes that come straight back"]["count"] == 1
    assert checks["Spikes that come straight back"]["sample"].index[0] == frame.index[150]
    assert checks["Large moves that hold"]["level"] == "note"
    assert frame.index[220] in checks["Large moves that hold"]["sample"].index


def test_an_unadjusted_split_is_a_problem_unless_there_is_an_adjusted_close():
    frame = daily()
    frame.iloc[200:, :4] /= 2                                              # a 2-for-1 split, unadjusted
    assert found(frame)["Jumps that look like a stock split"]["level"] == "problem"
    frame["adj_close"] = frame["close"].where(frame.index >= frame.index[200], frame["close"] / 2)
    assert found(frame)["Jumps that look like a stock split"]["level"] == "note"


def test_bars_that_do_not_fit_the_timeframe():
    hourly = daily(120)
    hourly.index = pd.date_range("2024-01-01", periods=120, freq="1h")
    assert found(hourly, "1d")["Bars closer together than the timeframe"]["count"] == 119
    assert "Bars further apart than the timeframe" in found(daily(), "1h")
    assert not {"Bars closer together than the timeframe", "Bars further apart than the timeframe"} & set(found(daily()))


def test_gaps_stale_prices_and_volume():
    frame = daily()
    missing = frame.index[110] - frame.index[99]
    frame = frame.drop(frame.index[100:110])                               # two weeks with no bars
    frame.iloc[200:206, :4] = frame.iloc[199, 3]                           # stuck at one price
    frame["volume"] = 0.0
    checks = found(frame)
    gap = checks["Stretches with no bars"]
    assert gap["level"] == "note" and gap["count"] == 1 and gap["sample"]["missing"].iloc[0] == missing
    assert checks["The same close for 5 bars or more in a row"]["level"] == "problem"
    assert checks["Bars where the price never moved"]["count"] == 6
    assert checks["No volume"]["count"] == len(frame) and "Bars with zero volume" not in checks
    assert "about" in D.data_summary(frame, "1d") and "with volume" in D.data_summary(frame, "1d")


def test_a_dataset_is_named_and_renamed(tmp_path):
    pytest.importorskip("pyarrow")
    from openstrategy_studio import store

    frame = daily(50)
    D.store_dataset(tmp_path, "SPY", "1d", frame, "yahoo")
    D.store_dataset(tmp_path, "SPY_csv", "1d", frame * 1.01, "file:spy.csv", ticker="SPY")
    listing = store.list_datasets(tmp_path)
    assert list(listing["symbol"]) == ["SPY", "SPY_csv"] and list(listing["ticker"]) == ["SPY", "SPY"]
    assert D.dataset_name_problem("SPY", ["SPY", "SPY_csv"]) is None              # adds a timeframe to it
    assert "only in capitals" in D.dataset_name_problem("spy", ["SPY"])
    assert "cannot name a dataset" in D.dataset_name_problem("SPY/1d", [])
    D.rename_dataset(tmp_path, "SPY_csv", "1d", "SPY_file")
    assert sorted(store.list_datasets(tmp_path)["symbol"]) == ["SPY", "SPY_file"]
    pd.testing.assert_frame_equal(D.load_dataset(tmp_path, "SPY_file", "1d"), frame * 1.01, check_freq=False)
    assert not D.price_path(tmp_path, "SPY_csv", "1d").exists()
    with pytest.raises(ValueError, match="already stored"):
        D.rename_dataset(tmp_path, "SPY_file", "1d", "SPY")


def test_an_older_catalogue_gains_the_ticker_column(tmp_path):
    import sqlite3
    from openstrategy_studio import store

    with sqlite3.connect(tmp_path / "catalogue.db") as conn:
        conn.execute("CREATE TABLE catalogue (symbol TEXT NOT NULL, timeframe TEXT NOT NULL, source TEXT, "
                     "first_bar TEXT, last_bar TEXT, n_bars INTEGER, adjusted INTEGER, tz TEXT, imported_at TEXT, "
                     "notes TEXT, path TEXT, PRIMARY KEY (symbol, timeframe))")
        conn.execute("INSERT INTO catalogue (symbol, timeframe, source) VALUES ('EURUSD=X', '1d', 'yahoo')")
    listing = store.list_datasets(tmp_path)
    assert listing["ticker"].tolist() == ["EURUSD=X"]


def test_clean_synthetic_data_has_no_problems():
    data = make_ohlcv(300)
    frame = pd.DataFrame({field: data[field]["AAA"] for field in ("open", "high", "low", "close", "volume")})
    assert D.validate_ohlcv(frame, "1d") == []


def test_mt4_export_mapping():
    raw = D.read_upload((FIXTURES / "mt4_eurusd_h1.csv").read_bytes(), "mt4.csv", header=False)
    mapping = {"date": 0, "time": 1, "open": 2, "high": 3, "low": 4, "close": 5, "volume": 6,
               "date_format": "%Y.%m.%d %H:%M"}
    frame, _ = D.normalise_ohlcv(D.apply_mapping(raw, mapping, "1h"), "1h")
    assert len(frame) == 48
    assert frame.index[0] == pd.Timestamp("2024-01-02 00:00")
    assert frame.index[1] - frame.index[0] == pd.Timedelta(hours=1)
    assert D.validate_ohlcv(frame, "1h") == []


def test_unparseable_dates_are_reported():
    raw = pd.DataFrame({"d": ["2024-01-01", "not a date"], "o": [1, 1], "h": [1, 1], "l": [1, 1], "c": [1, 1]})
    with pytest.raises(ValueError, match="could not be read"):
        D.apply_mapping(raw, {"date": "d", "open": "o", "high": "h", "low": "l", "close": "c"})


def test_resample_up_is_left_labelled():
    data = make_ohlcv(60, freq="1h")
    daily = D.resample_frames(data, "1D")
    first_day = data["close"].index[0].normalize()
    assert daily["close"].index[0] == first_day
    assert daily["open"].iloc[0, 0] == data["open"].iloc[0, 0]
    assert daily["high"].iloc[0, 0] == data["high"].iloc[:24, 0].max()
    assert daily["volume"].iloc[0, 0] == data["volume"].iloc[:24, 0].sum()


def test_combined_lse_and_us_load_common_days(tmp_path):
    pytest.importorskip("pyarrow")
    for symbol, name in (("VOD.L", "lse_daily.csv"), ("AAPL", "us_daily.csv")):
        clean, info = load_fixture(name)
        if symbol == "AAPL":
            clean = clean.drop(clean.index[4])            # a US holiday the LSE traded through
        D.store_dataset(tmp_path, symbol, "1d", clean, source="csv", tz=info["tz"])
    frames = D.load_frames(tmp_path, ["VOD.L", "AAPL"], "1d", "2024-01-01", "2024-12-31", adjusted=("AAPL",))
    assert len(frames["close"]) == 11                     # the common days, not zero rows
    assert list(frames["close"].columns) == ["VOD.L", "AAPL"]
    # Adjusted symbol: close is adj_close, and high/low scaled by the same factor keep the bracket.
    assert (frames["high"]["AAPL"] >= frames["close"]["AAPL"]).all()
    assert (frames["low"]["AAPL"] <= frames["close"]["AAPL"]).all()


def test_missing_dataset_message(tmp_path):
    with pytest.raises(ValueError, match="Import it on the Data page"):
        D.load_dataset(tmp_path, "NOPE", "1d")


def test_yahoo_download_takes_the_pages_dates(monkeypatch):
    """The Data page passes datetime.date (st.date_input); vectorbt's own date parsing must
    accept it, and the "To" date must be included (yfinance's end is exclusive).

    No network: yfinance.Ticker is replaced, so vectorbt's YFData runs its real date
    handling and then gets a canned frame.
    """
    pytest.importorskip("vectorbt")
    yfinance = pytest.importorskip("yfinance")
    from datetime import date

    asked = {}

    def fake_ticker(symbol, **kwargs):
        def history(period=None, start=None, end=None, **kwargs):
            asked.update(start=start, end=end)
            return raw_frame(n=3).assign(**{"Adj Close": 10.0})
        return types.SimpleNamespace(history=history)

    monkeypatch.setattr(yfinance, "Ticker", fake_ticker)
    frame = D.download_yahoo("SPY", "1d", date(2024, 1, 2), date(2024, 1, 5))
    assert {"Open", "High", "Low", "Close", "Adj Close"} <= set(frame.columns)
    # Local midnight on each side; the end is the day after the "To" date.
    assert asked["start"].replace(tzinfo=None) == pd.Timestamp("2024-01-02")
    assert asked["end"].replace(tzinfo=None) == pd.Timestamp("2024-01-06")
