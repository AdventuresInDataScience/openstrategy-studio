"""Test 10: data normalisation, validation and loading."""

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
    assert problems["High is below the open or close"]["count"] == 1
    assert problems["Zero or negative prices"]["count"] == 1
    assert len(problems["High is below the open or close"]["sample"]) == 1
    pd.testing.assert_frame_equal(frame, before)                                   # never repairs


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
