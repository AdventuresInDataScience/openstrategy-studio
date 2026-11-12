"""Test 2: no lookahead. Compiling on the full data and on data truncated at bar k
gives identical signals up to k, for every block, rule operator and filter."""

import copy

import pandas as pd
import pytest

from openstrategy_studio import blocks, compile as C
from openstrategy_studio.blocks import BLOCKS, block_defaults
from openstrategy_studio.spec import apply_params

from conftest import make_ohlcv

# Dense cut points: a leak only shows when the truncated value flips a condition,
# so a handful of cuts can miss it (a missing higher-timeframe shift did, with three).
CUTS = tuple(range(40, 400, 7))


def real_library(name):
    try:
        module = __import__(name)
    except ImportError:
        return False
    return not getattr(module, "STUB", False)


def available_blocks():
    names = ["momentum", "highest", "lowest", "zscore", "volatility", "python"]
    if real_library("vectorbt"):
        names += ["ma", "mstd", "bbands", "rsi", "stoch", "macd", "atr", "obv"]
    if real_library("ta_patterns"):
        names.append("pattern")
    return names


def signal_spec(features, entry, timeframe="1d"):
    return {"version": 1, "name": "t", "data": {"symbols": ["EURUSD=X"], "timeframe": timeframe,
                                                 "start": "2020-01-01", "end": "2030-01-01"},
            "instruments": {"EURUSD=X": {"mode": "spreadbet"}}, "features": features,
            "rules": {"long_entry": entry, "long_exit": None, "short_entry": copy.deepcopy(entry), "short_exit": None},
            "risk": {}, "account": {"init_cash": 1000, "freq": "D"}}


def assert_causal(spec, data):
    spec = apply_params(spec)
    full = C.compile_signals(spec, data)
    for cut in CUTS:
        part = C.compile_signals(spec, {field: frame.iloc[:cut] for field, frame in data.items()})
        for key in ("entries", "short_entries"):
            pd.testing.assert_frame_equal(full[key].iloc[:cut], part[key], obj=f"{key} cut at {cut}")


def test_every_block_is_causal(ohlcv):
    for name in available_blocks():
        feature = block_defaults(name)
        output = BLOCKS[name]["outputs"][0]
        entry = {"left": {"feature": "x", "output": output}, "op": ">",
                 "right": {"feature": "x", "output": output, "shift": 1}}
        assert_causal(signal_spec({"x": feature}, entry), ohlcv)


OPERATOR_RULES = {
    ">": {"left": {"feature": "m"}, "op": ">", "right": {"const": 0}},
    "<=": {"left": {"price": "close"}, "op": "<=", "right": {"feature": "hi", "shift": 1}},
    "between": {"left": {"feature": "z"}, "op": "between", "lower": {"const": -0.5}, "upper": {"const": 0.5}},
    "rising": {"left": {"feature": "m"}, "op": "rising", "bars": 3},
    "falling": {"left": {"price": "close"}, "op": "falling", "bars": 1},
    "leaf shift": {"left": {"feature": "m"}, "op": ">", "right": {"const": 0}, "shift": 2},
    "not/any": {"not": {"any": [{"left": {"feature": "m"}, "op": "<", "right": {"const": 0}},
                                {"left": {"feature": "z"}, "op": ">", "right": {"const": 1}}]}},
    "weekday": {"all": [{"filter": "weekday", "days": [0, 2, 4]},
                        {"left": {"feature": "m"}, "op": ">", "right": {"const": 0}}]},
    "month": {"all": [{"filter": "month", "months": [1, 2, 3, 10]},
                      {"left": {"feature": "m"}, "op": ">", "right": {"const": 0}}]},
    "higher timeframe": {"all": [{"filter": "higher_timeframe", "timeframe": "1w",
                                  "left": {"feature": "m"}, "op": ">", "right": {"const": 0}},
                                 {"left": {"feature": "z"}, "op": ">", "right": {"const": 0}}]},
    "higher timeframe monthly": {"filter": "higher_timeframe", "timeframe": "1M",
                                 "left": {"price": "close"}, "op": ">", "right": {"feature": "hi"}},
    "higher timeframe candle": {"filter": "higher_timeframe", "timeframe": "1w",
                                "left": {"price": "close"}, "op": ">", "right": {"price": "open"}},
}
FEATURES = {"m": dict(block_defaults("momentum"), window=10), "z": dict(block_defaults("zscore"), window=15),
            "hi": dict(block_defaults("highest"), window=5)}


def test_every_operator_is_causal(ohlcv):
    for name, entry in OPERATOR_RULES.items():
        assert_causal(signal_spec(FEATURES, entry), ohlcv)


def test_crossings_are_causal(ohlcv):
    pytest.importorskip("vectorbt")
    for op in ("crosses_above", "crosses_below"):
        entry = {"left": {"feature": "m"}, "op": op, "right": {"const": 0}}
        assert_causal(signal_spec(FEATURES, entry), ohlcv)


def test_pattern_leaf_is_causal(ohlcv, monkeypatch):
    # A stand-in detector that, like ta_patterns, uses only bars up to the current one.
    monkeypatch.setattr(blocks, "pattern_function",
                        lambda module, name: (lambda o, h, l, c, **kw: ((c > o) & (o > 0)).astype("int8")))
    entry = {"pattern": "anything", "module": "candle", "side": "bullish", "params": {}}
    assert_causal(signal_spec({}, entry), ohlcv)


def test_intraday_session_and_daily_filter_are_causal():
    data = make_ohlcv(600, ("EURUSD=X",), freq="1h", seed=5)
    entry = {"all": [{"filter": "session", "from": "07:00", "to": "16:00"},
                     {"filter": "higher_timeframe", "timeframe": "1d", "left": {"feature": "m"}, "op": ">",
                      "right": {"const": 0}}]}
    assert_causal(signal_spec(FEATURES, entry, timeframe="1h"), data)


def test_higher_timeframe_uses_only_completed_bars(ohlcv):
    """The weekly value on any day is the PREVIOUS completed week's, never the current one."""
    spec = apply_params(signal_spec({}, {"filter": "higher_timeframe", "timeframe": "1w",
                                         "left": {"price": "close"}, "op": ">", "right": {"price": "open"}}))
    signal = C.compile_signals(spec, ohlcv)["entries"]["EURUSD=X"]
    close, open_ = ohlcv["close"]["EURUSD=X"], ohlcv["open"]["EURUSD=X"]
    weekly_up = close.resample("W-SUN", label="left", closed="left").last() > \
        open_.resample("W-SUN", label="left", closed="left").first()
    for day in signal.index[10:60]:
        week_start = weekly_up.index[weekly_up.index <= day][-1]
        previous = weekly_up.index[weekly_up.index < week_start]
        expected = bool(weekly_up.loc[previous[-1]]) if len(previous) else False
        assert bool(signal.loc[day]) == expected, day


def test_the_compiler_shifts_no_signal(ohlcv):
    """The front door shifts all four signal arrays to the next open (vectorbtX issue 1
    fixed). The compiler must shift none of them, or they would be delayed two bars."""
    spec = apply_params(signal_spec(FEATURES, OPERATOR_RULES[">"]))
    compiled = C.compile_signals(spec, ohlcv)
    # signal_spec gives the long and the short entry the same rule, so if neither is
    # shifted the two frames are identical.
    assert compiled["entries"].to_numpy().any()
    pd.testing.assert_frame_equal(compiled["short_entries"], compiled["entries"])


def test_weights_are_causal(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = apply_params(momentum_spec)
    full = C.compile_weights(spec, ohlcv_multi)
    for cut in CUTS:
        part = C.compile_weights(spec, {field: frame.iloc[:cut] for field, frame in ohlcv_multi.items()})
        pd.testing.assert_frame_equal(full["weights"].iloc[:cut], part["weights"])
