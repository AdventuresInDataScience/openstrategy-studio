"""Test 2: no lookahead. Compiling on the full data and on data truncated at bar k
gives identical signals up to k, for features, every comparison, every logic
operator, every filter and the order timing.

Every catalogue entry is checked on its own in tests/test_indicators.py; this file
checks them as the compiler uses them: read by other features, compared, combined.
"""

import copy

import pandas as pd
import pytest

from openstrategy_studio import compile as C, indicators
from openstrategy_studio.spec import apply_params

from conftest import make_ohlcv

# Dense cut points: a leak only shows when the truncated value flips a condition,
# so a handful of cuts can miss it (a missing higher-timeframe shift did, with three).
CUTS = tuple(range(40, 400, 7))


def feature(indicator_id, inputs=None, **params):
    made = indicators.new_feature(indicator_id)
    made["params"].update(params)
    made["inputs"].update(inputs or {})
    return made


def rule(conditions, logic=None):
    """A rule from conditions given in order: lettered A, B, C ... and joined with "and" unless told."""
    lettered = {chr(ord("A") + index): condition for index, condition in enumerate(conditions)}
    return {"conditions": lettered, "logic": logic or " and ".join(lettered)}


def signal_spec(features, entry, timeframe="1d", orders=None):
    return {"version": 2, "name": "t", "setup": {"kind": "signals", "trading": "spreadbet"},
            "data": {"symbols": ["EURUSD=X"], "timeframe": timeframe, "start": "2020-01-01", "end": "2030-01-01"},
            "instruments": {"EURUSD=X": {}}, "features": features,
            "rules": {"long_entry": entry, "long_exit": None, "short_entry": copy.deepcopy(entry), "short_exit": None},
            "risk": {"orders": orders or {}}, "account": {"init_cash": 1000, "freq": "D"}}


def assert_causal(spec, data, keys=("entries", "short_entries")):
    spec = apply_params(spec)
    full = C.compile_signals(spec, data)
    for cut in CUTS:
        part = C.compile_signals(spec, {field: frame.iloc[:cut] for field, frame in data.items()})
        for key in keys:
            pd.testing.assert_frame_equal(full[key].iloc[:cut], part[key], obj=f"{key} cut at {cut}")


FEATURES = {"m": feature("talib:ROCP", timeperiod=10), "z": feature("pandas_ta:ZSCORE", length=15),
            "hi": feature("talib:MAX", {"close": "high"}, timeperiod=5)}


def above_zero(name):
    return {"left": {"feature": name}, "op": ">", "right": {"const": 0.0}}


def test_features_read_by_features_are_causal(ohlcv):
    """Chains as the Build page makes them: an RSI of an SMA, a band around an EMA, bars since a pattern."""
    pytest.importorskip("talib")
    features = {"ma": feature("talib:SMA", timeperiod=10),
                "rsi": feature("talib:RSI", {"close": {"feature": "ma"}}, timeperiod=7),
                "ema": feature("talib:EMA", timeperiod=12),
                "band": feature("studio:percent_band", {"source": {"feature": "ema"}}, percent=1.0),
                "kc": feature("talib:KC"),
                "prev": feature("studio:previous_period", period="week")}
    entry = rule([{"left": {"feature": "rsi"}, "op": ">", "right": {"const": 50.0}},
                  {"left": {"price": "close"}, "op": "between", "lower": {"feature": "band", "output": "lower"},
                   "upper": {"feature": "band", "output": "upper"}},
                  {"left": {"price": "close"}, "op": ">", "right": {"feature": "kc", "output": "lowerband"}},
                  {"left": {"price": "close"}, "op": ">", "right": {"feature": "prev", "output": "low"}}],
                 logic="(A or B) and (C or D)")
    assert_causal(signal_spec(features, entry), ohlcv)


OPERATOR_RULES = {
    ">": rule([above_zero("m")]),
    "<=": rule([{"left": {"price": "close"}, "op": "<=", "right": {"feature": "hi", "shift": 1}}]),
    "between": rule([{"left": {"feature": "z"}, "op": "between", "lower": {"const": -0.5}, "upper": {"const": 0.5}}]),
    "rising": rule([{"left": {"feature": "m"}, "op": "rising", "bars": 3}]),
    "falling": rule([{"left": {"price": "close"}, "op": "falling", "bars": 1}]),
    "bars ago": rule([{"left": {"feature": "m", "shift": 2}, "op": ">", "right": {"const": 0.0}}]),
    "not (or)": rule([{"left": {"feature": "m"}, "op": "<", "right": {"const": 0.0}},
                      {"left": {"feature": "z"}, "op": ">", "right": {"const": 1.0}}], logic="not (A or B)"),
    "xor nand nor xnor": rule([above_zero("m"), above_zero("z"), {"left": {"feature": "z"}, "op": ">",
                                                                    "right": {"const": 1.0}}],
                              logic="((A xor B) nand C) nor (A xnor C)"),
    "weekday": rule([{"filter": "weekday", "days": [0, 2, 4]}, above_zero("m")]),
    "month": rule([{"filter": "month", "months": [1, 2, 3, 10]}, above_zero("m")]),
    "higher timeframe": rule([{"filter": "higher_timeframe", "timeframe": "1w", **above_zero("m")},
                              above_zero("z")]),
    "higher timeframe monthly": rule([{"filter": "higher_timeframe", "timeframe": "1M", "left": {"price": "close"},
                                       "op": ">", "right": {"feature": "hi"}}]),
    "higher timeframe candle": rule([{"filter": "higher_timeframe", "timeframe": "1w", "left": {"price": "close"},
                                      "op": ">", "right": {"price": "open"}}]),
}


def test_every_operator_is_causal(ohlcv):
    for name, entry in OPERATOR_RULES.items():
        assert_causal(signal_spec(FEATURES, entry), ohlcv)


def test_crossings_are_causal(ohlcv):
    pytest.importorskip("vectorbt")
    for op in ("crosses_above", "crosses_below"):
        entry = rule([{"left": {"feature": "m"}, "op": op, "right": {"const": 0.0}}])
        assert_causal(signal_spec(FEATURES, entry), ohlcv)


def test_pattern_condition_is_causal(ohlcv, monkeypatch):
    # A stand-in detector that, like ta_patterns, uses only bars up to the current one.
    monkeypatch.setattr(indicators, "run_pattern",
                        lambda module, name, params, o, h, l, c, v: ((c > o) & (o > 0)).astype("int8"))
    features = {"hammer": indicators.new_feature("pattern:candle:hammer")}
    assert_causal(signal_spec(features, rule([{"pattern": "hammer", "side": "bullish"}])), ohlcv)


def test_intraday_session_and_daily_filter_are_causal():
    data = make_ohlcv(600, ("EURUSD=X",), freq="1h", seed=5)
    entry = rule([{"filter": "session", "from": "07:00", "to": "16:00"},
                  {"filter": "higher_timeframe", "timeframe": "1d", **above_zero("m")}])
    assert_causal(signal_spec(FEATURES, entry, timeframe="1h"), data)


def test_order_timing_is_causal_and_on_the_signal_bar():
    """Good for the day and flat by the day's end read the dates of the next two bars, never prices.

    ⚠️ A trader knows those dates in advance from the session calendar, but a backtest
    reads them from the data, so a truncated series treats its own last bar as a day's
    last bar (and the bar before it as the second-to-last). The comparison leaves those
    two bars out; everything earlier must agree.
    """
    data = make_ohlcv(600, ("EURUSD=X",), freq="1h", seed=5)
    spec = apply_params(signal_spec(FEATURES, OPERATOR_RULES[">"], timeframe="1h",
                                    orders={"good_for_day": True, "flat_by_day_end": True}))
    full = C.compile_signals(spec, data)
    for cut in CUTS:
        part = C.compile_signals(spec, {field: frame.iloc[:cut] for field, frame in data.items()})
        for key in ("entries", "exits"):
            pd.testing.assert_frame_equal(full[key].iloc[:cut - 2], part[key].iloc[:-2], obj=f"{key} cut at {cut}")
    index = data["close"].index
    last, second_last = C.day_boundaries(index)
    entries, exits = full["entries"]["EURUSD=X"].to_numpy(), full["exits"]["EURUSD=X"].to_numpy()
    # No entry on a day's last two bars (its fill would land on the last bar or the next day) ...
    assert not (entries & (last | second_last)).any()
    # ... and an exit on every day's second-to-last bar, which fills at the last bar's open.
    assert exits[second_last].all()


def test_higher_timeframe_uses_only_completed_bars(ohlcv):
    """The weekly value on any day is the PREVIOUS completed week's, never the current one."""
    spec = apply_params(signal_spec({}, rule([{"filter": "higher_timeframe", "timeframe": "1w",
                                                "left": {"price": "close"}, "op": ">", "right": {"price": "open"}}])))
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


def test_entry_priority_score_is_causal_and_unshifted(shared_spec, ohlcv_stocks):
    """A shared account's priority score: no lookahead, and on the signals' own bar. The front
    door shifts it with the signals, so a shift here would rank entries by a stale score."""
    pytest.importorskip("vectorbt")
    spec = apply_params(shared_spec)
    full = C.compile_signals(spec, ohlcv_stocks)
    close = ohlcv_stocks["close"]
    pd.testing.assert_frame_equal(full["entry_priority"], close / close.shift(63) - 1)
    for cut in CUTS:
        part = C.compile_signals(spec, {field: frame.iloc[:cut] for field, frame in ohlcv_stocks.items()})
        pd.testing.assert_frame_equal(full["entry_priority"].iloc[:cut], part["entry_priority"],
                                      obj=f"entry_priority cut at {cut}")


def test_weights_are_causal(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = apply_params(momentum_spec)
    full = C.compile_weights(spec, ohlcv_multi)
    for cut in CUTS:
        part = C.compile_weights(spec, {field: frame.iloc[:cut] for field, frame in ohlcv_multi.items()})
        pd.testing.assert_frame_equal(full["weights"].iloc[:cut], part["weights"])
