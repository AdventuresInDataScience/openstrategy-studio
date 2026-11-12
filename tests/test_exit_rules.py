"""Phase 3 exit templates: time stop, breakeven, step trailing.

Three layers:
1. The stop arithmetic, as plain functions.
2. A reference simulator that drives the callbacks bar by bar under the
   semantics the templates assume (adjust before the stop check; a trailing
   reference that follows completed bars' highs; gaps fill at the open). This
   checks the templates do what they promise IF vectorbt behaves that way.
3. Hand-built bars through vectorbtX (skipped without it): the phase 3 gate,
   which checks vectorbt really behaves that way.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import engine, exit_rules as X
from openstrategy_studio.compile import export_script
from openstrategy_studio.spec import apply_params, validate_spec

NAN = float("nan")


def test_protective_price_long_and_short():
    # Entry 100, initial stop 5 below, point 1.
    assert X.protective_price(100.0, 100.0, 5.0, 1.0, NAN, NAN, True) == 95.0
    assert X.protective_price(100.0, 104.0, 5.0, 1.0, 4.0, NAN, True) == 100.0      # breakeven reached
    assert X.protective_price(100.0, 103.9, 5.0, 1.0, 4.0, NAN, True) == 95.0       # not yet
    assert X.protective_price(100.0, 107.5, 5.0, 1.0, NAN, 2.0, True) == 101.0      # 3 steps of 2 up from 95
    assert X.protective_price(100.0, 92.5, 5.0, 1.0, NAN, 2.0, False) == 99.0       # short mirror
    assert X.protective_price(100.0, 96.0, 5.0, 1.0, 4.0, NAN, False) == 100.0
    assert X.stop_fraction(95.0, 100.0, True) == pytest.approx(0.05)
    assert X.stop_fraction(105.0, 100.0, False) == pytest.approx(0.05)
    assert X.stop_fraction(101.0, 100.0, True) == 0.0                                # never negative


def test_favourable_price_ignores_the_current_bar():
    high = np.array([[100.0], [103.0], [110.0]])
    low = np.array([[99.0], [97.0], [90.0]])
    assert X.favourable_price(high, low, 0, 2, 0, True, 100.0) == 103.0
    assert X.favourable_price(high, low, 0, 2, 0, False, 100.0) == 97.0
    assert X.favourable_price(high, low, 1, 1, 0, True, 100.0) == 100.0


def simulate(opens, highs, lows, entry_bar, side, stop_points, settings, point=1.0):
    """Drive the callbacks bar by bar. Returns (exit bar, exit price).

    The callbacks are called through `.py_func`, their plain-Python bodies: numba
    cannot type a SimpleNamespace context, and vectorbt hands them a NamedTuple
    (AdjustSLContext / AdjustTPContext) with the same field names.
    """
    adjust_sl = getattr(X.adjust_sl_nb, "py_func", X.adjust_sl_nb)
    adjust_tp = getattr(X.adjust_tp_nb, "py_func", X.adjust_tp_nb)
    o, h, l = (np.asarray(v, dtype=float).reshape(-1, 1) for v in (opens, highs, lows))
    long = side == "long"
    entry = o[entry_bar, 0]
    sl_args = (o, h, l, np.array([point]), np.array([float(stop_points)]), settings.get("time_bars", NAN),
               settings.get("breakeven_points", NAN), settings.get("step_points", NAN),
               np.full(1, -1, dtype=np.int64), np.full(1, np.nan))
    stop = stop_points * point / entry
    trail = False
    reference = entry
    target = 1e9
    for i in range(entry_bar, len(o)):
        context = SimpleNamespace(i=i, col=0, position_now=1.0 if long else -1.0, init_i=entry_bar, init_price=entry,
                                  curr_price=reference, curr_stop=stop, curr_trail=trail)
        stop, trail = adjust_sl(context, *sl_args)
        tp_context = SimpleNamespace(i=i, col=0, position_now=context.position_now, init_i=entry_bar,
                                     init_price=entry, curr_stop=target)
        target = adjust_tp(tp_context, o, settings.get("time_bars", NAN))
        base = reference if trail else entry
        stop_price = base * (1 - stop) if long else base * (1 + stop)
        target_price = entry * (1 + target) if long else entry * (1 - target)
        if long:
            if o[i, 0] <= stop_price or o[i, 0] >= target_price:
                return i, o[i, 0]
            if l[i, 0] <= stop_price:
                return i, stop_price
        else:
            if o[i, 0] >= stop_price or o[i, 0] <= target_price:
                return i, o[i, 0]
            if h[i, 0] >= stop_price:
                return i, stop_price
        if trail:
            reference = max(reference, h[i, 0]) if long else min(reference, l[i, 0])
    return None, None


def test_reference_time_stop_exits_at_the_open():
    flat = [100.0] * 10
    for opens in ([100, 100, 101, 102, 103, 104, 105, 106, 107, 108], [100, 100, 99, 98, 97, 97, 97, 97, 97, 97]):
        highs = [value + 0.5 for value in opens]
        lows = [value - 0.5 for value in opens]
        bar, price = simulate(opens, highs, lows, entry_bar=1, side="long", stop_points=20,
                              settings={"time_bars": 4})
        assert bar == 5 and price == opens[5], (opens, bar, price)
    bar, _ = simulate(flat, [101.0] * 10, [99.0] * 10, entry_bar=1, side="short", stop_points=20, settings={"time_bars": 3})
    assert bar == 4


def test_reference_breakeven_then_back_to_entry():
    opens = [100, 100, 102, 104, 105, 103, 101, 100.5, 98, 97]
    highs = [o + 0.6 for o in opens]
    lows = [o - 0.6 for o in opens]
    bar, price = simulate(opens, highs, lows, entry_bar=1, side="long", stop_points=5, settings={"breakeven_points": 4})
    # Highs reach 104.6 at bar 3, so from bar 4 the stop sits at the entry (100);
    # bar 7's low (99.9) touches it.
    assert bar == 7 and price == pytest.approx(100.0)
    bar, _ = simulate(opens, highs, lows, entry_bar=1, side="long", stop_points=5, settings={})
    assert bar is None                                       # without the template, 95 is never touched
    gapped = opens[:7] + [99.5, 98, 97]
    bar, price = simulate(gapped, [o + 0.6 for o in gapped], [o - 0.6 for o in gapped], entry_bar=1, side="long",
                          stop_points=5, settings={"breakeven_points": 4})
    assert bar == 7 and price == pytest.approx(99.5)         # a gap below the stop fills at the open


def test_reference_step_trailing_locks_in_steps():
    opens = [100, 100, 101, 102, 103, 104, 105, 106, 104, 102, 100, 98]
    highs = [o + 0.5 for o in opens]
    lows = [o - 0.5 for o in opens]
    bar, price = simulate(opens, highs, lows, entry_bar=1, side="long", stop_points=4, settings={"step_points": 2})
    # Best completed high 106.5 = 3 whole steps of 2 above 100, so the stop is 96 + 6 = 102.
    assert price == pytest.approx(102.0)
    assert opens[bar - 1] > 102 >= lows[bar]


def test_reference_short_step_trailing():
    opens = [100, 100, 99, 98, 97, 96, 95, 94, 96, 98, 100]
    highs = [o + 0.5 for o in opens]
    lows = [o - 0.5 for o in opens]
    _, price = simulate(opens, highs, lows, entry_bar=1, side="short", stop_points=4, settings={"step_points": 2})
    assert price == pytest.approx(98.0)                      # lows reach 93.5: 3 steps down from 104


def exits_spec(ma_spec, **exits):
    spec = copy.deepcopy(ma_spec)
    spec["risk"]["exits"] = exits
    return spec


def test_validation(ma_spec):
    assert validate_spec(exits_spec(ma_spec, time_bars=20, breakeven_points=40, step_points=25)) == []
    spec = exits_spec(ma_spec, time_bars=2.5)
    assert any("whole number" in p for p in validate_spec(spec))
    spec = exits_spec(ma_spec, breakeven_points=10)
    spec["risk"]["stop"] = {"kind": "trailing", "points": 30}
    assert any("need a fixed stop" in p for p in validate_spec(spec))
    assert any("Unknown extra exit" in p for p in validate_spec(exits_spec(ma_spec, nope=1)))


def test_call_plan_wiring(ma_spec):
    plan = engine.call_plan(apply_params(exits_spec(ma_spec, breakeven_points=40)))
    assert plan["args"]["adjust_sl_func_nb"] == ("var", "adjust_sl_func_nb")
    assert "adjust_tp_func_nb" not in plan["args"]
    spec = exits_spec(ma_spec, time_bars=10)
    spec["risk"]["target"] = None
    plan = engine.call_plan(apply_params(spec))
    assert plan["args"]["adjust_tp_func_nb"] == ("var", "adjust_tp_func_nb")
    assert plan["args"]["target_points"] == ("lit", X.FAR_TARGET_POINTS)
    assert engine.call_plan(apply_params(ma_spec))["args"].get("adjust_sl_func_nb") is None


def test_optimisable_exit_settings(ma_spec):
    spec = exits_spec(ma_spec, time_bars={"value": 10, "opt": {"min": 5, "max": 40, "int": True}})
    assert X.exit_settings(apply_params(spec, {"risk.exits.time_bars": 25}))["time_bars"] == 25.0


def test_export_includes_the_templates(ma_spec):
    import ast
    source = export_script(exits_spec(ma_spec, time_bars=10, step_points=20))
    ast.parse(source)
    assert "def adjust_sl_nb(" in source and "def adjust_tp_nb(" in source
    assert "@njit(cache=False)" in source and "adjust_sl_args = (_o, _h, _l" in source


# ---------------------------------------------------------------------------
# The phase 3 gate: hand-built bars through vectorbtX
# ---------------------------------------------------------------------------

def bars(opens, spread=0.5):
    index = pd.bdate_range("2024-01-01", periods=len(opens))
    frame = lambda values: pd.DataFrame({"X": np.asarray(values, dtype=float)}, index=index)
    return {"open": frame(opens), "high": frame([o + spread for o in opens]), "low": frame([o - spread for o in opens]),
            "close": frame(opens), "volume": frame([1000.0] * len(opens))}


def gate_spec(exits, stop_points, side="long"):
    # Zero spread: the gate compares the ORDER OF OPERATIONS with the reference
    # simulator, which models no costs. Costs on stop exits are pinned separately by
    # tests/test_engine.py::test_spread_is_paid_on_every_fill.
    code = "out = close * 0.0\nout.iloc[0] = 1.0\nreturn out"
    rules = {"long_entry": None, "long_exit": None, "short_entry": None, "short_exit": None}
    rules[f"{side}_entry"] = {"left": {"feature": "pulse"}, "op": ">", "right": {"const": 0.5}}
    return {"version": 1, "name": "gate", "data": {"symbols": ["X"], "timeframe": "1d", "start": "2000-01-01",
                                                    "end": "2100-01-01"},
            "instruments": {"X": {"mode": "equity", "point_size": 1.0, "quote_currency": "USD", "cost_currency": "USD",
                                  "spread_points": 0.0}},
            "features": {"pulse": {"block": "python", "code": code}}, "rules": rules,
            "risk": {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": stop_points}, "exits": exits},
            "account": {"init_cash": 10_000, "freq": "D"}}


def first_exit(run):
    trades = run["pf"].trades.records_readable
    when = next(c for c in trades.columns if "exit" in str(c).lower() and "timestamp" in str(c).lower())
    price = next(c for c in trades.columns if "exit" in str(c).lower() and "price" in str(c).lower())
    return trades.iloc[0][when], float(trades.iloc[0][price])


def test_gate_matches_the_reference_simulator():
    pytest.importorskip("vectorbtx")
    cases = [
        ({"time_bars": 4}, 20, [100, 100, 101, 102, 103, 104, 105, 106, 107, 108], "long"),
        ({"time_bars": 4}, 20, [100, 100, 99, 98, 97, 97, 97, 97, 97, 97], "long"),
        ({"breakeven_points": 4}, 5, [100, 100, 102, 104, 105, 103, 101, 100.5, 98, 97], "long"),
        ({"step_points": 2}, 4, [100, 100, 101, 102, 103, 104, 105, 106, 104, 102, 100, 98], "long"),
        ({"step_points": 2}, 4, [100, 100, 99, 98, 97, 96, 95, 94, 96, 98, 100], "short"),
    ]
    for exits, stop_points, opens, side in cases:
        data = bars(opens)
        run = engine.run_backtest(gate_spec(exits, stop_points, side), data)
        when, price = first_exit(run)
        bar, expected = simulate(opens, data["high"]["X"].tolist(), data["low"]["X"].tolist(), entry_bar=1,
                                 side=side, stop_points=stop_points, settings=exits)
        assert when == data["close"].index[bar], (exits, side, when, bar)
        assert price == pytest.approx(expected), (exits, side, price, expected)


def test_extra_exits_say_the_entry_bar_is_not_checked():
    """docs/UPSTREAM.md issue 14 (open). vectorbtX checks a stop on its entry bar by
    splitting each bar in two, which it cannot do with a custom callback, so with an
    extra exit active it falls back to whole bars and warns. This pins the warning, so
    the gap stays visible; when vectorbtX closes it, this test fails: delete it then."""
    pytest.importorskip("vectorbtx")
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.run_backtest(gate_spec({"time_bars": 4}, 20), bars([100.0] * 10))
    assert any("NOT checked on the bar a position is entered" in str(w.message) for w in caught)
