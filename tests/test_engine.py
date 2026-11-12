"""Tests 1, 3 and 4 (need vectorbtX installed), plus the pure call plan."""


import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import engine
from openstrategy_studio.spec import apply_params

from conftest import make_ohlcv


# ---------------------------------------------------------------------------
# The call plan (pure: runs without vectorbtX)
# ---------------------------------------------------------------------------

def without_pair(ma_spec):
    """The MA-cross spec as a non-FX spread bet, which has no base currency and so stays single."""
    spec = apply_params(ma_spec)
    instrument = spec["instruments"]["EURUSD=X"]
    instrument.pop("base_currency")
    instrument["asset_class"] = "stocks"
    return spec


def test_single_symbol_plan(ma_spec):
    plan = engine.call_plan(without_pair(ma_spec))
    args = plan["args"]
    assert plan["front_door"] == "SingleBacktest" and plan["single"]
    assert args["close"] == ("var", "close") and args["open"] == ("var", "open_")
    assert args["mode"] == ("lit", "spreadbet") and "instruments" not in args
    assert args["risk_pct"] == ("lit", 0.01)
    assert args["stop_points"] == ("lit", 60) and "trail_points" not in args
    assert args["target_points"] == ("lit", 120)
    assert args["short_entries"] == ("var", "short_entries")      # always the four-array form
    assert args["freq"] == ("lit", "D") and args["rates"] == ("var", "rates")
    assert args["upon_opposite_entry"] == ("lit", "reverse")
    assert "fees" not in args and "direction" not in args
    # The two vectorbt stop defaults the Studio overrides (see engine.STOP_ENTRY_PRICE).
    assert args["stop_entry_price"] == ("lit", "price") and args["stop_exit_price"] == ("lit", "stopmarket")


def test_stop_overrides_only_with_a_stop(momentum_spec):
    args = engine.call_plan(apply_params(momentum_spec))["args"]
    assert "stop_entry_price" not in args and "stop_exit_price" not in args


def test_fx_pair_routes_to_portfolio(ma_spec):
    """Phase 0 finding (VERIFY 5): a flat base_currency= is read as the ACCOUNT currency.

    So an FX pair keeps its base currency on the instrument, via PortfolioBacktest.
    """
    plan = engine.call_plan(apply_params(ma_spec))
    args = plan["args"]
    assert plan["front_door"] == "PortfolioBacktest" and not plan["single"]
    assert args["instruments"][1]["EURUSD=X"]["base_currency"] == "EUR"
    assert "base_currency" not in args and "mode" not in args


def test_fx_carry_uses_the_pair(ma_spec, ohlcv):
    """No "carry active but no currency pair" warning, and carry is the EUR/USD differential."""
    pytest.importorskip("vectorbtx")
    import warnings

    fast = {"features.fast.window": 5, "features.slow.window": 20}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = engine.run_backtest(ma_spec, ohlcv, params=fast)
    assert not [w for w in caught if "no currency pair" in str(w.message)]
    # One night's charge on a long: notional x (EUR 3% - USD 4.5%) / 360, before markup.
    costs = pd.DataFrame(run["pf"].holding_costs).iloc[:, 0]
    orders = run["pf"].orders.records_readable
    first = orders.iloc[0]
    if first["Side"] == "Buy":
        night = costs.loc[first["Timestamp"]]
        notional = first["Size"] * first["Price"]
        assert night == pytest.approx(-notional * 0.015 / 360, rel=0.05)


def test_trailing_stop_replaces_fixed(ma_spec):
    spec = apply_params(ma_spec)
    spec["risk"]["stop"] = {"kind": "trailing", "points": 40}
    args = engine.call_plan(spec)["args"]
    assert args["trail_points"] == ("lit", 40) and "stop_points" not in args


def test_portfolio_plan(momentum_spec):
    plan = engine.call_plan(apply_params(momentum_spec))
    args = plan["args"]
    assert plan["front_door"] == "PortfolioBacktest" and not plan["single"]
    assert set(args["instruments"][1]) == set(momentum_spec["data"]["symbols"])
    assert args["weights"] == ("var", "weights") and args["schedule"] == ("var", "schedule")
    assert "entries" not in args and "risk_pct" not in args


def test_reporting_currency_routes_to_portfolio(ma_spec):
    spec = without_pair(ma_spec)
    spec["account"]["base_currency"] = "GBP"
    assert engine.call_plan(spec)["front_door"] == "PortfolioBacktest"


def test_price_field_is_not_sent(ma_spec):
    for spec in (apply_params(ma_spec), without_pair(ma_spec)):
        spec["instruments"]["EURUSD=X"]["price_field"] = "close"
        args = engine.call_plan(spec)["args"]
        assert "price_field" not in args
        assert "price_field" not in args.get("instruments", ("lit", {}))[1].get("EURUSD=X", {})


def test_benchmark_is_the_market_unlevered(ma_spec, ohlcv, momentum_spec, ohlcv_multi):
    """Phase 0 finding (VERIFY 2.7): the buy-and-hold line is the market, not a hold at buying power.

    A front-door call with no signals held the 30x FX spread bet at 30x. The benchmark is
    now vectorbt's own: equal-weight closes, unlevered, from the first bar.
    """
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis

    fast = {"features.fast.window": 5, "features.slow.window": 20}      # 50/200 never trades in 400 bars
    for spec, data, params in ((ma_spec, ohlcv, fast), (momentum_spec, ohlcv_multi, None)):
        run = engine.run_backtest(spec, data, params=params)
        results = analysis.extract_results(run, benchmark=True)
        benchmark = results["series"]["benchmark"]
        close = run["data"]["close"]
        market = float((close.iloc[-1] / close.iloc[0]).mean() - 1)
        assert benchmark.iloc[-1] / benchmark.iloc[0] - 1 == pytest.approx(market, rel=1e-9)
        returns = results["stats"]["returns"]["all"]
        for label in ("Beta", "Alpha", "Treynor Ratio"):
            assert returns[label] is not None, f"{label} must be finite against the market"
        assert "benchmark" not in analysis.extract_results(run)["series"]


# ---------------------------------------------------------------------------
# Test 1: the golden strategy
# ---------------------------------------------------------------------------

def assert_stats_equal(left, right):
    pd.testing.assert_series_equal(left, right, check_exact=False, rtol=1e-9, atol=1e-12)


def test_golden_ma_cross(ma_spec, ohlcv):
    """The compiled MA-cross spec against the hand-written vectorbtX call.

    The owner will supply real numbers from examples/strategies/01_ma_cross_spreadbet_fx.py
    as text; until then this compares with the call written out by hand.
    """
    vbt = pytest.importorskip("vectorbtx")
    vectorbt = pytest.importorskip("vectorbt")
    run = engine.run_backtest(ma_spec, ohlcv)

    symbol = "EURUSD=X"
    # Frames, not Series: an FX pair goes through PortfolioBacktest so its base
    # currency stays on the instrument (test_fx_pair_routes_to_portfolio).
    close, open_ = ohlcv["close"][[symbol]], ohlcv["open"][[symbol]]
    high, low = ohlcv["high"][[symbol]], ohlcv["low"][[symbol]]
    fast = vectorbt.MA.run(close[symbol], window=50)
    slow = vectorbt.MA.run(close[symbol], window=200)
    entries = fast.ma_crossed_above(slow).to_frame(symbol)
    exits = fast.ma_crossed_below(slow).to_frame(symbol)
    no_shorts = pd.DataFrame(False, index=close.index, columns=[symbol])
    rates = vbt.x.static_rate_table(["EUR", "USD"], close.index, {"EUR": 0.03, "USD": 0.045})
    instrument = dict(mode="spreadbet", asset_class="fx", point_size=0.0001, base_currency="EUR",
                      quote_currency="USD", min_size=1.0, increment=0.01, rounding="floor", spread_points=0.9,
                      slippage=0.0002, leverage=30)
    hand = vbt.x.PortfolioBacktest(
        close=close, open=open_, high=high, low=low,
        entries=entries, exits=exits, short_entries=no_shorts, short_exits=no_shorts,
        instruments={symbol: instrument},
        risk_pct=0.01, stop_points=60, target_points=120, rates=rates,
        init_cash=10000, freq="D", upon_opposite_entry="reverse", seed=42,
        # Stops measured from the fill open and charged the spread (engine.STOP_ENTRY_PRICE).
        stop_entry_price="price", stop_exit_price="stopmarket",
    )
    assert_stats_equal(run["pf"].stats(), hand.stats())
    assert_stats_equal(run["pf"].trades.stats(), hand.trades.stats())
    assert_stats_equal(run["pf"].returns_stats(), hand.returns_stats())


# ---------------------------------------------------------------------------
# Tests 3 and 4: hand-built signals and bars
# ---------------------------------------------------------------------------

def pulse_spec(slot, bar, exit_slot, exit_bar, risk=None, symbol="X", spread_points=0.0):
    """A spec whose rule fires exactly once, at `bar`, via a Python block.

    The spread defaults to zero so fills are the bare open or stop price: these
    tests are about timing. Equity mode's own default is one point, charged as
    half a point on every fill; test_spread_is_paid_on_every_fill pins that.
    """
    code = (f"out = close * 0.0\nout.iloc[{bar}] = 1.0\nout.iloc[{exit_bar}] = 2.0\nreturn out")
    rules = {"long_entry": None, "long_exit": None, "short_entry": None, "short_exit": None}
    rules[slot] = {"left": {"feature": "pulse"}, "op": "between", "lower": {"const": 0.5}, "upper": {"const": 1.5}}
    rules[exit_slot] = {"left": {"feature": "pulse"}, "op": ">", "right": {"const": 1.5}}
    return {"version": 1, "name": "pulse", "data": {"symbols": [symbol], "timeframe": "1d",
                                                     "start": "2000-01-01", "end": "2100-01-01"},
            "instruments": {symbol: {"mode": "equity", "point_size": 1.0, "quote_currency": "USD",
                                     "cost_currency": "USD", "spread_points": spread_points}},
            "features": {"pulse": {"block": "python", "code": code}}, "rules": rules,
            "risk": risk or {"sizing": {"size": 1}, "stop": {"kind": "none"}},
            "account": {"init_cash": 10_000, "freq": "D"}}


def bars(opens, highs, lows, closes, symbol="X"):
    index = pd.bdate_range("2024-01-01", periods=len(opens))
    frame = lambda values: pd.DataFrame({symbol: np.asarray(values, dtype=float)}, index=index)
    return {"open": frame(opens), "high": frame(highs), "low": frame(lows), "close": frame(closes),
            "volume": frame([1000.0] * len(opens))}


def first_order(pf):
    orders = pf.orders.records_readable
    time_column = next(column for column in orders.columns if "timestamp" in str(column).lower())
    price_column = next(column for column in orders.columns if "price" in str(column).lower())
    return orders.iloc[0][time_column], float(orders.iloc[0][price_column])


def test_short_signal_fills_next_open():
    pytest.importorskip("vectorbtx")
    data = make_ohlcv(60, ("X",), seed=11, base=100.0)
    for bar in (10, 25):
        run = engine.run_backtest(pulse_spec("short_entry", bar, "short_exit", bar + 10), data)
        when, price = first_order(run["pf"])
        index = data["close"].index
        assert when == index[bar + 1], "a short signal at bar t must fill at bar t+1, never t"
        assert price == pytest.approx(data["open"]["X"].iloc[bar + 1], rel=1e-9)


def test_long_signal_fills_next_open():
    pytest.importorskip("vectorbtx")
    data = make_ohlcv(60, ("X",), seed=12, base=100.0)
    run = engine.run_backtest(pulse_spec("long_entry", 20, "long_exit", 30), data)
    when, price = first_order(run["pf"])
    assert when == data["close"].index[21]
    assert price == pytest.approx(data["open"]["X"].iloc[21], rel=1e-9)


def exit_of(pf):
    trades = pf.trades.records_readable
    time_column = next(c for c in trades.columns if "exit" in str(c).lower() and "timestamp" in str(c).lower())
    price_column = next(c for c in trades.columns if "exit" in str(c).lower() and "price" in str(c).lower())
    return trades.iloc[0][time_column], float(trades.iloc[0][price_column])


def test_gap_through_stop_fills_at_open():
    pytest.importorskip("vectorbtx")
    # Signal at bar 0, entry at bar 1's open (100). Stop 2 points below: 98.
    # Bar 3 opens at 95, already beyond the stop: the fill is the open, not 98.
    data = bars(opens=[100, 100, 100, 95, 95, 95], highs=[101, 101, 101, 96, 96, 96],
                lows=[99, 99, 99, 94, 94, 94], closes=[100, 100, 100, 95, 95, 95])
    risk = {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": 2}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 5, risk=risk), data)
    when, price = exit_of(run["pf"])
    assert when == data["close"].index[3]
    assert price == pytest.approx(95.0)


def test_same_bar_stop_and_target():
    pytest.importorskip("vectorbtx")
    # Entry at 100; stop at 98 and target at 102; bar 3's range (97..103) holds both.
    data = bars(opens=[100, 100, 100, 100, 100, 100], highs=[100.5, 100.5, 100.5, 103, 100.5, 100.5],
                lows=[99.5, 99.5, 99.5, 97, 99.5, 99.5], closes=[100, 100, 100, 100, 100, 100])
    risk = {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": 2}, "target": {"points": 2}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 5, risk=risk), data)
    when, price = exit_of(run["pf"])
    assert when == data["close"].index[3]
    observed = "stop" if price == pytest.approx(98.0) else "target" if price == pytest.approx(102.0) else price
    if engine.SAME_BAR_STOP_RULE is None:
        pytest.fail(f"Phase 0: the same-bar rule observed here is {observed!r}. Set "
                    f"engine.SAME_BAR_STOP_RULE = {observed!r} so the UI documents it.")
    assert observed == engine.SAME_BAR_STOP_RULE


def test_stop_is_measured_from_the_fill_open():
    """Phase 0 finding (VERIFY 2.8): vectorbt's default anchors the stop at the entry bar's CLOSE.

    Entry fills at bar 1's open (100); bar 1 closes at 103. A 2-point stop belongs at 98,
    which bar 2 (low 100.5) never reaches. Anchored at the close it would sit near 100.94,
    above the entry, and fire on bar 2.
    """
    pytest.importorskip("vectorbtx")
    data = bars(opens=[100, 100, 103, 103, 103, 103], highs=[100.5, 103.5, 103.5, 103.5, 103.5, 103.5],
                lows=[99.5, 99.5, 100.5, 102.5, 102.5, 102.5], closes=[100, 103, 103, 103, 103, 103])
    risk = {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": 2}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 5, risk=risk), data)
    trades = run["pf"].trades.records_readable
    assert len(trades) == 1 and trades.iloc[0]["Status"] == "Open", "the stop must not fire above the entry"


def test_spread_is_paid_on_every_fill():
    """Phase 0 finding (VERIFY 2.9): a one-point spread costs half a point on entry AND on exit.

    vectorbt's default fills stops with no slippage at all, so a stopped-out trade used to
    pay half the spread a signal exit pays.
    """
    pytest.importorskip("vectorbtx")
    flat = bars(opens=[100] * 6, highs=[100.5] * 6, lows=[99.5] * 6, closes=[100] * 6)
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 3, spread_points=1.0), flat)
    orders = run["pf"].orders.records_readable
    assert orders["Price"].tolist() == pytest.approx([100.5, 99.5]), "signal entry and exit"
    gapped = bars(opens=[100, 100, 100, 95, 95, 95], highs=[101, 101, 101, 96, 96, 96],
                  lows=[99, 99, 99, 94, 94, 94], closes=[100, 100, 100, 95, 95, 95])
    risk = {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": 2}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 5, risk=risk, spread_points=1.0), gapped)
    _, price = exit_of(run["pf"])
    assert price == pytest.approx(95.0 * (1 - 0.5 / 95.0)), "a gapped stop fills at the open, less half the spread"


def test_trailing_stop_follows_the_peak():
    pytest.importorskip("vectorbtx")
    rising = [100, 102, 104, 106, 108, 110]
    falling = [109, 108, 107, 106, 105]
    closes = rising + falling
    data = bars(opens=closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes], closes=closes)
    risk = {"sizing": {"size": 1}, "stop": {"kind": "trailing", "points": 3}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", len(closes) - 1, risk=risk), data)
    when, price = exit_of(run["pf"])
    assert when > data["close"].index[5], "the trailing stop must not fire while price rises"
    assert price < 110 and price > 104, "it exits a few points below the peak"
