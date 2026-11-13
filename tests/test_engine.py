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
    # vectorbtX's front door sets the stop anchor and stop fill price itself (issues 5, 6).
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
    # One night's charge on a long: notional x ((USD 4.5% - EUR 3%) + the carry markup) / 360.
    # vectorbtX ace29e2 dropped the markup (docs/UPSTREAM.md issue 4); it is now charged.
    import vectorbtx

    markup = vectorbtx.settings["vectorbtx"]["carry_markup"]
    assert markup > 0
    costs = pd.DataFrame(run["pf"].holding_costs).iloc[:, 0]
    orders = run["pf"].orders.records_readable
    first = orders.iloc[0]
    if first["Side"] == "Buy":
        night = costs.loc[first["Timestamp"]]
        notional = first["Size"] * first["Price"]
        assert night == pytest.approx(-notional * (0.015 + markup) / 360, rel=0.05)


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
    plan = engine.call_plan(spec)
    assert plan["front_door"] == "PortfolioBacktest"
    # The account goes by vectorbtX's unambiguous name: its base_currency= can mean a pair's base.
    assert plan["args"]["account_currency"] == ("lit", "GBP") and "base_currency" not in plan["args"]


def test_price_field_is_not_sent(ma_spec):
    for spec in (apply_params(ma_spec), without_pair(ma_spec)):
        spec["instruments"]["EURUSD=X"]["price_field"] = "close"
        args = engine.call_plan(spec)["args"]
        assert "price_field" not in args
        assert "price_field" not in args.get("instruments", ("lit", {}))[1].get("EURUSD=X", {})


def test_cash_plan(shared_spec):
    """Several symbols: the plan always names the cash model, so the export says it too."""
    spec = apply_params(shared_spec)
    args = engine.call_plan(spec)["args"]
    assert args["cash"] == ("lit", "shared") and args["when_cash_short"] == ("lit", "skip")
    assert args["entry_priority"] == ("var", "entry_priority")          # a score frame from compile_signals
    assert "cash_sharing" not in args and "group_by" not in args and "call_seq" not in args
    spec["account"]["entry_priority"] = {"by": "symbols", "order": ["NVDA"]}
    assert engine.call_plan(spec)["args"]["entry_priority"] == ("lit", ["NVDA"])
    spec["account"]["entry_priority"] = {"by": "symbols"}
    assert engine.call_plan(spec)["args"]["entry_priority"] == ("lit", ["AAPL", "MSFT", "NVDA"])
    spec["account"]["entry_priority"] = {"by": "random"}
    assert engine.call_plan(spec)["args"]["entry_priority"] == ("lit", "random")
    spec["account"] = {"init_cash": 10_000, "freq": "D"}
    args = engine.call_plan(spec)["args"]
    assert args["cash"] == ("lit", "per_instrument")
    assert "entry_priority" not in args and "when_cash_short" not in args
    # One symbol is one account either way, and SingleBacktest rejects cash= outright.
    spec["data"]["symbols"] = ["AAPL"]
    spec["instruments"] = {"AAPL": spec["instruments"]["AAPL"]}
    spec["account"]["cash"] = "shared"
    plan = engine.call_plan(spec)
    assert plan["single"] and "cash" not in plan["args"]


def competing_spec(account, score=None, symbols=("AAA", "BBB", "CCC")):
    """Every symbol enters on bar 1 (filling at bar 2's open) and exits on bar 4, each wanting 4,000."""
    pulse = "out = close * 0.0\nout.iloc[1] = 1.0\nout.iloc[4] = 2.0\nreturn out"
    features = {"pulse": {"block": "python", "code": pulse}}
    if score is not None:
        features["score"] = {"block": "python", "code": score}
    instrument = {"mode": "equity", "point_size": 1.0, "quote_currency": "USD", "cost_currency": "USD",
                  "spread_points": 0.0, "leverage": 1.0}
    rules = {"long_entry": {"left": {"feature": "pulse"}, "op": "between", "lower": {"const": 0.5},
                            "upper": {"const": 1.5}},
             "long_exit": {"left": {"feature": "pulse"}, "op": ">", "right": {"const": 1.5}},
             "short_entry": None, "short_exit": None}
    return {"version": 1, "name": "competing", "data": {"symbols": list(symbols), "timeframe": "1d",
                                                         "start": "2000-01-01", "end": "2100-01-01"},
            "instruments": {symbol: dict(instrument) for symbol in symbols}, "features": features, "rules": rules,
            "risk": {"sizing": {"notional": 4000.0}, "stop": {"kind": "none"}},
            "account": dict({"init_cash": 10_000, "freq": "D"}, **account), "engine_extra": {"seed": 7}}


def flat_bars(symbols, n=8, price=100.0):
    index = pd.bdate_range("2024-01-01", periods=n)
    frame = lambda value: pd.DataFrame({symbol: np.full(n, float(value)) for symbol in symbols}, index=index)
    return {"open": frame(price), "high": frame(price + 1), "low": frame(price - 1), "close": frame(price),
            "volume": frame(1000.0)}


def bought(spec, data):
    """Units bought per symbol, from the order records."""
    from openstrategy_studio.spec import validate_spec

    assert validate_spec(spec) == []
    orders = engine.run_backtest(spec, data)["pf"].orders.records_readable
    buys = orders[orders["Side"] == "Buy"]
    return {str(symbol): float(size) for symbol, size in buys.groupby("Column")["Size"].sum().items()}


def test_shared_cash_fills_competing_entries_by_priority():
    """Three same-bar entries each want 4,000. Separate accounts fill all three; one shared 10,000 fits two."""
    pytest.importorskip("vectorbtx")
    data = flat_bars(["AAA", "BBB", "CCC"])
    assert bought(competing_spec({}), data) == {"AAA": 40, "BBB": 40, "CCC": 40}
    shared = {"cash": "shared", "when_cash_short": "skip"}
    assert bought(competing_spec(shared), data) == {"AAA": 40, "BBB": 40}                # symbol order
    order = dict(shared, entry_priority={"by": "symbols", "order": ["CCC", "AAA"]})
    assert bought(competing_spec(order), data) == {"CCC": 40, "AAA": 40}
    # Filling part of it: CCC gets what is left, 2,000 at 100.
    assert bought(competing_spec(dict(shared, when_cash_short="partial")), data) == {"AAA": 40, "BBB": 40, "CCC": 20}
    score = "out = close * 0.0\nout['BBB'] = 3.0\nout['CCC'] = 2.0\nout['AAA'] = 1.0\nreturn out"
    by_score = {"by": "feature", "score": {"feature": "score"}}
    assert bought(competing_spec(dict(shared, entry_priority=by_score), score), data) == {"BBB": 40, "CCC": 40}
    lowest = dict(shared, entry_priority=dict(by_score, highest_first=False))
    assert bought(competing_spec(lowest, score), data) == {"AAA": 40, "CCC": 40}
    random = competing_spec(dict(shared, entry_priority={"by": "random"}))
    first = bought(random, data)
    assert len(first) == 2 and bought(random, data) == first                            # seeded: it repeats


def test_separate_accounts_are_reported_as_accounts(ma_spec):
    """Each symbol with its own cash: every account equals that symbol run alone, and the tables are
    the accounts taken together, not their average (vectorbt's stats() on ungrouped columns).

    FX spread bets at 30x with carry, so the margin loan and the holding costs are both in play."""
    pytest.importorskip("vectorbtx")
    import warnings

    from openstrategy_studio import analysis

    symbols = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
    data = make_ohlcv(400, symbols, seed=5)
    spec = apply_params(ma_spec, {"features.fast.window": 5, "features.slow.window": 20})   # enough trades
    pair = spec["instruments"]["EURUSD=X"]
    spec["data"]["symbols"] = symbols
    spec["instruments"] = {symbol: dict(pair, base_currency=symbol[:3]) for symbol in symbols}
    spec["account"]["rates"]["values"] = {"EUR": 0.03, "USD": 0.045, "GBP": 0.04, "AUD": 0.035}
    run = engine.run_backtest(spec, data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = analysis.extract_results(run)
    assert not [w for w in caught if "Aggregating" in str(w.message)], "no silent averaging of accounts"
    assert results["meta"]["accounts"] == {"model": "separate", "count": 3, "each": 10_000.0}
    table = results["per_symbol"]
    portfolio = results["stats"]["portfolio"]["all"]
    assert portfolio["Start Value"] == pytest.approx(30_000)
    assert portfolio["End Value"] == pytest.approx(table["End value"].sum(), rel=1e-12)
    for symbol in symbols:
        alone = dict(spec, data=dict(spec["data"], symbols=[symbol]), instruments={symbol: spec["instruments"][symbol]})
        pf = engine.run_backtest(alone, {field: frame[[symbol]] for field, frame in data.items()})["pf"]
        first = lambda value: float(np.asarray(value, dtype=float).ravel()[0])
        assert table.loc[symbol, "End value"] == pytest.approx(first(pf.final_value()), rel=1e-12)
        assert table.loc[symbol, "Sharpe ratio"] == pytest.approx(first(pf.sharpe_ratio()), rel=1e-12)
        assert table.loc[symbol, "Trades"] == first(pf.trades.count())
        assert table.loc[symbol, "Holding costs"] == pytest.approx(pd.DataFrame(pf.holding_costs).to_numpy().sum())


def test_shared_account_is_reported_as_one(shared_spec, ohlcv_stocks):
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis

    run = engine.run_backtest(shared_spec, ohlcv_stocks)
    results = analysis.extract_results(run)
    assert results["meta"]["accounts"] == {"model": "shared", "count": 1, "each": 10_000.0}
    assert results["stats"]["portfolio"]["all"]["Start Value"] == pytest.approx(10_000)
    table = results["per_symbol"]
    assert list(table.index) == ["AAPL", "MSFT", "NVDA"] and "End value" not in table
    assert table["Trades"].sum() > 0
    # 4,500 a position from one 10,000 account: never more than two open at once.
    held = pd.DataFrame(run["pf"].assets())                     # one column per symbol, grouped or not
    assert int((held > 0).sum(axis=1).max()) <= 2


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
    )
    assert_stats_equal(run["pf"].stats(), hand.stats())
    assert_stats_equal(run["pf"].trades.stats(), hand.trades.stats())
    assert_stats_equal(run["pf"].returns_stats(), hand.returns_stats())


def test_golden_numbers_from_the_vectorbtx_example(ma_spec):
    """The Studio's MA-cross template against vectorbtX's own example, on the same Yahoo prices.

    tests/fixtures/golden/make_golden.py ran examples/strategies/01_ma_cross_spreadbet_fx.py
    (SingleBacktest with flat FX terms) and saved its prices and its three stats tables.
    The Studio routes the pair through PortfolioBacktest(instruments=...): the numbers must
    agree. The example's static rates (its no-FRED line) replace the template's.

    ⚠️ The saved stats() leave carry out of Total Return and End Value (docs/UPSTREAM.md
    issue 15). When vectorbtX fixes that, this test fails: re-run make_golden.py and check
    that only the carry-dependent numbers moved.
    """
    pytest.importorskip("vectorbtx")
    import json
    from pathlib import Path

    folder = Path(__file__).parent / "fixtures" / "golden"
    golden = json.loads((folder / "ma_cross_01.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(folder / "eurusd_2019_2024_yahoo.csv", index_col="date", parse_dates=["date"])
    symbol = "EURUSD=X"
    data = {field: prices[[field]].rename(columns={field: symbol}) for field in ("open", "high", "low", "close")}
    data["volume"] = pd.DataFrame(0.0, index=prices.index, columns=[symbol])
    ma_spec["account"]["rates"]["values"] = dict(golden["rates"])
    pf = engine.run_backtest(ma_spec, data)["pf"]
    for name, table in (("stats", pf.stats()), ("trades", pf.trades.stats()), ("returns", pf.returns_stats())):
        assert sorted(map(str, table.index)) == sorted(golden[name]), name
        for label, value in table.items():
            expected = golden[name][str(label)]
            if isinstance(expected, float):
                assert float(value) == pytest.approx(expected, rel=1e-9, abs=1e-12), (name, label)
            elif expected is None:
                assert not np.isfinite(float(value)), (name, label)
            else:
                assert str(value) == expected, (name, label)


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


def test_fx_conversion_markup_is_charged_in_a_reporting_currency():
    """vectorbtX ace29e2 never charged the FX markup (docs/UPSTREAM.md issue 9). Through
    the Studio's own path (prepare_data -> to_base_currency -> the front door), a fill
    must now move by exactly the markup, including in a window (walk-forward), where the
    converted frames are sliced before the call."""
    pytest.importorskip("vectorbtx")
    data = bars(opens=[100.0] * 8, highs=[101.0] * 8, lows=[99.0] * 8, closes=[100.0] * 8)
    data["fx"] = pd.Series(1.25, index=data["close"].index, name="GBPUSD=X")
    spec = pulse_spec("long_entry", 1, "long_exit", 5)
    spec["account"].update(base_currency="GBP", fx={"symbol": "GBPUSD=X"})
    window = (data["close"].index[0], data["close"].index[-1])
    settings = engine.vectorbtx().settings["vectorbtx"]
    before = settings["fx_markup"]
    fills = {}
    try:
        for markup in (0.0, 0.05):
            engine.apply_vectorbtx_settings({"fx_markup": markup})
            for label, kwargs in (("full", {}), ("window", {"window": window})):
                run = engine.run_backtest(spec, data, **kwargs)
                fills[markup, label] = first_order(run["pf"])[1]
    finally:
        settings["fx_markup"] = before
    for label in ("full", "window"):
        assert fills[0.0, label] == pytest.approx(80.0)                     # 100 USD at 1.25
        assert fills[0.05, label] == pytest.approx(80.0 * 1.05)


def test_stop_is_checked_on_the_entry_bar():
    """vectorbtX 16293a2 checks a stop on the bar it was entered. Signal at bar 0, entry at
    bar 1's open (100), and bar 1 itself trades down to 97: the 2-point stop fills at 98
    on bar 1. On whole bars it was missed and the trade survived."""
    pytest.importorskip("vectorbtx")
    data = bars(opens=[100] * 6, highs=[101] * 6, lows=[99, 97, 99, 99, 99, 99], closes=[100] * 6)
    risk = {"sizing": {"size": 1}, "stop": {"kind": "fixed", "points": 2}}
    run = engine.run_backtest(pulse_spec("long_entry", 0, "long_exit", 5, risk=risk), data)
    when, price = exit_of(run["pf"])
    assert when == data["close"].index[1]
    assert price == pytest.approx(98.0)


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


# ---------------------------------------------------------------------------
# The borrow fee, the reserved financing field, and vectorbtX's warnings (9fa427e)
# ---------------------------------------------------------------------------

def test_a_direct_short_pays_its_borrow_rate_and_the_run_says_when_it_has_none():
    """vectorbtX charges a borrow fee on a direct (equity) short at the instrument's
    borrow_rate and only warns when none is set. Python sends warnings to the console, so
    the Studio collects each run's and keeps them with its results for the page to show."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis

    data = bars(opens=[100.0] * 12, highs=[101.0] * 12, lows=[99.0] * 12, closes=[100.0] * 12)
    risk = {"sizing": {"size": 100}, "stop": {"kind": "none"}}
    spec = pulse_spec("short_entry", 0, "short_exit", 11, risk=risk)     # short from bar 1, held
    run = engine.run_backtest(spec, data)
    assert any("borrow_rate" in note for note in run["warnings"])
    assert analysis.extract_results(run)["meta"]["warnings"] == run["warnings"]

    spec["instruments"]["X"]["borrow_rate"] = 0.036
    run = engine.run_backtest(spec, data)
    assert not any("borrow_rate" in note for note in run["warnings"])
    thursday = pd.DataFrame(run["pf"].holding_costs)["X"].iloc[3]
    assert float(thursday) == pytest.approx(-100 * 100.0 * 0.036 / 360)  # 10,000 short, 3.6% a year


def test_a_per_instrument_financing_value_is_not_passed_on():
    """vectorbtX keeps the field reserved: financing is the rate table plus the markup."""
    spec = pulse_spec("long_entry", 0, "long_exit", 5)
    spec["instruments"]["X"]["financing"] = 0.10
    assert "financing" not in engine.instrument_fields(spec)["X"]
