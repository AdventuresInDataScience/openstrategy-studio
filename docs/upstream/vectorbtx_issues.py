"""Reproductions of the vectorbtX issues in docs/UPSTREAM.md, using only vectorbtX, pandas and numpy.

Run it in any environment with vectorbtX installed (the Studio's, or vectorbtX's own):

    uv run python docs/upstream/vectorbtx_issues.py

Each check prints PRESENT (the behaviour described in UPSTREAM.md still happens) or
FIXED, with the numbers it measured. The exit code is the number of issues still
present, so the script can gate a vectorbtX release or a lock-file bump. Nothing
here imports the Studio, so it can be copied into vectorbtX's tests as it is.
"""

import inspect
import sys
import warnings

import numpy as np
import pandas as pd

import vectorbtx as vbt

warnings.simplefilter("ignore")


def frames(opens, highs=None, lows=None, closes=None, columns=("X",), start="2024-01-01"):
    """OHLC frames with the same values in every column."""
    index = pd.bdate_range(start, periods=len(opens))
    highs = highs if highs is not None else [value + 0.5 for value in opens]
    lows = lows if lows is not None else [value - 0.5 for value in opens]
    closes = closes if closes is not None else opens
    make = lambda values: pd.DataFrame({column: np.asarray(values, dtype=float) for column in columns}, index=index)
    return make(opens), make(highs), make(lows), make(closes)


def mask(index, columns, at):
    """A boolean frame that is True at (column, bar) pairs."""
    out = pd.DataFrame(False, index=index, columns=list(columns))
    for column, bar in at:
        out.iloc[bar, out.columns.get_loc(column)] = True
    return out


EQUITY = {"mode": "equity", "point_size": 1.0, "quote_currency": "USD", "spread_points": 0.0, "leverage": 1.0}
FX_SPREADBET = {"mode": "spreadbet", "asset_class": "fx", "point_size": 0.0001, "base_currency": "EUR",
                "quote_currency": "USD", "spread_points": 0.0, "leverage": 30}


# ---------------------------------------------------------------------------
# 1. Short signals are not shifted to the next bar
# ---------------------------------------------------------------------------

def check_short_signals_shifted():
    opens = [100, 101, 102, 103, 104, 105]
    o, h, l, c = frames(opens)
    none = mask(c.index, ["X"], [])
    fills = {}
    for side in ("long", "short"):
        signal = mask(c.index, ["X"], [("X", 1)])          # the signal is computed at bar 1's close
        kwargs = dict(entries=signal, exits=none) if side == "long" else dict(
            entries=none, exits=none, short_entries=signal, short_exits=none)
        pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": EQUITY}, size=1,
                                     init_cash=10_000, freq="D", **kwargs)
        order = pf.orders.records_readable.iloc[0]
        fills[side] = (c.index.get_loc(order["Timestamp"]), float(order["Price"]))
    present = fills["short"][0] != 2
    return present, f"signal at bar 1: long fills at bar {fills['long'][0]} ({fills['long'][1]}), " \
                    f"short fills at bar {fills['short'][0]} ({fills['short'][1]}); both should be bar 2 (102)"


# ---------------------------------------------------------------------------
# 2, 3 and 4. FX carry: SingleBacktest's base_currency, a missing currency leg, the markup
# ---------------------------------------------------------------------------

def _fx_run(single, instrument, settings_markup=None):
    opens = [1.10] * 12
    o, h, l, c = frames(opens, [1.1005] * 12, [1.0995] * 12, opens, columns=("EURUSD",), start="2024-01-01")
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    entries = mask(c.index, ["EURUSD"], [("EURUSD", 0)])
    exits = mask(c.index, ["EURUSD"], [])
    common = dict(size=10_000, rates=rates, init_cash=10_000, freq="D")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if single:
            fields = dict(instrument)
            pf = vbt.x.SingleBacktest(close=c["EURUSD"], open=o["EURUSD"], high=h["EURUSD"], low=l["EURUSD"],
                                      entries=entries["EURUSD"], exits=exits["EURUSD"], **fields, **common)
        else:
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=exits,
                                         instruments={"EURUSD": instrument}, **common)
    costs = pd.DataFrame(pf.holding_costs).iloc[:, 0]
    nights = costs[costs != 0].abs()
    one_night = float(nights.min()) if len(nights) else 0.0              # weekday nights carry multiplier 1
    notional = 10_000 * 1.10
    implied = one_night * 360 / notional                                   # annual rate charged, USD 360-day count
    pair_warning = any("no currency pair" in str(w.message) for w in caught)
    return implied, pair_warning


def check_single_backtest_base_currency():
    single, warned = _fx_run(True, FX_SPREADBET)
    portfolio, _ = _fx_run(False, FX_SPREADBET)
    present = abs(single - portfolio) > 1e-6 or warned
    return present, f"long EURUSD, EUR 3% / USD 4.5%: SingleBacktest(base_currency='EUR') charges {single:.2%} a year " \
                    f"(warning 'no currency pair': {warned}); PortfolioBacktest(instruments=...) charges {portfolio:.2%}"


def check_missing_currency_leg():
    instrument = {key: value for key, value in FX_SPREADBET.items() if key != "base_currency"}
    implied, warned = _fx_run(False, instrument)
    present = implied > 1e-6
    return present, f"FX instrument without base_currency: the warning says carry resolves to zero " \
                    f"(warned: {warned}), but a long is charged {implied:.2%} a year (the whole USD rate)"


def check_carry_markup_default():
    implied, _ = _fx_run(False, FX_SPREADBET)
    setting = vbt.settings["vectorbtx"]["carry_markup"]
    present = abs(implied - 0.015) < 1e-6 and setting > 0
    return present, f"settings carry_markup = {setting:.2%}, but the long is charged {implied:.2%} a year: the bare " \
                    f"1.50% differential, with no markup"


# ---------------------------------------------------------------------------
# 5 and 6. Stops: anchored at the entry bar's close; no spread on stop exits
# ---------------------------------------------------------------------------

def check_stop_anchor():
    # Entry fills at bar 1's open (100); bar 1 closes at 103; bar 2's low is 100.5.
    o, h, l, c = frames([100, 100, 103, 103, 103, 103], [100.5, 103.5, 103.5, 103.5, 103.5, 103.5],
                        [99.5, 99.5, 100.5, 102.5, 102.5, 102.5], [100, 103, 103, 103, 103, 103])
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=mask(c.index, ["X"], [("X", 0)]),
                                 exits=mask(c.index, ["X"], []), instruments={"X": EQUITY}, size=1, stop_points=2,
                                 init_cash=10_000, freq="D")
    orders = pf.orders.records_readable
    exits = orders[orders["Side"] == "Sell"]
    present = len(exits) > 0
    detail = f"exit at {float(exits.iloc[0]['Price'])}" if present else "no exit"
    return present, f"buy filled at 100, 2-point stop should sit at 98 and never fire; {detail}"


def check_stop_exit_spread():
    instrument = dict(EQUITY, spread_points=1.0)                        # half a point on each fill
    o, h, l, c = frames([100, 100, 100, 95, 95, 95], [101, 101, 101, 96, 96, 96], [99, 99, 99, 94, 94, 94])
    entries = mask(c.index, ["X"], [("X", 0)])
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=mask(c.index, ["X"], []),
                                 instruments={"X": instrument}, size=1, stop_points=2, init_cash=10_000, freq="D")
    stop_fill = float(pf.orders.records_readable.iloc[1]["Price"])
    flat = frames([100] * 6)
    pf = vbt.x.PortfolioBacktest(close=flat[3], open=flat[0], high=flat[1], low=flat[2],
                                 entries=mask(flat[3].index, ["X"], [("X", 0)]),
                                 exits=mask(flat[3].index, ["X"], [("X", 3)]), instruments={"X": instrument},
                                 size=1, init_cash=10_000, freq="D")
    signal_fill = float(pf.orders.records_readable.iloc[1]["Price"])
    present = abs(stop_fill - 95.0) < 1e-9
    return present, f"1-point spread: a signal exit at 100 fills at {signal_fill} (pays half), a stop gapped to 95 " \
                    f"fills at {stop_fill} (should be 94.5)"


# ---------------------------------------------------------------------------
# 7. Buy and hold (no signals, no weights) is funded at buying power
# ---------------------------------------------------------------------------

def check_hold_leverage():
    o, h, l, c = frames([1.10] * 10, [1.1005] * 10, [1.0995] * 10, columns=("EURUSD",))
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"EURUSD": FX_SPREADBET},
                                 init_cash=10_000, freq="D")
    value = pd.DataFrame(pf.value()).sum(axis=1)
    exposure = (pd.DataFrame(pf.assets()) * c).sum(axis=1)
    ratio = float(exposure.iloc[2] / value.iloc[2])
    return ratio > 1.5, f"a hold on a 30x spread bet carries {ratio:.1f}x the account's equity (expected 1x)"


# ---------------------------------------------------------------------------
# 8. Weights above 100%: filled in column order, not as the warning describes
# ---------------------------------------------------------------------------

def check_weights_over_one():
    o, h, l, c = frames([100] * 8, columns=("A", "B"))
    weights = pd.DataFrame(0.75, index=c.index, columns=["A", "B"])
    schedule = pd.Series(False, index=c.index)
    schedule.iloc[0] = True
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"A": EQUITY, "B": EQUITY},
                                 weights=weights, schedule=schedule, init_cash=100_000, freq="D")
    held = (pd.DataFrame(pf.assets()) * c).iloc[2] / float(pd.DataFrame(pf.value()).sum(axis=1).iloc[2])
    present = abs(float(held["A"]) - float(held["B"])) > 0.1
    return present, f"targets 0.75 / 0.75; held {float(held['A']):.2f} / {float(held['B']):.2f}. The warning says the " \
                    f"excess will not fill; in fact the first column fills in full and the last is starved"


# ---------------------------------------------------------------------------
# 9. The FX conversion markup is never charged
# ---------------------------------------------------------------------------

def check_fx_markup():
    o, h, l, c = frames([100] * 6, [101] * 6, [99] * 6)
    fx = pd.Series(1.25, index=c.index, name="GBPUSD")
    instrument = {"X": {"mode": "equity", "point_size": 0.01, "quote_currency": "USD", "spread_points": 0.0}}
    universe = vbt.x.load_universe({k: vbt.x.Instrument(**v) for k, v in instrument.items()})
    fills = {}
    for markup in (0.005, 0.05):
        converted = [vbt.x.to_base_currency(frame, base="GBP", fx=fx, instruments=universe, fx_markup=markup)
                     for frame in (o, h, l, c)]
        pf = vbt.x.PortfolioBacktest(close=converted[3], open=converted[0], high=converted[1], low=converted[2],
                                     entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"], [("X", 3)]),
                                     instruments=instrument, size=10, init_cash=10_000, freq="D",
                                     base_currency="GBP", fx=fx)
        fills[markup] = float(pf.orders.records_readable.iloc[0]["Price"])
    try:
        vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=mask(c.index, ["X"], [("X", 0)]),
                                exits=mask(c.index, ["X"], []), instruments=instrument, size=10, init_cash=10_000,
                                freq="D", fx_markup=0.05)
        argument = "accepted"
    except TypeError:
        argument = "raises TypeError"
    present = fills[0.005] == fills[0.05]
    return present, f"buy fill at 0.5% markup: {fills[0.005]}, at 5% markup: {fills[0.05]}; fx_markup= {argument}"


# ---------------------------------------------------------------------------
# 10. pf["column"] on a leveraged portfolio crashes every metric
# ---------------------------------------------------------------------------

def check_column_selection():
    o, h, l, c = frames([1.10, 1.11, 1.12, 1.11, 1.13, 1.14], columns=("A", "B"))
    instrument = dict(FX_SPREADBET)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=mask(c.index, ["A", "B"], [("A", 0), ("B", 0)]),
                                 exits=mask(c.index, ["A", "B"], []), instruments={"A": instrument, "B": instrument},
                                 size=10_000, init_cash=10_000, freq="D")
    try:
        pf["A"].total_return()
        return False, "pf['A'].total_return() works"
    except Exception as error:                                           # noqa: BLE001
        listed = pf[["A"]].total_return()
        return True, f"pf['A'].total_return() raises {type(error).__name__}; pf[['A']] works ({float(np.ravel(listed)[0]):.4f})"


# ---------------------------------------------------------------------------
# 11. pf.x.intrabar_drawdown(): shared cash and shorts
# ---------------------------------------------------------------------------

def _true_intrabar(pf):
    assets = pd.DataFrame(pf.assets())
    cash = pd.DataFrame(pf.cash()).sum(axis=1)                            # group level: counts shared cash once
    highs = pd.DataFrame(pf.high).reindex(columns=assets.columns)
    lows = pd.DataFrame(pf.low).reindex(columns=assets.columns)
    longs, shorts = assets.clip(lower=0), assets.clip(upper=0)
    best = (longs * highs + shorts * lows).sum(axis=1) + cash
    worst = (longs * lows + shorts * highs).sum(axis=1) + cash
    return float((1 - worst / best.cummax()).max())


def check_intrabar_drawdown():
    rng = np.random.default_rng(3)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (120, 4)), axis=0))
    index = pd.bdate_range("2024-01-01", periods=120)
    columns = ["A", "B", "C", "D"]
    c = pd.DataFrame(walk, index=index, columns=columns)
    o, h, l = c.shift(1).fillna(c), c * 1.004, c * 0.996
    weights = pd.DataFrame(0.25, index=index, columns=columns)
    schedule = pd.Series(False, index=index)
    schedule.iloc[0] = True
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={k: EQUITY for k in columns},
                                 weights=weights, schedule=schedule, init_cash=100_000, freq="D")
    value = pd.DataFrame(pf.value()).sum(axis=1)
    close_dd = float((1 - value / value.cummax()).max())
    shared = (pf.x.intrabar_drawdown(), _true_intrabar(pf))
    # A short of 10 at 100; bar 3 spikes to 104 and closes flat.
    so, sh, sl, sc = frames([100] * 8, [100.5, 100.5, 100.5, 104, 100.5, 100.5, 100.5, 100.5], [99.5] * 8, [100] * 8)
    none = mask(sc.index, ["X"], [])
    short = vbt.x.PortfolioBacktest(close=sc, open=so, high=sh, low=sl, entries=none, exits=none,
                                    short_entries=mask(sc.index, ["X"], [("X", 0)]), short_exits=none,
                                    instruments={"X": EQUITY}, size=10, init_cash=10_000, freq="D")
    shorted = (short.x.intrabar_drawdown(), _true_intrabar(short))
    present = shared[0] < close_dd - 1e-9 or abs(shorted[0] - shorted[1]) > 1e-6
    return present, f"4-asset weights book: close drawdown {close_dd:.2%}, library intrabar {shared[0]:.2%} " \
                    f"(shallower than close), shared cash counted once {shared[1]:.2%}; short with a spike to 104: " \
                    f"library {shorted[0]:.3%}, side-aware {shorted[1]:.3%}"


# ---------------------------------------------------------------------------
# 12. Housekeeping: pandas 4 removes DatetimeIndex.weekday
# ---------------------------------------------------------------------------

def check_weekday_deprecation():
    from vectorbtx.holding import accrual
    present = ".weekday" in inspect.getsource(accrual)
    return present, "vectorbtx/holding/accrual.py uses DatetimeIndex.weekday (use .day_of_week)"


# ---------------------------------------------------------------------------
# 13. Shared cash with an entry priority (a required feature, not a bug)
# ---------------------------------------------------------------------------

SHARED_CASH_ARGUMENTS = ("cash", "entry_priority", "when_cash_short")     # as proposed in UPSTREAM.md, issue 13


def check_shared_cash_priority():
    """The acceptance test for the shared-cash API. Rename SHARED_CASH_ARGUMENTS if the API's names differ.

    Scenario, 4 instruments at a flat 100, 10,000 cash, 40 shares (40%) per entry:
    bar 0 signals A and B (filled at bar 1, leaving 2,000). Bar 2 signals A to exit and
    C and D to enter (filled at bar 3). A's exit frees 4,000, so exactly one of C and D fits.
    """
    parameters = inspect.signature(vbt.x.PortfolioBacktest).parameters
    missing = [name for name in SHARED_CASH_ARGUMENTS if name not in parameters]
    if missing:
        return True, ("not implemented yet: PortfolioBacktest has no " + ", ".join(f"{name}=" for name in missing)
                      + " (the names proposed in UPSTREAM.md, issue 13; rename them here if the API differs)")
    columns = ["A", "B", "C", "D"]
    o, h, l, c = frames([100] * 8, [100.5] * 8, [99.5] * 8, columns=columns)
    entries = mask(c.index, columns, [("A", 0), ("B", 0), ("C", 2), ("D", 2)])
    exits = mask(c.index, columns, [("A", 2)])

    def bar_three(**kwargs):
        pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=exits,
                                     instruments={column: EQUITY for column in columns}, size=40,
                                     init_cash=10_000, freq="D", **kwargs)
        orders = pf.orders.records_readable
        filled = orders[orders["Timestamp"] == c.index[3]]
        return {str(row.Column): (row.Side, float(row.Size)) for row in filled.itertuples()}, pf

    failures = []
    expected = {"A": ("Sell", 40.0), "D": ("Buy", 40.0)}
    got, _ = bar_three(cash="shared", entry_priority=["D", "C", "A", "B"], when_cash_short="skip")
    if got != expected:
        failures.append(f"priority D first, skip: expected {expected} (A's exit funds D, C skipped), got {got}")
    # A per-bar score, aligned with the signals (the bar the signal is computed on) and shifted with them.
    scores = pd.DataFrame(0.0, index=c.index, columns=columns)
    scores.loc[c.index[2], ["C", "D"]] = [2.0, 1.0]
    got, _ = bar_three(cash="shared", entry_priority=scores, when_cash_short="skip")
    if got != {"A": ("Sell", 40.0), "C": ("Buy", 40.0)}:
        failures.append(f"score ranks C first: expected A sell and C buy, got {got}")
    got, _ = bar_three(cash="shared", entry_priority=["D", "C", "A", "B"], when_cash_short="partial")
    if got.get("C") != ("Buy", 20.0) or got.get("D") != ("Buy", 40.0):
        failures.append(f"partial: expected D 40 and C 20, got {got}")
    first, _ = bar_three(cash="shared", entry_priority="random", seed=7, when_cash_short="skip")
    second, _ = bar_three(cash="shared", entry_priority="random", seed=7, when_cash_short="skip")
    if first != second or len([side for side, _ in first.values() if side == "Buy"]) != 1:
        failures.append(f"random: the same seed must give the same single entry, got {first} then {second}")
    _, replicated = bar_three(cash="per_instrument")
    total = float(np.sum(np.asarray(replicated.get_init_cash(), dtype=float)))
    if abs(total - 40_000) > 1e-6:
        failures.append(f"per_instrument: expected each of 4 columns to hold 10,000 (40,000), got {total:,.0f}")
    return bool(failures), "; ".join(failures) or "shared cash, exits first, list/score/random priority, skip/partial"



# ---------------------------------------------------------------------------
# 15. Holding costs reach value() but not total_profit(), final_value(), total_return() or stats()
# ---------------------------------------------------------------------------

def check_holding_costs_in_totals():
    # A flat market, so the carry is the only thing that moves the account.
    opens = [1.10] * 12
    o, h, l, c = frames(opens, [1.1005] * 12, [1.0995] * 12, opens, columns=("EURUSD",))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"EURUSD": FX_SPREADBET},
                                 entries=mask(c.index, ["EURUSD"], [("EURUSD", 0)]),
                                 exits=mask(c.index, ["EURUSD"], []), size=10_000, rates=rates,
                                 init_cash=10_000, freq="D")
    charged = float(pd.DataFrame(pf.holding_costs).to_numpy().sum())
    last = float(pd.DataFrame(pf.value()).iloc[-1, 0])
    final = float(np.asarray(pf.final_value()).ravel()[0])
    end = float(pf.stats()["End Value"])
    total_return = float(np.asarray(pf.total_return()).ravel()[0])
    from_value = last / 10_000 - 1
    present = abs(final - last) > 1e-6 or abs(end - last) > 1e-6 or abs(total_return - from_value) > 1e-9
    return present, f"carry charged {charged:,.2f}: value()[-1] = {last:,.2f}, but final_value() = {final:,.2f} " \
                    f"and stats End Value = {end:,.2f}; total_return() = {total_return:.4%}, value() gives {from_value:.4%}"


# ---------------------------------------------------------------------------
# 16. Holding costs in a shared account, per instrument
# ---------------------------------------------------------------------------

EQUITY_2X = {"mode": "equity", "point_size": 1.0, "quote_currency": "GBP", "spread_points": 0.0, "leverage": 2.0}


def check_shared_account_holding_costs():
    failures = []
    # Carry: A holds a EURUSD spread bet, B holds nothing. B used to pay A's carry.
    o, h, l, c = frames([1.10] * 12, [1.1005] * 12, [1.0995] * 12, columns=("A", "B"))
    rates = vbt.x.static_rate_table(["EUR", "USD"], c.index, {"EUR": 0.03, "USD": 0.045})
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={k: FX_SPREADBET for k in "AB"},
                                 entries=mask(c.index, "AB", [("A", 0)]), exits=mask(c.index, "AB", []),
                                 size=10_000, rates=rates, init_cash=10_000, cash="shared", freq="D")
    carry = pd.DataFrame(pf.holding_costs).sum()
    if carry["B"] != 0:
        failures.append(f"carry: B holds nothing but paid {carry['B']:,.2f} (A paid {carry['A']:,.2f})")
    # Share dealing: A and B buy 8,000 each against 10,000 at 2x, so the account owes 6,000.
    o, h, l, c = frames([100.0] * 12, columns=("A", "B"))
    size = pd.DataFrame({"A": 80.0, "B": 80.0}, index=c.index)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={k: EQUITY_2X for k in "AB"},
                                 entries=mask(c.index, "AB", [("A", 0), ("B", 0)]), exits=mask(c.index, "AB", []),
                                 size=size, init_cash=10_000, cash="shared", freq="D")
    borrowed = pd.DataFrame(pf.x.borrowed_portion()).iloc[3].tolist()
    if not np.allclose(borrowed, [3_000, 3_000]):
        failures.append(f"share dealing: the account owes 6,000 but borrowed_portion() splits it as {borrowed}")
    return bool(failures), "; ".join(failures) or "carry per instrument; a 6,000 debit split 3,000 / 3,000"


# ---------------------------------------------------------------------------
# 17. Borrowing without a margin loan; leverage on the replay path
# ---------------------------------------------------------------------------

def check_borrowing_without_a_loan():
    failures = []
    # A fully paid 60/40 share book (the weights path runs without a margin loan).
    o, h, l, c = frames([100.0] * 12, columns=("A", "B"))
    rates = vbt.x.static_rate_table(["GBP"], c.index, {"GBP": 0.05})
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={k: EQUITY_2X for k in "AB"},
                                 weights=pd.DataFrame({"A": 0.6, "B": 0.4}, index=c.index), rates=rates,
                                 init_cash=10_000, freq="D")
    financing = float(pd.DataFrame(pf.holding_costs).to_numpy().sum())
    if financing != 0:
        failures.append(f"a fully paid 60/40 book paid {financing:,.2f} of margin financing")
    # The replay path: 150 shares at 100 on 10,000 at 2x.
    universe = vbt.load_universe({"X": vbt.Instrument(mode="equity", quote_currency="GBP", leverage=2.0)})
    size = pd.DataFrame(np.nan, index=c.index, columns=["X"])
    size.iloc[0] = 150.0
    replay = vbt.Portfolio.from_holding_costs(pd.DataFrame({"X": c["A"]}), size, universe, rates=rates,
                                              init_cash=10_000, freq="D")
    held = float(pd.DataFrame(replay.assets()).iloc[-1, 0])
    if abs(held - 150) > 1e-9:
        failures.append(f"Portfolio.from_holding_costs funded no leverage: 150 shares ordered, {held:g} held")
    return bool(failures), "; ".join(failures) or "a paid-for book borrows nothing; the replay funds 2x"


# ---------------------------------------------------------------------------
# 18. The borrow fee on a direct short (a feature, docs/dev/AUDIT-G.md G6 in vectorbtX)
# ---------------------------------------------------------------------------

def check_borrow_fee():
    o, h, l, c = frames([100.0] * 12)
    instrument = dict(EQUITY, quote_currency="USD", borrow_rate=0.036)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": instrument},
                                 entries=mask(c.index, ["X"], []), exits=mask(c.index, ["X"], []),
                                 short_entries=mask(c.index, ["X"], [("X", 0)]), short_exits=mask(c.index, ["X"], []),
                                 size=100, init_cash=10_000, freq="D")
    thursday = float(pd.DataFrame(pf.holding_costs).iloc[3, 0])
    failures = [] if abs(thursday + 1.0) < 1e-9 else [f"a 10,000 short at 3.6% paid {thursday:.4f} on a Thursday, not -1.00"]
    try:
        vbt.Instrument(mode="cfd", contract_size=1.0, borrow_rate=0.01)
        failures.append("a CFD accepts a borrow_rate, though its funding already covers the short")
    except ValueError:
        pass
    return bool(failures), "; ".join(failures) or "a direct short pays borrow_rate daily; a CFD has none"


# ---------------------------------------------------------------------------
# 19. A per-instrument financing value was silently ignored
# ---------------------------------------------------------------------------

def check_financing_field_says_it_is_unused():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vbt.Instrument(mode="cfd", contract_size=1.0, financing=0.10)
    said = any("financing" in str(w.message) for w in caught)
    return not said, "financing=0.10 warns that it is not used" if said else \
        "financing=0.10 is accepted in silence and never read"


CHECKS = [
    (1, "Short signals are not shifted to the next bar (lookahead)", check_short_signals_shifted),
    (2, "SingleBacktest(base_currency=) is read as the account currency", check_single_backtest_base_currency),
    (3, "A missing FX currency leg is priced at 0%, not skipped", check_missing_currency_leg),
    (4, "PortfolioBacktest's carry_markup=0.0 overrides the setting", check_carry_markup_default),
    (5, "Point stops are anchored at the entry bar's close", check_stop_anchor),
    (6, "Stop exits pay no spread", check_stop_exit_spread),
    (7, "Buy and hold is funded at buying power", check_hold_leverage),
    (8, "Weights above 100% fill in column order", check_weights_over_one),
    (9, "The FX conversion markup is never charged", check_fx_markup),
    (10, "pf['column'] crashes metrics on a leveraged portfolio", check_column_selection),
    (11, "intrabar_drawdown: shared cash counted per column; shorts", check_intrabar_drawdown),
    (12, "pandas 4: DatetimeIndex.weekday", check_weekday_deprecation),
    (13, "Shared cash with an entry priority (required feature)", check_shared_cash_priority),
    (15, "Holding costs missing from total_profit, final_value, total_return and stats()", check_holding_costs_in_totals),
    (16, "Shared account: carry on the whole account, share-dealing debit not split", check_shared_account_holding_costs),
    (17, "Borrowing counted without a loan; the replay path funds no leverage", check_borrowing_without_a_loan),
    (18, "The borrow fee on a direct short (feature)", check_borrow_fee),
    (19, "A per-instrument financing value is ignored in silence", check_financing_field_says_it_is_unused),
]


def main():
    import importlib.metadata
    print(f"vectorbtX {importlib.metadata.version('vectorbtx')}, vectorbt {importlib.metadata.version('vectorbt')}")
    still = 0
    for number, title, check in CHECKS:
        try:
            present, detail = check()
        except Exception as error:                                       # noqa: BLE001
            present, detail = True, f"the check itself failed: {type(error).__name__}: {error}"
        still += bool(present)
        print(f"{number:>2}. {'PRESENT' if present else 'FIXED  '}  {title}\n      {detail}")
    print(f"\n{still} of {len(CHECKS)} still present.")
    return still


if __name__ == "__main__":
    sys.exit(main())
