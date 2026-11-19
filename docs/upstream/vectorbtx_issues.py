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


# ---------------------------------------------------------------------------
# 21. min_trade never skips a trade
# ---------------------------------------------------------------------------

def check_min_trade_skips_small_trades():
    # A rises 1% a bar and B is flat; a fixed 60/40 book of fractional shares, rebalanced every
    # 5th bar. Each rebalance only trims the drift, about 1% of the account, so min_trade=0.5
    # ("skip any trade under half the account") should leave the opening purchase alone.
    n = 21
    index = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame({"A": 100 * 1.01 ** np.arange(n), "B": np.full(n, 100.0)}, index=index)
    instrument = dict(EQUITY, point_size=0.01, allow_fractional=True)
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=index)
    schedule = pd.Series(np.arange(n) % 5 == 0, index=index)
    counts = {}
    for min_trade in (None, 0.5):
        pf = vbt.x.PortfolioBacktest(close=close, open=close, high=close + 0.5, low=close - 0.5,
                                     instruments={"A": instrument, "B": instrument}, weights=weights,
                                     schedule=schedule, min_trade=min_trade, init_cash=10_000, freq="D")
        orders = pf.orders.records_readable
        value = pd.Series(np.asarray(pf.value()).ravel(), index=index).reindex(orders["Timestamp"]).to_numpy()
        share = (orders["Size"] * orders["Price"]).abs().to_numpy() / value
        later = share[(orders["Timestamp"] > orders["Timestamp"].min()).to_numpy()]
        counts[min_trade] = (len(orders), later)
    total, later = counts[0.5]
    present = bool((later < 0.5).any())
    return present, f"60/40 rebalanced every 5th bar: {counts[None][0]} orders with no min_trade, {total} with " \
                    f"min_trade=0.5, of which {len(later)} after the opening purchase, each at most " \
                    f"{later.max() if len(later) else 0:.2%} of the account (should be none)"


# ---------------------------------------------------------------------------
# 22. The sizing warnings measure every bar, not the bars an order is placed on
# ---------------------------------------------------------------------------

def check_sizing_warnings_measure_order_bars():
    # One entry, filled at 100; the price then rises to 1000 and nothing more is traded. A stake
    # of 300 a point is worth 30,000 at the fill, inside 5x leverage on 10,000 (50,000), so the
    # trade fills in full and the buying-power warning has nothing to say. A notional of 500 buys
    # 5 shares at 100, so the minimum-size warning has nothing to say either.
    n = 10
    opens = [100.0] * 5 + [1000.0] * 5
    o, h, l, c = frames(opens)
    entries = mask(c.index, ["X"], [("X", 1)])
    exits = mask(c.index, ["X"], [("X", 3)])
    said = []
    runs = (dict(mode="spreadbet", point_size=1.0, spread_points=0.0, leverage=5.0, stake=300.0),
            dict(EQUITY, notional=500.0))
    sizes = []
    for terms in runs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=exits,
                                         init_cash=10_000, freq="D", **terms)
        said += [str(item.message) for item in caught if "buying power" in str(item.message)
                 or "minimum tradeable size" in str(item.message)]
        sizes.append(float(pf.orders.records_readable["Size"].iloc[0]))
    present = bool(said)
    return present, (f"orders filled in full (sizes {sizes[0]:g} and {sizes[1]:g}), yet {len(said)} warning(s): "
                     + " | ".join(text[:90] for text in said) if said else
                     f"orders filled in full (sizes {sizes[0]:g} and {sizes[1]:g}) and no warning")


# ---------------------------------------------------------------------------
# 23. Risk sizing from the balance or the equity (a required feature)
# ---------------------------------------------------------------------------

def check_risk_sizing_from_equity():
    # Risk 1% over a 10-point stop. The first trade is stopped out for a loss of 100, leaving
    # 9,900. Sized from the equity (or the balance: no position is open in between), the second
    # trade risks 1% of 9,900, so 9.9 units, not the 10 that 1% of the starting cash gives.
    opens = [100.0] * 10
    lows = [value - 0.5 for value in opens]
    lows[3] = 89.0                                               # the first trade's stop (90) is hit
    o, h, l, c = frames(opens, lows=lows)
    entries = mask(c.index, ["X"], [("X", 1), ("X", 5)])
    none = mask(c.index, ["X"], [])
    terms = dict(EQUITY, allow_fractional=True)
    sizes = {}
    for basis in ("equity", "balance"):
        try:
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=none, risk_pct=0.01,
                                         stop_points=10, init_cash=10_000, freq="D", risk_of=basis, **terms)
        except Exception as error:                                       # noqa: BLE001
            return True, f"risk_of={basis!r} is not accepted ({type(error).__name__}: {str(error)[:80]})"
        buys = pf.orders.records_readable
        buys = buys[buys["Side"] == "Buy"]
        sizes[basis] = [round(float(size), 4) for size in buys["Size"]]
    present = any(found != [10.0, 9.9] for found in sizes.values())
    return present, f"buy sizes {sizes} (should be [10.0, 9.9] for both)"


# ---------------------------------------------------------------------------
# 24. Bought outright and FX spot default to 2x leverage
# ---------------------------------------------------------------------------

def check_unlevered_defaults():
    from vectorbtx.instruments.defaults import max_leverage

    found = {mode: max_leverage(mode) for mode in ("equity", "fx_spot")}
    limits = {f"{mode}/{asset}": max_leverage(mode, asset) for mode in ("spreadbet", "cfd")
              for asset in ("fx", "index", "commodities", "stocks")}
    wanted = {"spreadbet/fx": 30, "spreadbet/index": 10, "spreadbet/commodities": 10, "spreadbet/stocks": 5}
    wanted.update({key.replace("spreadbet", "cfd"): value for key, value in wanted.items()})
    present = any(value != 1.0 for value in found.values()) or limits != wanted
    return present, f"defaults {found} (should be 1x, no borrowing); spread bets and CFDs {limits}"


# ---------------------------------------------------------------------------
# 25. Point sizes: an index's point, and a share priced in pence
# ---------------------------------------------------------------------------

def check_point_size_defaults():
    found = {}
    for label, terms in (("spreadbet index", dict(mode="spreadbet", asset_class="index")),
                         ("cfd index", dict(mode="cfd", asset_class="index", contract_size=1.0)),
                         ("spreadbet GBp share", dict(mode="spreadbet", asset_class="stocks", quote_currency="GBp")),
                         ("spreadbet EURUSD", dict(mode="spreadbet", asset_class="fx", base_currency="EUR",
                                                   quote_currency="USD")),
                         ("spreadbet USDJPY", dict(mode="spreadbet", asset_class="fx", base_currency="USD",
                                                   quote_currency="JPY"))):
        try:
            found[label] = vbt.x.Instrument(**terms).filled().point_size
        except Exception as error:                                       # noqa: BLE001
            found[label] = f"{type(error).__name__}"
    wanted = {"spreadbet index": 1.0, "cfd index": 1.0, "spreadbet GBp share": 1.0, "spreadbet EURUSD": 0.0001,
              "spreadbet USDJPY": 0.01}
    return found != wanted, f"point sizes {found} (should be {wanted})"


# ---------------------------------------------------------------------------
# 26. Stops and targets as a vector: per symbol and per bar from the signal bar, with risk
# sizing from them (a required feature). A stop in % of the price is vectorbt's own sl_stop,
# which works already; what is missing is a value per column or per bar, shifted with the
# signals, and risk sizing from any distance but a fixed one in points. The API names are the
# ones UPSTREAM.md suggests; rename with them.
# ---------------------------------------------------------------------------

def check_stops_that_vary():
    # Every entry is signalled at bar 1 and filled at bar 2's open, 100. Bar 3's low is 97.5 and
    # bar 4's is 94.5, so a stop at 98 is hit on bar 3, and one at 97 or 95 on bar 4.
    opens = [100.0] * 8
    lows = [99.5, 99.5, 99.5, 97.5, 94.5, 99.5, 99.5, 99.5]
    index = pd.bdate_range("2024-01-01", periods=8)

    def per_bar(values):
        return pd.DataFrame({"X": [float(value) for value in values]}, index=index)

    def outcome(columns=("X",), **kwargs):
        """{column: (bought, (exit bar, exit price))}, or the error's name."""
        o, h, l, c = frames(opens, lows=lows, columns=columns)
        entries = mask(c.index, columns, [(column, 1) for column in columns])
        none = mask(c.index, columns, [])
        try:
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=none,
                                         instruments={column: EQUITY for column in columns}, init_cash=10_000,
                                         freq="D", **kwargs)
        except Exception as error:                                       # noqa: BLE001
            return type(error).__name__
        rows = pf.orders.records_readable
        found = {}
        for column in columns:
            mine = rows[rows["Column"] == column]
            buys, sells = mine[mine["Side"] == "Buy"], mine[mine["Side"] == "Sell"]
            bought = round(float(buys["Size"].iloc[0]), 4) if len(buys) else None
            sold = (c.index.get_loc(sells["Timestamp"].iloc[0]), round(float(sells["Price"].iloc[0]), 2)) \
                if len(sells) else None
            found[column] = (bought, sold)
        return found

    cases = {
        # One distance per symbol: 2 points on X, 5 on Y.
        "one stop per column": (dict(columns=("X", "Y"), size=1, stop_points={"X": 2, "Y": 5}),
                                {"X": (1.0, (3, 98.0)), "Y": (1.0, (4, 95.0))}),
        # ⚠️ Aligned with the signals and shifted with them: 2 at the signal bar is the stop, not the
        # 5 at the fill bar. (A native per-bar sl_stop is read at the fill bar today: lookahead for
        # a stop worked out from the signal bar's ATR.)
        "a stop per bar, from the signal bar": (dict(size=1, stop_points=per_bar([9, 2, 5, 9, 9, 9, 9, 9])),
                                                {"X": (1.0, (3, 98.0))}),
        # The same for vectorbt's own fractions given per bar: 2% at the signal bar, not 5% at the fill bar.
        "a % stop per bar, from the signal bar": (dict(size=1, sl_stop=per_bar([0.09, 0.02, 0.05, 0.09, 0.09, 0.09,
                                                                                0.09, 0.09])),
                                                  {"X": (1.0, (3, 98.0))}),
        # Risk sizing from whichever distance is given: 100 at risk over 2 (2% of 100, or 2 points) is 50.
        "risk sized from a % stop": (dict(risk_amount=100, sl_stop=0.02), {"X": (50.0, (3, 98.0))}),
        "risk sized from a stop per bar": (dict(risk_amount=100, stop_points=per_bar([9, 2, 5, 9, 9, 9, 9, 9])),
                                           {"X": (50.0, (3, 98.0))}),
        "risk sized from a trailing stop": (dict(risk_amount=100, trail_points=2), {"X": (50.0, None)}),
    }
    wrong = {}
    for name, (kwargs, wanted) in cases.items():
        found = outcome(**kwargs)
        if name == "risk sized from a trailing stop" and isinstance(found, dict):
            found = {"X": (found["X"][0], None)}                   # the trail moves the exit; the size is the point
        if found != wanted:
            wrong[name] = found
    detail = "; ".join(f"{name}: {found}" for name, found in wrong.items()) or "every case as wanted"
    return bool(wrong), f"{len(wrong)} of {len(cases)} cases not as wanted ({detail})"


# ---------------------------------------------------------------------------
# 27. A trailing stop in points widens as the price rises
# ---------------------------------------------------------------------------

def check_trailing_points_stay_a_distance():
    # Filled at 120 (bar 2's open), trail 2 points. The high reaches 200.01 on bar 6 and bar 7 dips
    # to 190, so the stop should fill 2 points behind the peak, at 198.01. A fraction of the fill
    # price (2 / 120) applied to the peak puts it at 196.68 instead.
    opens = [100.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 199.0, 199.0]
    highs = [value + 0.01 for value in opens]
    lows = [value - 0.01 for value in opens]
    lows[7] = 190.0
    o, h, l, c = frames(opens, highs, lows, opens)
    entries = mask(c.index, ["X"], [("X", 1)])
    none = mask(c.index, ["X"], [])
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=none,
                                 instruments={"X": EQUITY}, size=1, trail_points=2, init_cash=10_000, freq="D")
    sells = pf.orders.records_readable
    sells = sells[sells["Side"] == "Sell"]
    price = float(sells["Price"].iloc[0]) if len(sells) else float("nan")
    present = not abs(price - 198.01) < 1e-6
    return present, f"the trailing stop fills at {price:.2f}, {200.01 - price:.2f} points behind the peak of 200.01 " \
                    f"(should be 198.01, 2 points behind)"


# ---------------------------------------------------------------------------
# 28. Risk sizing without a stop: sized from the leverage
# ---------------------------------------------------------------------------

def check_risk_sizing_without_a_stop():
    # Risk 1% of 10,000 (100) with no stop. The rule asked for: the stop is the move that takes
    # the trade's whole margin, 1 / leverage of the price, so the position is worth the risk times
    # the leverage. Bought outright at 1x that is 100 of stock (the price must fall to zero to
    # lose it); at 5x it is 500 (a 20% fall loses the 100).
    opens = [100.0] * 6
    o, h, l, c = frames(opens)
    entries = mask(c.index, ["X"], [("X", 1)])
    none = mask(c.index, ["X"], [])
    values = {}
    for leverage in (1.0, 5.0):
        terms = dict(EQUITY, allow_fractional=True, leverage=leverage)
        try:
            pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=none, risk_pct=0.01,
                                         init_cash=10_000, freq="D", **terms)
        except Exception as error:                                       # noqa: BLE001
            return True, f"risk_pct with no stop raises ({type(error).__name__}: {str(error)[:90]})"
        buys = pf.orders.records_readable
        buys = buys[buys["Side"] == "Buy"]
        values[leverage] = round(float((buys["Size"] * buys["Price"]).sum()), 2)
    present = values != {1.0: 100.0, 5.0: 500.0}
    return present, f"position values {values} (should be 100 at 1x and 500 at 5x)"


# ---------------------------------------------------------------------------
# 29. Points on a price converted to the account currency
# ---------------------------------------------------------------------------

def check_points_on_converted_prices():
    # A USD share at 100, converted to a GBP account at 0.8 (80 pounds). A point is a cent,
    # so a 50-point stop is 50 cents (99.50 dollars, 79.60 pounds), 100 pounds at risk over
    # it is 250 shares (40 pence a share), and a 1-point spread buys at 100.005 dollars
    # (80.004 pounds). Points applied to the converted price make the stop 50 pence instead.
    index = pd.bdate_range("2024-01-01", periods=8)
    native = pd.DataFrame({"X": [100.0] * 8}, index=index)
    low = native * 0.9975
    low.iloc[3] = 99.0
    terms = {"mode": "equity", "point_size": 0.01, "quote_currency": "USD", "spread_points": 0.0,
             "leverage": 1.0, "allow_fractional": True}
    universe = vbt.x.load_universe({"X": vbt.x.Instrument(**terms)})
    fx = pd.Series(0.8, index=index, name="USDGBP")
    o, h, l, c = [vbt.x.to_base_currency(f, base="GBP", fx=fx, instruments=universe, fx_markup=0.0)
                  for f in (native, native * 1.0025, low, native)]
    entries = mask(index, ["X"], [("X", 0)])
    none = mask(index, ["X"], [])
    common = dict(close=c, open=o, high=h, low=l, entries=entries, exits=none, account_currency="GBP",
                  fx=fx, freq="D")
    pf = vbt.x.PortfolioBacktest(instruments={"X": terms}, size=10, stop_points=50, init_cash=10_000, **common)
    stop = float(pf.orders.records_readable["Price"].iloc[1])
    pf = vbt.x.PortfolioBacktest(instruments={"X": terms}, risk_amount=100, stop_points=50, init_cash=100_000,
                                 **common)
    risked = float(pf.orders.records_readable["Size"].iloc[0])
    pf = vbt.x.PortfolioBacktest(instruments={"X": dict(terms, spread_points=1.0)}, size=1, init_cash=10_000,
                                 **common)
    spread = float(pf.orders.records_readable["Price"].iloc[0])
    present = abs(stop - 79.6) > 1e-6 or abs(risked - 250) > 1e-6 or abs(spread - 80.004) > 1e-6
    return present, f"USD share at 100 in a GBP account at 0.8: 50-point stop fills at {stop:.4f} (should be " \
                    f"79.60), 100 at risk over it buys {risked:g} (should be 250), a 1-point spread buys at " \
                    f"{spread:.4f} (should be 80.004)"


# ---------------------------------------------------------------------------
# 30. notional_pct: a position worth a share of the account (issue 23's optional rule)
# ---------------------------------------------------------------------------

def check_notional_pct():
    # Half the account each time. The first trade (50 shares at 100) is stopped out at 90
    # for a loss of 500, so from the equity the second is half of 9,500: 47.5 shares.
    lows = [99.5] * 10
    lows[3] = 89.0
    o, h, l, c = frames([100.0] * 10, lows=lows)
    entries = mask(c.index, ["X"], [("X", 1), ("X", 5)])
    none = mask(c.index, ["X"], [])
    try:
        pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, entries=entries, exits=none, notional_pct=0.5,
                                     stop_points=10, risk_of="equity", init_cash=10_000, freq="D",
                                     **dict(EQUITY, allow_fractional=True))
    except Exception as error:                                           # noqa: BLE001
        return True, f"notional_pct= is not accepted ({type(error).__name__}: {str(error)[:80]})"
    buys = pf.orders.records_readable
    sizes = [round(float(size), 4) for size in buys[buys["Side"] == "Buy"]["Size"]]
    return sizes != [50.0, 47.5], f"buy sizes {sizes} (should be [50.0, 47.5])"


# ---------------------------------------------------------------------------
# 31. Stops at a price level (issue 26's optional item 4)
# ---------------------------------------------------------------------------

def check_stop_levels():
    # The signal bar closes at 101 with its EMA at 97.3; the next bar opens at 100. The stop
    # should sit on the EMA (97.3), not 3.7 below the fill (96.3, a distance taken at the
    # close). A long whose stop level is above its fill is rejected. A trailing level moves
    # up with each completed bar and never back.
    index = pd.bdate_range("2024-01-01", periods=8)
    closes = [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    o, h, l, c = frames([100.0] * 8, [101.5] * 8, [99.5, 99.5, 99.5, 97.0, 96.0, 99.5, 99.5, 99.5], closes)
    entries = mask(index, ["X"], [("X", 1)])
    none = mask(index, ["X"], [])
    common = dict(close=c, open=o, high=h, low=l, entries=entries, exits=none, instruments={"X": EQUITY},
                  size=1, init_cash=10_000, freq="D")
    failures = []
    try:
        level = pd.DataFrame({"X": [90.0, 97.3, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]}, index=index)
        pf = vbt.x.PortfolioBacktest(stop_price=level, **common)
        sells = pf.orders.records_readable.query("Side == 'Sell'")
        stop = float(sells["Price"].iloc[0]) if len(sells) else None
        if stop != 97.3:
            failures.append(f"stop at the signal bar's EMA: sold at {stop}, should be 97.3")
        pf = vbt.x.PortfolioBacktest(stop_price=101.0, **common)
        if len(pf.orders.records):
            failures.append("a long with its stop level above the fill was placed; it should be rejected")
        opens = [100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 107.0, 105.0, 103.0, 101.0]
        to, th, tl, tc = frames(opens, [v + 0.5 for v in opens], [v - 0.5 for v in opens])
        trail = pd.DataFrame({"X": [v - 3 for v in opens]}, index=tc.index)
        trail.iloc[6] = 104.6                                     # a dip that must not loosen it
        pf = vbt.x.PortfolioBacktest(close=tc, open=to, high=th, low=tl, entries=mask(tc.index, ["X"], [("X", 1)]),
                                     exits=mask(tc.index, ["X"], []), instruments={"X": EQUITY}, size=1,
                                     trail_price=trail, init_cash=10_000, freq="D")
        sells = pf.orders.records_readable.query("Side == 'Sell'")
        exit_price = float(sells["Price"].iloc[0]) if len(sells) else None
        if exit_price != 105.0:
            failures.append(f"trailing level: sold at {exit_price}, should be 105 (the stop never loosens)")
    except Exception as error:                                           # noqa: BLE001
        return True, f"stop_price= / trail_price= not accepted ({type(error).__name__}: {str(error)[:80]})"
    return bool(failures), "; ".join(failures) or "on the level across the gap; wrong side rejected; trail ratchets"


# ---------------------------------------------------------------------------
# 32, 33 and 34. The leverage cap (max_size): closes, the starting cash, the increment
# ---------------------------------------------------------------------------

def _orders(pf):
    """[(bar, side, size)] for a one-column portfolio."""
    records = pf.orders.records_readable
    return [(int(pf.wrapper.index.get_loc(stamp)), side, round(float(size), 4))
            for stamp, side, size in zip(records["Timestamp"], records["Side"], records["Size"])]


def _held_at_end(pf):
    return round(float(np.asarray(pf.assets()).ravel()[-1]), 4)


def check_cap_cuts_closing_orders():
    # 100 shares bought at 100 (a notional of 10,000 at 1x). The exit fills at 100.5, where
    # 1 x 10,000 buys 99.50 shares: the exit must still sell all 100, and the second round trip
    # must happen. Then a weights book: a target of 1.0 bought at 100, cut to 0 at 150, must
    # sell everything, not the 66.67 shares 10,000 buys at 150.
    opens = [100, 100, 100, 100.5, 100.5, 100.5, 100, 100, 100, 100]
    o, h, l, c = frames(opens)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": EQUITY},
                                 entries=mask(c.index, ["X"], [("X", 0), ("X", 6)]),
                                 exits=mask(c.index, ["X"], [("X", 2), ("X", 8)]),
                                 notional=10_000, init_cash=10_000, freq="D")
    rule_orders = _orders(pf)
    rule_held = _held_at_end(pf)
    opens = [100, 100, 120, 140, 150, 150, 150, 150]
    o, h, l, c = frames(opens)
    weights = pd.DataFrame({"X": [1.0] * 5 + [0.0] * 3}, index=c.index)
    schedule = pd.Series([True, False, False, False, False, True, False, False], index=c.index)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l,
                                 instruments={"X": dict(EQUITY, allow_fractional=True)},
                                 weights=weights, schedule=schedule, leverage_budget=False,
                                 init_cash=10_000, freq="D")
    book_held = _held_at_end(pf)
    want = [(1, "Buy", 100.0), (3, "Sell", 100.0), (7, "Buy", 100.0), (9, "Sell", 100.0)]
    present = rule_orders != want or rule_held != 0 or book_held != 0
    return present, (f"rule strategy: orders {rule_orders}, {rule_held} held at the end "
                     f"(should be {want}, flat); weights cut to 0: {book_held} held (should be 0)")


def check_cap_from_the_account():
    # A short from 100 covered at 80 leaves equity of 12,000. The long that follows, at 80, is
    # worth the whole equity (notional_pct=1.0, risk_of="equity"): 150 shares, inside 1x of
    # 12,000. A cap of 1 x the starting cash (10,000 / 80) allows only 125.
    opens = [100, 100, 90, 80, 80, 80, 80, 80]
    o, h, l, c = frames(opens)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": EQUITY},
                                 entries=mask(c.index, ["X"], [("X", 4)]), exits=mask(c.index, ["X"], []),
                                 short_entries=mask(c.index, ["X"], [("X", 0)]),
                                 short_exits=mask(c.index, ["X"], [("X", 2)]),
                                 notional_pct=1.0, risk_of="equity", init_cash=10_000, freq="D")
    equity = round(float(np.asarray(pf.value()).ravel()[4]), 2)
    buys = [size for bar, side, size in _orders(pf) if side == "Buy" and bar == 5]
    return buys != [150.0], f"equity {equity} at the signal bar; the long bought {buys} (should be [150.0])"


def check_cut_orders_on_the_increment():
    # 100 whole shares asked for at 150 with 10,000 of cash, cut down to what is funded
    # (when_cash_short="partial"): 66 shares, a whole number, not 66.67.
    o, h, l, c = frames([150.0] * 6)
    pf = vbt.x.PortfolioBacktest(close=c, open=o, high=h, low=l, instruments={"X": EQUITY},
                                 entries=mask(c.index, ["X"], [("X", 0)]), exits=mask(c.index, ["X"], []),
                                 size=100, when_cash_short="partial", init_cash=10_000, freq="D")
    bought = [size for _, side, size in _orders(pf) if side == "Buy"]
    return bought != [66.0], f"bought {bought} shares (should be [66.0]: the increment is one share)"


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
    (21, "min_trade never skips a trade: it compares targets, not holdings", check_min_trade_skips_small_trades),
    (22, "The sizing warnings measure every bar, not the bars an order is placed on",
     check_sizing_warnings_measure_order_bars),
    (23, "Risk sizing from the balance or the equity (required feature)", check_risk_sizing_from_equity),
    (24, "Bought outright and FX spot default to 2x leverage", check_unlevered_defaults),
    (25, "An index's point size defaults to 0.01; a share priced in pence is not recognised",
     check_point_size_defaults),
    (26, "Stops and targets as a vector (per symbol, per bar), and risk sized from them (required feature)",
     check_stops_that_vary),
    (27, "A trailing stop in points widens as the price rises", check_trailing_points_stay_a_distance),
    (28, "Risk sizing without a stop, from the leverage (required feature)", check_risk_sizing_without_a_stop),
    (29, "Points on a converted price are applied in the account currency", check_points_on_converted_prices),
    (30, "notional_pct: a position worth a share of the account (issue 23, optional)", check_notional_pct),
    (31, "Stops and targets at a price level (issue 26, item 4)", check_stop_levels),
    (32, "The leverage cap (max_size) cuts the orders that close a position", check_cap_cuts_closing_orders),
    (33, "The leverage cap is leverage x the starting cash, not the account", check_cap_from_the_account),
    (34, "An order cut down to what is funded is not rounded to the increment", check_cut_orders_on_the_increment),
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
