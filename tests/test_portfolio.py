"""Test 9: portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import compile as C, engine
from openstrategy_studio.spec import apply_params


def one(condition):
    """A rule of one condition: the shape every rule slot takes (conditions and a logic line)."""
    return {"conditions": {"A": condition}, "logic": "A"}


def test_inverse_vol_rows_sum_to_one(ohlcv_multi):
    weights = C.inverse_vol_weight(ohlcv_multi["close"], window=20).dropna()
    assert np.allclose(weights.sum(axis=1), 1.0)
    # The least volatile asset gets the largest weight.
    vol = ohlcv_multi["close"].pct_change(fill_method=None).rolling(20).std().loc[weights.index]
    assert (weights.idxmax(axis=1) == vol.idxmin(axis=1)).all()


def test_equal_weight_monthly_matches_pandas(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = apply_params(momentum_spec)
    out = C.compile_weights(spec, ohlcv_multi)
    weights, schedule = out["weights"], out["schedule"]

    # Independent calculation: top 2 by 126-bar momentum with score >= 0, equal weight.
    close = ohlcv_multi["close"]
    momentum = close / close.shift(126) - 1
    rank = momentum.rank(axis=1, ascending=False, method="first")
    selected = (rank <= 2) & (momentum >= 0) & momentum.notna()
    expected = selected.astype(float).div(selected.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pd.testing.assert_frame_equal(weights, expected, check_names=False)

    sums = weights.sum(axis=1).round(12)
    assert set(sums.unique()) <= {0.0, 1.0}, "rows sum to 1 (invested) or 0 (cash)"
    first_of_month = close.index.to_series().groupby(close.index.to_period("M")).transform("min") == close.index
    pd.testing.assert_series_equal(schedule.astype(bool), pd.Series(first_of_month.to_numpy(), index=close.index),
                                   check_names=False)


def test_calendar_and_threshold_uses_drift_anchor(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = apply_params(momentum_spec)
    spec["schedule"]["threshold"] = 0.02
    calendar_only = C.compile_weights(dict(spec, schedule=dict(spec["schedule"], threshold=None)), ohlcv_multi)
    combined = C.compile_weights(spec, ohlcv_multi)
    assert not (combined["schedule"] & ~calendar_only["schedule"]).any(), "'and' never adds dates"
    assert combined["schedule"].sum() < calendar_only["schedule"].sum(), \
        "with the drift anchor the threshold is not always true"


def test_intrabar_drawdown_brackets_the_close_drawdown(ma_spec, ohlcv, momentum_spec, ohlcv_multi):
    """The intrabar curve is never shallower than the close-to-close one, at every bar.

    It is vectorbtX's own number, with or without shared cash (every weights portfolio
    shares it). vectorbtX ace29e2 counted shared starting cash once per column, which
    made its figure shallower than the close-to-close one; 16293a2 fixed that
    (docs/UPSTREAM.md issue 11), so the two now agree everywhere.
    """
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis

    fast = {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20}       # enough trades on 400 bars
    single = engine.run_backtest(ma_spec, ohlcv, params=fast)
    shared = engine.run_backtest(momentum_spec, ohlcv_multi)
    assert not single["pf"].cash_sharing and shared["pf"].cash_sharing
    for run in (single, shared):
        series = analysis.extract_results(run)["series"]
        assert (series["intrabar_drawdown"] <= series["drawdown"] + 1e-12).all(), "intrabar is never shallower"
    series = analysis.extract_results(single)["series"]
    assert -series["intrabar_drawdown"].min() == pytest.approx(single["pf"].x.intrabar_drawdown(), rel=1e-12)
    series = analysis.extract_results(shared)["series"]
    assert -series["intrabar_drawdown"].min() == pytest.approx(shared["pf"].x.intrabar_drawdown(), rel=1e-12)
    # ...and on shared cash it is deeper than the close-to-close drawdown, as it must be.
    assert -series["intrabar_drawdown"].min() >= -series["drawdown"].min()


def test_intrabar_drawdown_of_a_short_uses_the_high(ma_spec):
    """A short is hurt by the high: a bar that spikes up must show in the intrabar curve."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis

    index = pd.bdate_range("2024-01-01", periods=8)
    closes = [100.0] * 8
    highs = [100.5, 100.5, 100.5, 104.0, 100.5, 100.5, 100.5, 100.5]          # bar 3 spikes up, closes flat
    frame = lambda values: pd.DataFrame({"X": values}, index=index)
    data = {"open": frame(closes), "high": frame(highs), "low": frame([99.5] * 8), "close": frame(closes),
            "volume": frame([1000.0] * 8)}
    code = "out = close * 0.0\nout.iloc[0] = 1.0\nreturn out"
    spec = {"version": 2, "name": "short", "setup": {"kind": "signals", "trading": "equity"},
            "data": {"symbols": ["X"], "timeframe": "1d", "start": "2000-01-01", "end": "2100-01-01"},
            "instruments": {"X": {"point_size": 1.0, "quote_currency": "USD",
                                  "cost_currency": "USD", "spread_points": 0.0}},
            "features": {"pulse": {"indicator": "studio:python", "inputs": {}, "params": {"code": code}}},
            "rules": {"long_entry": None, "long_exit": None, "short_exit": None,
                      "short_entry": one({"left": {"feature": "pulse"}, "op": ">", "right": {"const": 0.5}})},
            "risk": {"sizing": {"size": 10}, "stop": {"kind": "none"}}, "account": {"init_cash": 10_000, "freq": "D"}}
    series = analysis.extract_results(engine.run_backtest(spec, data))["series"]
    # Short 10 shares from 100: at the 104 high the book is down 40 of 10,000 (plus the 0.5 bar-2 swing).
    assert series["intrabar_drawdown"].min() == pytest.approx(-(40 + 5) / (10_000 + 5), rel=1e-9)
    assert series["drawdown"].min() == pytest.approx(0.0, abs=1e-12), "the closes never move"


def test_min_trade_suppression_shows_in_drift(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = momentum_spec
    spec["schedule"]["min_trade"] = 0.2           # suppress most rebalances
    run = engine.run_backtest(spec, ohlcv_multi)
    drift = run["pf"].x.drift()
    assert np.abs(np.asarray(drift, dtype=float)).sum() > 0
