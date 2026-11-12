"""Test 9: portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import compile as C, engine
from openstrategy_studio.spec import apply_params


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


def test_min_trade_suppression_shows_in_drift(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    spec = momentum_spec
    spec["schedule"]["min_trade"] = 0.2           # suppress most rebalances
    run = engine.run_backtest(spec, ohlcv_multi)
    drift = run["pf"].x.drift()
    assert np.abs(np.asarray(drift, dtype=float)).sum() > 0
