"""Shared fixtures. Tests never touch the network.

make_ohlcv is a seeded random walk in which high >= max(open, close) and
low <= min(open, close) always hold.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import spec as spec_module

FIXTURES = Path(__file__).parent / "fixtures"


def make_ohlcv(n=400, symbols=("AAA",), seed=0, freq="B", start="2020-01-01", base=1.1):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq)
    fields = {field: {} for field in ("open", "high", "low", "close", "volume")}
    for number, symbol in enumerate(symbols):
        close = base * (1 + 0.1 * number) * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, n - 1))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
        fields["open"][symbol] = open_
        fields["high"][symbol] = high
        fields["low"][symbol] = low
        fields["close"][symbol] = close
        fields["volume"][symbol] = rng.integers(1_000, 5_000, n).astype(float)
    return {field: pd.DataFrame(values, index=index) for field, values in fields.items()}


@pytest.fixture
def ohlcv():
    return make_ohlcv(400, ("EURUSD=X",))


@pytest.fixture
def ohlcv_multi():
    return make_ohlcv(400, ("AAA", "BBB", "CCC", "DDD"), seed=3, base=100.0)


@pytest.fixture
def ma_spec():
    """The golden MA-cross spec (plan section 5, example 1) with a static rate table."""
    return copy.deepcopy(spec_module.load_template("ma_cross"))


@pytest.fixture
def shared_spec():
    """Three stocks trading from one shared account; competing entries go strongest momentum first."""
    return copy.deepcopy(spec_module.load_template("shared_account_ma_cross"))


@pytest.fixture
def ohlcv_stocks():
    return make_ohlcv(400, ("AAPL", "MSFT", "NVDA"), seed=3, base=100.0)


@pytest.fixture
def momentum_spec():
    spec = copy.deepcopy(spec_module.load_template("momentum_top_n"))
    spec["data"]["symbols"] = ["AAA", "BBB", "CCC", "DDD"]
    instrument = next(iter(spec["instruments"].values()))
    spec["instruments"] = {symbol: dict(instrument) for symbol in spec["data"]["symbols"]}
    spec["selection"]["top_n"] = 2
    return spec
