"""Tests 7 and 11: every block compiles; the pattern cache keys on keywords."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import blocks, compile as C
from openstrategy_studio.blocks import BLOCKS, block_defaults

PANDAS_BLOCKS = ["momentum", "highest", "lowest", "zscore", "volatility", "python"]


def check_output(name, outputs, close):
    assert set(outputs) == set(BLOCKS[name]["outputs"]), name
    for output, frame in outputs.items():
        assert frame.shape == close.shape, (name, output)
        assert list(frame.columns) == list(close.columns), (name, output)
        values = frame.to_numpy(dtype=float)
        assert not np.isinf(values).any(), (name, output)
        # NaN only in the warm-up: once a column has a value, it keeps having one.
        for column in range(values.shape[1]):
            finite = np.isfinite(values[:, column])
            if finite.any():
                first = int(np.argmax(finite))
                assert finite[first:].all(), (name, output, "NaN after warm-up")


def compile_one(name, data):
    features = {"x": block_defaults(name)}
    return C.compile_features(features, data)["x"]


def test_pandas_blocks_compile(ohlcv_multi):
    for name in PANDAS_BLOCKS:
        check_output(name, compile_one(name, ohlcv_multi), ohlcv_multi["close"])


def test_every_block_compiles(ohlcv_multi):
    pytest.importorskip("vectorbt")
    pytest.importorskip("ta_patterns")
    import vectorbt
    if getattr(vectorbt, "STUB", False):
        pytest.skip("local stand-in, not the real vectorbt")
    for name in BLOCKS:
        check_output(name, compile_one(name, ohlcv_multi), ohlcv_multi["close"])


def test_feature_of_feature(ohlcv):
    features = {"base": block_defaults("momentum"), "z": dict(block_defaults("zscore"), source={"feature": "base"})}
    out = C.compile_features(features, ohlcv)
    expected = (out["base"]["momentum"] - out["base"]["momentum"].rolling(20).mean()) / \
        out["base"]["momentum"].rolling(20).std()
    pd.testing.assert_frame_equal(out["z"]["zscore"], expected)


def test_pattern_cache_keys_on_keywords(ohlcv, monkeypatch):
    calls = []

    def fake_detector(o, h, l, c, shadow_factor=2.0, require_trend=False):
        calls.append(shadow_factor)
        return (c > o).astype("int8")

    monkeypatch.setattr(blocks, "pattern_function", lambda module, name: fake_detector)
    C.FEATURE_CACHE.clear()
    fingerprint = C.data_fingerprint(ohlcv)
    leaf = {"pattern": "hammer", "module": "candle", "side": "bullish", "params": {"shadow_factor": 1.5}}
    C._pattern_signal(leaf, ohlcv, fingerprint)
    C._pattern_signal(dict(leaf, params={"shadow_factor": 2.5}), ohlcv, fingerprint)
    pattern_keys = [key for key in C.FEATURE_CACHE if key[0] == "pattern"]
    assert len(pattern_keys) == 2
    C._pattern_signal(leaf, ohlcv, fingerprint)          # identical keywords: a cache hit
    assert calls == [1.5, 2.5]


def test_python_block_rejects_wrong_shape(ohlcv):
    feature = {"block": "python", "code": "return close.iloc[:10]"}
    with pytest.raises(ValueError, match="shaped like"):
        C.compile_features({"bad": feature}, ohlcv)
