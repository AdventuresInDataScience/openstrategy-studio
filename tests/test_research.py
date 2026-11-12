"""Pattern research and the pipeline diagram (phase 3)."""

import numpy as np
import pytest

from openstrategy_studio import blocks, research
from openstrategy_studio.spec import list_templates, load_template


@pytest.fixture
def every_tenth_bar(monkeypatch):
    """A stand-in detector: bullish on bars 10, 20, 30 ... (completed bars only)."""
    def detector(o, h, l, c, **keywords):
        out = np.zeros(len(c), dtype="int8")
        out[10::10] = 1
        return out
    monkeypatch.setattr(blocks, "pattern_function", lambda module, name: detector)


def test_forward_returns_start_at_the_next_open(ohlcv):
    forward = research.forward_returns(ohlcv, 3)
    t = 50
    expected = ohlcv["close"].iloc[t + 3, 0] / ohlcv["open"].iloc[t + 1, 0] - 1
    assert forward.iloc[t, 0] == pytest.approx(expected)
    assert forward.iloc[-3:].isna().all().all()


def test_study_measures_signed_returns(ohlcv, every_tenth_bar):
    table = research.study(ohlcv, [{"module": "candle", "name": "fake", "direction": "bullish", "params": {}}],
                           horizons=(1, 5), min_hits=5)
    row = table.iloc[0]
    hits = list(range(10, len(ohlcv["close"]), 10))
    assert row["hits"] == len(hits) and row["bullish"] == len(hits)
    forward = research.forward_returns(ohlcv, 5)["EURUSD=X"]
    values = forward.iloc[hits].dropna()
    assert row["mean 5"] == pytest.approx(values.mean())
    assert row["hit rate 5"] == pytest.approx((values > 0).mean())
    assert row["excess 5"] == pytest.approx(values.mean() - np.nanmean(forward))


def test_bearish_signals_are_judged_as_shorts(ohlcv, monkeypatch):
    monkeypatch.setattr(blocks, "pattern_function",
                        lambda module, name: (lambda o, h, l, c, **k: np.where(np.arange(len(c)) % 10 == 5, -1, 0).astype("int8")))
    table = research.study(ohlcv, [{"module": "candle", "name": "bear", "params": {}}], horizons=(3,), min_hits=5)
    forward = research.forward_returns(ohlcv, 3)["EURUSD=X"]
    hits = [t for t in range(5, len(forward), 10)]
    assert table.iloc[0]["mean 3"] == pytest.approx(-forward.iloc[hits].dropna().mean())


def test_event_path(ohlcv, every_tenth_bar):
    path = research.event_path(ohlcv, "candle", "fake", horizon=5)
    assert list(path.index) == [1, 2, 3, 4, 5]
    assert path["n"].iloc[0] >= path["n"].iloc[-1] > 0
    assert (path["p25"] <= path["p75"]).all()


def test_add_pattern_to_rule(ma_spec):
    research.add_pattern_to_rule(ma_spec, "long_entry", "candle", "hammer", "bullish")
    assert ma_spec["rules"]["long_entry"]["all"][-1]["pattern"] == "hammer"
    research.add_pattern_to_rule(ma_spec, "short_entry", "chart", "double_top", "bearish")
    assert ma_spec["rules"]["short_entry"] == {"all": [{"pattern": "double_top", "module": "chart", "side": "bearish",
                                                        "params": {}}]}


def test_slow_patterns():
    assert research.is_slow("hs_top") and research.is_slow("busted_hs_top") and not research.is_slow("hammer")


def test_pipeline_diagram_for_every_template():
    pytest.importorskip("plotly")
    from openstrategy_studio import analysis
    for name in list_templates():
        dot = analysis.pipeline_dot(load_template(name))
        assert dot.startswith("digraph pipeline {") and dot.endswith("}")
        assert dot.count("{") == dot.count("}")
        assert "-> door;" in dot
    dot = analysis.pipeline_dot(load_template("ma_cross"))
    assert "feature_fast -> rule_long_entry;" in dot and "rule_long_entry -> door;" in dot
    dot = analysis.pipeline_dot(load_template("momentum_top_n"))
    assert "feature_mom -> selection;" in dot and "schedule -> door;" in dot
    assert "alternatives searched" in analysis.pipeline_dot(load_template("structure_search"))
