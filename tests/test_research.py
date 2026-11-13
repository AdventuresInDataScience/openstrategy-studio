"""Pattern research and the pipeline diagram (phase 3)."""

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import indicators, research
from openstrategy_studio.spec import list_templates, load_template, validate_spec


@pytest.fixture
def every_tenth_bar(monkeypatch):
    """A stand-in detector: bullish on bars 10, 20, 30 ... (completed bars only)."""
    def detector(module, name, params, o, h, l, c, v):
        out = np.zeros(len(c), dtype="int8")
        out[10::10] = 1
        return pd.DataFrame({column: out for column in c.columns}, index=c.index)
    monkeypatch.setattr(indicators, "run_pattern", detector)


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
    def detector(module, name, params, o, h, l, c, v):
        values = np.where(np.arange(len(c)) % 10 == 5, -1, 0).astype("int8")
        return pd.DataFrame({column: values for column in c.columns}, index=c.index)
    monkeypatch.setattr(indicators, "run_pattern", detector)
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
    assert ma_spec["features"]["hammer"]["indicator"] == "pattern:candle:hammer"
    assert ma_spec["rules"]["long_entry"]["conditions"]["B"] == {"pattern": "hammer", "side": "bullish"}
    assert ma_spec["rules"]["long_entry"]["logic"] == "A and B"
    research.add_pattern_to_rule(ma_spec, "short_entry", "chart", "double_top", "bearish")
    assert ma_spec["rules"]["short_entry"] == {"conditions": {"A": {"pattern": "double_top", "side": "bearish"}},
                                               "logic": "A"}
    assert validate_spec(ma_spec) == []


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
    assert "versions searched" in analysis.pipeline_dot(load_template("structure_search"))
    dot = analysis.pipeline_dot(load_template("shared_account_ma_cross"))
    assert "one shared account" in dot and "feature_strength -> door" in dot


def test_every_pattern_row_comes_from_the_catalogue():
    rows = research.pattern_rows()
    assert len(rows) == sum(1 for item in indicators.catalogue().values() if item["source"] == "pattern")
    assert {"module", "name", "direction", "family"} <= set(rows[0])
