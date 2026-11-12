"""Test 8: the exported script and the in-app run agree."""

import ast

import pandas as pd
import pytest

from openstrategy_studio import engine, spec as S
from openstrategy_studio.compile import export_script


def test_every_template_exports_valid_python():
    for name in S.list_templates():
        source = export_script(S.load_template(name))
        ast.parse(source)
        assert "import vectorbtx as vbt" in source
        assert "def run(open_, high, low, close, volume, fx=None):" in source


def test_export_shifts_shorts_and_names_every_argument(ma_spec):
    source = export_script(ma_spec)
    assert "short_entries = short_entries.shift(1, fill_value=False)" in source
    for keyword in engine.call_plan(S.apply_params(ma_spec))["args"]:
        assert f"    {keyword}=" in source


def run_exported(spec, data):
    namespace = {"__name__": "exported"}
    exec(compile(export_script(spec), "<exported>", "exec"), namespace)
    return namespace["run"](data["open"], data["high"], data["low"], data["close"], data["volume"],
                            fx=data.get("fx"))


def test_export_equivalence_signals(ma_spec, ohlcv):
    pytest.importorskip("vectorbtx")
    in_app = engine.run_backtest(ma_spec, ohlcv)["pf"]
    exported = run_exported(ma_spec, ohlcv)
    pd.testing.assert_series_equal(in_app.stats(), exported.stats())


def test_export_equivalence_portfolio(momentum_spec, ohlcv_multi):
    pytest.importorskip("vectorbtx")
    in_app = engine.run_backtest(momentum_spec, ohlcv_multi)["pf"]
    exported = run_exported(momentum_spec, ohlcv_multi)
    pd.testing.assert_series_equal(in_app.stats(), exported.stats())


def test_export_equivalence_every_template_shape(ohlcv):
    """Rules with patterns, filters, shifts and channels export the same signals."""
    pytest.importorskip("vectorbtx")
    for name in ("donchian_breakout", "rsi_reversion"):
        spec = S.load_template(name)
        symbol = spec["data"]["symbols"][0]
        data = {field: frame.rename(columns={"EURUSD=X": symbol}) for field, frame in ohlcv.items()}
        in_app = engine.run_backtest(spec, data)["pf"]
        pd.testing.assert_series_equal(in_app.stats(), run_exported(spec, data).stats())
