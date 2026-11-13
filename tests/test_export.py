"""Test 8: the exported script and the in-app run agree."""

import ast
import copy

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


def test_export_names_every_argument_and_shifts_nothing(ma_spec):
    source = export_script(ma_spec)
    # The front door shifts all four signal arrays itself; a shift here would double it.
    assert "short_entries = short_entries.shift" not in source
    assert "short_exits = short_exits.shift" not in source
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


def test_export_equivalence_shared_account(shared_spec, ohlcv_stocks):
    """One account across three symbols, competing entries ranked by a feature's score frame."""
    pytest.importorskip("vectorbtx")
    source = export_script(shared_spec)
    assert "cash='shared'" in source and "entry_priority=entry_priority" in source
    assert "entry_priority = full(F['strength__real'], close)" in source
    in_app = engine.run_backtest(shared_spec, ohlcv_stocks)["pf"]
    exported = run_exported(shared_spec, ohlcv_stocks)
    pd.testing.assert_series_equal(in_app.stats(), exported.stats())
    lowest = dict(shared_spec, account=dict(shared_spec["account"],
                                            entry_priority=dict(shared_spec["account"]["entry_priority"],
                                                                highest_first=False)))
    assert "entry_priority = -full(" in export_script(lowest)
    pd.testing.assert_series_equal(engine.run_backtest(lowest, ohlcv_stocks)["pf"].stats(),
                                   run_exported(lowest, ohlcv_stocks).stats())


def test_export_equivalence_every_template_shape(ohlcv):
    """Rules with patterns, filters, shifts and channels export the same signals."""
    pytest.importorskip("vectorbtx")
    for name in ("donchian_breakout", "rsi_reversion", "candlestick_reversal"):
        spec = S.load_template(name)
        symbol = spec["data"]["symbols"][0]
        data = {field: frame.rename(columns={"EURUSD=X": symbol}) for field, frame in ohlcv.items()}
        in_app = engine.run_backtest(spec, data)["pf"]
        pd.testing.assert_series_equal(in_app.stats(), run_exported(spec, data).stats())


def test_export_equivalence_catalogue_logic_and_order_timing(ma_spec):
    """TA-Lib, pandas-ta and the Studio's own entries, a feature of a feature, a logic line with
    xor and not, and the intraday order masks: the script and the app agree."""
    pytest.importorskip("vectorbtx")
    from conftest import make_ohlcv
    from openstrategy_studio import indicators

    spec = copy.deepcopy(ma_spec)
    spec["data"]["timeframe"] = "1h"
    spec["account"]["freq"] = "1h"
    spec["features"]["kc"] = indicators.new_feature("talib:KC")
    spec["features"]["ce"] = indicators.new_feature("pandas_ta:CE")
    spec["features"]["band"] = dict(indicators.new_feature("studio:percent_band"), inputs={"source": {"feature": "slow"}})
    entry = spec["rules"]["long_entry"]
    entry["conditions"]["B"] = {"left": {"price": "close"}, "op": ">", "right": {"feature": "kc", "output": "lowerband"}}
    entry["conditions"]["C"] = {"left": {"price": "close"}, "op": "<", "right": {"feature": "band", "output": "upper"}}
    entry["conditions"]["D"] = {"left": {"price": "close"}, "op": ">", "right": {"feature": "ce", "output": "ce_l"}}
    entry["logic"] = "(A or (B xor C)) and not D"
    spec["risk"]["orders"] = {"good_for_day": True, "flat_by_day_end": True}
    assert S.validate_spec(spec) == []
    data = make_ohlcv(600, ("EURUSD=X",), freq="1h", seed=5)
    source = export_script(spec)
    assert "run_library_indicator('pandas_ta', 'CE'" in source and "day_boundaries(close.index)" in source
    in_app = engine.run_backtest(spec, data, params={"features.fast.params.timeperiod": 5,
                                                     "features.slow.params.timeperiod": 20})
    applied = S.apply_params(spec, {"features.fast.params.timeperiod": 5, "features.slow.params.timeperiod": 20})
    pd.testing.assert_series_equal(in_app["pf"].stats(), run_exported(applied, data).stats())
