"""The feature catalogue: complete, current, faithful to the libraries, and free of lookahead."""

import json

import numpy as np
import pandas as pd
import pytest

from openstrategy_studio import indicators as I


def test_talib_ma_types_match_the_library():
    talib = pytest.importorskip("talib")
    lookup = talib.MA_Type._lookup
    assert len(I.TALIB_MA_TYPES) == len(lookup)
    # Spot-check the order: the integer TA-Lib takes is the list position.
    for position, words in ((0, "Simple"), (1, "Exponential"), (9, "Hull"), (13, "Wilder"), (15, "Legoux")):
        assert words in lookup[position], (position, lookup[position])


def test_every_talib_function_has_one_place():
    talib = pytest.importorskip("talib")
    from talib import abstract
    wanted = {name for name in talib.get_functions() if abstract.Function(name).info["group"] != "Pattern Recognition"}
    placed = [name for names in I.TALIB_MENU.values() for name in names]
    assert len(placed) == len(set(placed)), "a TA-Lib function is placed twice"
    assert set(placed) == wanted, (sorted(wanted - set(placed)), sorted(set(placed) - wanted))


def test_every_pandas_ta_indicator_is_offered_or_excluded():
    pytest.importorskip("pandas_ta_classic")
    vectorbt = pytest.importorskip("vectorbt")
    talib = pytest.importorskip("talib")
    names = set(vectorbt.IndicatorFactory.get_pandas_ta_indicators())
    placed = [name for values in I.PANDAS_TA_MENU.values() for name in values]
    assert len(placed) == len(set(placed)), "a pandas-ta indicator is placed twice"
    excluded = {key.split(":", 1)[1] for key in I.EXCLUDED if key.startswith("pandas_ta:")}
    duplicates = names & set(talib.get_functions())
    assert not set(placed) & duplicates, "a pandas-ta duplicate of a TA-Lib function is offered"
    assert names - duplicates == set(placed) | excluded, sorted((names - duplicates) ^ (set(placed) | excluded))


def test_every_pattern_is_offered_once():
    from openstrategy_studio.indicators import catalogue
    patterns = [item for item in catalogue().values() if item["source"] == "pattern"]
    candles = {item["function"].split(":")[1] for item in patterns if item["menu"][0] == "Candlestick"}
    charts = {item["function"].split(":")[1] for item in patterns if item["menu"][0] == "Chart pattern"}
    # phase 0: 106 candlesticks and 193 chart patterns have functions in ta_patterns 1.2.1;
    # the four flag and pennant detectors are left out for looking ahead (docs/UPSTREAM.md P1).
    left_out = {key.split(":")[2] for key in I.EXCLUDED if key.startswith("pattern:chart:")}
    assert len(candles) == 106 and len(charts) == 193 - len(left_out) and not charts & left_out
    assert not any(item["source"] in ("talib", "pandas_ta") and item["function"].upper().startswith("CDL")
                   and item["function"] != "CDL_Z" for item in catalogue().values())


def test_the_snapshot_is_current():
    """indicator_catalogue.json is what the installed libraries and this module's decisions give."""
    pytest.importorskip("talib")
    pytest.importorskip("pandas_ta_classic")
    pytest.importorskip("ta_patterns")
    stored = json.loads(I.CATALOGUE_FILE.read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(I.build_catalogue()))
    assert [row["id"] for row in fresh] == [row["id"] for row in stored]
    for new, old in zip(fresh, stored):
        assert new == old, f"{new['id']} changed: run `python -m openstrategy_studio.indicators`"


def _inputs_for(item, data):
    frames = []
    for name in item["inputs"]:
        field = item["defaults"].get(name)
        frames.append(data[field] if field else pd.DataFrame(10.0, index=data["close"].index,
                                                             columns=data["close"].columns))
    return frames


def _variants(item):
    """The default parameters, plus each other value of a pattern's mode (forming may look ahead)."""
    feature = I.new_feature(item["id"])
    yield feature
    mode = item["params"].get("mode")
    if mode and mode["type"] == "choice":
        for option in mode["options"]:
            if option != mode["default"]:
                yield dict(feature, params=dict(feature["params"], mode=option))


def _another_future(data, cut):
    """The same bars up to `cut`, then a different future: prices rescaled by a steady trend."""
    trend = np.ones(len(data["close"]))
    trend[cut:] = np.linspace(1.3, 0.6, len(trend) - cut)
    # Volume too, with the same factor: it must change only after the cut.
    return {field: frame.mul(trend, axis=0) for field, frame in data.items()}


def test_no_entry_looks_ahead():
    """⚠️ Every entry gives the same value on a bar whether or not later bars exist.

    Each is computed on the full sample, on truncated copies, and with the bars after
    the cut replaced by a different future; the shared bars must agree exactly (NaN
    with NaN). A failure is lookahead: move the entry to EXCLUDED with the reason, or
    hide the parameter that causes it.

    A leak only shows when the leaked value changes the output, so this proves a
    leak, never its absence: it found ta_patterns' flag detectors reading the last
    close (docs/UPSTREAM.md P1) because one bear flag fired on bar 40.
    """
    pytest.importorskip("talib")
    pytest.importorskip("pandas_ta_classic")
    pytest.importorskip("ta_patterns")
    data = I._sample_data(n=400)
    failures = []
    for item in I.catalogue().values():
        if item["id"] == "studio:python":
            continue
        for feature in _variants(item):
            full = I.compute(feature, _inputs_for(item, data), data)
            for cut in (150, 260, 333):
                part_data = {field: frame.iloc[:cut] for field, frame in data.items()}
                other_data = _another_future(data, cut)
                part = I.compute(feature, _inputs_for(item, part_data), part_data)
                other = I.compute(feature, _inputs_for(item, other_data), other_data)
                for name, values in part.items():
                    b = values.to_numpy(dtype=float)
                    for label, against in (("truncated", full), ("another future", other)):
                        a = against[name].to_numpy(dtype=float)[:cut]
                        same = np.isclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True)
                        if not same.all():
                            failures.append(f"{item['id']} {feature['params'].get('mode', '')} {name}: bar "
                                            f"{int(np.argmin(same))} differs at cut {cut} ({label})")
                            break
    assert not failures, "\n".join(sorted(set(failures)))


def test_compute_matches_the_library_called_directly():
    talib = pytest.importorskip("talib")
    pta = pytest.importorskip("pandas_ta_classic")
    data = I._sample_data(n=300)
    close = data["close"]
    out = I.compute({"indicator": "talib:BBANDS", "params": {"timeperiod": 10, "nbdevup": 1.5}}, [close], data)
    upper, middle, lower = talib.BBANDS(close["X"].to_numpy(), timeperiod=10, nbdevup=1.5, nbdevdn=2.0, matype=0)
    np.testing.assert_array_equal(out["upperband"]["X"].to_numpy(), upper)
    np.testing.assert_array_equal(out["lowerband"]["X"].to_numpy(), lower)
    feature = I.new_feature("pandas_ta:CE")
    out = I.compute(feature, [data["high"], data["low"], close], data)
    direct = pta.ce(data["high"]["X"], data["low"]["X"], close["X"], length=22, multiplier=3.0, mamode="rma")
    np.testing.assert_allclose(out["ce_l"]["X"].to_numpy(), direct.iloc[:, 0].to_numpy(), equal_nan=True)
    assert list(out["ce_l"].columns) == ["X"] and out["ce_l"].index.equals(close.index)


def test_simple_moving_average_equals_vectorbts():
    """The golden MA-cross template uses TA-Lib's SMA; the golden test builds its signals with vectorbt.MA.

    ⚠️ Not bit for bit in general: TA-Lib keeps a running sum and vectorbt a rolling one,
    and they part in the last bits (3.6e-13 here). On the golden EURUSD prices they are
    identical. A crossing cannot flip at that size, so the golden numbers hold.
    """
    vectorbt = pytest.importorskip("vectorbt")
    pytest.importorskip("talib")
    data = I._sample_data(n=300)
    ours = I.compute({"indicator": "talib:SMA", "params": {"timeperiod": 50}}, [data["close"]], data)["real"]
    theirs = vectorbt.MA.run(data["close"], window=50).ma.to_numpy()
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-12, atol=0, equal_nan=True)


def test_studio_entries():
    data = I._sample_data(n=120)
    close = data["close"]
    band = I.compute({"indicator": "studio:percent_band", "params": {"percent": 5}}, [close], data)
    np.testing.assert_allclose(band["upper"], close * 1.05)
    distance = I.compute({"indicator": "studio:distance", "params": {}}, [close * 1.1, close], data)["value"]
    np.testing.assert_allclose(distance, 10.0)
    ago = I.compute({"indicator": "studio:bars_ago", "params": {"bars": 3}}, [close], data)["value"]
    assert ago["X"].iloc[10] == close["X"].iloc[7]
    signal = pd.DataFrame(0.0, index=close.index, columns=["X"])
    signal.iloc[[5, 9]] = 1.0
    since = I.compute({"indicator": "studio:bars_since", "params": {}}, [signal], data)["value"]["X"]
    assert np.isnan(since.iloc[4]) and since.iloc[5] == 0 and since.iloc[8] == 3 and since.iloc[9] == 0
    previous = I.compute({"indicator": "studio:previous_period", "params": {"period": "week"}}, [], data)
    # A bar in week 3 sees week 2's high: never its own unfinished week.
    weekly = data["high"]["X"].resample("W-SUN", label="left", closed="left").max()
    bar = close.index[20]
    week_start = bar.normalize() - pd.Timedelta(days=(bar.dayofweek + 1) % 7)
    assert previous["high"]["X"].loc[bar] == weekly.loc[week_start - pd.Timedelta(days=7)]


def test_menu_is_complete_and_ordered():
    menu = I.menu()
    assert list(menu)[:3] == ["Indicator", "Statistical", "Price levels"]
    ids = [indicator_id for families in menu.values() for members in families.values() for indicator_id in members]
    assert len(ids) == len(set(ids)) == len(I.catalogue())
    for wanted in ("talib:KC", "talib:AROON", "studio:envelope", "talib:ICHIMOKU", "pandas_ta:CE", "talib:SUPERTREND"):
        assert wanted in ids
    assert I.outputs("talib:AROON") == ["aroondown", "aroonup"]
