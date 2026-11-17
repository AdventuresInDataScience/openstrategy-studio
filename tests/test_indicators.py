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


def test_price_types_compute_what_their_labels_say():
    """Each price type's label carries its formula (ENTRY_TEXT); the library must agree with it."""
    pytest.importorskip("talib")
    data = I._sample_data(n=60)
    o, h, l, c = (data[name] for name in ("open", "high", "low", "close"))
    formulas = {"talib:AVGPRICE": ([o, h, l, c], (o + h + l + c) / 4), "talib:MEDPRICE": ([h, l], (h + l) / 2),
                "talib:TYPPRICE": ([h, l, c], (h + l + c) / 3), "talib:WCLPRICE": ([h, l, c], (h + l + 2 * c) / 4)}
    for indicator, (inputs, expected) in formulas.items():
        got = I.compute({"indicator": indicator, "params": {}}, inputs, data)["real"]
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), err_msg=indicator)
        assert "(" in I.entry(indicator)["label"], indicator
    highest = I.compute({"indicator": "talib:MAX", "params": {"timeperiod": 5}}, [h], data)["real"]["X"]
    assert highest.iloc[10] == h["X"].iloc[6:11].max()           # this bar included, as its label says


def test_highest_high_and_lowest_low():
    data = I._sample_data(n=80)
    high, low = data["high"], data["low"]
    before = I.compute({"indicator": "studio:highest_lowest", "params": {"bars": 10, "include_current": False}},
                       [high, low], data)
    with_bar = I.compute({"indicator": "studio:highest_lowest", "params": {"bars": 10, "include_current": True}},
                         [high, low], data)
    for bar in (10, 33, 79):
        assert before["upper"]["X"].iloc[bar] == high["X"].iloc[bar - 10:bar].max()      # the 10 bars BEFORE
        assert with_bar["upper"]["X"].iloc[bar] == high["X"].iloc[bar - 9:bar + 1].max()  # this bar and 9 before
        assert before["lower"]["X"].iloc[bar] == low["X"].iloc[bar - 10:bar].min()
        window = high["X"].iloc[bar - 9:bar + 1].to_numpy()
        assert with_bar["bars_since_high"]["X"].iloc[bar] == 9 - int(np.flatnonzero(window == window.max())[-1])
        assert before["bars_since_high"]["X"].iloc[bar] == with_bar["bars_since_high"]["X"].iloc[bar - 1] + 1
    assert np.isnan(before["upper"]["X"].iloc[9]) and np.isnan(with_bar["upper"]["X"].iloc[8])
    np.testing.assert_allclose(before["middle"], (before["upper"] + before["lower"]) / 2)


def test_pivot_points_come_from_the_previous_period():
    data = {name: frame for name, frame in I._sample_data(n=60).items()}
    previous = I.compute({"indicator": "studio:previous_period", "params": {"period": "week"}}, [], data)
    h, l, c = previous["high"], previous["low"], previous["close"]
    classic = I.compute({"indicator": "studio:pivots", "params": {"period": "week", "method": "classic"}}, [], data)
    pivot = (h + l + c) / 3
    np.testing.assert_allclose(classic["pivot"], pivot)
    np.testing.assert_allclose(classic["r1"], 2 * pivot - l)
    np.testing.assert_allclose(classic["s2"], pivot - (h - l))
    woodie = I.compute({"indicator": "studio:pivots", "params": {"period": "week", "method": "woodie"}}, [], data)
    np.testing.assert_allclose(woodie["pivot"], (h + l + 2 * c) / 4)
    camarilla = I.compute({"indicator": "studio:pivots", "params": {"period": "week", "method": "camarilla"}}, [], data)
    np.testing.assert_allclose(camarilla["r3"], c + (h - l) * 1.1 / 4)
    fibonacci = I.compute({"indicator": "studio:pivots", "params": {"period": "week", "method": "fibonacci"}}, [], data)
    np.testing.assert_allclose(fibonacci["s1"], pivot - 0.382 * (h - l))


def test_period_so_far():
    from conftest import make_ohlcv
    data = make_ohlcv(72, ("X",), freq="1h", seed=2)
    today = I.compute({"indicator": "studio:period_to_date", "params": {"period": "day"}}, [], data)
    index = data["close"].index
    for bar in (0, 5, 30, 71):
        same_day = (index.normalize() == index[bar].normalize()) & (index <= index[bar])
        assert today["high"]["X"].iloc[bar] == data["high"]["X"][same_day].max()
        assert today["low"]["X"].iloc[bar] == data["low"]["X"][same_day].min()
        assert today["open"]["X"].iloc[bar] == data["open"]["X"][index.normalize() == index[bar].normalize()].iloc[0]


def test_menu_is_complete_and_ordered():
    menu = I.menu()
    assert list(menu)[:3] == ["Indicator", "Statistical", "Price levels"]
    ids = [indicator_id for families in menu.values() for members in families.values() for indicator_id in members]
    assert len(ids) == len(set(ids)) == len(I.catalogue())
    for wanted in ("talib:KC", "talib:AROON", "studio:envelope", "talib:ICHIMOKU", "pandas_ta:CE", "talib:SUPERTREND"):
        assert wanted in ids
    assert I.outputs("talib:AROON") == ["aroondown", "aroonup"]


def test_every_setting_has_a_tooltip():
    """param_help explains every setting, and every curated description is reachable.

    A PARAM_HELP name that no entry of that library has (a typo, or a parameter a library
    bump renamed) would never show, so it fails here.
    """
    used = {}
    for item in I.catalogue().values():
        for name in item["params"]:
            used.setdefault(item["source"], set()).add(name)
            text = I.param_help(item["id"], name)
            assert text.strip(), (item["id"], name)
    for source, described in I.PARAM_HELP.items():
        assert set(described) <= used.get(source, set()), (source, sorted(set(described) - used.get(source, set())))
    assert I.param_help("talib:SMA", "timeperiod") == ("How many bars it looks back over. SMA (TA-Lib) takes it as "
                                                       "`timeperiod`. Default: 30.")


def test_moving_average_reproduces_each_single_type_average():
    """The feature card swaps these for talib:MA at the same type (SAME_AS_MA), so the
    optimiser can pick the type. The swap keeps the inputs and the period, so it must give
    exactly the same numbers, through compute() as the compiler calls it."""
    pytest.importorskip("talib")
    data = I._sample_data(n=400)
    every_type = I.entry("talib:MA")["params"]["matype"]["options"]
    for indicator, kind in I.SAME_AS_MA.items():
        item = I.entry(indicator)
        assert kind in every_type, indicator
        assert list(item["params"]) == ["timeperiod"] and item["outputs"] == ["real"], indicator
        assert item["inputs"] == I.entry("talib:MA")["inputs"], indicator
        for period in (2, 9, 50):
            own = I.compute({"indicator": indicator, "params": {"timeperiod": period}}, [data["close"]], data)
            ma = I.compute({"indicator": "talib:MA", "params": {"timeperiod": period, "matype": kind}},
                           [data["close"]], data)
            np.testing.assert_array_equal(own["real"].to_numpy(), ma["real"].to_numpy(), err_msg=f"{indicator} {period}")
