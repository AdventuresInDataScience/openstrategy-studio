"""The feature catalogue: every indicator, pattern and transform the Build page offers.

Where they come from
* TA-Lib: 176 functions, every one but its 61 candlestick functions.
* pandas-ta-classic: the indicators it has that TA-Lib does not (Chandelier exit,
  Squeeze, QQE, Jurik MA, TD Sequential ...). One that duplicates a TA-Lib
  function is left out, so each indicator is offered once.
* ta_patterns: every candlestick and chart pattern. Patterns come from here only,
  never from TA-Lib's or pandas-ta's own candle functions.
* The Studio's own (STUDIO below): MA envelopes, the previous day, week or month,
  percentage offsets and bands, distance between lines, bars ago, bars since, and
  Python code.

Two layers
* The decisions are dict literals in this file: where each indicator sits in the
  menu (TALIB_MENU, PANDAS_TA_MENU), what is left out and why (EXCLUDED), the
  parameters never shown (HIDDEN_PARAMS) and the Studio's own entries (STUDIO).
* The facts are a generated snapshot, indicator_catalogue.json: each entry's
  label, inputs, parameters with defaults, and outputs. pandas-ta only reveals its
  outputs by running, which takes up to 2.5 s an indicator, so the picker reads
  the snapshot. `python -m openstrategy_studio.indicators` rewrites it from the
  installed libraries; tests/test_indicators.py regenerates it and compares, and
  runs every entry through the lookahead check.

⚠️ No lookahead. Every entry must give the same value on a bar whether or not
later bars exist. tests/test_indicators.py proves it for every entry, by
computing on the full data and on truncated copies. What failed is in EXCLUDED
(pandas-ta's centred DPO, Ichimoku's chikou span, TOS_STDEVALL's whole-sample
fit), and pandas-ta's `offset` parameter, which can shift values backwards in
time, is never shown.

Runtime: compute() calls the library once per symbol, with plain numpy arrays
(TA-Lib) or Series (pandas-ta), and rebuilds one frame per output on the close
frame's index and columns. export_source() gives the same functions' source for
the exported script, so the two paths cannot drift apart.
"""

import functools
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGUE_FILE = Path(__file__).with_name("indicator_catalogue.json")
PRICE_FIELDS = ["open", "high", "low", "close", "volume"]

# TA-Lib's moving-average types, by the integer its matype parameters take (TA-Lib 0.8
# has 16; tests/test_indicators.py checks this list against talib.MA_Type). NONE is no
# smoothing at all; DEFAULT is the function's own choice.
TALIB_MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3", "HMA", "NONE", "DEFAULT",
                  "ZLEMA", "RMA", "VIDYA", "ALMA"]
# pandas-ta's moving averages, by the name its mamode parameters take.
# ⚠️ VERIFY on a pandas-ta-classic bump: read from pandas_ta_classic.ma's own list.
PANDAS_TA_MA_TYPES = ["sma", "ema", "dema", "tema", "wma", "hma", "rma", "fwma", "pwma", "sinwma", "swma", "t3",
                      "trima", "vidya", "linreg", "midpoint", "zlma"]

# The menu: (category, family) for every TA-Lib function outside Pattern Recognition.
# tests/test_indicators.py checks that each such function appears exactly once.
TALIB_MENU = {
    ("Indicator", "Moving averages"): [
        "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3", "MA", "HMA", "ALMA", "FRAMA", "VIDYA",
        "VWMA", "ZLEMA", "RMA", "MCGD", "HT_TRENDLINE", "MAVP", "MIDPOINT", "MIDPRICE", "SWAK_BUTTER", "SWAK_GAUSS"],
    ("Indicator", "Bands and channels"): ["BBANDS", "KC", "DONCHIAN", "ACCBANDS", "PERCENTB", "BBW"],
    ("Indicator", "Trend and stops"): ["SAR", "SAREXT", "SUPERTREND", "CKSP", "ICHIMOKU"],
    ("Indicator", "Trend strength"): [
        "ADX", "ADXR", "DX", "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "AROON", "AROONOSC", "VHF", "CHOP",
        "CHOPTR", "VORTEX", "ER"],
    ("Indicator", "Oscillators"): [
        "RSI", "STOCH", "STOCHF", "STOCHRSI", "WILLR", "CCI", "CMO", "CMOU", "ULTOSC", "CRSI", "CTI", "FISHER", "IMI",
        "KDJ", "PSO", "SMI", "STC", "WAVETREND"],
    ("Indicator", "Momentum"): [
        "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "MACD", "MACDEXT", "MACDFIX", "PPO", "APO", "TRIX", "TSI", "KST",
        "KSTEXT", "COPPOCK", "AO", "AC", "BOP", "QSTICK", "SQZMOM", "DOSC", "DPO", "ERI", "ASI", "SI", "IBS", "CG",
        "FOSC"],
    ("Indicator", "Volatility"): ["ATR", "NATR", "TRANGE", "ADR", "CVI", "MASSI", "RVI", "RVIR", "ROGERSSATCHELL"],
    ("Indicator", "Volume"): [
        "AD", "ADOSC", "CMF", "EFI", "EMV", "KVO", "MARKETFI", "NVI", "OBV", "PVI", "PVO", "PVT", "RVOL", "VWAP",
        "MFI", "WAD"],
    ("Indicator", "Cycles"): [
        "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE", "HT_TRENDMODE", "SWAK_2PHP", "SWAK_BP", "SWAK_HP"],
    ("Statistical", "Regression"): ["LINEARREG", "LINEARREG_ANGLE", "LINEARREG_INTERCEPT", "LINEARREG_SLOPE", "TSF"],
    ("Statistical", "Spread and distribution"): [
        "STDDEV", "VAR", "AVGDEV", "KURTOSIS", "MEDIAN", "PERCENTILE", "PERCENTRANK"],
    ("Statistical", "Correlation"): ["BETA", "CORREL"],
    ("Price levels", "Highs and lows over n bars"): ["MAX", "MIN", "MINMAX", "MAXINDEX", "MININDEX", "MINMAXINDEX"],
    ("Price levels", "Swing points"): ["FRACTAL", "ZIGZAG"],
    ("Price levels", "Price types"): ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "HA"],
    ("Transform", "Arithmetic"): ["ADD", "SUB", "MULT", "DIV", "SUM", "CUMSUM"],
    ("Transform", "Functions"): [
        "LN", "LOG10", "EXP", "SQRT", "CEIL", "FLOOR", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SINH", "COSH",
        "TANH"],
}

# The same for pandas-ta's indicators that TA-Lib does not have (vectorbt's names, upper case).
PANDAS_TA_MENU = {
    ("Indicator", "Moving averages"): [
        "FWMA", "HWMA", "JMA", "PWMA", "SINWMA", "SSF", "SWMA", "MMAR", "RAINBOW"],
    ("Indicator", "Bands and channels"): ["ABERRATION", "HWC"],
    ("Indicator", "Trend and stops"): ["CE", "PMAX", "HILO", "AMAT"],
    ("Indicator", "Oscillators"): ["LRSI", "RSX", "QQE", "PSL", "INERTIA", "RVGI"],
    ("Indicator", "Momentum"): ["BIAS", "BRAR", "CFO", "PGO", "PO", "TRIXH", "SQUEEZE", "SQUEEZE_PRO"],
    ("Indicator", "Volatility"): ["HVOL", "AVOLUME", "UI", "THERMO", "PDIST"],
    ("Indicator", "Volume"): ["AOBV", "PVOL", "PVR", "VFI", "VOSC", "VWMACD"],
    ("Indicator", "Cycles"): ["DSP", "EBSW", "MSW"],
    ("Statistical", "Regression"): ["SLOPE", "STDERR"],
    ("Statistical", "Spread and distribution"): ["MAD", "MD", "QUANTILE", "SKEW", "ENTROPY", "ZSCORE", "CDL_Z"],
    ("Statistical", "Returns"): ["LOG_RETURN", "PERCENT_RETURN", "DRAWDOWN"],
    ("Price levels", "Swing points"): ["SMC_SWEEP"],
    ("Price levels", "Pivots and previous periods"): ["CPR"],
    ("Price levels", "Runs and counts"): ["INCREASING", "DECREASING", "LONG_RUN", "SHORT_RUN", "TD_SEQ"],
    ("Transform", "Arithmetic"): ["DECAY", "EDECAY"],
    ("Transform", "Functions"): ["NPABS", "NPROUND", "TRUNC", "TODEG", "TORAD"],
}

# Left out, and why. A pandas-ta indicator with the same name as a TA-Lib function is
# left out without being listed: TA-Lib's is offered instead.
EXCLUDED = {
    "pandas_ta:CDL_DOJI": "candlestick patterns come from ta_patterns",
    "pandas_ta:CDL_INSIDE": "candlestick patterns come from ta_patterns",
    "pandas_ta:CDL_PATTERN": "candlestick patterns come from ta_patterns",
    "pandas_ta:TOS_STDEVALL": "⚠️ looks ahead: it fits one regression over the whole history",
    "pandas_ta:TSIGNALS": "turns a trend series into trades, which is what the rules do",
    "pandas_ta:TTM_TREND": "does not run through vectorbt's pandas-ta wrapper (a 3-D output)",
    "pandas_ta:DM": "the same as TA-Lib's PLUS_DM and MINUS_DM",
    "pandas_ta:EOM": "the same as TA-Lib's EMV",
    "pandas_ta:HL2": "the same as TA-Lib's MEDPRICE",
    "pandas_ta:HLC3": "the same as TA-Lib's TYPPRICE",
    "pandas_ta:OHLC4": "the same as TA-Lib's AVGPRICE",
    "pandas_ta:WCP": "the same as TA-Lib's WCLPRICE",
    "pandas_ta:LINREG": "the same as TA-Lib's LINEARREG",
    "pandas_ta:LINREGANGLE": "the same as TA-Lib's LINEARREG_ANGLE",
    "pandas_ta:LINREGINTERCEPT": "the same as TA-Lib's LINEARREG_INTERCEPT",
    "pandas_ta:LINREGSLOPE": "the same as TA-Lib's LINEARREG_SLOPE",
    "pandas_ta:STDEV": "the same as TA-Lib's STDDEV",
    "pandas_ta:VARIANCE": "the same as TA-Lib's VAR",
    "pandas_ta:ROLLING_MAX": "the same as TA-Lib's MAX",
    "pandas_ta:ROLLING_MIN": "the same as TA-Lib's MIN",
    "pandas_ta:ROLLING_SUM": "the same as TA-Lib's SUM",
    "pandas_ta:TRUE_RANGE": "the same as TA-Lib's TRANGE",
    "pandas_ta:UO": "the same as TA-Lib's ULTOSC",
    "pandas_ta:PSAR": "the same as TA-Lib's SAR",
    "pandas_ta:ZLMA": "the same as TA-Lib's ZLEMA",
    # ⚠️ ta_patterns 1.2.1: _flagpole reads c[-1], the last close, on each detector's first
    # bar (docs/UPSTREAM.md, P1). Restore them when ta_patterns is fixed.
    "pattern:chart:flag_bull": "⚠️ looks ahead on its first bar (ta_patterns _flagpole; docs/UPSTREAM.md P1)",
    "pattern:chart:flag_bear": "⚠️ looks ahead on its first bar (ta_patterns _flagpole; docs/UPSTREAM.md P1)",
    "pattern:chart:pennant_bull": "⚠️ looks ahead on its first bar (ta_patterns _flagpole; docs/UPSTREAM.md P1)",
    "pattern:chart:pennant_bear": "⚠️ looks ahead on its first bar (ta_patterns _flagpole; docs/UPSTREAM.md P1)",
}

# Parameters never shown, so never set. ⚠️ pandas-ta's `offset` shifts the output by
# that many bars, and a negative one moves the future into the present. `talib` makes
# pandas-ta hand the work to TA-Lib; the default is kept.
HIDDEN_PARAMS = {"offset", "talib"}

# The candlestick and chart pattern families: (module path in ta_patterns, family label).
PATTERN_FAMILIES = {
    "candle": [("single", "Single bar"), ("two_bar", "Two bars"), ("three_bar", "Three bars"),
               ("multi_bar", "Four or more bars")],
    "chart": [("classic", "Classic"), ("double_multi", "Double and multiple tops and bottoms"),
              ("harmonic", "Harmonic"), ("busted", "Busted patterns"), ("short", "Short-term"),
              ("volume", "Volume patterns")],
}

# The Studio's own entries. Their compute functions are below; export_source() copies them.
STUDIO = {
    "studio:envelope": {
        "label": "Moving average envelope", "menu": ["Indicator", "Bands and channels"],
        "inputs": ["close"],
        "params": {"timeperiod": {"type": "int", "default": 20}, "matype": {"type": "choice", "default": "SMA",
                                                                            "options": TALIB_MA_TYPES},
                   "percent": {"type": "float", "default": 2.5}},
        "outputs": ["upper", "middle", "lower"],
        "description": "A moving average with bands a fixed percentage above and below it.",
    },
    "studio:previous_period": {
        "label": "Previous day, week or month", "menu": ["Price levels", "Pivots and previous periods"],
        "inputs": [],
        "params": {"period": {"type": "choice", "default": "day", "options": ["day", "week", "month"]}},
        "outputs": ["open", "high", "low", "close"],
        "description": "The last completed day's (week's, month's) open, high, low and close, held until the next "
                       "one completes. It never shows a period that is still forming.",
    },
    "studio:offset": {
        "label": "Offset a line by a percentage", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"percent": {"type": "float", "default": 5.0}},
        "outputs": ["value"],
        "description": "The line times (1 + percent / 100). Use a negative percent to go below it.",
    },
    "studio:percent_band": {
        "label": "Band a percentage around a line", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"percent": {"type": "float", "default": 5.0}},
        "outputs": ["upper", "lower"],
        "description": "Two lines, percent above and percent below the input. 'Within 5% of the EMA' is a price "
                       "between the lower and upper lines of a 5% band around the EMA.",
    },
    "studio:distance": {
        "label": "Distance between two lines (%)", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["a", "b"],
        "params": {},
        "outputs": ["value"],
        "description": "How far line a is above line b, as a percentage of b: (a / b - 1) x 100.",
    },
    "studio:bars_ago": {
        "label": "A line some bars ago", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {"bars": {"type": "int", "default": 1}},
        "outputs": ["value"],
        "description": "The input's value that many bars back. History only: it never moves a value forward.",
    },
    "studio:bars_since": {
        "label": "Bars since a signal", "menu": ["Transform", "Offsets and bands"],
        "inputs": ["source"],
        "params": {},
        "outputs": ["value"],
        "description": "How many bars ago the input was last non-zero, such as a pattern firing. Blank before the "
                       "first time.",
    },
    "studio:python": {
        "label": "Python code", "menu": ["Custom", "Python"],
        "inputs": [],
        "params": {"code": {"type": "code", "default": "return close.rolling(10).mean()"}},
        "outputs": ["value"],
        "description": "Trusted local code. The function body receives open, high, low, close, volume, np and pd, "
                       "and returns a frame shaped like close.",
    },
}


# ---------------------------------------------------------------------------
# Reading the catalogue
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def catalogue():
    """{indicator id: entry} from the snapshot. An entry is
    {"id", "source", "function", "label", "menu": [category, family], "inputs": [names],
     "defaults": {input: price name or None}, "params": {name: {"type", "default", "options"?}},
     "outputs": [names], "description", "direction" (patterns)}."""
    entries = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


def entry(indicator_id):
    return catalogue()[indicator_id]


def menu():
    """{category: {family: [indicator ids, by label]}}, categories and families in menu order."""
    order = ["Indicator", "Statistical", "Price levels", "Candlestick", "Chart pattern", "Transform", "Custom"]
    out = {category: {} for category in order}
    for item in catalogue().values():
        category, family = item["menu"]
        out.setdefault(category, {}).setdefault(family, []).append(item["id"])
    for families in out.values():
        for family, ids in families.items():
            ids.sort(key=lambda indicator_id: catalogue()[indicator_id]["label"].lower())
    return {category: families for category, families in out.items() if families}


def outputs(indicator_id):
    return list(entry(indicator_id)["outputs"])


def new_feature(indicator_id):
    """A new feature dict for the spec: the indicator, its default inputs and its default parameters."""
    item = entry(indicator_id)
    feature = {"indicator": indicator_id, "inputs": {}, "params": {}}
    for name in item["inputs"]:
        feature["inputs"][name] = item["defaults"].get(name) or "close"
    for name, declaration in item["params"].items():
        if declaration.get("default") is not None:
            feature["params"][name] = declaration["default"]
    return feature


def describe(feature):
    """A short text form, for summaries and the pipeline diagram: 'EMA(close, 50)'."""
    item = catalogue().get(feature.get("indicator"))
    if item is None:
        return str(feature.get("indicator"))
    parts = []
    for name in item["inputs"]:
        value = (feature.get("inputs") or {}).get(name, item["defaults"].get(name))
        if isinstance(value, dict):
            value = value["feature"] + (f".{value['output']}" if value.get("output") else "")
        if value is not None:
            parts.append(str(value))
    for name, value in (feature.get("params") or {}).items():
        if name != "code":
            parts.append(f"{value:g}" if isinstance(value, float) else str(value))
    short = item["function"] if item["source"] in ("talib", "pandas_ta") else item["label"]
    return f"{short}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Computing one feature
# ---------------------------------------------------------------------------
# The library calls are plain functions with their imports inside, so export_source()
# can copy them into the exported script word for word.

def run_library_indicator(source, function, input_names, inputs, params, output_names):
    """A TA-Lib or pandas-ta indicator on every column: {output name: DataFrame}.

    `inputs` are frames shaped like close, in the function's input order. The call is
    made once per column with that column's values, because both libraries take one
    series at a time.
    """
    like = inputs[0]
    columns = {name: {} for name in output_names}
    for column in like.columns:
        arrays = [np.ascontiguousarray(frame[column].to_numpy(dtype=float)) for frame in inputs]
        if source == "talib":
            import talib
            result = getattr(talib, function)(*arrays, **params)
            values = list(result) if isinstance(result, (tuple, list)) else [result]
        else:
            import pandas_ta_classic
            series = {name: pd.Series(array, index=like.index) for name, array in zip(input_names, arrays)}
            result = getattr(pandas_ta_classic, function.lower())(**series, **params)
            if isinstance(result, pd.Series):
                values = [result]
            else:
                # ⚠️ Outputs are matched by position with the catalogue's names: pandas-ta
                # names its columns after the parameters (CEl_22_3.0). Non-numeric
                # columns were left out of the catalogue, and are dropped here too.
                numeric = [result[name] for name in result.columns if pd.api.types.is_numeric_dtype(result[name])]
                values = numeric
        if len(values) != len(output_names):
            raise ValueError(f"{function} gave {len(values)} outputs where the catalogue lists {len(output_names)}; "
                             "the library has changed. Run `python -m openstrategy_studio.indicators`.")
        for name, value in zip(output_names, values):
            columns[name][column] = np.asarray(value, dtype=float)
    return {name: pd.DataFrame(values, index=like.index, columns=like.columns) for name, values in columns.items()}


def run_pattern(module, name, params, open_, high, low, close, volume):
    """A ta_patterns detector on every column: an int8 frame, +1 bullish, -1 bearish, 0 none.

    Signals sit on the bar that COMPLETES the pattern and use only past bars, so the
    front door's next-open fill is correct execution.
    """
    import inspect as _inspect
    import ta_patterns
    import ta_patterns.chart_patterns
    detector = getattr(ta_patterns.chart_patterns if module == "chart" else ta_patterns, name)
    # Each detector names the prices it reads (black_candle takes only o and c), and five
    # volume patterns run only when v is given: pass exactly the ones it names.
    wanted = [key for key in _inspect.signature(detector).parameters if key in ("o", "h", "l", "c", "v")]
    frames = {"o": open_, "h": high, "l": low, "c": close, "v": volume}
    out = pd.DataFrame(0, index=close.index, columns=close.columns, dtype="int8")
    for symbol in close.columns:
        prices = {key: frames[key][symbol].to_numpy(dtype=float) for key in wanted}
        out[symbol] = np.asarray(detector(**prices, **params), dtype="int8")
    return out


def run_studio(function, params, inputs, data):
    """One of the Studio's own entries: {output name: DataFrame}."""
    close = data["close"]
    if function == "envelope":
        import talib
        ma_type = TALIB_MA_TYPES.index(params["matype"])
        source = inputs[0]
        middle = pd.DataFrame({column: talib.MA(np.ascontiguousarray(source[column].to_numpy(dtype=float)),
                                                timeperiod=int(params["timeperiod"]), matype=ma_type)
                               for column in source.columns}, index=source.index)
        share = float(params["percent"]) / 100.0
        return {"upper": middle * (1 + share), "middle": middle, "lower": middle * (1 - share)}
    if function == "previous_period":
        # ⚠️ No higher-timeframe leakage: bins labelled and closed on the left, then
        # shifted one period, so a bar only ever sees periods that have finished.
        rule = {"day": "1D", "week": "W-SUN", "month": "MS"}[params["period"]]
        out = {}
        for field, how in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")):
            binned = getattr(data[field].resample(rule, label="left", closed="left"), how)()
            binned = binned[binned.notna().any(axis=1)].shift(1)
            out[field] = binned.reindex(close.index, method="ffill")
        return out
    if function == "offset":
        return {"value": inputs[0] * (1 + float(params["percent"]) / 100.0)}
    if function == "percent_band":
        share = float(params["percent"]) / 100.0
        return {"upper": inputs[0] * (1 + share), "lower": inputs[0] * (1 - share)}
    if function == "distance":
        return {"value": (inputs[0] / inputs[1] - 1.0) * 100.0}
    if function == "bars_ago":
        return {"value": inputs[0].shift(int(params["bars"]))}
    if function == "bars_since":
        fired = inputs[0].fillna(0.0) != 0
        position = pd.DataFrame(np.arange(len(fired))[:, None].repeat(fired.shape[1], axis=1),
                                index=fired.index, columns=fired.columns, dtype=float)
        last = position.where(fired).ffill()
        return {"value": position - last}
    if function == "python":
        # ⚠️ Runs arbitrary code in-process. The UI labels it as trusted local code and
        # warns before running an imported spec that contains one.
        body = "\n".join("    " + line for line in params["code"].splitlines()) or "    pass"
        namespace = {}
        exec(f"def block(open, high, low, close, volume, np, pd):\n{body}\n", namespace)
        result = namespace["block"](data["open"], data["high"], data["low"], close, data["volume"], np, pd)
        if isinstance(result, pd.Series):
            result = result.to_frame(close.columns[0]) if close.shape[1] == 1 else None
        if not isinstance(result, pd.DataFrame) or result.shape != close.shape:
            raise ValueError("Python code must return a DataFrame shaped like `close` "
                             "(one column per symbol, same index).")
        return {"value": result.astype(float)}
    raise ValueError(f"Unknown Studio entry {function}.")


def call_params(feature):
    """The keyword arguments for the library: defaults filled in, choices turned into the
    library's own values, and parameters left blank (None) not passed at all."""
    item = entry(feature["indicator"])
    given = feature.get("params") or {}
    out = {}
    for name, declaration in item["params"].items():
        value = given.get(name, declaration.get("default"))
        if value is None:
            continue
        kind = declaration["type"]
        if kind == "choice" and item["source"] == "talib":
            value = TALIB_MA_TYPES.index(value)          # TA-Lib's matype parameters take an integer
        elif kind == "int":
            value = int(value)
        elif kind == "float":
            value = float(value)
        elif kind == "bool":
            value = bool(value)
        out[name] = value
    return out


def compute(feature, inputs, data):
    """{output name: DataFrame shaped like close} for one feature.

    `inputs` holds a frame for each of the entry's inputs, in order (price fields or
    other features' outputs, resolved by the compiler).
    """
    item = entry(feature["indicator"])
    params = call_params(feature)
    if item["source"] in ("talib", "pandas_ta"):
        return run_library_indicator(item["source"], item["function"], item["inputs"], inputs, params,
                                     item["outputs"])
    if item["source"] == "pattern":
        module, name = item["function"].split(":")
        return {"signal": run_pattern(module, name, params, data["open"], data["high"], data["low"], data["close"],
                                      data["volume"])}
    return run_studio(item["function"], params, inputs, data)


def export_source():
    """The compute functions' own source, for the exported script."""
    return "\n\n".join([f"TALIB_MA_TYPES = {TALIB_MA_TYPES!r}",
                        inspect.getsource(run_library_indicator), inspect.getsource(run_pattern),
                        inspect.getsource(run_studio)])


# ---------------------------------------------------------------------------
# Building the snapshot (offline: `python -m openstrategy_studio.indicators`)
# ---------------------------------------------------------------------------

def _talib_entries():
    import talib
    from talib import abstract

    placed = {name: place for place, names in TALIB_MENU.items() for name in names}
    rows = []
    for function in sorted(talib.get_functions()):
        info = abstract.Function(function).info
        if info["group"] == "Pattern Recognition":
            continue            # candlesticks come from ta_patterns
        inputs = []
        for value in info["input_names"].values():
            inputs += list(value) if isinstance(value, (list, tuple)) else [value]
        params = {}
        for name, default in info["parameters"].items():
            if "matype" in name:
                params[name] = {"type": "choice", "default": TALIB_MA_TYPES[int(default)], "options": TALIB_MA_TYPES}
            elif isinstance(default, int):
                params[name] = {"type": "int", "default": default}
            else:
                params[name] = {"type": "float", "default": float(default)}
        category, family = placed[function]       # KeyError: a new TA-Lib function needs a menu place
        rows.append({"id": f"talib:{function}", "source": "talib", "function": function,
                     "label": info["display_name"], "menu": [category, family], "inputs": inputs,
                     "defaults": {name: name if name in PRICE_FIELDS else None for name in inputs},
                     "params": params, "outputs": list(info["output_names"]),
                     "description": info["group"]})
    return rows


def _pandas_ta_default(function, name, doc, source):
    """A pandas-ta parameter's real default: most signatures say None and set it in the body."""
    import ast
    match = re.search(rf"^\s*{re.escape(name)}\s*\(([^)]*)\):(.*?)(?=^\s*\w+\s*\(|\Z)", doc, re.M | re.S)
    kind, default = None, None
    if match:
        kind = match.group(1).split(",")[0].strip().lower()
        found = re.search(r"Default:?\s*(-?[\w.'\"]+)", match.group(2))
        if found:
            text = found.group(1).rstrip(".")
            try:
                default = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                default = text.strip("'\"")
    if default is None:
        found = re.search(rf"^\s*{re.escape(name)}\s*=.*?\belse\s+([^\n#]+)", source, re.M)
        if found:
            try:
                default = ast.literal_eval(found.group(1).strip().rstrip(")"))
            except (ValueError, SyntaxError):
                default = None
    return kind, default


def _pandas_ta_entries(talib_names):
    import warnings

    import pandas_ta_classic
    import vectorbt

    placed = {name: place for place, names in PANDAS_TA_MENU.items() for name in names}
    sample = _sample_data()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = sorted(vectorbt.IndicatorFactory.get_pandas_ta_indicators())
        for name in names:
            if name in talib_names or f"pandas_ta:{name}" in EXCLUDED:
                continue
            factory = vectorbt.IndicatorFactory.from_pandas_ta(name)
            function = getattr(pandas_ta_classic, name.lower())
            doc = inspect.getdoc(function) or ""
            try:
                code = inspect.getsource(function)
            except (OSError, TypeError):
                code = ""
            params = {}
            for param in factory.param_names:
                if param in HIDDEN_PARAMS:
                    continue
                kind, default = _pandas_ta_default(name, param, doc, code)
                if param == "mamode":
                    params[param] = {"type": "choice", "default": default if default in PANDAS_TA_MA_TYPES else None,
                                     "options": PANDAS_TA_MA_TYPES}
                elif isinstance(default, bool) or kind == "bool":
                    params[param] = {"type": "bool", "default": default if isinstance(default, bool) else None}
                elif isinstance(default, int) or kind == "int":
                    params[param] = {"type": "int", "default": default if isinstance(default, int) else None}
                elif isinstance(default, str) or kind == "str":
                    params[param] = {"type": "str", "default": default if isinstance(default, str) else None}
                else:
                    # A number, or no default and no type in the docs (Holt-Winter's na, nb ...):
                    # a number left blank, which means "the library's own default".
                    params[param] = {"type": "float", "default": float(default) if isinstance(default, (int, float))
                                     else None}
            # Which outputs are numbers: run once on the sample, positionally like compute().
            inputs = list(factory.input_names)
            series = {key: sample[_price_for(key)]["X"] for key in inputs}
            result = function(**series)
            if isinstance(result, pd.Series):
                numeric = [True]
            else:
                numeric = [pd.api.types.is_numeric_dtype(result[column]) for column in result.columns]
            output_names = [out for out, keep in zip(factory.output_names, numeric) if keep]
            first = doc.splitlines()[0].strip() if doc else name
            category, family = placed[name]      # KeyError: a new pandas-ta indicator needs a place or a reason
            rows.append({"id": f"pandas_ta:{name}", "source": "pandas_ta", "function": name,
                         "label": first.rstrip("."), "menu": [category, family], "inputs": inputs,
                         "defaults": {key: _price_for(key) for key in inputs}, "params": params,
                         "outputs": output_names, "description": " ".join(doc.split("\n\n")[1:2]).strip()[:300]})
    return rows


def _price_for(input_name):
    key = input_name.lower().rstrip("_")
    return key if key in PRICE_FIELDS else "close"


def _pattern_entries():
    import ta_patterns
    import ta_patterns.chart_patterns as chart
    import ta_patterns.scanner as scanner

    rows = []
    for module, families in PATTERN_FAMILIES.items():
        package = chart if module == "chart" else ta_patterns
        classes = chart if module == "chart" else scanner
        labels = ["BULLISH", "BEARISH", "DIRECTIONAL", "BIDIRECTIONAL", "NON_DIRECTIONAL"]
        known = set()
        for label in labels:
            known |= set(getattr(classes, label, ()))
        for submodule, family in families:
            source = getattr(package, submodule)
            for name in sorted(vars(source)):
                function = vars(source)[name]
                if name.startswith("_") or not callable(function) or getattr(function, "__module__", "") != source.__name__:
                    continue
                if name not in known or not hasattr(package, name) or f"pattern:{module}:{name}" in EXCLUDED:
                    continue
                direction = next(label.lower() for label in labels if name in getattr(classes, label, ()))
                params = {}
                for key, parameter in inspect.signature(function).parameters.items():
                    if key in ("o", "h", "l", "c", "v") or parameter.default is inspect.Parameter.empty:
                        continue
                    default = parameter.default
                    if key == "mode":
                        params[key] = {"type": "choice", "default": default, "options": ["confirmed", "forming"]}
                    elif isinstance(default, bool):
                        params[key] = {"type": "bool", "default": default}
                    elif isinstance(default, int):
                        params[key] = {"type": "int", "default": default}
                    elif isinstance(default, float) or default is None:
                        params[key] = {"type": "float", "default": default}
                    else:
                        params[key] = {"type": "str", "default": default}
                doc = (inspect.getdoc(function) or "").replace("**", "")
                rows.append({"id": f"pattern:{module}:{name}", "source": "pattern", "function": f"{module}:{name}",
                             "label": name.replace("_", " ").capitalize(),
                             "menu": ["Candlestick" if module == "candle" else "Chart pattern", family],
                             "inputs": [], "defaults": {}, "params": params, "outputs": ["signal"],
                             "direction": direction, "description": " ".join(doc.split())[:300]})
    return rows


def _studio_entries():
    rows = []
    for indicator_id, item in STUDIO.items():
        rows.append(dict(item, id=indicator_id, source="studio", function=indicator_id.split(":", 1)[1],
                         defaults={name: "close" for name in item["inputs"]}))
    return rows


def _sample_data(n=400, seed=3):
    """A small synthetic OHLCV set for the snapshot and the lookahead check."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return {name: pd.DataFrame({"X": values}, index=index) for name, values in
            (("open", open_), ("high", high), ("low", low), ("close", close), ("volume", volume))}


def build_catalogue():
    """Every entry, from the installed libraries and the decisions above, sorted by id."""
    talib_rows = _talib_entries()
    talib_names = {row["function"] for row in talib_rows}
    rows = talib_rows + _pandas_ta_entries(talib_names) + _pattern_entries() + _studio_entries()
    return sorted(rows, key=lambda row: row["id"])


def write_catalogue():
    rows = build_catalogue()
    CATALOGUE_FILE.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    catalogue.cache_clear()
    return len(rows)


if __name__ == "__main__":
    print(f"Wrote {write_catalogue()} entries to {CATALOGUE_FILE}")
