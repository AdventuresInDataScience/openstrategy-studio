"""Feature blocks: one dict literal, BLOCKS, holding every block.

Each entry declares its parameters (type, bounds, default, label), its outputs,
its compile function and its export snippet. The Build page generates forms
from the parameter declarations, the optimiser reads the bounds as defaults for
``opt``, and export_script pastes the snippet. Adding a block is one dict entry
and one function, with no UI code.

compile(params, data, source) -> {output name: DataFrame}
    params  every declared parameter, defaults filled in, no "opt" nodes
    data    the dict of five frames (open, high, low, close, volume), one column
            per symbol on one shared index
    source  the frame named by the "source" parameter (a price field or another
            feature's output), or None for blocks without one

Indicator names, inputs, parameters and outputs are read from
vectorbt/indicators/basic.py. ⚠️ Pass scalar parameters only: a list makes
vectorbt build a column MultiIndex that the compiler does not expect.
"""

import inspect

import numpy as np
import pandas as pd


def _flat(values, like):
    """Rebuild an indicator output as a frame with `like`'s index and columns.

    ⚠️ vectorbt indicator outputs carry a parameter level in their columns
    (for example (50, "EURUSD=X")), so two MAs with different windows do not
    align under plain pandas arithmetic. Every vectorbt block is passed through
    this so that every feature has exactly the close frame's labels.
    """
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(like.shape),
                        index=like.index, columns=like.columns)


def _vbt():
    # Imported lazily so that spec validation and the pandas blocks work without
    # loading numba. vectorbt (not vectorbtX) provides the indicators.
    import vectorbt
    return vectorbt


# ---------------------------------------------------------------------------
# vectorbt indicator blocks
# ---------------------------------------------------------------------------

def compile_ma(params, data, source):
    out = _vbt().MA.run(source, window=int(params["window"]), ewm=bool(params["ewm"]))
    return {"ma": _flat(out.ma, source)}


def compile_mstd(params, data, source):
    out = _vbt().MSTD.run(source, window=int(params["window"]), ewm=bool(params["ewm"]))
    return {"mstd": _flat(out.mstd, source)}


def compile_bbands(params, data, source):
    out = _vbt().BBANDS.run(source, window=int(params["window"]), ewm=bool(params["ewm"]),
                            alpha=float(params["alpha"]))
    return {"upper": _flat(out.upper, source), "middle": _flat(out.middle, source),
            "lower": _flat(out.lower, source)}


def compile_rsi(params, data, source):
    out = _vbt().RSI.run(source, window=int(params["window"]), ewm=bool(params["ewm"]))
    return {"rsi": _flat(out.rsi, source)}


def compile_stoch(params, data, source):
    close = data["close"]
    out = _vbt().STOCH.run(data["high"], data["low"], close, k_window=int(params["k_window"]),
                           d_window=int(params["d_window"]), d_ewm=bool(params["d_ewm"]))
    return {"percent_k": _flat(out.percent_k, close), "percent_d": _flat(out.percent_d, close)}


def compile_macd(params, data, source):
    out = _vbt().MACD.run(source, fast_window=int(params["fast_window"]),
                          slow_window=int(params["slow_window"]),
                          signal_window=int(params["signal_window"]),
                          macd_ewm=bool(params["macd_ewm"]), signal_ewm=bool(params["signal_ewm"]))
    return {"macd": _flat(out.macd, source), "signal": _flat(out.signal, source)}


def compile_atr(params, data, source):
    close = data["close"]
    # ⚠️ ATR's ewm defaults to True in vectorbt, unlike every other indicator here.
    out = _vbt().ATR.run(data["high"], data["low"], close, window=int(params["window"]),
                         ewm=bool(params["ewm"]))
    return {"atr": _flat(out.atr, close), "tr": _flat(out.tr, close)}


def compile_obv(params, data, source):
    close = data["close"]
    out = _vbt().OBV.run(close, data["volume"])
    return {"obv": _flat(out.obv, close)}


# ---------------------------------------------------------------------------
# pandas blocks
# ---------------------------------------------------------------------------

def compile_momentum(params, data, source):
    window = int(params["window"])
    return {"momentum": source / source.shift(window) - 1}


def compile_highest(params, data, source):
    return {"highest": source.rolling(int(params["window"])).max()}


def compile_lowest(params, data, source):
    return {"lowest": source.rolling(int(params["window"])).min()}


def compile_zscore(params, data, source):
    rolling = source.rolling(int(params["window"]))
    return {"zscore": (source - rolling.mean()) / rolling.std()}


def compile_volatility(params, data, source):
    # fill_method=None: a NaN price stays NaN instead of producing a 0% return.
    returns = source.pct_change(fill_method=None)
    return {"volatility": returns.rolling(int(params["window"])).std()}


# ---------------------------------------------------------------------------
# Pattern block (ta_patterns)
# ---------------------------------------------------------------------------

def pattern_function(module, name):
    """The ta_patterns detector. Candlesticks live on the package, chart patterns
    in ta_patterns.chart_patterns. Both module and name are stored in the spec,
    because scan_all_patterns prefixes colliding chart names with cp_ and direct
    calls do not."""
    import ta_patterns
    import ta_patterns.chart_patterns
    return getattr(ta_patterns.chart_patterns if module == "chart" else ta_patterns, name)


def compile_pattern(params, data, source):
    detector = pattern_function(params["module"], params["name"])
    keywords = dict(params["params"])
    # Five volume patterns run only when `v` is given; pass it only if accepted.
    wants_volume = "v" in inspect.signature(detector).parameters
    close = data["close"]
    out = pd.DataFrame(0, index=close.index, columns=close.columns, dtype="int8")
    # ta_patterns takes 1-D arrays, so call the detector once per symbol and
    # rebuild the frame. Output is int8: +1 bullish, -1 bearish, 0 none, always
    # the input length. Signals sit on the bar that COMPLETES the pattern and use
    # only past bars, so the front door's next-open fill is correct execution.
    for symbol in close.columns:
        arrays = [data[field][symbol].to_numpy(dtype=float) for field in ("open", "high", "low", "close")]
        if wants_volume:
            keywords["v"] = data["volume"][symbol].to_numpy(dtype=float)
        out[symbol] = np.asarray(detector(*arrays, **keywords), dtype="int8")
    return {"signal": out}


def pattern_catalogue():
    """Every detector for the picker: module, name, direction class and keyword defaults.

    ⚠️ Assumes ta_patterns.scanner's classification sets hold candlestick names and
    ta_patterns.chart_patterns' sets hold chart names (plan annex D). Verify.
    """
    import ta_patterns
    import ta_patterns.chart_patterns as chart
    import ta_patterns.scanner as scanner

    rows = []
    groups = [("candle", ta_patterns, scanner, ["BULLISH", "BEARISH", "DIRECTIONAL", "NON_DIRECTIONAL"]),
              ("chart", chart, chart, ["BULLISH", "BEARISH", "DIRECTIONAL", "BIDIRECTIONAL", "NON_DIRECTIONAL"])]
    for module_name, module, classes, labels in groups:
        names = set()
        for label in labels:
            names |= set(getattr(classes, label, ()))
        for name in sorted(names):
            if not hasattr(module, name):
                continue
            direction = next(label.lower() for label in labels if name in getattr(classes, label, ()))
            keywords = {}
            for key, parameter in inspect.signature(getattr(module, name)).parameters.items():
                if parameter.default is not inspect.Parameter.empty and key != "v":
                    keywords[key] = parameter.default
            rows.append({"module": module_name, "name": name, "direction": direction, "params": keywords})
    return rows


# ---------------------------------------------------------------------------
# Python block (trusted local code, phase 3)
# ---------------------------------------------------------------------------

def compile_python(params, data, source):
    # ⚠️ Runs arbitrary code in-process. The UI labels it as trusted local code and
    # warns before running an imported spec that contains one.
    body = "\n".join("    " + line for line in params["code"].splitlines()) or "    pass"
    namespace = {}
    exec(f"def block(open, high, low, close, volume, np, pd):\n{body}\n", namespace)
    close = data["close"]
    result = namespace["block"](data["open"], data["high"], data["low"], close, data["volume"], np, pd)
    if isinstance(result, pd.Series):
        result = result.to_frame(close.columns[0]) if close.shape[1] == 1 else None
    if not isinstance(result, pd.DataFrame) or result.shape != close.shape:
        raise ValueError("A Python block must return a DataFrame shaped like `close` "
                         "(one column per symbol, same index).")
    return {"value": result.astype(float)}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# Parameter types: int, float, bool, choice, source (a price field or a feature
# reference), str, dict, code. `min`/`max` are the optimiser's default bounds.

SOURCE = {"type": "source", "default": "close", "label": "Input"}


def _window(default, label="Window (bars)"):
    return {"type": "int", "min": 2, "max": 500, "default": default, "label": label}


BLOCKS = {
    "ma": {
        "label": "Moving average", "group": "Trend",
        "params": {"source": SOURCE, "window": _window(20),
                   "ewm": {"type": "bool", "default": False, "label": "Exponential"}},
        "outputs": ["ma"],
        "compile": compile_ma,
        "export": "{name}__ma = flat(vectorbt.MA.run({src}, window={window}, ewm={ewm}).ma)",
    },
    "mstd": {
        "label": "Moving standard deviation", "group": "Volatility",
        "params": {"source": SOURCE, "window": _window(20),
                   "ewm": {"type": "bool", "default": False, "label": "Exponential"}},
        "outputs": ["mstd"],
        "compile": compile_mstd,
        "export": "{name}__mstd = flat(vectorbt.MSTD.run({src}, window={window}, ewm={ewm}).mstd)",
    },
    "bbands": {
        "label": "Bollinger bands", "group": "Volatility",
        "params": {"source": SOURCE, "window": _window(20),
                   "ewm": {"type": "bool", "default": False, "label": "Exponential"},
                   "alpha": {"type": "float", "min": 0.5, "max": 5.0, "default": 2.0, "label": "Width (std devs)"}},
        "outputs": ["upper", "middle", "lower"],
        "compile": compile_bbands,
        "export": ("_bb = vectorbt.BBANDS.run({src}, window={window}, ewm={ewm}, alpha={alpha})\n"
                   "{name}__upper, {name}__middle, {name}__lower = flat(_bb.upper), flat(_bb.middle), flat(_bb.lower)"),
    },
    "rsi": {
        "label": "RSI", "group": "Momentum",
        "params": {"source": SOURCE, "window": _window(14),
                   "ewm": {"type": "bool", "default": False, "label": "Exponential"}},
        "outputs": ["rsi"],
        "compile": compile_rsi,
        "export": "{name}__rsi = flat(vectorbt.RSI.run({src}, window={window}, ewm={ewm}).rsi)",
    },
    "stoch": {
        "label": "Stochastic", "group": "Momentum",
        "params": {"k_window": _window(14, "%K window"), "d_window": _window(3, "%D window"),
                   "d_ewm": {"type": "bool", "default": False, "label": "Exponential %D"}},
        "outputs": ["percent_k", "percent_d"],
        "compile": compile_stoch,
        "export": ("_st = vectorbt.STOCH.run(high, low, close, k_window={k_window}, d_window={d_window}, d_ewm={d_ewm})\n"
                   "{name}__percent_k, {name}__percent_d = flat(_st.percent_k), flat(_st.percent_d)"),
    },
    "macd": {
        "label": "MACD", "group": "Momentum",
        "params": {"source": SOURCE, "fast_window": _window(12, "Fast window"),
                   "slow_window": _window(26, "Slow window"), "signal_window": _window(9, "Signal window"),
                   "macd_ewm": {"type": "bool", "default": False, "label": "Exponential MACD"},
                   "signal_ewm": {"type": "bool", "default": False, "label": "Exponential signal"}},
        "outputs": ["macd", "signal"],
        "compile": compile_macd,
        "export": ("_mc = vectorbt.MACD.run({src}, fast_window={fast_window}, slow_window={slow_window}, "
                   "signal_window={signal_window}, macd_ewm={macd_ewm}, signal_ewm={signal_ewm})\n"
                   "{name}__macd, {name}__signal = flat(_mc.macd), flat(_mc.signal)"),
    },
    "atr": {
        "label": "Average true range", "group": "Volatility",
        "params": {"window": _window(14),
                   "ewm": {"type": "bool", "default": True, "label": "Exponential"}},
        "outputs": ["atr", "tr"],
        "compile": compile_atr,
        "export": ("_at = vectorbt.ATR.run(high, low, close, window={window}, ewm={ewm})\n"
                   "{name}__atr, {name}__tr = flat(_at.atr), flat(_at.tr)"),
    },
    "obv": {
        "label": "On-balance volume", "group": "Volume",
        "params": {},
        "outputs": ["obv"],
        "compile": compile_obv,
        "export": "{name}__obv = flat(vectorbt.OBV.run(close, volume).obv)",
    },
    "momentum": {
        "label": "Momentum (return over window)", "group": "Momentum",
        "params": {"source": SOURCE, "window": _window(20)},
        "outputs": ["momentum"],
        "compile": compile_momentum,
        "export": "{name}__momentum = {src} / {src}.shift({window}) - 1",
    },
    "highest": {
        "label": "Highest value", "group": "Channels",
        "params": {"source": dict(SOURCE, default="high"), "window": _window(20)},
        "outputs": ["highest"],
        "compile": compile_highest,
        "export": "{name}__highest = {src}.rolling({window}).max()",
    },
    "lowest": {
        "label": "Lowest value", "group": "Channels",
        "params": {"source": dict(SOURCE, default="low"), "window": _window(20)},
        "outputs": ["lowest"],
        "compile": compile_lowest,
        "export": "{name}__lowest = {src}.rolling({window}).min()",
    },
    "zscore": {
        "label": "Z-score", "group": "Statistics",
        "params": {"source": SOURCE, "window": _window(20)},
        "outputs": ["zscore"],
        "compile": compile_zscore,
        "export": "{name}__zscore = ({src} - {src}.rolling({window}).mean()) / {src}.rolling({window}).std()",
    },
    "volatility": {
        "label": "Volatility (std of returns)", "group": "Volatility",
        "params": {"source": SOURCE, "window": _window(20)},
        "outputs": ["volatility"],
        "compile": compile_volatility,
        "export": "{name}__volatility = {src}.pct_change(fill_method=None).rolling({window}).std()",
    },
    "pattern": {
        "label": "Chart or candlestick pattern", "group": "Patterns",
        "params": {"module": {"type": "choice", "options": ["candle", "chart"], "default": "candle",
                              "label": "Pattern family"},
                   "name": {"type": "str", "default": "hammer", "label": "Pattern"},
                   "params": {"type": "dict", "default": {}, "label": "Detector settings"}},
        "outputs": ["signal"],
        "compile": compile_pattern,
        "export": "{name}__signal = pattern_signal(open_, high, low, close, volume, {module}, {detector}, {keywords})",
    },
    "python": {
        "label": "Python (trusted local code)", "group": "Custom",
        "params": {"code": {"type": "code", "default": "return close.rolling(10).mean()",
                            "label": "Function body: receives open, high, low, close, volume, np, pd"}},
        "outputs": ["value"],
        "compile": compile_python,
        "export": ("def _{name}(open, high, low, close, volume, np, pd):\n{code}\n"
                   "{name}__value = _{name}(open_, high, low, close, volume, np, pd).astype(float)"),
    },
}


def block_defaults(block_name):
    """A new feature dict for a block, with every parameter at its default."""
    feature = {"block": block_name}
    for key, declaration in BLOCKS[block_name]["params"].items():
        default = declaration["default"]
        feature[key] = dict(default) if isinstance(default, dict) else default
    return feature
