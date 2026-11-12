"""The compiler: spec + data -> signals or weights.

``compile_signals`` and ``compile_weights`` each read top to bottom with a
comment per step (plan section 7). ``export_script`` turns a spec into Python
source that reproduces the backtest with plain vectorbtX calls; the export
equivalence test (section 12, test 8) keeps the two paths from drifting apart.

`data` is always a dict of five DataFrames (open, high, low, close, volume),
one column per symbol, on one shared index. It may carry an extra "fx" Series,
which only the engine reads.
"""

import hashlib
import textwrap

import numpy as np
import pandas as pd

from . import engine
from .blocks import BLOCKS
from .data import resample_frames
from .spec import (PRICE_FIELDS, RESAMPLE_RULE, RULE_SLOTS, apply_params, canonical_json, feature_order,
                   strategy_kind)

# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
# Optimisation reruns the compiler thousands of times, so feature results are
# memoised per process, keyed by (block, canonical JSON of its resolved params,
# data fingerprint). Pattern features include every detector keyword in the key.
# Whole backtests are NOT memoised: vectorbt is fast enough and stale portfolio
# objects are memory-heavy.
# ⚠️ Cached frames are shared: never modify a feature frame in place.
FEATURE_CACHE = {}
CACHE_LIMIT = 256


def _cache_put(key, value):
    FEATURE_CACHE[key] = value
    while len(FEATURE_CACHE) > CACHE_LIMIT:
        FEATURE_CACHE.pop(next(iter(FEATURE_CACHE)))   # oldest first


def data_fingerprint(data):
    """A cheap content hash of the five frames (well under a millisecond per 10k bars)."""
    digest = hashlib.blake2b(digest_size=16)
    close = data["close"]
    digest.update(close.index.asi8.tobytes())
    digest.update(canonical_json([str(column) for column in close.columns]).encode())
    for field in PRICE_FIELDS:
        digest.update(np.ascontiguousarray(data[field].to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()


def _merged_params(feature):
    """Every declared parameter of the feature's block, defaults filled in."""
    params = {}
    for key, declaration in BLOCKS[feature["block"]]["params"].items():
        params[key] = feature.get(key, declaration["default"])
    return params


def _output_name(features, ref):
    return ref.get("output") or BLOCKS[features[ref["feature"]]["block"]]["outputs"][0]


def compile_features(features, data, fingerprint=None):
    """{feature name: {output name: DataFrame}} for every feature, in dependency order."""
    fingerprint = fingerprint or data_fingerprint(data)
    computed = {}
    keys = {}
    for name in feature_order(features):
        feature = features[name]
        block = BLOCKS[feature["block"]]
        params = _merged_params(feature)
        key_params = dict(params)
        source = None
        if "source" in block["params"]:
            if isinstance(params["source"], dict):
                # A feature of a feature (an RSI of a moving average, say). Key on
                # the upstream's own cache key so a change upstream misses here too.
                upstream = params["source"]
                output = _output_name(features, upstream)
                source = computed[upstream["feature"]][output]
                key_params["source"] = f"{keys[upstream['feature']]}/{output}"
            else:
                source = data[params["source"]]
        key = (feature["block"], canonical_json(key_params), fingerprint)
        if key not in FEATURE_CACHE:
            _cache_put(key, block["compile"](params, data, source))
        computed[name] = FEATURE_CACHE[key]
        keys[name] = hashlib.blake2b(repr(key).encode(), digest_size=8).hexdigest()
    return computed


def _pattern_signal(leaf, data, fingerprint):
    """A pattern rule leaf's detector output, through the same cache as features."""
    params = {"module": leaf.get("module", "candle"), "name": leaf["pattern"],
              "params": dict(leaf.get("params") or {})}
    key = ("pattern", canonical_json(params), fingerprint)
    if key not in FEATURE_CACHE:
        _cache_put(key, BLOCKS["pattern"]["compile"](params, data, None))
    return FEATURE_CACHE[key]["signal"]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _full(value, like):
    """Broadcast a scalar to a frame shaped like `like`; frames pass through."""
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(float(value), index=like.index, columns=like.columns)


def _mask_frame(mask, like):
    """Broadcast a 1-D boolean mask over the index to every column."""
    values = np.repeat(np.asarray(mask, dtype=bool)[:, None], like.shape[1], axis=1)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def resolve_ref(ref, data, features, spec_features):
    """A feature output, a price frame or a constant float, shifted if asked."""
    if "feature" in ref:
        value = features[ref["feature"]][_output_name(spec_features, ref)]
    elif "price" in ref:
        value = data[ref["price"]]
    else:
        value = float(ref["const"])
    shift = int(ref.get("shift", 0))
    if shift and isinstance(value, pd.DataFrame):
        value = value.shift(shift)          # the value `shift` bars ago: history only
    return value


def _session_mask(index, start, end):
    times = index.time
    start = pd.Timestamp(f"2000-01-01 {start}").time()
    end = pd.Timestamp(f"2000-01-01 {end}").time()
    if start <= end:
        return (times >= start) & (times < end)
    return (times >= start) | (times < end)      # a session that wraps past midnight


def evaluate_rule(node, data, features, spec_features, fingerprint):
    """A boolean DataFrame (shaped like close) for one rule tree."""
    close = data["close"]
    if "all" in node or "any" in node:
        combine_all = "all" in node
        result = None
        for child in node["all" if combine_all else "any"]:
            signal = evaluate_rule(child, data, features, spec_features, fingerprint)
            if result is None:
                result = signal
            else:
                result = (result & signal) if combine_all else (result | signal)
        return result
    if "not" in node:
        return ~evaluate_rule(node["not"], data, features, spec_features, fingerprint)
    if "pattern" in node:
        signal = _pattern_signal(node, data, fingerprint)
        side = node.get("side", "either")
        if side == "bullish":
            return signal == 1
        if side == "bearish":
            return signal == -1
        return signal != 0
    if "filter" in node:
        kind = node["filter"]
        if kind == "session":
            return _mask_frame(_session_mask(close.index, node["from"], node["to"]), close)
        if kind == "weekday":
            return _mask_frame(close.index.weekday.isin(node["days"]), close)
        if kind == "month":
            return _mask_frame(close.index.month.isin(node["months"]), close)
        # Higher timeframe: compute the condition on resampled bars, then shift by
        # one higher bar and forward-fill back onto the base index.
        # ⚠️ The resample MUST be label="left", closed="left" followed by
        # .shift(1). pandas aliases such as W and ME default to right-labelled
        # bins, and without both settings a weekly value leaks into the days
        # before the week completes. resample_frames does the left/left part.
        higher = resample_frames(data, RESAMPLE_RULE[node["timeframe"]])
        higher_fp = data_fingerprint(higher)
        higher_features = compile_features(spec_features, higher, higher_fp)
        condition = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars", "shift")
                     if key in node}
        signal = evaluate_rule(condition, higher, higher_features, spec_features, higher_fp)
        signal = signal.shift(1, fill_value=False)
        signal = signal.astype(float).reindex(close.index, method="ffill")
        return signal.fillna(0.0).astype(bool)

    # A comparison leaf.
    op = node["op"]
    left = resolve_ref(node["left"], data, features, spec_features)
    if op in ("rising", "falling"):
        bars = int(node.get("bars", 1))
        left = _full(left, close)
        result = left > left.shift(bars) if op == "rising" else left < left.shift(bars)
    elif op == "between":
        lower = resolve_ref(node["lower"], data, features, spec_features)
        upper = resolve_ref(node["upper"], data, features, spec_features)
        left = _full(left, close)
        result = (left >= lower) & (left <= upper)
    elif op in ("crosses_above", "crosses_below"):
        engine.vectorbt()   # registers the .vbt accessor
        right = _full(resolve_ref(node["right"], data, features, spec_features), close)
        left = _full(left, close)
        # vectorbt's generic accessor. Its crossing rule needs the left side to
        # have been strictly below (above) first, which differs from the naive
        # (a > b) & (a.shift() <= b.shift()) when values touch.
        if op == "crosses_above":
            result = left.vbt.crossed_above(right)
        else:
            result = left.vbt.crossed_below(right)
    else:
        right = resolve_ref(node["right"], data, features, spec_features)
        left = _full(left, close)
        if op == ">":
            result = left > right
        elif op == "<":
            result = left < right
        elif op == ">=":
            result = left >= right
        else:
            result = left <= right
    # Indicator warm-up periods are NaN, and NaN compares False. fillna keeps the
    # dtype boolean whatever the inputs were.
    result = pd.DataFrame(result).fillna(False).astype(bool)
    shift = int(node.get("shift", 0))
    if shift:
        result = result.shift(shift, fill_value=False)
    return result


# ---------------------------------------------------------------------------
# Signal strategies
# ---------------------------------------------------------------------------

def compile_signals(spec, data, shift_shorts=True):
    """{"entries", "exits", "short_entries", "short_exits"}: boolean frames like close.

    `spec` must already have its parameters applied (no "opt" nodes).
    shift_shorts=False is only for the live preview, which marks signal bars.
    """
    close = data["close"]
    fingerprint = data_fingerprint(data)
    spec_features = spec.get("features") or {}

    # 1. Frames: `data` is already the five aligned frames.
    # 2. Features, in dependency order.
    features = compile_features(spec_features, data, fingerprint)

    # 3. Rules. A missing rule is all-False. Filters live inside entry rules
    #    only (validate_spec rejects them in exits), so they gate entries and
    #    never trap a position open.
    rules = spec.get("rules") or {}
    names = {"long_entry": "entries", "long_exit": "exits",
             "short_entry": "short_entries", "short_exit": "short_exits"}
    out = {}
    for slot in RULE_SLOTS:
        node = rules.get(slot)
        if node:
            out[names[slot]] = evaluate_rule(node, data, features, spec_features, fingerprint)
        else:
            out[names[slot]] = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 4. Shift short signals by one bar.
    # ⚠️ PortfolioBacktest shifts entries, exits and schedule itself (_next_bar),
    # but short_entries/short_exits travel through **kwargs to
    # Portfolio.from_signals UNSHIFTED. A short computed at bar t's close would
    # fill at bar t's open: lookahead. So shorts are shifted here, and long
    # signals must NOT be shifted again (the front door does it).
    if shift_shorts:
        out["short_entries"] = out["short_entries"].shift(1, fill_value=False)
        out["short_exits"] = out["short_exits"].shift(1, fill_value=False)
    return out


# ---------------------------------------------------------------------------
# Portfolio strategies
# ---------------------------------------------------------------------------

def inverse_vol_weight(prices, window=20):
    """Weights proportional to 1 / rolling volatility, normalised per row.

    vectorbtX's API.md documents inverse_vol_weight but it does not exist in the
    package yet; this matches the documented signature so it can be swapped for
    vbt.x.inverse_vol_weight once it is added.
    """
    volatility = prices.pct_change(fill_method=None).rolling(window).std()
    inverse = 1.0 / volatility.replace(0.0, np.nan)
    return inverse.div(inverse.sum(axis=1), axis=0)


def compile_weights(spec, data):
    """{"weights", "schedule", "min_trade"} for the weights/schedule route.

    `spec` must already have its parameters applied.
    """
    close = data["close"]
    fingerprint = data_fingerprint(data)
    spec_features = spec.get("features") or {}

    # 1. Features, then the score frame named in selection.score.
    features = compile_features(spec_features, data, fingerprint)

    # 2. Select. Rank across columns per row and keep the top (or bottom) N with
    #    a score at or above min_score. A row where nothing qualifies selects
    #    nothing, which means holding cash. No selection means every asset.
    selection = spec.get("selection")
    if selection:
        score = _full(resolve_ref(selection["score"], data, features, spec_features), close)
        mode = selection.get("mode", "top")
        if mode in ("top", "bottom"):
            # method="first" breaks ties by column order, so exactly N are kept.
            rank = score.rank(axis=1, ascending=(mode == "bottom"), method="first")
            mask = rank <= int(selection["top_n"])
        else:
            mask = score >= float(selection["min_score"])
        if selection.get("min_score") is not None:
            mask = mask & (score >= float(selection["min_score"]))
        mask = mask & score.notna()
    else:
        mask = close.notna()
    mask = mask.fillna(False).astype(bool)

    # 3. Weight. Each scheme gives raw weights, masked to the selection, then
    #    divided by the row sum. Rows summing to zero stay zero (cash).
    weighting = spec["weighting"]
    scheme = weighting["scheme"]
    if scheme == "equal":
        raw = mask.astype(float)
    elif scheme == "inverse_vol":
        raw = inverse_vol_weight(close, window=int(weighting.get("vol_window", 20))).where(mask, 0.0)
    elif scheme == "price":
        raw = engine.price_weight(close).where(mask, 0.0)
    else:
        # A fixed table is used as given (a 60/40 split, or 90% invested with 10%
        # cash), so it is masked but NOT re-normalised.
        table = pd.Series(weighting.get("table") or {}, dtype=float).reindex(close.columns).fillna(0.0)
        raw = pd.DataFrame(np.tile(table.to_numpy(), (len(close), 1)), index=close.index, columns=close.columns)
        raw = raw.where(mask, 0.0)
    if scheme == "fixed":
        weights = raw.fillna(0.0)
    else:
        total = raw.sum(axis=1)
        weights = raw.div(total.where(total > 0), axis=0).fillna(0.0)

    # 4. Schedule. Calendar dates, optionally combined with a drift threshold.
    #    ⚠️ Always pass schedule= to drifted_weights, or drift accumulates from the
    #    first bar and the threshold test becomes always-true.
    schedule_spec = spec["schedule"]
    calendar = schedule_spec.get("calendar")
    if calendar:
        schedule = engine.calendar_schedule(close.index, freq=calendar["freq"], on=calendar.get("on", "first"))
        if schedule_spec.get("threshold") is not None:
            drifted = engine.drifted_weights(weights, close, schedule=schedule)
            breached = engine.threshold_schedule(drifted, weights, tolerance=float(schedule_spec["threshold"]))
            if schedule_spec.get("combine", "and") == "and":
                schedule = schedule & breached
            else:
                schedule = schedule | breached
    else:
        schedule = engine.always_schedule(close.index)

    # 5. Weights stay a full-shape frame. The library turns them into
    #    NaN-off-schedule targets itself. ⚠️ Never forward-fill them: that would
    #    rebalance every bar.
    return {"weights": weights, "schedule": schedule, "min_trade": float(schedule_spec.get("min_trade", 0.0))}


# ---------------------------------------------------------------------------
# Code export
# ---------------------------------------------------------------------------

_SCRIPT_HELPERS = '''
def full(value, like):
    """Broadcast a scalar to a frame shaped like `like`."""
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(float(value), index=like.index, columns=like.columns)


def as_bool(frame):
    return pd.DataFrame(frame).fillna(False).astype(bool)


def mask_frame(mask, like):
    values = np.repeat(np.asarray(mask, dtype=bool)[:, None], like.shape[1], axis=1)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def session_mask(index, start, end):
    times = index.time
    start = pd.Timestamp(f"2000-01-01 {start}").time()
    end = pd.Timestamp(f"2000-01-01 {end}").time()
    return (times >= start) & (times < end) if start <= end else (times >= start) | (times < end)


def resample(frames, rule):
    # ⚠️ label="left", closed="left", then .shift(1) after computing: no leakage.
    def bins(frame):
        return frame.resample(rule, label="left", closed="left")
    out = {"open": bins(frames["open"]).first(), "high": bins(frames["high"]).max(),
           "low": bins(frames["low"]).min(), "close": bins(frames["close"]).last(),
           "volume": bins(frames["volume"]).sum(min_count=1)}
    keep = out["close"].notna().any(axis=1)
    return {field: frame[keep] for field, frame in out.items()}


def higher_timeframe(signal, base_index):
    signal = signal.shift(1, fill_value=False).astype(float).reindex(base_index, method="ffill")
    return signal.fillna(0.0).astype(bool)


def pattern_signal(open_, high, low, close, volume, module, name, keywords):
    import inspect
    import ta_patterns
    import ta_patterns.chart_patterns
    detector = getattr(ta_patterns.chart_patterns if module == "chart" else ta_patterns, name)
    keywords = dict(keywords)
    wants_volume = "v" in inspect.signature(detector).parameters
    out = pd.DataFrame(0, index=close.index, columns=close.columns, dtype="int8")
    for symbol in close.columns:
        arrays = [frame[symbol].to_numpy(dtype=float) for frame in (open_, high, low, close)]
        if wants_volume:
            keywords["v"] = volume[symbol].to_numpy(dtype=float)
        out[symbol] = np.asarray(detector(*arrays, **keywords), dtype="int8")
    return out
'''

_PRICE_VARS = {"open": "open_", "high": "high", "low": "low", "close": "close", "volume": "volume"}


def _feature_code(spec_features):
    """The body of compute_features(open_, high, low, close, volume) -> dict."""
    lines = []
    produced = []
    for name in feature_order(spec_features):
        feature = spec_features[name]
        block = BLOCKS[feature["block"]]
        params = _merged_params(feature)
        fields = {"name": name}
        for key, value in params.items():
            # Plain Python literals in the script (a numpy int would print as np.int64(50)).
            kind = block["params"][key]["type"]
            if kind == "int":
                value = int(value)
            elif kind == "float":
                value = float(value)
            elif kind == "bool":
                value = bool(value)
            fields[key] = repr(value)
        if "source" in block["params"]:
            source = params["source"]
            if isinstance(source, dict):
                fields["src"] = f"{source['feature']}__{_output_name(spec_features, source)}"
            else:
                fields["src"] = _PRICE_VARS[source]
        if feature["block"] == "pattern":
            # The detector's own "name" parameter would clash with the feature name.
            fields = {"name": name, "module": repr(params["module"]), "detector": repr(params["name"]),
                      "keywords": repr(dict(params["params"]))}
        elif feature["block"] == "python":
            fields = {"name": name, "code": textwrap.indent(params["code"], "    ")}
        lines.append(block["export"].format(**fields))
        produced += [f"{name}__{output}" for output in block["outputs"]]
    lines.append("return {" + ", ".join(f"{var!r}: {var}" for var in produced) + "}")
    return lines


def _ref_code(ref, spec_features, frames="P", feats="F"):
    if "feature" in ref:
        code = f"{feats}[{(ref['feature'] + '__' + _output_name(spec_features, ref))!r}]"
    elif "price" in ref:
        code = f"{frames}[{ref['price']!r}]"
    else:
        return repr(float(ref["const"]))
    shift = int(ref.get("shift", 0))
    return f"{code}.shift({shift})" if shift else code


def _rule_code(node, spec_features, frames="P", feats="F"):
    """Python source for one rule tree, mirroring evaluate_rule exactly."""
    close = f"{frames}['close']"
    if "all" in node or "any" in node:
        joiner = " & " if "all" in node else " | "
        parts = [_rule_code(child, spec_features, frames, feats) for child in node["all" if "all" in node else "any"]]
        return "(" + joiner.join(parts) + ")"
    if "not" in node:
        return "(~" + _rule_code(node["not"], spec_features, frames, feats) + ")"
    if "pattern" in node:
        call = (f"pattern_signal({frames}['open'], {frames}['high'], {frames}['low'], {close}, {frames}['volume'], "
                f"{node.get('module', 'candle')!r}, {node['pattern']!r}, {dict(node.get('params') or {})!r})")
        side = node.get("side", "either")
        return f"({call} {'== 1' if side == 'bullish' else '== -1' if side == 'bearish' else '!= 0'})"
    if "filter" in node:
        kind = node["filter"]
        if kind == "session":
            return f"mask_frame(session_mask({close}.index, {node['from']!r}, {node['to']!r}), {close})"
        if kind == "weekday":
            return f"mask_frame({close}.index.weekday.isin({list(node['days'])!r}), {close})"
        if kind == "month":
            return f"mask_frame({close}.index.month.isin({list(node['months'])!r}), {close})"
        condition = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars", "shift")
                     if key in node}
        rule = RESAMPLE_RULE[node["timeframe"]]
        inner = _rule_code(condition, spec_features, "PH", "FH")
        return (f"(lambda PH: (lambda FH: higher_timeframe({inner}, {close}.index))"
                f"(compute_features(PH['open'], PH['high'], PH['low'], PH['close'], PH['volume'])))"
                f"(resample({frames}, {rule!r}))")
    op = node["op"]
    left = _ref_code(node["left"], spec_features, frames, feats)
    if op in ("rising", "falling"):
        bars = int(node.get("bars", 1))
        sign = ">" if op == "rising" else "<"
        code = f"(lambda x: x {sign} x.shift({bars}))(full({left}, {close}))"
    elif op == "between":
        lower = _ref_code(node["lower"], spec_features, frames, feats)
        upper = _ref_code(node["upper"], spec_features, frames, feats)
        code = f"(lambda x: (x >= {lower}) & (x <= {upper}))(full({left}, {close}))"
    elif op in ("crosses_above", "crosses_below"):
        right = _ref_code(node["right"], spec_features, frames, feats)
        code = f"full({left}, {close}).vbt.{'crossed_above' if op == 'crosses_above' else 'crossed_below'}(full({right}, {close}))"
    else:
        right = _ref_code(node["right"], spec_features, frames, feats)
        code = f"(full({left}, {close}) {op} {right})"
    code = f"as_bool({code})"
    shift = int(node.get("shift", 0))
    return f"{code}.shift({shift}, fill_value=False)" if shift else code


def _portfolio_code(spec):
    spec_features = spec.get("features") or {}
    lines = []
    selection = spec.get("selection")
    if selection:
        lines.append(f"score = full({_ref_code(selection['score'], spec_features)}, close)")
        mode = selection.get("mode", "top")
        if mode in ("top", "bottom"):
            lines.append(f"rank = score.rank(axis=1, ascending={mode == 'bottom'}, method='first')")
            lines.append(f"mask = rank <= {int(selection['top_n'])}")
        else:
            lines.append(f"mask = score >= {float(selection['min_score'])!r}")
        if selection.get("min_score") is not None:
            lines.append(f"mask = mask & (score >= {float(selection['min_score'])!r})")
        lines.append("mask = mask & score.notna()")
    else:
        lines.append("mask = close.notna()")
    lines.append("mask = mask.fillna(False).astype(bool)")
    weighting = spec["weighting"]
    scheme = weighting["scheme"]
    if scheme == "equal":
        lines.append("raw = mask.astype(float)")
    elif scheme == "inverse_vol":
        window = int(weighting.get("vol_window", 20))
        lines += ["vol = close.pct_change(fill_method=None).rolling(%d).std()" % window,
                  "inverse = 1.0 / vol.replace(0.0, np.nan)",
                  "raw = inverse.div(inverse.sum(axis=1), axis=0).where(mask, 0.0)"]
    elif scheme == "price":
        lines.append("raw = vbt.x.price_weight(close).where(mask, 0.0)")
    else:
        lines += [f"table = pd.Series({dict(weighting.get('table') or {})!r}, dtype=float).reindex(close.columns).fillna(0.0)",
                  "raw = pd.DataFrame(np.tile(table.to_numpy(), (len(close), 1)), index=close.index, columns=close.columns)",
                  "raw = raw.where(mask, 0.0)"]
    if scheme == "fixed":
        lines.append("weights = raw.fillna(0.0)")
    else:
        lines += ["total = raw.sum(axis=1)", "weights = raw.div(total.where(total > 0), axis=0).fillna(0.0)"]
    schedule = spec["schedule"]
    calendar = schedule.get("calendar")
    if calendar:
        lines.append(f"schedule = vbt.x.calendar_schedule(close.index, freq={calendar['freq']!r}, "
                     f"on={calendar.get('on', 'first')!r})")
        if schedule.get("threshold") is not None:
            lines.append("drifted = vbt.x.drifted_weights(weights, close, schedule=schedule)   # schedule= is essential")
            lines.append(f"breached = vbt.x.threshold_schedule(drifted, weights, tolerance={float(schedule['threshold'])!r})")
            lines.append(f"schedule = schedule {'&' if schedule.get('combine', 'and') == 'and' else '|'} breached")
    else:
        lines.append("schedule = vbt.x.always_schedule(close.index)")
    return lines


def export_script(spec, data_dir=None):
    """Python source reproducing the backtest with plain vectorbtX calls."""
    spec = apply_params(spec)
    spec_features = spec.get("features") or {}
    kind = strategy_kind(spec)
    plan = engine.call_plan(spec, data_dir=data_dir)
    account = spec.get("account") or {}
    lines = [
        f'"""{spec["name"]}',
        "",
        "Exported from OpenStrategy Studio. It reproduces the Studio's backtest with",
        "plain vectorbtX calls; run it directly to backtest the stored data again.",
        '"""',
        "",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import vectorbt                  # indicators and the .vbt pandas accessors",
        "import vectorbtx as vbt          # lower-case import name; vbt.x holds the front doors",
        "",
        _SCRIPT_HELPERS.strip("\n"),
        "",
        "",
        "def compute_features(open_, high, low, close, volume):",
        "    def flat(values, like=close):",
        "        # vectorbt outputs carry a parameter column level; rebuild with close's labels.",
        "        return pd.DataFrame(np.asarray(values, dtype=float).reshape(like.shape), index=like.index, columns=like.columns)",
        "",
    ]
    lines += ["    " + line for code in _feature_code(spec_features) for line in code.splitlines()]
    if "adjust_sl_func_nb" in plan["args"]:
        from . import exit_rules
        lines += ["", "", "# --- Extra exits: fixed numba templates (time stop, breakeven, step trailing) ---",
                  "from numba import njit", "", exit_rules.export_source()]
    lines += ["", "", "def run(open_, high, low, close, volume, fx=None):",
              "    P = {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}"]
    body = []
    if account.get("fx") and account.get("base_currency"):
        body += ["# Convert open, high, low AND close with the same FX table BEFORE any signal.",
                 f"universe = {engine.universe_code(spec, data_dir)}",
                 "for _field in ('open', 'high', 'low', 'close'):",
                 f"    P[_field] = vbt.x.to_base_currency(P[_field], base={account['base_currency']!r}, fx=fx, instruments=universe)",
                 "open_, high, low, close = P['open'], P['high'], P['low'], P['close']"]
    body.append("F = compute_features(open_, high, low, close, volume)")
    if kind == "signals":
        rules = spec["rules"]
        names = {"long_entry": "entries", "long_exit": "exits",
                 "short_entry": "short_entries", "short_exit": "short_exits"}
        for slot in RULE_SLOTS:
            if rules.get(slot):
                body.append(f"{names[slot]} = {_rule_code(rules[slot], spec_features)}")
            else:
                body.append(f"{names[slot]} = pd.DataFrame(False, index=close.index, columns=close.columns)")
        body += ["# ⚠️ The front door shifts entries and exits by one bar itself, but NOT the",
                 "# short arrays (they pass through **kwargs unshifted), so shift them here.",
                 "short_entries = short_entries.shift(1, fill_value=False)",
                 "short_exits = short_exits.shift(1, fill_value=False)"]
    elif kind == "portfolio":
        body += _portfolio_code(spec)
        body.append(f"min_trade = {float(spec['schedule'].get('min_trade', 0.0))!r}")
    rates = account.get("rates")
    if rates:
        body.append(f"rates = {engine.rates_code(spec, data_dir)}")
    if plan["single"]:
        symbol = spec["data"]["symbols"][0]
        body.append("# SingleBacktest takes Series and flat instrument keywords.")
        for var in ("close", "open_", "high", "low", "volume", "entries", "exits", "short_entries", "short_exits"):
            if var in ("close", "open_", "high", "low", "volume") or kind == "signals":
                body.append(f"{var} = {var}[{symbol!r}]")
    if "adjust_sl_func_nb" in plan["args"]:
        from . import exit_rules
        settings = exit_rules.exit_settings(spec)
        sizes = engine.point_sizes(spec, list(spec["data"]["symbols"]), data_dir)

        def literal(value):
            return 'float("nan")' if value != value else repr(value)

        body += ["# Extra exits: the Studio's numba templates (defined above), with fresh state for this run.",
                 "_o, _h, _l = [np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(len(x), -1)) "
                 "for x in (open_, high, low)]",
                 "adjust_sl_func_nb = adjust_sl_nb",
                 f"adjust_sl_args = (_o, _h, _l, np.array({sizes!r}), np.full(_o.shape[1], "
                 f"{float(spec['risk']['stop']['points'])!r}), {literal(settings['time_bars'])}, "
                 f"{literal(settings['breakeven_points'])}, {literal(settings['step_points'])}, "
                 "np.full(_o.shape[1], -1, dtype=np.int64), np.full(_o.shape[1], np.nan))"]
        if "adjust_tp_func_nb" in plan["args"]:
            body += ["adjust_tp_func_nb = adjust_tp_nb", f"adjust_tp_args = (_o, {literal(settings['time_bars'])})"]
    call = [f"pf = vbt.x.{plan['front_door']}("]
    for keyword, (how, value) in plan["args"].items():
        call.append(f"    {keyword}={value if how == 'var' else repr(value)},")
    call.append(")")
    body += call + ["return pf"]
    lines += ["    " + line for line in body]

    # The __main__ block reads the Studio's own parquet store, aligned the same way.
    data = spec["data"]
    lines += ["", "", "if __name__ == '__main__':",
              f"    DATA_DIR = Path({str(data_dir) if data_dir else '.'!r})",
              f"    SYMBOLS, TIMEFRAME = {list(data['symbols'])!r}, {data['timeframe']!r}",
              f"    START, END = {data['start']!r}, {data['end']!r}",
              f"    ADJUSTED = {engine.adjusted_symbols(spec)!r}   # equity symbols read adj_close",
              "    frames = {}",
              "    for symbol in SYMBOLS:",
              "        safe = ''.join('_' if ch in '\\\\/:*?\"<>|' else ch for ch in symbol)",
              "        raw = pd.read_parquet(DATA_DIR / 'prices' / safe / f'{TIMEFRAME}.parquet').loc[START:END]",
              "        if symbol in ADJUSTED and 'adj_close' in raw:",
              "            factor = raw['adj_close'] / raw['close']",
              "            raw = raw.assign(open=raw['open'] * factor, high=raw['high'] * factor,",
              "                             low=raw['low'] * factor, close=raw['adj_close'])",
              "        frames[symbol] = raw",
              "    # Align on the common trading days, as the Studio does.",
              "    close = pd.concat({s: f['close'] for s, f in frames.items()}, axis=1).dropna(how='any')",
              "    fields = {}",
              "    for field in ('open', 'high', 'low', 'close', 'volume'):",
              "        fields[field] = pd.concat({s: f[field] if field in f else pd.Series(np.nan, index=f.index)",
              "                                   for s, f in frames.items()}, axis=1).reindex(close.index)"]
    if account.get("fx"):
        fx_symbol = account["fx"]["symbol"]
        safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in fx_symbol)
        lines += [f"    fx = pd.read_parquet(DATA_DIR / 'prices' / {safe!r} / f'{{TIMEFRAME}}.parquet')['close']",
                  f"    fx = fx.rename({fx_symbol!r})"]
    else:
        lines.append("    fx = None")
    lines += ["    pf = run(fields['open'], fields['high'], fields['low'], fields['close'], fields['volume'], fx=fx)",
              "    print(pf.stats())", ""]
    return "\n".join(lines)

