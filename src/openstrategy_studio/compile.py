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

import numpy as np
import pandas as pd

from . import engine, indicators
from .data import resample_frames
from .expressions import parse_logic
from .spec import (PRICE_FIELDS, RESAMPLE_RULE, RULE_SLOTS, apply_params, canonical_json, cash_model,
                   feature_order, strategy_kind)

# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
# Optimisation reruns the compiler thousands of times, so feature results are
# memoised per process, keyed by (indicator, canonical JSON of its parameters and
# inputs, data fingerprint). Whole backtests are NOT memoised: vectorbt is fast
# enough and stale portfolio objects are memory-heavy.
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


def _output_name(features, ref):
    """The output a reference reads: its own, else the feature's first."""
    return ref.get("output") or indicators.outputs(features[ref["feature"]]["indicator"])[0]


def _input_ref(feature, name):
    item = indicators.entry(feature["indicator"])
    return (feature.get("inputs") or {}).get(name, item["defaults"].get(name) or "close")


def compile_features(features, data, fingerprint=None):
    """{feature name: {output name: DataFrame}} for every feature, in dependency order."""
    fingerprint = fingerprint or data_fingerprint(data)
    computed = {}
    keys = {}
    for name in feature_order(features):
        feature = features[name]
        item = indicators.entry(feature["indicator"])
        frames = []
        input_keys = []
        for input_name in item["inputs"]:
            ref = _input_ref(feature, input_name)
            if isinstance(ref, dict):
                # A feature of a feature (an RSI of a moving average, say). Key on the
                # upstream's own cache key so a change upstream misses here too.
                output = _output_name(features, ref)
                frames.append(computed[ref["feature"]][output])
                input_keys.append(f"{keys[ref['feature']]}/{output}")
            else:
                frames.append(data[ref])
                input_keys.append(ref)
        key = (feature["indicator"], canonical_json({"params": indicators.call_params(feature), "inputs": input_keys}),
               fingerprint)
        if key not in FEATURE_CACHE:
            _cache_put(key, indicators.compute(feature, frames, data))
        computed[name] = FEATURE_CACHE[key]
        keys[name] = hashlib.blake2b(repr(key).encode(), digest_size=8).hexdigest()
    return computed


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
    shift = int(ref.get("shift", 0) or 0)
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


def evaluate_condition(node, data, features, spec_features, fingerprint):
    """A boolean DataFrame (shaped like close) for one condition."""
    close = data["close"]
    if "pattern" in node:
        signal = features[node["pattern"]]["signal"]
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
            return _mask_frame(close.index.day_of_week.isin(node["days"]), close)
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
        comparison = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars") if key in node}
        signal = evaluate_condition(comparison, higher, higher_features, spec_features, higher_fp)
        signal = signal.shift(1, fill_value=False)
        signal = signal.astype(float).reindex(close.index, method="ffill")
        return signal.fillna(0.0).astype(bool)

    # A comparison.
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
    return pd.DataFrame(result).fillna(False).astype(bool)


def evaluate_logic(tree, masks):
    """Combine the conditions' masks as the logic line says (expressions.parse_logic's tree)."""
    if isinstance(tree, str):
        return masks[tree]
    if "not" in tree:
        return ~evaluate_logic(tree["not"], masks)
    values = [evaluate_logic(arg, masks) for arg in tree["args"]]
    op = tree["op"]
    combined = values[0]
    for value in values[1:]:
        if op in ("and", "nand"):
            combined = combined & value
        elif op in ("or", "nor"):
            combined = combined | value
        else:
            combined = combined ^ value          # xor and xnor: true for an odd number
    return ~combined if op in ("nand", "nor", "xnor") else combined


def evaluate_rule(rule, data, features, spec_features, fingerprint):
    """A boolean DataFrame (shaped like close) for one rule: its conditions, combined by its logic line."""
    masks = {letter: evaluate_condition(condition, data, features, spec_features, fingerprint)
             for letter, condition in rule["conditions"].items()}
    return evaluate_logic(parse_logic(rule["logic"]), masks)


def day_boundaries(index):
    """(next bar is another day, next-but-one bar is another day) as boolean arrays, by UTC date.

    ⚠️ VERIFY for a market whose trading day crosses midnight UTC (FX rolls at 17:00 New
    York): its "day" here is the calendar date of the bars, which the Risk step says.
    """
    dates = np.asarray(index.normalize())
    last = np.ones(len(dates), dtype=bool)
    last[:-1] = dates[1:] != dates[:-1]
    second_last = np.zeros(len(dates), dtype=bool)
    second_last[:-1] = last[1:] & ~last[:-1]
    return last, second_last


# ---------------------------------------------------------------------------
# Signal strategies
# ---------------------------------------------------------------------------

def compile_signals(spec, data):
    """{"entries", "exits", "short_entries", "short_exits"}: boolean frames like close.

    Plus "entry_priority", a float frame like close, when a shared account ranks
    competing entries by a feature.

    `spec` must already have its parameters applied. All of them are on the bar the
    signal is computed on: the front door shifts every one of them to the next bar's
    open itself, so none is shifted here.
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
        rule = rules.get(slot)
        if rule:
            out[names[slot]] = evaluate_rule(rule, data, features, spec_features, fingerprint)
        else:
            out[names[slot]] = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 4. Order timing (intraday). Each signal fills at the NEXT bar's open, so:
    #    * good for the day: an entry whose fill would be on the next day lapses, so an
    #      entry on a day's last bar is dropped;
    #    * flat by the day's end: an exit fires on the day's second-to-last bar (it fills
    #      at the last bar's open), and entries that would fill on that last bar or later
    #      are dropped. Masks on the signal bar: nothing is shifted.
    #    ⚠️ An approximation of real order types, which vectorbtX does not have yet
    #    (docs/UPSTREAM.md issue 20): the exit fills at the last bar's OPEN, not the close.
    orders = (spec.get("risk") or {}).get("orders") or {}
    if orders.get("good_for_day") or orders.get("flat_by_day_end"):
        last, second_last = day_boundaries(close.index)
        blocked = last.copy()
        if orders.get("flat_by_day_end"):
            blocked = blocked | second_last
            closing = _mask_frame(second_last, close)
            out["exits"] = out["exits"] | closing
            out["short_exits"] = out["short_exits"] | closing
        allowed = _mask_frame(~blocked, close)
        out["entries"] = out["entries"] & allowed
        out["short_entries"] = out["short_entries"] & allowed

    # 5. The entry priority of a shared account, when it ranks competing entries by a
    #    feature: higher goes first, so "lowest first" is the negated score. A missing
    #    score (warm-up) goes last. Ties keep symbol order (vectorbtX's rule).
    priority = (spec.get("account") or {}).get("entry_priority") or {}
    if cash_model(spec) == "shared" and priority.get("by") == "feature":
        score = _full(resolve_ref(priority["score"], data, features, spec_features), close)
        out["entry_priority"] = score if priority.get("highest_first", True) else -score

    # ⚠️ No shift here, for any of them. The front door shifts entries, exits, short
    # entries and short exits to the next bar's open itself (vectorbtX 16293a2,
    # docs/UPSTREAM.md issue 1, which used to leave the shorts unshifted), and shifts a
    # priority score with them. Shifting here as well would delay every signal two bars.
    # test_engine.py::test_short_signal_fills_next_open checks the outcome.
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


def first_invested_bar(weights):
    """A schedule that is True only on the first bar with any weight: buy and hold.

    Not simply the first bar: inverse-volatility weights are empty while their window
    warms up, and buying then would hold cash for ever.
    """
    invested = weights.sum(axis=1) > 0
    schedule = pd.Series(False, index=weights.index)
    if invested.any():
        schedule.loc[invested.idxmax()] = True
    else:
        schedule.iloc[0] = True
    return schedule


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

    # 4. Schedule: never (buy and hold), every bar, or calendar dates optionally
    #    combined with a drift threshold.
    #    ⚠️ Always pass schedule= to drifted_weights, or drift accumulates from the
    #    first bar and the threshold test becomes always-true.
    schedule_spec = spec["schedule"]
    rebalance = schedule_spec["rebalance"]
    if rebalance == "never":
        schedule = first_invested_bar(weights)
    elif rebalance == "every_bar":
        schedule = engine.always_schedule(close.index)
    else:
        calendar = schedule_spec["calendar"]
        schedule = engine.calendar_schedule(close.index, freq=calendar["freq"], on=calendar.get("on", "first"))
        if schedule_spec.get("threshold") is not None:
            drifted = engine.drifted_weights(weights, close, schedule=schedule)
            breached = engine.threshold_schedule(drifted, weights, tolerance=float(schedule_spec["threshold"]))
            if schedule_spec.get("combine", "and") == "and":
                schedule = schedule & breached
            else:
                schedule = schedule | breached

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


def day_boundaries(index):
    dates = np.asarray(index.normalize())
    last = np.ones(len(dates), dtype=bool)
    last[:-1] = dates[1:] != dates[:-1]
    second_last = np.zeros(len(dates), dtype=bool)
    second_last[:-1] = last[1:] & ~last[:-1]
    return last, second_last
'''

_PRICE_VARS = {"open": "open_", "high": "high", "low": "low", "close": "close", "volume": "volume"}


def _python(value):
    """A literal for the script: numpy scalars become plain Python (np.int64(50) would print as such)."""
    return value.item() if isinstance(value, np.generic) else value


def _feature_code(spec_features):
    """The body of compute_features(open_, high, low, close, volume) -> dict of frames."""
    lines = ["P = {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}", "F = {}"]
    for name in feature_order(spec_features):
        feature = spec_features[name]
        item = indicators.entry(feature["indicator"])
        params = {key: _python(value) for key, value in indicators.call_params(feature).items()}
        inputs = []
        for input_name in item["inputs"]:
            ref = _input_ref(feature, input_name)
            if isinstance(ref, dict):
                inputs.append(f"F[{(ref['feature'] + '__' + _output_name(spec_features, ref))!r}]")
            else:
                inputs.append(_PRICE_VARS[ref])
        listed = "[" + ", ".join(inputs) + "]"
        if item["source"] in ("talib", "pandas_ta"):
            call = (f"run_library_indicator({item['source']!r}, {item['function']!r}, {item['inputs']!r}, {listed}, "
                    f"{params!r}, {item['outputs']!r})")
        elif item["source"] == "pattern":
            module, detector = item["function"].split(":")
            call = (f"{{'signal': run_pattern({module!r}, {detector!r}, {params!r}, open_, high, low, close, "
                    f"volume)}}")
        else:
            call = f"run_studio({item['function']!r}, {params!r}, {listed}, P)"
        lines.append(f"for _output, _frame in {call}.items():")
        lines.append(f"    F[{name!r} + '__' + _output] = _frame")
    lines.append("return F")
    return lines


def _ref_code(ref, spec_features, frames="P", feats="F"):
    if "feature" in ref:
        code = f"{feats}[{(ref['feature'] + '__' + _output_name(spec_features, ref))!r}]"
    elif "price" in ref:
        code = f"{frames}[{ref['price']!r}]"
    else:
        return repr(float(ref["const"]))
    shift = int(ref.get("shift", 0) or 0)
    return f"{code}.shift({shift})" if shift else code


def _condition_code(node, spec_features, frames="P", feats="F"):
    """Python source for one condition, mirroring evaluate_condition exactly."""
    close = f"{frames}['close']"
    if "pattern" in node:
        signal = f"{feats}[{(node['pattern'] + '__signal')!r}]"
        side = node.get("side", "either")
        return f"({signal} {'== 1' if side == 'bullish' else '== -1' if side == 'bearish' else '!= 0'})"
    if "filter" in node:
        kind = node["filter"]
        if kind == "session":
            return f"mask_frame(session_mask({close}.index, {node['from']!r}, {node['to']!r}), {close})"
        if kind == "weekday":
            return f"mask_frame({close}.index.day_of_week.isin({list(node['days'])!r}), {close})"
        if kind == "month":
            return f"mask_frame({close}.index.month.isin({list(node['months'])!r}), {close})"
        comparison = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars") if key in node}
        rule = RESAMPLE_RULE[node["timeframe"]]
        inner = _condition_code(comparison, spec_features, "PH", "FH")
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
    return f"as_bool({code})"


def _logic_code(tree, codes):
    """The logic line as Python operators on the conditions' masks, mirroring evaluate_logic."""
    if isinstance(tree, str):
        return codes[tree]
    if "not" in tree:
        return f"(~{_logic_code(tree['not'], codes)})"
    joiner = {"and": " & ", "nand": " & ", "or": " | ", "nor": " | ", "xor": " ^ ", "xnor": " ^ "}[tree["op"]]
    text = "(" + joiner.join(_logic_code(arg, codes) for arg in tree["args"]) + ")"
    return f"(~{text})" if tree["op"] in ("nand", "nor", "xnor") else text


def _rule_code(rule, spec_features):
    codes = {letter: _condition_code(condition, spec_features) for letter, condition in rule["conditions"].items()}
    return _logic_code(parse_logic(rule["logic"]), codes)


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
    if schedule["rebalance"] == "never":
        lines += ["# Buy and hold: the target weights are bought once, on the first bar with any weight.",
                  "invested = weights.sum(axis=1) > 0",
                  "schedule = pd.Series(False, index=weights.index)",
                  "schedule.loc[invested.idxmax() if invested.any() else weights.index[0]] = True"]
    elif schedule["rebalance"] == "every_bar":
        lines.append("schedule = vbt.x.always_schedule(close.index)")
    else:
        calendar = schedule["calendar"]
        lines.append(f"schedule = vbt.x.calendar_schedule(close.index, freq={calendar['freq']!r}, "
                     f"on={calendar.get('on', 'first')!r})")
        if schedule.get("threshold") is not None:
            lines.append("drifted = vbt.x.drifted_weights(weights, close, schedule=schedule)   # schedule= is essential")
            lines.append(f"breached = vbt.x.threshold_schedule(drifted, weights, tolerance={float(schedule['threshold'])!r})")
            lines.append(f"schedule = schedule {'&' if schedule.get('combine', 'and') == 'and' else '|'} breached")
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
        "Indicators come from TA-Lib and pandas-ta-classic, patterns from ta_patterns,",
        "called one symbol at a time exactly as the Studio calls them.",
        '"""',
        "",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import vectorbt                  # the .vbt pandas accessors (crossings)",
        "import vectorbtx as vbt          # lower-case import name; vbt.x holds the front doors",
        "",
        _SCRIPT_HELPERS.strip("\n"),
        "",
        "",
        "# --- The Studio's feature functions, copied from openstrategy_studio.indicators ---",
        indicators.export_source(),
        "",
        "",
        "def compute_features(open_, high, low, close, volume):",
    ]
    lines += ["    " + line for line in _feature_code(spec_features)]
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
        orders = (spec.get("risk") or {}).get("orders") or {}
        if orders.get("good_for_day") or orders.get("flat_by_day_end"):
            body += ["# Order timing, on the signal bar (see the Studio's compile_signals).",
                     "_last, _second_last = day_boundaries(close.index)",
                     "_blocked = _last.copy()"]
            if orders.get("flat_by_day_end"):
                body += ["_blocked = _blocked | _second_last",
                         "exits = exits | mask_frame(_second_last, close)",
                         "short_exits = short_exits | mask_frame(_second_last, close)"]
            body += ["entries = entries & mask_frame(~_blocked, close)",
                     "short_entries = short_entries & mask_frame(~_blocked, close)"]
        if plan["args"].get("entry_priority") == ("var", "entry_priority"):
            priority = account["entry_priority"]
            sign = "" if priority.get("highest_first", True) else "-"
            body.append("# One shared account: competing entries are filled highest score first.")
            body.append(f"entry_priority = {sign}full({_ref_code(priority['score'], spec_features)}, close)")
        body.append("# The front door shifts all four signal arrays (and a priority score) to the next bar's open itself.")
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
              f"    ADJUSTED = {engine.adjusted_symbols(spec)!r}   # symbols bought outright read adj_close",
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
