"""The strategy spec (version 2).

A spec is a plain dict, stored as one JSON file per strategy. Every editor in
the UI reads and writes it; the compiler, the optimiser and the code export
consume it. There is no other source of truth.

Its shape
    setup        {"kind": "signals" | "portfolio", "trading": "equity" | "cfd" | "spreadbet" | "fx_spot"}
                 What is being built, and how it is traded. One trading mode per
                 strategy (the owner's rule: mixing them gets confusing), so the
                 instruments carry no mode of their own.
    data         {"symbols", "timeframe", "start", "end"}
    instruments  {symbol: Instrument fields} or {"from_file": universe file}
    features     {name: {"indicator": catalogue id, "inputs": {input: ref}, "params": {...}}}
    rules        {slot: rule or None}, slot in RULE_SLOTS (signals only). A rule is
                 {"conditions": {"A": condition, "B": ...}, "logic": "A and (B or C)"},
                 or alternatives {"choose": label, "options": {label: rule}}.
    selection, weighting, schedule   (portfolio only)
    risk         sizing, stop, target, extra exits, order timing (signals only)
    account, engine_extra
    optimise     {"variables": {dotted path: {"start", "stop", "step"} | {"values": [...]}}}

Every value in the spec is a plain literal. The search ranges live only in
optimise.variables, keyed by the dotted path of the value they vary
(features.fast.params.timeperiod), so the compiler never sees a range.
apply_params(spec, params) writes a trial's values at those paths and resolves
every alternative to its chosen option.

validate_spec(spec) returns plain-English problems; an empty list means valid.
It is called when a spec is loaded and when Run is pressed, nowhere else.
"""

import copy
import hashlib
import json
import math
from importlib import resources
from pathlib import Path

import pandas as pd

SPEC_VERSION = 2

# Bar timeframes stored natively (plan section 8). Tick data is out of scope.
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
                     "1d": 1440, "1w": 10080, "1M": 43200}
# The vectorbt `freq` string for each timeframe. ⚠️ Always pass freq= to a
# backtest: without it annualised metrics are filtered out and fail confusingly.
# Weekly is "7D" because a fixed Timedelta is needed for annualisation.
FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
        "4h": "4h", "1d": "D", "1w": "7D"}
# pandas resample rules used when resampling UP (and by the higher-timeframe
# filter). Always used with label="left", closed="left".
RESAMPLE_RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
                 "4h": "4h", "1d": "1D", "1w": "W-SUN", "1M": "MS"}

KINDS = ["signals", "portfolio"]
# How a strategy is traded: vectorbtX's instrument modes, one per strategy.
TRADING = ["equity", "cfd", "spreadbet", "fx_spot"]
PRICE_FIELDS = ["open", "high", "low", "close", "volume"]
SIZING_KEYS = ["risk_pct", "risk_amount", "stake", "notional", "size"]
EXIT_KEYS = ["time_bars", "breakeven_points", "step_points"]     # the numba exit templates (exit_rules.py)
# Order timing for intraday strategies (docs/UPSTREAM.md issue 20 explains the approximations).
ORDER_KEYS = ["good_for_day", "flat_by_day_end"]
COMPARISON_OPS = [">", "<", ">=", "<=", "crosses_above", "crosses_below", "between", "rising", "falling"]
FILTER_KINDS = ["session", "weekday", "month", "higher_timeframe"]
PATTERN_SIDES = ["bullish", "bearish", "either"]
RULE_SLOTS = ["long_entry", "long_exit", "short_entry", "short_exit"]
WEIGHT_SCHEMES = ["equal", "inverse_vol", "price", "fixed"]
SELECTION_MODES = ["top", "bottom", "threshold"]
CALENDAR_FREQS = ["W", "M", "Q", "Y"]
# How often a portfolio goes back to its target weights. "never" is buy and hold: the
# weights are bought on the first bar and then left to drift.
REBALANCE = ["never", "every_bar", "calendar"]
# The account, for a rule strategy on several symbols (vectorbtX's cash=, entry_priority=,
# when_cash_short=; docs/UPSTREAM.md issue 13). "per_instrument" gives each symbol its own
# copy of init_cash (three symbols on 10,000 trade 30,000); "shared" is one account.
CASH_MODELS = ["per_instrument", "shared"]
# Who is filled first when entries compete for a shared account's cash.
PRIORITY_KINDS = ["symbols", "random", "feature"]
# What an entry the cash cannot cover does: not placed at all, or filled as far as the cash goes.
WHEN_CASH_SHORT = ["skip", "partial"]
# vectorbt passthroughs that would fight the account's cash model if set in engine_extra.
CASH_ENGINE_KEYS = ["cash", "cash_sharing", "group_by", "call_seq", "allow_partial", "entry_priority",
                    "when_cash_short"]


# ---------------------------------------------------------------------------
# Small accessors used everywhere
# ---------------------------------------------------------------------------

def is_choice(node):
    """True for structural alternatives {"choose": label, "options": {label: rule, ...}}.

    The optimiser can choose between whole rules (structure search, plan phase 3) when
    the path of "choose" is one of the optimise variables.
    """
    return isinstance(node, dict) and "choose" in node and "options" in node


def resolve_choices(node):
    """Replace every choice by its selected option, all the way down."""
    if is_choice(node):
        return resolve_choices(node["options"][node["choose"]])
    if isinstance(node, dict):
        return {key: resolve_choices(value) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_choices(value) for value in node]
    return node


def collect_choices(spec):
    """Every choice node in the spec, as {dotted path: node}."""
    found = {}
    stack = [("", spec)]
    while stack:
        prefix, node = stack.pop()
        if is_choice(node):
            found[prefix] = node
        if isinstance(node, dict):
            children = node.items()
        elif isinstance(node, list):
            children = enumerate(node)
        else:
            continue
        for key, child in children:
            if key != "optimise":
                stack.append((f"{prefix}.{key}" if prefix else str(key), child))
    return dict(sorted(found.items()))


def _split(path):
    # "rules.long_entry.conditions.A.right.const" -> ["rules", "long_entry", "conditions", "A", "right", "const"]
    return [int(part) if part.isdigit() else part for part in path.split(".")]


def get_path(spec, path):
    node = spec
    for part in _split(path):
        node = node[part]
    return node


def has_path(spec, path):
    try:
        get_path(spec, path)
    except (KeyError, IndexError, TypeError):
        return False
    return True


def set_path(spec, path, value):
    """Write one value into the spec in place. Every UI widget writes through this."""
    parts = _split(path)
    node = spec
    for part in parts[:-1]:
        if isinstance(node, dict) and part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def canonical_json(obj):
    # Sorted keys and fixed separators: the same dict always gives the same text,
    # which is what the spec hash and the feature cache keys rely on.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec):
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:16]


def strategy_kind(spec):
    """"signals" (entry and exit rules) or "portfolio" (weights and a rebalance schedule)."""
    return (spec.get("setup") or {}).get("kind", "signals")


def trading_mode(spec):
    """How the strategy is traded: one of TRADING, for every instrument."""
    return (spec.get("setup") or {}).get("trading", "equity")


def cash_model(spec):
    """How a rule strategy on several symbols is funded: "per_instrument" or "shared".

    None when the question does not arise, and then nothing about cash is passed to
    vectorbtX: one symbol is one account either way, and a portfolio's weights are
    shares of one account by construction.
    """
    if strategy_kind(spec) != "signals" or len(spec["data"]["symbols"]) < 2:
        return None
    return (spec.get("account") or {}).get("cash") or "per_instrument"


def is_intraday(spec):
    return TIMEFRAME_MINUTES.get((spec.get("data") or {}).get("timeframe"), 0) < 1440


# ---------------------------------------------------------------------------
# Loading, saving and templates
# ---------------------------------------------------------------------------

def load_spec(path):
    """Read a spec from JSON. The caller runs validate_spec and shows the problems."""
    with open(Path(path), encoding="utf-8") as handle:
        return json.load(handle)


def save_spec(spec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 keeps the file readable and git-friendly.
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def list_templates():
    """Starter strategies shipped as package data: {file stem: display name}."""
    found = {}
    folder = resources.files("openstrategy_studio") / "templates"
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if entry.name.endswith(".json"):
            found[entry.name[:-5]] = json.loads(entry.read_text(encoding="utf-8"))["name"]
    return found


def load_template(name):
    entry = resources.files("openstrategy_studio") / "templates" / f"{name}.json"
    return json.loads(entry.read_text(encoding="utf-8"))


def default_spec():
    # The MA-cross template is the golden strategy (plan section 12, test 1).
    return load_template("ma_cross")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def ref_feature(ref):
    """The feature a reference reads, or None for a price or a number."""
    return ref["feature"] if isinstance(ref, dict) and "feature" in ref else None


def feature_dependencies(features):
    """{feature: set of features its inputs read}."""
    deps = {}
    for name, feature in features.items():
        deps[name] = {ref_feature(ref) for ref in (feature.get("inputs") or {}).values() if ref_feature(ref)}
    return deps


def feature_order(features):
    """Feature names in dependency order (Kahn's algorithm).

    If there is a cycle the returned list is shorter than `features`;
    validate_spec reports that, so the compiler can assume a full order.
    """
    deps = feature_dependencies(features)
    order = []
    ready = sorted(name for name, needs in deps.items() if not needs)
    remaining = {name: set(needs) for name, needs in deps.items() if needs}
    while ready:
        name = ready.pop(0)
        order.append(name)
        for other in sorted(remaining):
            remaining[other].discard(name)
            if not remaining[other]:
                del remaining[other]
                ready.append(other)
    return order


def feature_outputs(features, name):
    """The outputs of a feature, from the catalogue (empty for an unknown indicator)."""
    from .indicators import catalogue
    item = catalogue().get((features.get(name) or {}).get("indicator"))
    return list(item["outputs"]) if item else []


# ---------------------------------------------------------------------------
# Optimisation: the values that can vary, and applying a trial's values
# ---------------------------------------------------------------------------

SLOT_LABELS = {"long_entry": "Long entry", "long_exit": "Long exit", "short_entry": "Short entry",
               "short_exit": "Short exit"}


def _rule_inputs(out, prefix, rule, where):
    """The numbers inside one rule's conditions: constants, bars, and bars ago."""
    for letter, condition in (rule.get("conditions") or {}).items():
        base = f"{prefix}.conditions.{letter}"
        for side in ("left", "right", "lower", "upper"):
            ref = condition.get(side)
            if isinstance(ref, dict) and "const" in ref:
                out.append({"path": f"{base}.{side}.const", "group": "Rules", "type": "float", "value": ref["const"],
                            "label": f"{where} › {letter} › the number ({side})"})
            if isinstance(ref, dict) and ref.get("shift"):
                out.append({"path": f"{base}.{side}.shift", "group": "Rules", "type": "int", "value": ref["shift"],
                            "label": f"{where} › {letter} › bars ago ({side})"})
        if "bars" in condition:
            out.append({"path": f"{base}.bars", "group": "Rules", "type": "int", "value": condition["bars"],
                        "label": f"{where} › {letter} › bars"})


def tunable_inputs(spec):
    """Every value the optimiser can vary, in the order the Optimise page lists them.

    [{"path", "group", "label", "type": "int" | "float" | "choice", "value", "options"?}].
    Risk dials (stop, target, size, time stop, breakeven, trailing step) are here
    beside the features' parameters and the rules' numbers.
    """
    from .indicators import catalogue

    out = []
    for name, feature in (spec.get("features") or {}).items():
        item = catalogue().get(feature.get("indicator"))
        if item is None:
            continue
        for param, declaration in item["params"].items():
            kind = declaration["type"]
            if kind not in ("int", "float", "choice", "bool"):
                continue
            value = (feature.get("params") or {}).get(param, declaration.get("default"))
            row = {"path": f"features.{name}.params.{param}", "group": "Features", "value": value,
                   "label": f"{name} › {param}", "type": "choice" if kind in ("choice", "bool") else kind}
            if kind == "choice":
                row["options"] = list(declaration["options"])
            elif kind == "bool":
                row["options"] = [True, False]
            out.append(row)
    rules = spec.get("rules") or {}
    for slot in RULE_SLOTS:
        node = rules.get(slot)
        if is_choice(node):
            out.append({"path": f"rules.{slot}.choose", "group": "Rules", "type": "choice", "value": node["choose"],
                        "options": list(node["options"]), "label": f"{SLOT_LABELS[slot]} › which version"})
            for option, rule in node["options"].items():
                _rule_inputs(out, f"rules.{slot}.options.{option}", rule, f"{SLOT_LABELS[slot]} ({option})")
        elif node:
            _rule_inputs(out, f"rules.{slot}", node, SLOT_LABELS[slot])
    risk = spec.get("risk") or {}
    for key, value in (risk.get("sizing") or {}).items():
        out.append({"path": f"risk.sizing.{key}", "group": "Risk", "type": "float", "value": value,
                    "label": f"Position size ({key})"})
    if (risk.get("stop") or {}).get("kind") in ("fixed", "trailing"):
        out.append({"path": "risk.stop.points", "group": "Risk", "type": "float", "value": risk["stop"]["points"],
                    "label": "Stop distance (points)"})
    if risk.get("target"):
        out.append({"path": "risk.target.points", "group": "Risk", "type": "float", "value": risk["target"]["points"],
                    "label": "Take profit (points)"})
    labels = {"time_bars": "Time stop (bars held)", "breakeven_points": "Breakeven trigger (points)",
              "step_points": "Trailing step (points)"}
    for key in EXIT_KEYS:
        value = (risk.get("exits") or {}).get(key)
        if value is not None:
            out.append({"path": f"risk.exits.{key}", "group": "Risk", "type": "int" if key == "time_bars" else "float",
                        "value": value, "label": labels[key]})
    selection = spec.get("selection") or {}
    if selection.get("top_n") is not None and selection.get("mode", "top") in ("top", "bottom"):
        out.append({"path": "selection.top_n", "group": "Portfolio", "type": "int", "value": selection["top_n"],
                    "label": "Selection: how many"})
    if selection.get("min_score") is not None:
        out.append({"path": "selection.min_score", "group": "Portfolio", "type": "float",
                    "value": selection["min_score"], "label": "Selection: minimum score"})
    weighting = spec.get("weighting") or {}
    if weighting:
        out.append({"path": "weighting.scheme", "group": "Portfolio", "type": "choice", "value": weighting["scheme"],
                    "options": [scheme for scheme in WEIGHT_SCHEMES if scheme != "fixed"], "label": "Weighting"})
    if weighting.get("scheme") == "inverse_vol":
        out.append({"path": "weighting.vol_window", "group": "Portfolio", "type": "int",
                    "value": weighting.get("vol_window", 20), "label": "Volatility window (bars)"})
    schedule = spec.get("schedule") or {}
    if schedule.get("threshold") is not None:
        out.append({"path": "schedule.threshold", "group": "Portfolio", "type": "float",
                    "value": schedule["threshold"], "label": "Drift threshold"})
    return out


def variables(spec):
    """The optimise variables: {dotted path: {"start", "stop", "step"} | {"values": [...]}}."""
    return dict(((spec.get("optimise") or {}).get("variables")) or {})


def variable_values(variable, kind):
    """The values one variable takes on a grid: its list, or start to stop by step."""
    if "values" in variable:
        return list(variable["values"])
    start, stop = variable["start"], variable["stop"]
    step = variable.get("step") or (1 if kind == "int" else (stop - start) / 9)
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = [start + index * step for index in range(max(count, 1))]
    return [int(round(value)) for value in values] if kind == "int" else [float(value) for value in values]


def apply_params(spec, params=None):
    """Deep copy of the spec with a trial's values written in and every alternative resolved.

    `params` maps dotted paths to values. A path the spec cannot vary means the spec
    changed under a stored optimisation: it fails loudly rather than being ignored.
    """
    params = params or {}
    known = {row["path"] for row in tunable_inputs(spec)}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"These values cannot be varied in this strategy: {', '.join(unknown)}")
    out = copy.deepcopy(spec)
    for path, value in params.items():
        set_path(out, path, value)
    out.pop("optimise", None)          # the compiler never sees the ranges
    # Choices last: a trial's values may sit inside any option, chosen or not.
    return resolve_choices(out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_number(problems, where, value, minimum=None, integer=False, allow_zero=True):
    """Check a numeric value. Returns it, or None when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        problems.append(f"{where} must be a number.")
        return None
    if integer and int(value) != value:
        problems.append(f"{where} must be a whole number.")
    if minimum is not None and (value < minimum or (value == minimum and not allow_zero)):
        problems.append(f"{where} must be {'at least' if allow_zero else 'above'} {minimum}.")
    return value


def _check_ref(problems, where, ref, features, allow_const=True):
    if not isinstance(ref, dict):
        problems.append(f"{where}: expected a feature, a price or a number.")
        return
    if "feature" in ref:
        name = ref["feature"]
        if name not in features:
            problems.append(f"{where} refers to a feature called '{name}' that does not exist.")
            return
        outputs = feature_outputs(features, name)
        output = ref.get("output")
        if outputs and output is not None and output not in outputs:
            problems.append(f"{where}: feature '{name}' has no output '{output}' (it has {', '.join(outputs)}).")
    elif "price" in ref:
        if ref["price"] not in PRICE_FIELDS:
            problems.append(f"{where}: '{ref['price']}' is not a price field.")
    elif "const" in ref and allow_const:
        _check_number(problems, f"{where} number", ref["const"])
    else:
        problems.append(f"{where}: expected a feature, a price{' or a number' if allow_const else ''}.")
    if "shift" in ref:
        # A reference's own shift reads history (the channel high of the bar before).
        _check_number(problems, f"{where} bars ago", ref["shift"], minimum=0, integer=True)


def _check_condition(problems, where, node, features, timeframe, entry):
    if not isinstance(node, dict):
        problems.append(f"{where}: a condition must be an object.")
        return
    if "pattern" in node:
        name = node["pattern"]
        feature = features.get(name)
        if feature is None:
            problems.append(f"{where} uses a pattern feature called '{name}' that does not exist.")
        elif not str(feature.get("indicator", "")).startswith("pattern:"):
            problems.append(f"{where}: '{name}' is not a pattern; compare it with a condition instead.")
        if node.get("side", "either") not in PATTERN_SIDES:
            problems.append(f"{where}: the pattern side must be bullish, bearish or either.")
        return
    if "filter" in node:
        kind = node["filter"]
        # ⚠️ Filters gate entries only. Gating an exit would trap a position open.
        if not entry:
            problems.append(f"{where}: filters can only be used in entry rules; filtering an exit would trap "
                            "positions open.")
        if kind not in FILTER_KINDS:
            problems.append(f"{where}: unknown filter '{kind}'.")
        elif kind == "session":
            if TIMEFRAME_MINUTES.get(timeframe, 0) >= 1440:
                problems.append(f"{where}: a session filter means nothing on daily or weekly bars.")
            for key in ("from", "to"):
                try:
                    pd.Timestamp(f"2000-01-01 {node.get(key, '')}")
                except ValueError:
                    problems.append(f"{where}: session '{key}' must be a time such as 07:00.")
        elif kind == "weekday":
            days = node.get("days") or []
            if not days or any(day not in range(7) for day in days):
                problems.append(f"{where}: pick weekdays as numbers 0 (Monday) to 6 (Sunday).")
        elif kind == "month":
            months = node.get("months") or []
            if not months or any(month not in range(1, 13) for month in months):
                problems.append(f"{where}: pick months as numbers 1 to 12.")
        elif kind == "higher_timeframe":
            higher = node.get("timeframe")
            if TIMEFRAME_MINUTES.get(higher, 0) <= TIMEFRAME_MINUTES.get(timeframe, 0):
                problems.append(f"{where}: the higher timeframe must be above the data's {timeframe}.")
            comparison = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars") if key in node}
            _check_condition(problems, f"{where} condition", comparison, features, timeframe, entry)
        return
    # A comparison.
    op = node.get("op")
    if op not in COMPARISON_OPS:
        problems.append(f"{where}: unknown comparison '{op}'.")
        return
    _check_ref(problems, f"{where} left side", node.get("left"), features)
    if op == "between":
        _check_ref(problems, f"{where} lower bound", node.get("lower"), features)
        _check_ref(problems, f"{where} upper bound", node.get("upper"), features)
    elif op in ("rising", "falling"):
        _check_number(problems, f"{where} bars", node.get("bars", 1), minimum=1, integer=True)
    else:
        _check_ref(problems, f"{where} right side", node.get("right"), features)


def _check_rule(problems, where, rule, features, timeframe, entry):
    """One rule: lettered conditions and a logic line that uses every one of them."""
    from .expressions import logic_letters, parse_logic

    if not isinstance(rule, dict) or not isinstance(rule.get("conditions"), dict):
        problems.append(f"{where}: a rule needs conditions and a logic line.")
        return
    conditions = rule["conditions"]
    if not conditions:
        problems.append(f"{where}: add at least one condition.")
        return
    for letter, condition in conditions.items():
        if not (letter.isalpha() and letter.isupper()):
            problems.append(f"{where}: name conditions with capital letters (A, B ... AA), not '{letter}'.")
        _check_condition(problems, f"{where} › {letter}", condition, features, timeframe, entry)
    try:
        tree = parse_logic(rule.get("logic", ""))
    except ValueError as error:
        problems.append(f"{where} logic: {error}")
        return
    used = logic_letters(tree)
    missing = sorted(used - set(conditions))
    unused = sorted(set(conditions) - used)
    if missing:
        problems.append(f"{where} logic mentions {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} "
                        "not a condition.")
    if unused:
        problems.append(f"{where}: condition {', '.join(unused)} {'is' if len(unused) == 1 else 'are'} not in the "
                        "logic line; add it there or delete it.")


def _check_features(problems, features):
    from .indicators import catalogue

    for name, feature in features.items():
        where = f"Feature '{name}'"
        if not name.isidentifier() or name in PRICE_FIELDS:
            problems.append(f"{where}: use letters, digits and underscores, starting with a letter, and not a price "
                            "name such as close.")
        item = catalogue().get((feature or {}).get("indicator"))
        if item is None:
            problems.append(f"{where} uses an unknown indicator '{(feature or {}).get('indicator')}'.")
            continue
        for key, value in (feature.get("params") or {}).items():
            declaration = item["params"].get(key)
            if declaration is None:
                problems.append(f"{where}: '{key}' is not a setting of {item['label']}.")
                continue
            kind = declaration["type"]
            if value is None:
                continue                      # blank: the library's own default
            if kind in ("int", "float"):
                _check_number(problems, f"{where} {key}", value, integer=kind == "int")
            elif kind == "choice" and value not in declaration["options"]:
                problems.append(f"{where} {key} must be one of {', '.join(map(str, declaration['options']))}.")
            elif kind == "bool" and not isinstance(value, bool):
                problems.append(f"{where} {key} must be true or false.")
        for key, ref in (feature.get("inputs") or {}).items():
            if key not in item["inputs"]:
                problems.append(f"{where}: '{key}' is not an input of {item['label']}.")
            elif isinstance(ref, dict):
                _check_ref(problems, f"{where} input {key}", ref, features, allow_const=False)
            elif ref not in PRICE_FIELDS:
                problems.append(f"{where} input {key} must be a price field or another feature.")
        for key in item["inputs"]:
            if key not in (feature.get("inputs") or {}) and item["defaults"].get(key) is None:
                problems.append(f"{where}: choose what its '{key}' input reads.")
    if len(feature_order(features)) < len(features):
        problems.append("Some features read each other in a loop; break the cycle.")


def _check_risk(problems, spec, kind):
    risk = spec.get("risk") or {}
    sizing = risk.get("sizing") or {}
    used = [key for key in SIZING_KEYS if key in sizing]
    stop = risk.get("stop") or {"kind": "none"}
    stop_kind = stop.get("kind", "none")
    exits = risk.get("exits") or {}
    used_exits = [key for key in EXIT_KEYS if exits.get(key) is not None]
    orders = risk.get("orders") or {}
    if kind == "portfolio":
        if used or stop_kind != "none" or risk.get("target") or used_exits or any(orders.values()):
            problems.append("A portfolio is sized by its weights; remove the sizing, stops, target and order settings.")
        return
    # ⚠️ vectorbtX raises BacktestError for two sizing arguments: catch it here first.
    if len(used) > 1:
        problems.append(f"Choose one way to size positions, not {len(used)} ({', '.join(used)}).")
    for key in used:
        _check_number(problems, f"Sizing {key}", sizing[key], minimum=0, allow_zero=False)
    extra = sorted(set(sizing) - set(SIZING_KEYS))
    if extra:
        problems.append(f"Unknown sizing setting: {', '.join(extra)}.")
    if stop_kind not in ("none", "fixed", "trailing"):
        problems.append("The stop must be none, fixed or trailing.")
    if stop_kind in ("fixed", "trailing"):
        _check_number(problems, "Stop distance (points)", stop.get("points"), minimum=0, allow_zero=False)
        if "trailing_points" in stop:
            # ⚠️ trail_points overwrites stop_points in vectorbtX: they cannot combine.
            problems.append("A fixed stop and a trailing stop cannot be used together; pick one.")
    if used and used[0] in ("risk_pct", "risk_amount") and stop_kind != "fixed":
        problems.append(f"{used[0]} sizes from the stop distance, so it needs a fixed stop.")
    target = risk.get("target")
    if target:
        _check_number(problems, "Take profit (points)", target.get("points"), minimum=0, allow_zero=False)
    unknown = sorted(set(exits) - set(EXIT_KEYS))
    if unknown:
        problems.append(f"Unknown extra exit: {', '.join(unknown)}.")
    labels = {"time_bars": "Time stop (bars)", "breakeven_points": "Breakeven trigger (points)",
              "step_points": "Step size (points)"}
    for key in used_exits:
        _check_number(problems, labels[key], exits[key], minimum=1 if key == "time_bars" else 0,
                      integer=key == "time_bars", allow_zero=False)
    if used_exits and stop_kind != "fixed":
        # The templates adjust an existing stop: vectorbt only calls them while a stop is active.
        problems.append("The extra exits (time stop, breakeven, step trailing) adjust a stop, so they need a fixed stop.")
    unknown = sorted(set(orders) - set(ORDER_KEYS))
    if unknown:
        problems.append(f"Unknown order setting: {', '.join(unknown)}.")
    if any(orders.get(key) for key in ORDER_KEYS) and not is_intraday(spec):
        problems.append("Good-for-the-day entries and closing before the end of the day need intraday bars "
                        "(under one day).")


def _check_portfolio(problems, spec, features):
    symbols = spec["data"]["symbols"]
    selection = spec.get("selection")
    weighting = spec.get("weighting")
    schedule = spec.get("schedule")
    if weighting is None or schedule is None:
        problems.append("A portfolio needs a weighting and a rebalance schedule.")
        return
    if selection is not None:
        _check_ref(problems, "Selection score", selection.get("score"), features)
        mode = selection.get("mode", "top")
        if mode not in SELECTION_MODES:
            problems.append(f"Selection mode must be one of {', '.join(SELECTION_MODES)}.")
        if mode in ("top", "bottom"):
            count = _check_number(problems, "Selection count", selection.get("top_n"), minimum=1, integer=True)
            if count is not None and count > len(symbols):
                problems.append(f"Selection keeps {count} assets but the universe has only {len(symbols)}.")
        if selection.get("min_score") is not None:
            _check_number(problems, "Selection minimum score", selection["min_score"])
    scheme = weighting.get("scheme")
    if scheme not in WEIGHT_SCHEMES:
        problems.append(f"Weighting must be one of {', '.join(WEIGHT_SCHEMES)}.")
    if scheme == "inverse_vol" or "vol_window" in weighting:
        _check_number(problems, "Volatility window", weighting.get("vol_window", 20), minimum=2, integer=True)
    if scheme == "fixed":
        table = weighting.get("table") or {}
        missing = [symbol for symbol in table if symbol not in symbols]
        if missing:
            problems.append(f"Fixed weights name symbols outside the universe: {', '.join(missing)}.")
        if any(value < 0 for value in table.values()):
            problems.append("Fixed weights must not be negative.")
        # ⚠️ Gross weights above 1 behave inconsistently in vectorbtX (plan section 3,
        # item 7, a TEST item), so leverage stays off the portfolio path for now.
        if sum(table.values()) > 1 + 1e-9:
            problems.append("Fixed weights add up to more than 100%; leverage is not supported on portfolios yet.")
    rebalance = schedule.get("rebalance")
    if rebalance not in REBALANCE:
        problems.append("Rebalance never (buy and hold), on every bar, or on a calendar.")
    calendar = schedule.get("calendar")
    if rebalance == "calendar":
        if not calendar:
            problems.append("A calendar rebalance needs its frequency.")
        else:
            if calendar.get("freq") not in CALENDAR_FREQS:
                problems.append(f"Rebalance frequency must be one of {', '.join(CALENDAR_FREQS)}.")
            if calendar.get("on", "first") not in ("first", "last"):
                problems.append("Rebalance day must be the first or the last of the period.")
    if schedule.get("threshold") is not None:
        _check_number(problems, "Drift threshold", schedule["threshold"], minimum=0, allow_zero=False)
        if rebalance != "calendar":
            # Drift is measured from the last calendar rebalance; without an anchor it
            # grows from the first bar and the threshold test is always true.
            problems.append("A drift threshold needs a calendar schedule to measure drift from.")
    if schedule.get("combine", "and") not in ("and", "or"):
        problems.append("Combine the calendar and the threshold with 'and' or 'or'.")
    _check_number(problems, "Minimum trade", schedule.get("min_trade", 0.0), minimum=0)


def _check_cash(problems, spec, kind, features):
    """The account's cash model: an account per symbol, or one shared account with an entry priority."""
    account = spec.get("account") or {}
    cash = account.get("cash")
    priority = account.get("entry_priority")
    short = account.get("when_cash_short")
    if cash is not None and cash not in CASH_MODELS:
        problems.append("Cash must be per_instrument (each symbol trades its own copy of the account) or shared "
                        "(one account for every symbol).")
    if short is not None and short not in WHEN_CASH_SHORT:
        problems.append("When cash runs short, choose skip (the entry is not placed) or partial (it is filled as far "
                        "as the cash goes).")
    if kind == "portfolio" and cash == "per_instrument":
        problems.append("A portfolio's weights are shares of one account, so each symbol cannot have its own cash. "
                        "Make the account shared, or remove the cash setting.")
    if kind == "portfolio" and (priority or short):
        problems.append("The entry priority and the cash-short rule choose between competing entry signals, so only "
                        "entry and exit rule strategies have them. Remove them.")
    elif kind == "signals" and cash != "shared" and (priority or short):
        problems.append("The entry priority and the cash-short rule only apply to a shared account. Make the account "
                        "shared, or remove them.")
    if priority:
        by = priority.get("by")
        symbols = (spec.get("data") or {}).get("symbols") or []
        if by not in PRIORITY_KINDS:
            problems.append(f"The entry priority must be by {', '.join(PRIORITY_KINDS)}.")
        elif by == "symbols":
            order = priority.get("order") or []
            unknown = [symbol for symbol in order if symbol not in symbols]
            if unknown:
                problems.append(f"The entry priority names symbols that are not in the list: {', '.join(unknown)}.")
            if len(set(order)) != len(order):
                problems.append("The entry priority lists a symbol twice.")
        elif by == "random":
            # ⚠️ Unseeded, every run (and every optimisation trial) would draw a different
            # order, so two runs of the same spec would disagree and trials would not compare.
            if (spec.get("engine_extra") or {}).get("seed") is None:
                problems.append("A random entry priority needs a random seed, so every run draws the same order. "
                                "Set one on the Account step.")
        else:
            score = priority.get("score")
            if isinstance(score, dict) and "const" in score:
                problems.append("Rank competing entries by a feature or a price: a number ranks them all the same.")
            else:
                _check_ref(problems, "Entry priority score", score, features)
            if not isinstance(priority.get("highest_first", True), bool):
                problems.append("The entry priority must say whether the highest or the lowest score goes first.")
    if kind == "signals":
        clashing = sorted(set(spec.get("engine_extra") or {}) & set(CASH_ENGINE_KEYS))
        if clashing:
            problems.append("Set how the account shares cash on the Account step, not as engine settings: remove "
                            f"{', '.join(clashing)} from engine_extra.")


def _check_choices(problems, spec):
    for path, node in collect_choices(spec).items():
        options = node["options"]
        where = f"The versions at {path}"
        if not isinstance(options, dict) or not options:
            problems.append(f"{where} need at least one option.")
            continue
        if node["choose"] not in options:
            problems.append(f"{where}: the chosen version '{node['choose']}' does not exist.")


def _check_variables(problems, spec):
    """The optimise variables: each must name a value the strategy has, with a usable range or list."""
    rows = {row["path"]: row for row in tunable_inputs(spec)}
    for path, variable in variables(spec).items():
        row = rows.get(path)
        where = f"Optimise variable {path}"
        if row is None:
            problems.append(f"{where} no longer exists in the strategy; remove it on the Optimise page.")
            continue
        if not isinstance(variable, dict):
            problems.append(f"{where}: give a range or a list of values.")
            continue
        if "values" in variable:
            values = variable["values"]
            if not isinstance(values, list) or not values:
                problems.append(f"{where}: list at least one value.")
            elif row["type"] == "choice" and any(value not in row["options"] for value in values):
                problems.append(f"{where}: every value must be one of {', '.join(map(str, row['options']))}.")
            elif row["type"] in ("int", "float"):
                for value in values:
                    _check_number(problems, f"{where} value", value, integer=row["type"] == "int")
            continue
        if row["type"] == "choice":
            problems.append(f"{where}: a choice is optimised over a list of values, not a range.")
            continue
        start = _check_number(problems, f"{where} start", variable.get("start"), integer=row["type"] == "int")
        stop = _check_number(problems, f"{where} stop", variable.get("stop"), integer=row["type"] == "int")
        if start is not None and stop is not None and not start < stop:
            problems.append(f"{where}: the start must be below the stop.")
        if variable.get("step") is not None:
            _check_number(problems, f"{where} step", variable["step"], minimum=0, allow_zero=False,
                          integer=row["type"] == "int")


def validate_spec(spec):
    """Every problem with the spec, in plain English. An empty list means valid.

    With alternatives (choices), the spec is checked as it stands now and then
    once with each alternative selected in turn, so a version the optimiser may
    pick is checked before a job spends hours finding it broken.
    """
    if not isinstance(spec, dict):
        return ["The strategy file must hold a JSON object."]
    problems = []
    if spec.get("version") != SPEC_VERSION:
        return [f"This file is strategy version {spec.get('version')}; the Studio reads version {SPEC_VERSION}."]
    _check_choices(problems, spec)
    if problems:
        return problems
    _check_variables(problems, spec)
    problems += _validate_resolved(resolve_choices({key: value for key, value in spec.items() if key != "optimise"}))
    for path, node in collect_choices(spec).items():
        current = node["choose"]
        for label in node["options"]:
            if label == current:
                continue
            variant = copy.deepcopy(spec)
            variant.pop("optimise", None)
            set_path(variant, f"{path}.choose", label)
            for problem in _validate_resolved(resolve_choices(variant)):
                text = f"With version '{label}' at {path}: {problem}"
                if problem not in problems and text not in problems:
                    problems.append(text)
    return problems


def _validate_resolved(spec):
    """validate_spec for a spec with no choices left in it."""
    problems = []
    if not str(spec.get("name", "")).strip():
        problems.append("Give the strategy a name.")
    setup = spec.get("setup") or {}
    kind = setup.get("kind")
    if kind not in KINDS:
        problems.append("Choose what you are building on the Setup step: trading on signals, or holding a portfolio.")
    if setup.get("trading") not in TRADING:
        problems.append("Choose how you are trading it on the Setup step: buying outright, CFDs, spread bets or FX.")
    trading = setup.get("trading")

    data = spec.get("data") or {}
    symbols = data.get("symbols") or []
    if not symbols:
        problems.append("Pick at least one symbol.")
    if len(set(symbols)) != len(symbols):
        problems.append("A symbol is listed twice.")
    timeframe = data.get("timeframe")
    if timeframe not in TIMEFRAMES:
        problems.append(f"Timeframe must be one of {', '.join(TIMEFRAMES)}.")
    try:
        if pd.Timestamp(data.get("start")) >= pd.Timestamp(data.get("end")):
            problems.append("The start date must be before the end date.")
    except (TypeError, ValueError):
        problems.append("Start and end must be dates such as 2020-01-31.")

    instruments = spec.get("instruments") or {}
    if "from_file" not in instruments:
        for symbol in symbols:
            fields = instruments.get(symbol)
            if fields is None:
                problems.append(f"'{symbol}' has no instrument settings.")
                continue
            if "mode" in fields:
                problems.append(f"'{symbol}' names its own mode; the trading mode is set once, on the Setup step.")
            if (fields.get("asset_class") == "fx" or trading == "fx_spot") \
                    and not (fields.get("base_currency") and fields.get("quote_currency")):
                # ⚠️ An FX position is long one currency and short the other. With a leg
                # missing, vectorbtX skips the carry with only a warning (docs/UPSTREAM.md
                # issue 3), so the run quietly shows no holding costs at all.
                problems.append(f"'{symbol}' is an FX instrument, so name both of its currencies: base_currency and "
                                "quote_currency (EUR and USD for EURUSD). Carry needs both.")
            rate = fields.get("borrow_rate")
            if rate is not None:
                # The borrow fee follows how a thing is traded (vectorbtX 9fa427e): shorting
                # what you trade directly borrows it; a CFD or spread bet short pays its
                # funding instead. vectorbtX rejects the rest at construction; say it here.
                if trading != "equity":
                    problems.append(f"'{symbol}' sets borrow_rate, which applies only when buying outright: a CFD or "
                                    "spread bet short borrows nothing, and its funding is its cost. Clear borrow_rate.")
                elif isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
                    problems.append(f"'{symbol}' borrow_rate must be a yearly rate of 0 or more "
                                    "(0.005 is 0.5% a year).")
        for symbol in instruments:
            if symbol not in symbols:
                problems.append(f"Instrument settings for '{symbol}', which is not in the symbol list.")

    features = spec.get("features") or {}
    _check_features(problems, features)

    if kind == "signals":
        rules = spec.get("rules") or {}
        if not rules.get("long_entry") and not rules.get("short_entry"):
            problems.append("Add a long or a short entry rule.")
        extra = sorted(set(rules) - set(RULE_SLOTS))
        if extra:
            problems.append(f"Unknown rule slot: {', '.join(extra)}.")
        for slot in RULE_SLOTS:
            if rules.get(slot):
                _check_rule(problems, SLOT_LABELS[slot], rules[slot], features, timeframe,
                            entry=slot.endswith("entry"))
        if any(key in spec for key in ("selection", "weighting", "schedule")):
            problems.append("A signal strategy has no portfolio settings; remove selection, weighting and schedule.")
    if kind == "portfolio":
        if any((spec.get("rules") or {}).get(slot) for slot in RULE_SLOTS):
            problems.append("A portfolio has no entry or exit rules; remove them.")
        if symbols:
            _check_portfolio(problems, spec, features)
    if kind in KINDS:
        _check_risk(problems, spec, kind)

    account = spec.get("account") or {}
    _check_number(problems, "Starting cash", account.get("init_cash"), minimum=0, allow_zero=False)
    if not account.get("freq"):
        problems.append("Set the bar frequency (freq); annualised metrics need it.")
    fx = account.get("fx")
    if fx and not fx.get("symbol"):
        problems.append("Name the stored FX dataset to convert with (for example GBPUSD=X).")
    if fx and not account.get("base_currency"):
        problems.append("Converting with FX needs a reporting currency.")
    rates = account.get("rates")
    if rates and rates.get("source") not in ("fred", "static"):
        problems.append("Rates come from FRED or a static table.")
    if rates and rates.get("source") == "static" and not rates.get("values"):
        problems.append("A static rate table needs at least one currency and rate.")
    _check_cash(problems, spec, kind, features)
    return problems
