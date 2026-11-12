"""The strategy spec.

A spec is a plain dict, stored as one JSON file per strategy. Every editor in
the UI reads and writes it; the compiler, the optimiser and the code export
consume it. There is no other source of truth.

Conventions (plan section 5):

* A numeric or categorical parameter is either a literal (``50``) or an
  optimisable node ``{"value": 50, "opt": {...}}``. ``opt`` uses BUTChC's
  search-space dialect verbatim: ``{"min", "max"}`` plus ``"int": true`` or
  ``"log": true``, or ``{"values": [...]}``. The Studio adds one key of its own,
  ``"step"``, used only by the grid backend and stripped before BUTChC sees it.
* A parameter's name is its dotted path in the spec (``features.fast.window``).
* ``apply_params(spec, params)`` returns a deep copy with every optimisable node
  replaced by a literal. The compiler never sees ``opt``.
* ``validate_spec(spec)`` returns plain-English problems; an empty list means
  valid. It is called when a spec is loaded and when Run is pressed, nowhere else.
"""

import copy
import hashlib
import json
import math
from importlib import resources
from pathlib import Path

import pandas as pd

SPEC_VERSION = 1

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

PRICE_FIELDS = ["open", "high", "low", "close", "volume"]
SIZING_KEYS = ["risk_pct", "risk_amount", "stake", "notional", "size"]
EXIT_KEYS = ["time_bars", "breakeven_points", "step_points"]     # the numba exit templates (exit_rules.py)
COMPARISON_OPS = [">", "<", ">=", "<=", "crosses_above", "crosses_below", "between", "rising", "falling"]
FILTER_KINDS = ["session", "weekday", "month", "higher_timeframe"]
RULE_SLOTS = ["long_entry", "long_exit", "short_entry", "short_exit"]
WEIGHT_SCHEMES = ["equal", "inverse_vol", "price", "fixed"]
SELECTION_MODES = ["top", "bottom", "threshold"]
CALENDAR_FREQS = ["W", "M", "Q", "Y"]


# ---------------------------------------------------------------------------
# Small accessors used everywhere
# ---------------------------------------------------------------------------

def is_opt(node):
    """True for an optimisable parameter node {"value": ..., "opt": {...}}."""
    return isinstance(node, dict) and "value" in node and "opt" in node


def value_of(node):
    """The literal value of a parameter, whether or not it is optimisable."""
    return node["value"] if is_opt(node) else node


def is_choice(node):
    """True for a structural alternative {"choose": label, "options": {label: subtree, ...}}.

    The selector ("choose") is a literal label or an optimisable categorical over
    the labels, so the optimiser can search between whole rule trees, features
    or settings (structure search, plan phase 3).
    """
    return isinstance(node, dict) and "choose" in node and "options" in node


def resolve_choices(node):
    """Replace every choice by its selected option, all the way down."""
    if is_choice(node):
        return resolve_choices(node["options"][value_of(node["choose"])])
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
            if not is_opt(child):
                stack.append((f"{prefix}.{key}" if prefix else str(key), child))
    return dict(sorted(found.items()))


def _split(path):
    # "rules.long_entry.all.0.left" -> ["rules", "long_entry", "all", 0, "left"]
    return [int(part) if part.isdigit() else part for part in path.split(".")]


def get_path(spec, path):
    node = spec
    for part in _split(path):
        node = node[part]
    return node


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
# Optimisable parameters
# ---------------------------------------------------------------------------

def collect_opts(spec):
    """Every optimisable node in the spec, as {dotted path: node}.

    Paths are unique by construction, which satisfies BUTChC's unique-name rule.
    """
    found = {}
    stack = [("", spec)]
    while stack:
        prefix, node = stack.pop()
        if is_opt(node):
            found[prefix] = node
            continue
        if isinstance(node, dict):
            children = node.items()
        elif isinstance(node, list):
            children = enumerate(node)
        else:
            continue
        for key, child in children:
            stack.append((f"{prefix}.{key}" if prefix else str(key), child))
    # Sorted so the parameter order (and so Sobol dimensions) is stable.
    return dict(sorted(found.items()))


def apply_params(spec, params=None):
    """Deep copy of the spec with every optimisable node replaced by a literal
    and every choice replaced by its selected option.

    A trial's value is used where params names the path; otherwise the node's
    own "value". The result contains no "opt" keys and no choices, so the
    compiler never sees either.
    """
    params = params or {}
    opts = collect_opts(spec)
    unknown = sorted(set(params) - set(opts))
    if unknown:
        # A trial naming a path the spec does not have means the spec changed
        # under a stored optimisation: fail loudly rather than ignore it.
        raise ValueError(f"These parameters are not optimisable in this spec: {', '.join(unknown)}")
    out = copy.deepcopy(spec)
    for path, node in opts.items():
        set_path(out, path, params.get(path, node["value"]))
    # Choices last: a trial's params may sit inside any option, chosen or not.
    return resolve_choices(out)


# ---------------------------------------------------------------------------
# Strategy kind and feature order
# ---------------------------------------------------------------------------

def strategy_kind(spec):
    """Which PortfolioBacktest route the spec takes, mirroring the front door's dispatch.

    rules -> "signals" (entries/exits); weighting + schedule -> "portfolio"
    (weights/schedule); neither -> "hold" (buy and hold); both -> "conflict".
    """
    rules = spec.get("rules") or {}
    has_rules = any(rules.get(slot) for slot in RULE_SLOTS)
    has_portfolio = any(key in spec for key in ("selection", "weighting", "schedule"))
    if has_rules and has_portfolio:
        return "conflict"
    if has_rules:
        return "signals"
    if has_portfolio:
        return "portfolio"
    return "hold"


def feature_dependencies(features):
    """{feature: set of features it reads through its "source" parameter}."""
    deps = {}
    for name, feature in features.items():
        source = value_of(feature.get("source"))
        deps[name] = {source["feature"]} if isinstance(source, dict) and "feature" in source else set()
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_opt(problems, where, opt):
    if not isinstance(opt, dict):
        problems.append(f"{where}: the optimisation settings must be an object.")
        return
    if "values" in opt:
        if not isinstance(opt["values"], list) or not opt["values"]:
            problems.append(f"{where}: list at least one allowed value to optimise over.")
        return
    if "min" not in opt or "max" not in opt:
        problems.append(f"{where}: give both a minimum and a maximum to optimise over.")
        return
    if not opt["min"] < opt["max"]:
        problems.append(f"{where}: the optimisation minimum must be below the maximum.")
    if opt.get("int") and opt.get("log"):
        problems.append(f"{where}: choose integer steps or a log scale, not both.")
    if opt.get("log") and opt["min"] <= 0:
        problems.append(f"{where}: a log scale needs a minimum above zero.")
    if "step" in opt and not (isinstance(opt["step"], (int, float)) and opt["step"] > 0):
        problems.append(f"{where}: the grid step must be a positive number.")


def _check_number(problems, where, node, minimum=None, integer=False, allow_zero=True):
    """Check a numeric parameter, literal or optimisable. Returns the literal or None."""
    if is_opt(node):
        _check_opt(problems, where, node["opt"])
    value = value_of(node)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        problems.append(f"{where} must be a number.")
        return None
    if integer and int(value) != value:
        problems.append(f"{where} must be a whole number.")
    if minimum is not None and (value < minimum or (value == minimum and not allow_zero)):
        problems.append(f"{where} must be {'at least' if allow_zero else 'above'} {minimum}.")
    return value


def _check_ref(problems, where, ref, features, blocks):
    if not isinstance(ref, dict):
        problems.append(f"{where}: expected a feature, a price or a constant.")
        return
    if "feature" in ref:
        name = ref["feature"]
        if name not in features:
            problems.append(f"{where} refers to a feature called '{name}' that does not exist.")
            return
        block = blocks.get(features[name].get("block"))
        output = ref.get("output")
        if block is not None and output is not None and output not in block["outputs"]:
            problems.append(f"{where}: feature '{name}' has no output '{output}' "
                            f"(it has {', '.join(block['outputs'])}).")
    elif "price" in ref:
        if ref["price"] not in PRICE_FIELDS:
            problems.append(f"{where}: '{ref['price']}' is not a price field.")
    elif "const" in ref:
        _check_number(problems, f"{where} constant", ref["const"])
    else:
        problems.append(f"{where}: expected a feature, a price or a constant.")
    if "shift" in ref:
        # A reference's own shift reads history (the channel high of the bar before).
        _check_number(problems, f"{where} shift", ref["shift"], minimum=0, integer=True)


def _check_rule(problems, where, node, features, blocks, timeframe, entry):
    if not isinstance(node, dict):
        problems.append(f"{where}: a rule must be an object.")
        return
    if "all" in node or "any" in node:
        key = "all" if "all" in node else "any"
        children = node[key]
        if not isinstance(children, list) or not children:
            problems.append(f"{where}: an '{key}' group needs at least one condition.")
            return
        for index, child in enumerate(children):
            _check_rule(problems, f"{where} › {key} {index + 1}", child, features, blocks, timeframe, entry)
        return
    if "not" in node:
        _check_rule(problems, f"{where} › not", node["not"], features, blocks, timeframe, entry)
        return
    if "pattern" in node:
        if node.get("module", "candle") not in ("candle", "chart"):
            problems.append(f"{where}: pattern module must be 'candle' or 'chart'.")
        if node.get("side", "either") not in ("bullish", "bearish", "either"):
            problems.append(f"{where}: pattern side must be bullish, bearish or either.")
        for key, value in (node.get("params") or {}).items():
            if not isinstance(value_of(value), (str, bool)):
                _check_number(problems, f"{where} pattern parameter {key}", value)
        return
    if "filter" in node:
        kind = node["filter"]
        # ⚠️ Filters gate entries only. Gating an exit would trap a position open.
        if not entry:
            problems.append(f"{where}: filters can only be used in entry rules; "
                            "filtering an exit would trap positions open.")
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
            condition = {key: node[key] for key in ("left", "op", "right", "lower", "upper", "bars") if key in node}
            _check_rule(problems, f"{where} condition", condition, features, blocks, timeframe, entry)
        return
    # A comparison leaf.
    op = node.get("op")
    if op not in COMPARISON_OPS:
        problems.append(f"{where}: unknown operator '{op}'.")
        return
    _check_ref(problems, f"{where} left side", node.get("left"), features, blocks)
    if op == "between":
        _check_ref(problems, f"{where} lower bound", node.get("lower"), features, blocks)
        _check_ref(problems, f"{where} upper bound", node.get("upper"), features, blocks)
    elif op in ("rising", "falling"):
        _check_number(problems, f"{where} bars", node.get("bars", 1), minimum=1, integer=True)
    else:
        _check_ref(problems, f"{where} right side", node.get("right"), features, blocks)
    # shift means "the value n bars ago". It never moves a signal forward.
    _check_number(problems, f"{where} shift", node.get("shift", 0), minimum=0, integer=True)


def _check_features(problems, features, blocks):
    for name, feature in features.items():
        where = f"Feature '{name}'"
        if not isinstance(feature, dict) or feature.get("block") not in blocks:
            problems.append(f"{where} uses an unknown block '{(feature or {}).get('block')}'.")
            continue
        declared = blocks[feature["block"]]["params"]
        for key, value in feature.items():
            if key == "block":
                continue
            if key not in declared:
                problems.append(f"{where}: '{key}' is not a parameter of the {feature['block']} block.")
                continue
            kind = declared[key]["type"]
            if kind in ("int", "float"):
                _check_number(problems, f"{where} {key}", value, minimum=declared[key].get("min"),
                              integer=kind == "int")
            elif kind == "choice":
                if is_opt(value):
                    _check_opt(problems, f"{where} {key}", value["opt"])
                if value_of(value) not in declared[key]["options"]:
                    problems.append(f"{where} {key} must be one of {', '.join(map(str, declared[key]['options']))}.")
            elif kind == "source":
                source = value_of(value)
                if isinstance(source, dict):
                    _check_ref(problems, f"{where} source", source, features, blocks)
                elif source not in PRICE_FIELDS:
                    problems.append(f"{where} source must be a price field or another feature.")
    if len(feature_order(features)) < len(features):
        problems.append("Some features refer to each other in a loop; break the cycle.")


def _check_risk(problems, spec, kind):
    risk = spec.get("risk") or {}
    sizing = risk.get("sizing") or {}
    used = [key for key in SIZING_KEYS if key in sizing]
    # ⚠️ vectorbtX raises BacktestError for two sizing arguments: catch it here first.
    if len(used) > 1:
        problems.append(f"Choose one way to size positions, not {len(used)} ({', '.join(used)}).")
    for key in used:
        _check_number(problems, f"Sizing {key}", sizing[key], minimum=0, allow_zero=False)
    extra = sorted(set(sizing) - set(SIZING_KEYS))
    if extra:
        problems.append(f"Unknown sizing setting: {', '.join(extra)}.")
    stop = risk.get("stop") or {"kind": "none"}
    stop_kind = stop.get("kind", "none")
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
    exits = risk.get("exits") or {}
    unknown = sorted(set(exits) - set(EXIT_KEYS))
    if unknown:
        problems.append(f"Unknown extra exit: {', '.join(unknown)}.")
    labels = {"time_bars": "Time stop (bars)", "breakeven_points": "Breakeven trigger (points)",
              "step_points": "Step size (points)"}
    used_exits = [key for key in EXIT_KEYS if exits.get(key) is not None]
    for key in used_exits:
        _check_number(problems, labels[key], exits[key], minimum=1 if key == "time_bars" else 0,
                      integer=key == "time_bars", allow_zero=False)
    if used_exits and stop_kind != "fixed":
        # The templates adjust an existing stop: vectorbt only calls them while a stop is active.
        problems.append("The extra exits (time stop, breakeven, step trailing) adjust a stop, so they need a fixed stop.")
    if kind == "portfolio" and (used or stop_kind != "none" or target or used_exits):
        problems.append("Portfolio strategies are sized by their weights; remove the sizing, stops and target.")


def _check_portfolio(problems, spec, features, blocks):
    symbols = spec["data"]["symbols"]
    selection = spec.get("selection")
    weighting = spec.get("weighting")
    schedule = spec.get("schedule")
    if weighting is None or schedule is None:
        problems.append("A portfolio strategy needs both a weighting and a schedule.")
        return
    if selection is not None:
        _check_ref(problems, "Selection score", selection.get("score"), features, blocks)
        mode = selection.get("mode", "top")
        if mode not in SELECTION_MODES:
            problems.append(f"Selection mode must be one of {', '.join(SELECTION_MODES)}.")
        if mode in ("top", "bottom"):
            count = _check_number(problems, "Selection count", selection.get("top_n"), minimum=1, integer=True)
            if count is not None and count > len(symbols):
                problems.append(f"Selection keeps {count} assets but the universe has only {len(symbols)}.")
        if selection.get("min_score") is not None:
            _check_number(problems, "Selection minimum score", selection["min_score"])
    scheme = value_of(weighting.get("scheme"))
    if is_opt(weighting.get("scheme")):
        _check_opt(problems, "Weighting scheme", weighting["scheme"]["opt"])
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
    calendar = schedule.get("calendar")
    if calendar is not None:
        if calendar.get("freq") not in CALENDAR_FREQS:
            problems.append(f"Rebalance frequency must be one of {', '.join(CALENDAR_FREQS)}.")
        if calendar.get("on", "first") not in ("first", "last"):
            problems.append("Rebalance day must be the first or the last of the period.")
    if schedule.get("threshold") is not None:
        _check_number(problems, "Drift threshold", schedule["threshold"], minimum=0, allow_zero=False)
        if calendar is None:
            # Drift is measured from the last calendar rebalance; without an anchor it
            # grows from the first bar and the threshold test is always true.
            problems.append("A drift threshold needs a calendar schedule to measure drift from.")
    if schedule.get("combine", "and") not in ("and", "or"):
        problems.append("Combine the calendar and the threshold with 'and' or 'or'.")
    _check_number(problems, "Minimum trade", schedule.get("min_trade", 0.0), minimum=0)


def _check_choices(problems, spec):
    for path, node in collect_choices(spec).items():
        options = node["options"]
        where = f"The alternatives at {path}"
        if not isinstance(options, dict) or not options:
            problems.append(f"{where} need at least one option.")
            continue
        if value_of(node["choose"]) not in options:
            problems.append(f"{where}: the chosen option '{value_of(node['choose'])}' does not exist.")
        if is_opt(node["choose"]):
            allowed = node["choose"]["opt"].get("values") or []
            unknown = [label for label in allowed if label not in options]
            if unknown or not allowed:
                problems.append(f"{where}: the optimiser may only pick existing options "
                                f"({', '.join(map(str, unknown)) or 'none listed'}).")


def validate_spec(spec):
    """Every problem with the spec, in plain English. An empty list means valid.

    With alternatives (choices), the spec is checked as it stands now and then
    once with each alternative selected in turn, so an option the optimiser may
    pick is checked before a job spends hours finding it broken.
    """
    if not isinstance(spec, dict):
        return ["The strategy file must hold a JSON object."]
    problems = []
    _check_choices(problems, spec)
    if problems:
        return problems
    problems = _validate_resolved(resolve_choices(spec))
    for path, node in collect_choices(spec).items():
        current = value_of(node["choose"])
        for label in node["options"]:
            if label == current:
                continue
            variant = copy.deepcopy(spec)
            set_path(variant, f"{path}.choose", label)
            for problem in _validate_resolved(resolve_choices(variant)):
                text = f"With '{label}' chosen at {path}: {problem}"
                if problem not in problems and text not in problems:
                    problems.append(text)
    return problems


def _validate_resolved(spec):
    """validate_spec for a spec with no choices left in it."""
    from .blocks import BLOCKS   # local import: blocks.py does not import this module's callers

    problems = []
    if spec.get("version") != SPEC_VERSION:
        problems.append(f"This file is spec version {spec.get('version')}; the Studio reads version {SPEC_VERSION}.")
    if not str(spec.get("name", "")).strip():
        problems.append("Give the strategy a name.")

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
            if symbol not in instruments:
                problems.append(f"'{symbol}' has no instrument settings.")
            elif not instruments[symbol].get("mode"):
                problems.append(f"'{symbol}' needs a mode (equity, cfd, spreadbet ...).")
        for symbol in instruments:
            if symbol not in symbols:
                problems.append(f"Instrument settings for '{symbol}', which is not in the symbol list.")

    features = spec.get("features") or {}
    _check_features(problems, features, BLOCKS)

    kind = strategy_kind(spec)
    if kind == "conflict":
        problems.append("A strategy has either entry/exit rules or a portfolio (weighting and schedule), not both.")
    if kind == "signals":
        rules = spec["rules"]
        if not rules.get("long_entry") and not rules.get("short_entry"):
            problems.append("Add a long or a short entry rule.")
        extra = sorted(set(rules) - set(RULE_SLOTS))
        if extra:
            problems.append(f"Unknown rule slot: {', '.join(extra)}.")
        for slot in RULE_SLOTS:
            if rules.get(slot):
                _check_rule(problems, slot.replace("_", " ").capitalize(), rules[slot],
                            features, BLOCKS, timeframe, entry=slot.endswith("entry"))
    if kind == "portfolio" and symbols:
        _check_portfolio(problems, spec, features, BLOCKS)
    if kind in ("signals", "hold", "portfolio"):
        _check_risk(problems, spec, kind)

    account = spec.get("account") or {}
    _check_number(problems, "Starting cash", account.get("init_cash"), minimum=0, allow_zero=False)
    if not account.get("freq"):
        problems.append("Set the bar frequency (freq); annualised metrics need it.")
    fx = account.get("fx")
    if fx and not fx.get("symbol"):
        problems.append("Name the stored FX dataset to convert with (for example GBPUSD=X).")
    if fx and not account.get("base_currency"):
        problems.append("Converting with FX needs a base currency.")
    rates = account.get("rates")
    if rates and rates.get("source") not in ("fred", "static"):
        problems.append("Rates come from FRED or a static table.")
    if rates and rates.get("source") == "static" and not rates.get("values"):
        problems.append("A static rate table needs at least one currency and rate.")
    return problems
