"""Text forms of a strategy: a rule's logic line, its conditions in words, and the At a glance summary.

The logic line combines a rule's named conditions, for example
``(A and (B or C)) and not D``. New conditions take the next free letter; once renamed
the line reads in their names: ``trend and not overbought``.

    and, or, xor, nand, nor, xnor    between two conditions (or bracketed groups)
    not                              in front of a condition or a group
    ( )                              grouping

⚠️ Brackets are required wherever two different operators meet at one level:
"A and B or C" is refused, with a message asking which was meant. There are no
precedence rules to remember (the owner's call: explicit is always better).

One operator repeated chains: ``A and B and C``. Over three or more:
* xor is true when an odd number are true (as Python's ^ chains), xnor is its
  opposite,
* nand is "not all of them", nor is "none of them".

Parsing is a small tokenizer and a recursive descent over the names and these
words, so text is never evaluated.
"""

import re

LOGIC_OPS = ["and", "or", "xor", "nand", "nor", "xnor"]
LOGIC_WORDS = {"and": "all are true", "or": "any is true", "xor": "an odd number are true",
               "nand": "not all are true", "nor": "none is true", "xnor": "an even number are true"}
# Words no condition may be called, in any case: the logic line reads them as operators.
RESERVED_NAMES = LOGIC_OPS + ["not"]
# A condition's name: a letter, then letters, digits or underscores (A, B ... by default; trend
# or rsi_low once renamed). Matched exactly, capitals included, so "a" is not "A".
NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_]*"

OP_WORDS = {">": "is above", "<": "is below", ">=": "is at or above", "<=": "is at or below",
            "crosses_above": "crosses above", "crosses_below": "crosses below", "between": "is between",
            "rising": "is rising over", "falling": "is falling over"}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TRADING_WORDS = {"equity": "bought outright", "cfd": "CFDs", "spreadbet": "spread bets", "fx_spot": "FX spot"}


# ---------------------------------------------------------------------------
# The logic line
# ---------------------------------------------------------------------------

def _tokens(text):
    tokens = []
    for piece in re.findall(rf"\(|\)|{NAME_PATTERN}|\S+", text or ""):
        if piece in ("(", ")"):
            tokens.append(piece)
        elif piece.lower() in RESERVED_NAMES:
            tokens.append(piece.lower())
        elif re.fullmatch(NAME_PATTERN, piece):
            tokens.append(piece)
        else:
            raise ValueError(f"'{piece}' is not a condition name, a bracket or one of and, or, xor, nand, nor, "
                             "xnor, not.")
    return tokens


def condition_name_problem(name, taken):
    """Why `name` cannot name a condition in a rule whose other conditions are `taken`; None if it can."""
    if not re.fullmatch(NAME_PATTERN, name or ""):
        return (f"'{name}' cannot name a condition: use letters, digits and underscores, starting with a letter, "
                "such as trend or rsi_low.")
    if name.lower() in RESERVED_NAMES:
        return f"'{name}' would read as an operator in the logic line; pick another name."
    if name in taken:
        return f"This rule already has a condition called {name}; pick another name."
    return None


def parse_logic(text):
    """The logic line as a tree: a condition's name, {"not": tree}, or {"op": op, "args": [trees]}.

    Raises ValueError with a message that says what to change.
    """
    tokens = _tokens(text)
    if not tokens:
        raise ValueError("write which conditions must hold, such as A and B.")
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def take():
        nonlocal position
        position += 1
        return tokens[position - 1]

    def operand():
        token = peek()
        if token is None:
            raise ValueError("it ends too soon: a condition or a bracket is missing at the end.")
        if token == "not":
            take()
            return {"not": operand()}
        if token == "(":
            take()
            tree = expression()
            if peek() != ")":
                raise ValueError("a bracket is opened and never closed.")
            take()
            return tree
        if token in LOGIC_OPS or token == ")":
            raise ValueError(f"'{token}' needs a condition or a bracketed group before it.")
        return take()

    def expression():
        first = operand()
        args, op = [first], None
        while peek() in LOGIC_OPS:
            this = take()
            if op is not None and this != op:
                raise ValueError(f"'{op}' and '{this}' meet without brackets. Add brackets to say which comes "
                                 f"first, for example ({' '.join(['A', op, 'B'])}) {this} C.")
            op = this
            args.append(operand())
        if op is None:
            return first
        return {"op": op, "args": args}

    tree = expression()
    if peek() is not None:
        token = peek()
        raise ValueError("a bracket is closed that was never opened." if token == ")"
                         else f"'{token}' follows without an operator between; add and, or ... before it.")
    return tree


def logic_names(tree):
    """The conditions a parsed logic line uses."""
    if isinstance(tree, str):
        return {tree}
    if "not" in tree:
        return logic_names(tree["not"])
    found = set()
    for arg in tree["args"]:
        found |= logic_names(arg)
    return found


def logic_text(tree, top=True):
    """A tree back to text, bracketing every group below the top."""
    if isinstance(tree, str):
        return tree
    if "not" in tree:
        inner = tree["not"]
        return "not " + (logic_text(inner, top=False) if isinstance(inner, str) or "not" in inner
                         else f"({logic_text(inner)})")
    text = f" {tree['op']} ".join(logic_text(arg, top=False) for arg in tree["args"])
    return text if top else f"({text})"


def next_letter(existing):
    """The first free condition letter: A ... Z, then AA, AB ..."""
    index = 0
    while True:
        name, value = "", index
        while True:
            name = chr(ord("A") + value % 26) + name
            value = value // 26 - 1
            if value < 0:
                break
        if name not in existing:
            return name
        index += 1


def next_condition_name(existing):
    """The first free default name for a new condition: condition1, condition2 ...

    Words, not letters (the owner's call, 20 November 2026: "A" read as confusing). Any name
    can still be typed in its Name box, and a strategy's older lettered names keep working.
    """
    number = 1
    while f"condition{number}" in existing:
        number += 1
    return f"condition{number}"


def append_letter(logic, letter, op="and"):
    """The logic line with a new condition joined on with `op`."""
    if not (logic or "").strip():
        return letter
    try:
        tree = parse_logic(logic)
    except ValueError:
        return f"{logic} {op} {letter}"
    if isinstance(tree, dict) and tree.get("op") == op:
        return f"{logic} {op} {letter}"
    if isinstance(tree, str) or "not" in tree:
        return f"{logic} {op} {letter}"
    return f"({logic}) {op} {letter}"


def remove_letter(logic, letter):
    """The logic line without one condition; "" when nothing is left. A broken line is returned as it was."""
    try:
        tree = parse_logic(logic)
    except ValueError:
        return logic

    def drop(node):
        if isinstance(node, str):
            return None if node == letter else node
        if "not" in node:
            inner = drop(node["not"])
            return None if inner is None else {"not": inner}
        args = [kept for kept in (drop(arg) for arg in node["args"]) if kept is not None]
        if not args:
            return None
        return args[0] if len(args) == 1 else {"op": node["op"], "args": args}

    kept = drop(tree)
    return "" if kept is None else logic_text(kept)


def rename_in_logic(logic, old, new):
    """The logic line with condition `old` called `new`, and the rest exactly as written.

    Only whole names are replaced (renaming A leaves AA alone), and a broken line is
    renamed too, so the user's half-typed text keeps up with the conditions.
    """
    return re.sub(NAME_PATTERN, lambda match: new if match.group(0) == old else match.group(0), logic or "")


# ---------------------------------------------------------------------------
# Conditions in words
# ---------------------------------------------------------------------------

def ref_text(ref):
    if not isinstance(ref, dict):
        return "?"
    if "feature" in ref:
        text = ref["feature"] + (f".{ref['output']}" if ref.get("output") else "")
    elif "price" in ref:
        text = ref["price"]
    else:
        value = ref.get("const")
        text = f"{value:g}" if isinstance(value, float) else str(value)
    shift = int(ref.get("shift", 0) or 0)
    return f"{text} {shift} bar{'s' if shift != 1 else ''} ago" if shift else text


def timing_text(condition):
    """A condition's timing in words (compile.held), "" when it has none."""
    for key, words in (("for_bars", "on each of the last {} bars"), ("within_bars", "on any of the last {} bars")):
        if condition.get(key):
            return ", " + words.format(condition[key])
    if condition.get("bars_ago"):
        bars = condition["bars_ago"]
        return f", {bars} bar{'s' if bars != 1 else ''} ago"
    return ""


def condition_text(condition):
    """One condition in words: 'fast crosses above slow', 'hammer appears (bullish), on any of the last 3 bars'."""
    if "pattern" in condition:
        side = condition.get("side", "either")
        return f"{condition['pattern']} appears" + ("" if side == "either" else f" ({side})") + timing_text(condition)
    if "filter" in condition:
        kind = condition["filter"]
        if kind == "weekday":
            return "on " + ", ".join(DAY_NAMES[day] for day in condition.get("days", []))
        if kind == "month":
            return "in " + ", ".join(MONTH_NAMES[month - 1] for month in condition.get("months", []))
        if kind == "session":
            return f"between {condition.get('from')} and {condition.get('to')} UTC"
        comparison = {key: condition[key] for key in ("left", "op", "right", "lower", "upper", "bars")
                      if key in condition}
        return f"on {condition.get('timeframe')} bars, {condition_text(comparison)}{timing_text(condition)}"
    op = condition.get("op")
    left = ref_text(condition.get("left"))
    if op == "between":
        text = f"{left} is between {ref_text(condition.get('lower'))} and {ref_text(condition.get('upper'))}"
    elif op in ("rising", "falling"):
        bars = condition.get("bars", 1)
        text = f"{left} {OP_WORDS[op]} {bars} bar{'s' if bars != 1 else ''}"
    else:
        text = f"{left} {OP_WORDS.get(op, op)} {ref_text(condition.get('right'))}"
    return text + timing_text(condition)


# ---------------------------------------------------------------------------
# At a glance
# ---------------------------------------------------------------------------

def _number(value):
    if isinstance(value, float):
        # A whole number reads as one (10,000, not 1e+04, which ".4g" gave for a starting cash).
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.6g}" if 1e-4 <= abs(value) < 1e9 else f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def logic_words(tree, conditions):
    """A rule's logic line in words, each condition by what it says: "close crosses above slow and rsi is
    below 30". A part joined by another operator is bracketed, as the logic line itself must be."""
    if isinstance(tree, str):
        condition = conditions.get(tree)
        return condition_text(condition) if condition else tree
    if "not" in tree:
        inner = logic_words(tree["not"], conditions)
        return f"not {inner}" if isinstance(tree["not"], str) else f"not ({inner})"
    parts = []
    for arg in tree["args"]:
        text = logic_words(arg, conditions)
        parts.append(text if isinstance(arg, str) or "not" in arg else f"({text})")
    if tree["op"] in ("and", "or"):
        return f" {tree['op']} ".join(parts)
    return f"{LOGIC_WORDS[tree['op']]} of: " + "; ".join(parts)


def _rule_words(rule):
    """(text, items) for one rule: its logic in words, or the logic line and each condition as an item when a
    sentence would be hard to read (more than two conditions, or operators mixed); "(no logic line)"."""
    if rule.get("choose") is not None and rule.get("options"):
        text, items = _rule_words(rule["options"][rule["choose"]])
        return f"{text} (version {rule['choose']} of {', '.join(rule['options'])})", items
    conditions = rule.get("conditions") or {}
    if not (rule.get("logic") or "").strip():
        return "(no logic line yet)", []
    try:
        tree = parse_logic(rule["logic"])
    except ValueError:
        return f"{rule.get('logic') or '(no logic line)'} (the logic line does not read)", []
    if len(conditions) <= 2 or (isinstance(tree, dict) and "op" in tree and tree["op"] in ("and", "or")
                                and all(isinstance(arg, str) for arg in tree["args"])):
        if len(conditions) <= 2 or isinstance(tree, str):
            return logic_words(tree, conditions), []
        # Three or more joined by one operator: a list reads better than one long sentence.
        lead = "all of these are true" if tree["op"] == "and" else "any of these is true"
        return lead + ":", [condition_text(conditions[name]) for name in tree["args"] if name in conditions]
    return f"{rule.get('logic')}, where:", [f"{name}: {condition_text(condition)}"
                                           for name, condition in conditions.items()]


def _distance_words(node, anchor):
    """How far a stop or target sits, in words: "13.14% from the entry", "50 points from the entry",
    "3 x atr from the entry", "at ema". `anchor` is what a distance is measured from."""
    unit = node.get("unit", "points")
    if unit == "pct":
        return f"{_percent(node['pct'])} {anchor}"
    if unit == "multiple":
        return f"{_number(node['multiple'])} x {ref_text(node.get('line') or {})} {anchor}"
    if unit == "level":
        return f"at {ref_text(node.get('line') or {})}"
    return f"{_number(node['points'])} points {anchor}"


# A timeframe as a run of bars reads ("daily bars"), by spec.TIMEFRAMES and the higher ones.
TIMEFRAME_WORDS = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "30m": "30-minute", "1h": "hourly",
                   "4h": "4-hour", "1d": "daily", "1w": "weekly", "1M": "monthly"}
# What each rule slot does, as the start of its sentence.
SLOT_WORDS = {"long_entry": "Buy", "long_exit": "Sell", "short_entry": "Sell short", "short_exit": "Buy back"}


def plan(spec):
    """The strategy on a page, as it would be told: a list of sentences, each {"tone", "lead", "text",
    "items"} (a part's tone, the sentence's first words in bold, the rest, and a list under it), and
    "after", a sentence after the list, on an exit that closes at a stop too.

    "Spread bets on SPY, long only, on daily bars from 2010-01-01 to 2026-11-19, from 10,000. Buy when the
    close crosses above JMA_slow. Sell when the low crosses below JMA_fast, or at the stop or the target.
    Each trade risks 20.00% of the balance, with a stop loss 13.14% from the entry and a take profit 7.54%
    from the entry." The Build page's At a glance panel reads it (the owner, 25 November 2026: the panel
    listed every setting, but did not read as a strategy).
    """
    from .indicators import describe
    from .spec import strategy_kind, trading_mode, tunable_inputs, variables

    out = []
    data = spec.get("data") or {}
    symbols = ", ".join(data.get("symbols") or []) or "no symbols yet"
    kind = strategy_kind(spec)
    mode = trading_mode(spec)
    rules = spec.get("rules") or {}
    if not data.get("symbols"):
        market = f"No symbols yet ({TRADING_WORDS.get(mode, mode)})"
        if kind == "signals" and not any(rules.get(f"{side}_entry") for side in ("long", "short")):
            market += ", no entry rule yet"
    elif kind == "signals":
        market = {"equity": f"{symbols}, bought outright", "cfd": f"CFDs on {symbols}",
                  "spreadbet": f"Spread bets on {symbols}", "fx_spot": f"FX spot on {symbols}"}.get(mode, symbols)
        sides = [side for side in ("long", "short") if rules.get(f"{side}_entry")]
        market += {("long",): ", long only", ("short",): ", short only", ("long", "short"): ", long and short",
                   (): ", with no entry rule yet"}[tuple(sides)]
    else:
        market = f"A portfolio of {symbols}" + {"equity": ", bought outright", "cfd": ", as CFDs",
                                                "spreadbet": ", as spread bets", "fx_spot": ", as FX spot"}.get(mode, "")
    account = spec.get("account") or {}
    money = _number(account.get("init_cash")) + (f" {account['base_currency']}" if account.get("base_currency") else "")
    # Leverage as the instrument grid sets it: said when every symbol names it (a blank one is the usual for the
    # market, which the grid shows), once when they agree.
    given = [item.get("leverage") for item in (spec.get("instruments") or {}).values() if isinstance(item, dict)]
    levered = ""
    if given and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in given):
        if len(set(given)) > 1:
            levered = ", with leverage set per symbol"
        elif float(given[0]) > 1:
            levered = f", at up to {_number(float(given[0]))}x leverage"
    shared = ", one account for every symbol" if account.get("cash") == "shared" and kind == "signals" else ""
    timeframe = TIMEFRAME_WORDS.get(data.get("timeframe"), str(data.get("timeframe")))
    out.append({"tone": "setup", "lead": market, "items": [],
                "text": f", on {timeframe} bars from {data.get('start')} to {data.get('end')}, starting with "
                        f"{money}{levered}{shared}."})

    if kind == "signals":
        risk = spec.get("risk") or {}
        stop = risk.get("stop") or {}
        target = risk.get("target") or {}
        exits = risk.get("exits") or {}
        # What closes a position whatever its exit rule says (validate_spec: a stop's kind says whether there
        # is one; never the dict).
        closes = []
        if stop.get("kind") in ("fixed", "trailing"):
            closes.append("the trailing stop" if stop["kind"] == "trailing" else "the stop")
        if target.get("kind") == "fixed":
            closes.append("the target")
        if exits.get("time_bars") is not None:
            closes.append(f"after {exits['time_bars']} bars")
        for slot, verb in SLOT_WORDS.items():
            rule = rules.get(slot)
            side = slot.split("_")[0]
            if slot.endswith("_exit") and not rules.get(f"{side}_entry"):
                continue
            tone = "entry" if slot.endswith("_entry") else "exit"
            if rule is None and slot.endswith("_entry"):
                continue
            if rule is None:
                if closes:
                    out.append({"tone": tone, "lead": verb, "items": [],
                                "text": f" only at {_joined(closes, 'or')}: there is no {side} exit rule."})
                else:
                    out.append({"tone": tone, "lead": verb, "items": [],
                                "text": f" never: there is no {side} exit rule, stop or target, so each {side} is "
                                        "held until an opposite entry or the end of the history."})
                continue
            text, items = _rule_words(rule)
            block = {"tone": tone, "lead": verb, "text": " when " + text, "items": items}
            if slot.endswith("_exit") and closes:
                # After the list when there is one: "or" inside "all of these are true" would change it.
                if items:
                    block["after"] = f"Or at {_joined(closes, 'or')}."
                else:
                    block["text"] += f", or at {_joined(closes, 'or')}"
            if not items:
                block["text"] += "."
            out.append(block)

        sizing = risk.get("sizing") or {}
        of = {"balance": "the balance", "equity": "the equity"}.get(sizing.get("risk_of"), "the account")
        size = "no size set"
        for key, value in sizing.items():
            if key == "risk_of":
                continue
            share = _percent(value) if isinstance(value, (int, float)) else str(value)
            unit = {"equity": "share", "fx_spot": "lot", "cfd": "contract"}.get(mode, "unit")
            unit += "" if value == 1 else "s"
            size = {"risk_pct": f"risks {share} of {of}", "risk_amount": f"risks {_number(value)}",
                    "stake": (f"is a stake of {_number(value)} a point" if mode == "spreadbet"
                              else f"is {_number(value)} {unit}"),
                    "notional": f"is worth {_number(value)}", "notional_pct": f"is worth {share} of {of}",
                    "size": f"is {_number(value)} {unit}"}.get(key, f"{key} {value}")
        guards = []
        if stop.get("kind") == "fixed":
            guards.append("a stop loss " + _distance_words(stop, "from the entry"))
        elif stop.get("kind") == "trailing":
            guards.append("a trailing stop " + _distance_words(stop, "behind the best price"))
        if target.get("kind") == "fixed":
            guards.append("a take profit " + _distance_words(target, "from the entry"))
        if exits.get("breakeven_points") is not None:
            guards.append(f"the stop moved to the entry after {_number(exits['breakeven_points'])} points")
        if exits.get("step_points") is not None:
            guards.append(f"the stop stepped up every {_number(exits['step_points'])} points")
        orders = risk.get("orders") or {}
        timing = []
        if orders.get("good_for_day"):
            timing.append("Entries are good for the day only.")
        if orders.get("flat_by_day_end"):
            timing.append("Every position is closed by the day's end.")
        out.append({"tone": "risk", "lead": "Each trade", "items": [],
                    "text": f" {size}" + (f", with {_joined(guards)}" if guards else ", with no stop or target")
                            + "." + "".join(f" {sentence}" for sentence in timing)})
    else:
        parts = []
        selection = spec.get("selection")
        if selection:
            how = (f"the {selection.get('mode', 'top')} {selection.get('top_n')}" if selection.get("mode", "top")
                   in ("top", "bottom") else f"those scoring at least {selection.get('min_score')}")
            parts.append(f"holds {how} by {ref_text(selection.get('score'))}")
        scheme = (spec.get("weighting") or {}).get("scheme")
        parts.append({"equal": "weights them equally", "inverse_vol": "weights them by the inverse of their "
                      "volatility", "price": "weights them by price", "fixed": "holds fixed weights"}
                     .get(scheme, f"weights them by {scheme}"))
        schedule = spec.get("schedule") or {}
        rebalance = schedule.get("rebalance")
        if rebalance == "calendar":
            calendar = schedule.get("calendar") or {}
            period = {"W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(calendar.get("freq"),
                                                                                 str(calendar.get("freq")))
            text = f"rebalances on the {calendar.get('on', 'first')} bar of each {period}"
            threshold = schedule.get("threshold")
            if threshold is not None:
                # spec: an asset's weight drifting by more than this share of its target (0.05 = 5%).
                drift = f"a weight has drifted more than {_percent(threshold)} from its target"
                text += (f", when {drift}" if schedule.get("combine", "and") == "and"
                         else f", and whenever {drift}")
        else:
            text = {"never": "never rebalances (buy and hold)", "every_bar": "rebalances every bar"}.get(
                rebalance, f"rebalances {rebalance}")
        parts.append(text)
        out.append({"tone": "entry", "lead": "It", "text": " " + _joined(parts) + ".", "items": []})

    features = spec.get("features") or {}
    if features:
        out.append({"tone": "feature", "lead": "Where", "text": "", "items": [
            f"{name} is {describe(feature)}" for name, feature in features.items()]})
    found = variables(spec)
    if found:
        labels = {row["path"]: row["label"] for row in tunable_inputs(spec)}
        items = []
        for path, variable in found.items():
            span = (", ".join(map(_number, variable["values"])) if "values" in variable else
                    f"{_number(variable.get('start'))} to {_number(variable.get('stop'))}"
                    + (f" by {_number(variable['step'])}" if variable.get("step") else ""))
            items.append(f"{labels.get(path, path)}: {span}")
        out.append({"tone": None, "lead": "Optimising", "items": items,
                    "text": f" {len(found)} setting{'s' if len(found) != 1 else ''}, on the Optimise page:"})
    return out


def _joined(parts, word="and"):
    """"a", "a and b", "a, b and c" (or "or")."""
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" {word} " + parts[-1]


def _percent(value):
    """A fraction as a percentage with no trailing zeros: 0.2 is "20%", 0.13138 is "13.14%"."""
    return f"{value * 100:.2f}".rstrip("0").rstrip(".") + "%"
