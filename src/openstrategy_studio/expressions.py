"""Text forms of a strategy: a rule's logic line, its conditions in words, and the At a glance summary.

The logic line combines a rule's lettered conditions, for example
``(A and (B or C)) and not D``.

    and, or, xor, nand, nor, xnor    between two letters (or bracketed groups)
    not                              in front of a letter or a group
    ( )                              grouping

⚠️ Brackets are required wherever two different operators meet at one level:
"A and B or C" is refused, with a message asking which was meant. There are no
precedence rules to remember (the owner's call: explicit is always better).

One operator repeated chains: ``A and B and C``. Over three or more:
* xor is true when an odd number are true (as Python's ^ chains), xnor is its
  opposite,
* nand is "not all of them", nor is "none of them".

Parsing is a small tokenizer and a recursive descent over the letters and these
words, so text is never evaluated.
"""

import re

LOGIC_OPS = ["and", "or", "xor", "nand", "nor", "xnor"]
LOGIC_WORDS = {"and": "all are true", "or": "any is true", "xor": "an odd number are true",
               "nand": "not all are true", "nor": "none is true", "xnor": "an even number are true"}

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
    for piece in re.findall(r"\(|\)|[A-Za-z]+|\S+", text or ""):
        if piece in ("(", ")"):
            tokens.append(piece)
        elif piece.lower() in LOGIC_OPS or piece.lower() == "not":
            tokens.append(piece.lower())
        elif piece.isalpha() and piece.isupper():
            tokens.append(piece)
        else:
            raise ValueError(f"'{piece}' is not a condition letter, a bracket or one of and, or, xor, nand, nor, "
                             "xnor, not.")
    return tokens


def parse_logic(text):
    """The logic line as a tree: a letter, {"not": tree}, or {"op": op, "args": [trees]}.

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
            raise ValueError("it ends too soon: a letter or a bracket is missing at the end.")
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
            raise ValueError(f"'{token}' needs a letter or a bracketed group before it.")
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


def logic_letters(tree):
    if isinstance(tree, str):
        return {tree}
    if "not" in tree:
        return logic_letters(tree["not"])
    found = set()
    for arg in tree["args"]:
        found |= logic_letters(arg)
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


def condition_text(condition):
    """One condition in words: 'fast crosses above slow', 'hammer appears (bullish)'."""
    if "pattern" in condition:
        side = condition.get("side", "either")
        return f"{condition['pattern']} appears" + ("" if side == "either" else f" ({side})")
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
        return f"on {condition.get('timeframe')} bars, {condition_text(comparison)}"
    op = condition.get("op")
    left = ref_text(condition.get("left"))
    if op == "between":
        return f"{left} is between {ref_text(condition.get('lower'))} and {ref_text(condition.get('upper'))}"
    if op in ("rising", "falling"):
        bars = condition.get("bars", 1)
        return f"{left} {OP_WORDS[op]} {bars} bar{'s' if bars != 1 else ''}"
    return f"{left} {OP_WORDS.get(op, op)} {ref_text(condition.get('right'))}"


# ---------------------------------------------------------------------------
# At a glance
# ---------------------------------------------------------------------------

def _number(value):
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _rule_rows(label, rule):
    rows = []
    if rule is None:
        return rows
    if "choose" in rule and "options" in rule:
        rows.append((label, f"version {rule['choose']} of {', '.join(rule['options'])}"))
        rule = rule["options"][rule["choose"]]
    else:
        rows.append((label, rule.get("logic") or "(no logic line)"))
    for letter, condition in (rule.get("conditions") or {}).items():
        rows.append((f"  {letter}", condition_text(condition)))
    return rows


def summary(spec):
    """The strategy as (label, text) rows, for the Build page's At a glance panel."""
    from .indicators import describe
    from .spec import SLOT_LABELS, strategy_kind, trading_mode, variables

    rows = []
    data = spec.get("data") or {}
    rows.append(("Market", f"{', '.join(data.get('symbols') or [])} · {data.get('timeframe')} · "
                           f"{data.get('start')} to {data.get('end')}"))
    kind = strategy_kind(spec)
    rows.append(("Trades", f"{'on signals' if kind == 'signals' else 'a portfolio'}, "
                           f"{TRADING_WORDS.get(trading_mode(spec), trading_mode(spec))}"))
    for name, feature in (spec.get("features") or {}).items():
        rows.append(("Feature", f"{name} = {describe(feature)}"))
    if kind == "signals":
        for slot, label in SLOT_LABELS.items():
            rows += _rule_rows(label, (spec.get("rules") or {}).get(slot))
        risk = spec.get("risk") or {}
        parts = []
        for key, value in (risk.get("sizing") or {}).items():
            parts.append({"risk_pct": f"risk {value:.2%} a trade" if isinstance(value, (int, float)) else str(value),
                          "risk_amount": f"risk {_number(value)} a trade", "stake": f"stake {_number(value)}",
                          "notional": f"positions of {_number(value)}", "size": f"size {_number(value)}"}[key])
        stop = risk.get("stop") or {}
        if stop.get("kind") in ("fixed", "trailing"):
            parts.append(f"{stop['kind']} stop {_number(stop['points'])} pts")
        if risk.get("target"):
            parts.append(f"target {_number(risk['target']['points'])} pts")
        exits = risk.get("exits") or {}
        if exits.get("time_bars") is not None:
            parts.append(f"out after {exits['time_bars']} bars")
        if exits.get("breakeven_points") is not None:
            parts.append(f"breakeven at {_number(exits['breakeven_points'])} pts")
        if exits.get("step_points") is not None:
            parts.append(f"step {_number(exits['step_points'])} pts")
        orders = risk.get("orders") or {}
        if orders.get("good_for_day"):
            parts.append("entries good for the day")
        if orders.get("flat_by_day_end"):
            parts.append("flat by the day's end")
        rows.append(("Risk", " · ".join(parts) or "no sizing set"))
    else:
        selection = spec.get("selection")
        if selection:
            how = (f"{selection.get('mode', 'top')} {selection.get('top_n')}" if selection.get("mode", "top")
                   in ("top", "bottom") else f"score at least {selection.get('min_score')}")
            rows.append(("Selection", f"{how} by {ref_text(selection.get('score'))}"))
        weighting = spec.get("weighting") or {}
        rows.append(("Weighting", str(weighting.get("scheme"))))
        schedule = spec.get("schedule") or {}
        rebalance = schedule.get("rebalance")
        if rebalance == "calendar":
            calendar = schedule.get("calendar") or {}
            text = f"every {calendar.get('freq')} ({calendar.get('on', 'first')} day)"
            if schedule.get("threshold") is not None:
                text += f" {schedule.get('combine', 'and')} drift over {schedule['threshold']}"
        else:
            text = {"never": "never (buy and hold)", "every_bar": "every bar"}.get(rebalance, str(rebalance))
        rows.append(("Rebalance", text))
    account = spec.get("account") or {}
    text = f"{_number(account.get('init_cash'))}"
    if account.get("base_currency"):
        text += f" in {account['base_currency']}"
    if account.get("rates"):
        text += f" · rates: {account['rates'].get('source')}"
    if account.get("cash") == "shared" and kind == "signals":
        text += " · one shared account"
    rows.append(("Account", text))
    found = variables(spec)
    if found:
        rows.append(("Optimise", f"{len(found)} variable{'s' if len(found) != 1 else ''}"))
        for path, variable in found.items():
            span = (", ".join(map(_number, variable["values"])) if "values" in variable else
                    f"{_number(variable.get('start'))} to {_number(variable.get('stop'))}"
                    + (f" step {_number(variable['step'])}" if variable.get("step") else ""))
            rows.append(("  ", f"{path.replace('features.', '').replace('.params', '')}: {span}"))
    return rows
