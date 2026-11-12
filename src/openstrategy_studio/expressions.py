"""The expression editor: text such as ``rsi(14) < 30 and close > ma(200)``.

Text is parsed with ast.parse(mode="eval") and walked against a whitelist of
node types, names and functions; anything else is rejected with a message.
⚠️ Raw text is NEVER evaluated.

It converts to and from the rule tree, so the builder and the text edit one
spec. An indicator call such as ``ma(200)`` becomes a feature named ``ma_200``
(added to the spec's features if missing); a feature with such a name is
written back as the call.

Grammar
    a and b, a or b, not a              all / any / not
    a > b, a < b, a >= b, a <= b        comparisons (chains become "all")
    crosses_above(a, b), crosses_below(a, b), between(x, lo, hi)
    rising(x, n), falling(x, n)
    x[n]                                the value n bars ago
    fast, bb.upper, close, 30           a feature, a feature output, a price, a constant
    ma(50), ema(20), rsi(14), bbands(20).upper, momentum(126) ...
    pattern("hammer", side="bullish", module="candle", shadow_factor=1.5)
    session("07:00", "16:00"), weekday(0, 1, 2, 3, 4), month(1, 2, 3)
"""

import ast

from .blocks import BLOCKS, block_defaults
from .spec import PRICE_FIELDS, value_of

ALIASES = {"sma": ("ma", {"ewm": False}), "ema": ("ma", {"ewm": True})}
OPERATORS = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<="}
ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Gt,
           ast.Lt, ast.GtE, ast.LtE, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.Subscript,
           ast.keyword)


def _numeric_params(block_name):
    """The block's numeric parameters in declaration order: the positional arguments of a call."""
    return [key for key, declaration in BLOCKS[block_name]["params"].items()
            if declaration["type"] in ("int", "float")]


def auto_name(block_name, params):
    numbers = [str(value_of(params[key])).replace(".", "_") for key in _numeric_params(block_name) if key in params]
    prefix = "ema" if block_name == "ma" and value_of(params.get("ewm")) else block_name
    return "_".join([prefix] + numbers)


def _constant(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    raise ValueError(f"Expected a number or text, found '{ast.unparse(node)}'.")


def parse(text, features):
    """(rule tree, new features to add) from expression text. Raises ValueError on anything unsupported."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"That expression does not parse: {error.msg} (column {error.offset}).") from None
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"'{type(node).__name__}' is not allowed in an expression.")
    new_features = {}

    def feature_call(call):
        name = call.func.id
        block, fixed = ALIASES.get(name, (name, {}))
        if block not in BLOCKS or block in ("pattern", "python"):
            raise ValueError(f"'{name}' is not an indicator.")
        feature = block_defaults(block)
        feature.update(fixed)
        for key, argument in zip(_numeric_params(block), call.args):
            feature[key] = _constant(argument)
        if len(call.args) > len(_numeric_params(block)):
            raise ValueError(f"{name}() takes at most {len(_numeric_params(block))} numbers.")
        for keyword in call.keywords:
            if keyword.arg not in BLOCKS[block]["params"]:
                raise ValueError(f"{name}() has no setting '{keyword.arg}'.")
            feature[keyword.arg] = _constant(keyword.value)
        feature_name = auto_name(block, feature)
        existing = features.get(feature_name) or new_features.get(feature_name)
        if existing is not None and {k: value_of(v) for k, v in existing.items()} != feature:
            raise ValueError(f"A different feature is already called '{feature_name}'.")
        if existing is None:
            new_features[feature_name] = feature
        return feature_name

    def ref(node):
        shift = 0
        if isinstance(node, ast.Subscript):
            shift = _constant(node.slice)
            if not isinstance(shift, int) or shift < 0:
                raise ValueError("x[n] needs a whole number n of bars ago (0 or more).")
            node = node.value
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            out = {"const": _constant(node)}
        elif isinstance(node, ast.Name) and node.id in PRICE_FIELDS:
            out = {"price": node.id}
        elif isinstance(node, ast.Name):
            if node.id not in features and node.id not in new_features:
                raise ValueError(f"There is no feature called '{node.id}'.")
            out = {"feature": node.id}
        elif isinstance(node, ast.Attribute):
            base = node.value
            name = feature_call(base) if isinstance(base, ast.Call) else getattr(base, "id", None)
            if name is None or (name not in features and name not in new_features):
                raise ValueError(f"'{ast.unparse(base)}' is not a feature.")
            out = {"feature": name, "output": node.attr}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out = {"feature": feature_call(node)}
        else:
            raise ValueError(f"'{ast.unparse(node)}' is not a feature, a price or a number.")
        if shift:
            out["shift"] = shift
        return out

    def rule(node):
        if isinstance(node, ast.BoolOp):
            return {"all" if isinstance(node.op, ast.And) else "any": [rule(value) for value in node.values]}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return {"not": rule(node.operand)}
        if isinstance(node, ast.Compare):
            parts = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                parts.append({"left": ref(left), "op": OPERATORS[type(op)], "right": ref(right)})
                left = right
            return parts[0] if len(parts) == 1 else {"all": parts}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = node.args
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if name in ("crosses_above", "crosses_below") and len(args) == 2:
                return {"left": ref(args[0]), "op": name, "right": ref(args[1])}
            if name == "between" and len(args) == 3:
                return {"left": ref(args[0]), "op": "between", "lower": ref(args[1]), "upper": ref(args[2])}
            if name in ("rising", "falling") and len(args) in (1, 2):
                return {"left": ref(args[0]), "op": name, "bars": _constant(args[1]) if len(args) == 2 else 1}
            if name == "pattern" and len(args) == 1:
                leaf = {"pattern": _constant(args[0]), "module": "candle", "side": "either", "params": {}}
                for key, value in keywords.items():
                    if key in ("side", "module"):
                        leaf[key] = _constant(value)
                    elif isinstance(value, ast.Constant) and isinstance(value.value, bool):
                        leaf["params"][key] = value.value
                    else:
                        leaf["params"][key] = _constant(value)
                return leaf
            if name == "session" and len(args) == 2:
                return {"filter": "session", "from": _constant(args[0]), "to": _constant(args[1])}
            if name == "weekday":
                return {"filter": "weekday", "days": [_constant(arg) for arg in args]}
            if name == "month":
                return {"filter": "month", "months": [_constant(arg) for arg in args]}
            raise ValueError(f"'{name}(...)' is not a condition. Compare it with something, "
                             "or use crosses_above, between, rising, pattern ...")
        raise ValueError(f"'{ast.unparse(node)}' is not a condition.")

    return rule(tree.body), new_features


# ---------------------------------------------------------------------------
# Rule tree -> text
# ---------------------------------------------------------------------------

def _ref_text(ref, features):
    if "const" in ref:
        text = repr(value_of(ref["const"]))
    elif "price" in ref:
        text = ref["price"]
    else:
        name = ref["feature"]
        feature = features.get(name)
        text = name
        if feature is not None and feature.get("block") in BLOCKS and name == auto_name(feature["block"], feature):
            # Written back as the call it came from.
            block = feature["block"]
            numbers = ", ".join(repr(value_of(feature[key])) for key in _numeric_params(block) if key in feature)
            call = "ema" if block == "ma" and value_of(feature.get("ewm")) else block
            text = f"{call}({numbers})"
        if ref.get("output"):
            text += f".{ref['output']}"
    if ref.get("shift"):
        text += f"[{int(value_of(ref['shift']))}]"
    return text


def to_text(node, features, top=True):
    """Expression text for a rule tree (the inverse of parse, up to feature naming)."""
    if node is None:
        return ""
    if "choose" in node:
        raise ValueError("Alternatives have no text form; edit them in the builder.")
    for key, joiner in (("all", " and "), ("any", " or ")):
        if key in node:
            parts = [to_text(child, features, top=False) for child in node[key]]
            text = joiner.join(parts)
            return text if top or len(parts) == 1 else f"({text})"
    if "not" in node:
        return f"not {to_text(node['not'], features, top=False)}"
    if "pattern" in node:
        extras = [f"side={node.get('side', 'either')!r}", f"module={node.get('module', 'candle')!r}"]
        extras += [f"{key}={value_of(value)!r}" for key, value in (node.get("params") or {}).items()]
        return f"pattern({node['pattern']!r}, {', '.join(extras)})"
    if "filter" in node:
        kind = node["filter"]
        if kind == "session":
            return f"session({node['from']!r}, {node['to']!r})"
        if kind == "weekday":
            return f"weekday({', '.join(map(str, node['days']))})"
        if kind == "month":
            return f"month({', '.join(map(str, node['months']))})"
        raise ValueError("The higher-timeframe filter has no text form; edit it in the rule builder.")
    op = node["op"]
    left = _ref_text(node["left"], features)
    if node.get("shift"):
        raise ValueError("A whole-condition shift has no text form; shift the values instead (x[1]).")
    if op in ("crosses_above", "crosses_below"):
        return f"{op}({left}, {_ref_text(node['right'], features)})"
    if op == "between":
        return f"between({left}, {_ref_text(node['lower'], features)}, {_ref_text(node['upper'], features)})"
    if op in ("rising", "falling"):
        return f"{op}({left}, {int(value_of(node.get('bars', 1)))})"
    return f"{left} {op} {_ref_text(node['right'], features)}"
