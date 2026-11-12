"""Shared Streamlit helpers. Pages are a thin layer: they read and write the one
spec dict and call functions from the other modules.

State rules (plan section 11):
* st.session_state["spec"] is the single source of truth.
* Every widget is seeded from the spec on every run and writes back through
  set_value(), so no widget holds its own copy. This also survives Streamlit
  dropping the state of widgets that were not rendered on a run (the classic
  way a multi-tab form loses input).
* Widget keys are built from the spec path plus a revision number. Replacing the
  spec (a template, a file, Apply to spec) bumps the revision, so every widget
  re-seeds instead of showing a stale value.
* Every edit also writes drafts/current.json, because session state vanishes on
  a browser refresh. Home offers to restore it.
"""

import copy
import json
import traceback
from pathlib import Path

import streamlit as st

from . import config, data as data_module
from .blocks import BLOCKS
from .spec import (COMPARISON_OPS, PRICE_FIELDS, TIMEFRAME_MINUTES, default_spec, get_path, is_choice, is_opt,
                   set_path, value_of)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
OP_LABELS = {">": "is above", "<": "is below", ">=": "is at or above", "<=": "is at or below",
             "crosses_above": "crosses above", "crosses_below": "crosses below", "between": "is between",
             "rising": "is rising over", "falling": "is falling over"}


# ---------------------------------------------------------------------------
# Start-up and state
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vectorbtX (the first start compiles numba code)")
def _apply_settings_once(settings_json):
    # Once per process (and again whenever the settings change).
    config.apply_settings(json.loads(settings_json))
    return True


def boot(title):
    """Call first on every page. Returns (settings, data_dir)."""
    st.set_page_config(page_title=f"{title} · OpenStrategy Studio", layout="wide")
    if "settings" not in st.session_state:
        st.session_state["settings"] = config.load_settings()
    settings = st.session_state["settings"]
    _apply_settings_once(json.dumps(settings, sort_keys=True))
    data_dir = config.data_dir(settings)
    if "spec" not in st.session_state:
        st.session_state["spec"] = default_spec()
        st.session_state["spec_rev"] = 0
    return settings, data_dir


def current():
    return st.session_state["spec"]


def key(path, suffix=""):
    """A widget key from the spec path: unique, stable, and renewed when the spec is replaced."""
    return f"r{st.session_state.get('spec_rev', 0)}:{path}{suffix}"


def save_draft():
    folder = config.data_dir(st.session_state["settings"]) / "drafts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "current.json").write_text(json.dumps(current(), indent=2), encoding="utf-8")


def set_value(path, value):
    set_path(current(), path, value)
    save_draft()


def delete_key(path):
    parent, _, last = path.rpartition(".")
    node = get_path(current(), parent) if parent else current()
    if isinstance(node, dict) and last in node:
        del node[last]
        save_draft()


def replace_spec(spec):
    """Swap in a whole new spec (template, file, JSON edit, Apply to spec)."""
    st.session_state["spec"] = spec
    st.session_state["spec_rev"] = st.session_state.get("spec_rev", 0) + 1
    st.session_state["untrusted_python"] = False
    save_draft()


def has_python_block(spec):
    return any(feature.get("block") == "python" for feature in (spec.get("features") or {}).values())


def _get(path, default):
    try:
        return get_path(current(), path)
    except (KeyError, IndexError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def chart(fig):
    # ⚠️ theme=None ALWAYS: the default theme="streamlit" overrides the template's
    # colours and defeats plotly-themes. width="stretch" replaces the deprecated
    # use_container_width (Streamlit 1.50+; confirmed on 1.67, where it is the default).
    st.plotly_chart(fig, theme=None, width="stretch", config={"displaylogo": False})


def table(frame, **options):
    st.dataframe(frame, width="stretch", hide_index=options.pop("hide_index", True), **options)


def font_css():
    """Inject plotly-themes' bundled fonts once per page that draws charts."""
    from . import analysis
    st.markdown(f"<style>{analysis.font_css()}</style>", unsafe_allow_html=True)


def show_problems(problems, heading="Fix these before running"):
    if problems:
        st.warning(heading + ":\n\n" + "\n".join(f"- {problem}" for problem in problems))


def run_guarded(label, fn):
    """The one try/except around a run: the message in an error box, the traceback in an expander.

    vectorbtX's own errors already name the offending argument, so the message is shown as it is.
    """
    with st.spinner(label):
        try:
            return fn()
        except Exception as error:   # noqa: BLE001
            st.error(f"{type(error).__name__}: {error}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc(), language="text")
            return None


def theme_picker(settings, prefix):
    """Theme selector in three groups, plus the glow checkbox for line figures."""
    from . import analysis

    groups = analysis.theme_groups()
    options = [theme for names in groups.values() for theme in names]
    labels = {theme: f"{group}: {theme.replace('_', ' ')}" for group, names in groups.items() for theme in names}
    default = settings.get("default_theme", "plotly_white")
    left, right = st.columns([3, 1])
    theme = left.selectbox("Chart theme", options, index=options.index(default) if default in options else 0,
                           format_func=labels.get, key=f"{prefix}:theme")
    glow = right.checkbox("Glow", value=theme in analysis.GLOW_THEMES, key=f"{prefix}:glow",
                          help="Adds a glow to line charts; suits star wars, cyberpunk and synthwave.")
    return theme, glow


# ---------------------------------------------------------------------------
# Spec-bound inputs
# ---------------------------------------------------------------------------

def _opt_controls(path, declaration, value, current_opt):
    """Min / max / step / log for numbers, allowed values for choices. Returns the opt dict."""
    kind = declaration["type"]
    if kind == "choice":
        allowed = st.multiselect("Values to try", declaration["options"],
                                 default=(current_opt or {}).get("values") or declaration["options"],
                                 key=key(path, ":opt:values"))
        return {"values": allowed}
    integer = kind == "int"
    low_default = declaration.get("min", value / 2 if value else 0)
    high_default = declaration.get("max", value * 2 if value else 1)
    opt = current_opt or {"min": low_default, "max": high_default}
    columns = st.columns(4)
    cast = int if integer else float
    number = {"step": 1} if integer else {"format": "%g"}
    low = columns[0].number_input("Min", value=cast(opt.get("min", low_default)), key=key(path, ":opt:min"), **number)
    high = columns[1].number_input("Max", value=cast(opt.get("max", high_default)), key=key(path, ":opt:max"),
                                   **number)
    step = columns[2].number_input("Grid step", value=cast(opt.get("step") or 0), min_value=cast(0),
                                   key=key(path, ":opt:step"), help="Grid backend only; 0 means ten points.",
                                   **number)
    out = {"min": cast(low), "max": cast(high)}
    if integer:
        out["int"] = True
    elif columns[3].checkbox("Log scale", value=bool(opt.get("log")), key=key(path, ":opt:log")):
        out["log"] = True
    if step:
        out["step"] = cast(step)
    return out


def param_input(path, declaration, label=None, help=None, optimisable=True):
    """One parameter bound to the spec, with an Optimise toggle beside every numeric
    or categorical field. Turning it on reveals the search range and stores the
    {"value", "opt"} node from plan section 5."""
    node = _get(path, declaration.get("default"))
    if declaration["type"] == "float" and is_opt(node) and node["opt"].get("int"):
        # The spec already searches this in whole numbers: keep it that way.
        declaration = dict(declaration, type="int")
    kind = declaration["type"]
    value = value_of(node)
    label = label or declaration.get("label") or path.rsplit(".", 1)[-1].replace("_", " ").capitalize()
    can_optimise = optimisable and kind in ("int", "float", "choice")
    if can_optimise:
        left, right = st.columns([4, 1])
    else:
        left, right = st.container(), None
    if kind == "int":
        new = left.number_input(label, value=int(value if value is not None else 0), step=1, help=help,
                                key=key(path))
    elif kind == "float":
        new = left.number_input(label, value=float(value if value is not None else 0.0), format="%g", help=help,
                                key=key(path))
    elif kind == "bool":
        new = left.checkbox(label, value=bool(value), help=help, key=key(path))
    elif kind == "choice":
        options = declaration["options"]
        new = left.selectbox(label, options, index=options.index(value) if value in options else 0, help=help,
                             key=key(path))
    elif kind == "code":
        new = left.text_area(label, value=value or "", height=160, help=help, key=key(path))
    else:
        new = left.text_input(label, value="" if value is None else str(value), help=help, key=key(path))
    result = new
    if can_optimise:
        right.write("")
        if right.toggle("Optimise", value=is_opt(node), key=key(path, ":opt")):
            previous = node["opt"] if is_opt(node) else None
            result = {"value": new, "opt": _opt_controls(path, declaration, new, previous)}
    if result != node:
        set_value(path, result)
    return result


def source_input(path, features, own_name):
    """A block's input: a price field or another feature's output."""
    node = _get(path, "close")
    options = list(PRICE_FIELDS)
    for name, feature in features.items():
        if name != own_name and feature.get("block") in BLOCKS:
            options += [f"{name}.{output}" for output in BLOCKS[feature["block"]]["outputs"]]
    if isinstance(node, dict):
        output = node.get("output") or BLOCKS[features[node["feature"]]["block"]]["outputs"][0]
        current_label = f"{node['feature']}.{output}"
    else:
        current_label = node
    chosen = st.selectbox("Input", options, index=options.index(current_label) if current_label in options else 0,
                          key=key(path))
    if chosen in PRICE_FIELDS:
        new = chosen
    else:
        name, output = chosen.split(".", 1)
        new = {"feature": name, "output": output}
    if new != node:
        set_value(path, new)


def ref_input(path, features, label):
    """One side of a comparison: a feature output, a price, or a number (optimisable)."""
    ref = _get(path, {"price": "close"})
    kinds = (["feature"] if features else []) + ["price", "const"]
    names = {"feature": "Feature", "price": "Price", "const": "Number"}
    kind = next((name for name in ("feature", "price", "const") if name in ref), "price")
    chosen = st.selectbox(label, kinds, index=kinds.index(kind) if kind in kinds else 0, format_func=names.get,
                          key=key(path, ":kind"))
    if chosen != kind:
        first = next(iter(features), None)
        set_value(path, {"feature": first} if chosen == "feature" else {"price": "close"} if chosen == "price"
                  else {"const": 0})
        st.rerun()
    if chosen == "feature":
        options = [f"{name}.{output}" for name, feature in features.items() if feature.get("block") in BLOCKS
                   for output in BLOCKS[feature["block"]]["outputs"]]
        output = ref.get("output") or (BLOCKS[features[ref["feature"]]["block"]]["outputs"][0]
                                       if ref.get("feature") in features else None)
        current_label = f"{ref.get('feature')}.{output}"
        picked = st.selectbox("Which", options, index=options.index(current_label) if current_label in options else 0,
                              key=key(path, ":feature"), label_visibility="collapsed")
        name, output = picked.split(".", 1)
        new = {"feature": name}
        if len(BLOCKS[features[name]["block"]]["outputs"]) > 1:
            new["output"] = output
    elif chosen == "price":
        new = {"price": st.selectbox("Which", PRICE_FIELDS, index=PRICE_FIELDS.index(ref.get("price", "close")),
                                     key=key(path, ":price"), label_visibility="collapsed")}
    else:
        param_input(f"{path}.const", {"type": "float", "default": 0.0}, label="Value")
        return
    shift = st.number_input("Bars ago", min_value=0, step=1, value=int(ref.get("shift", 0)), key=key(path, ":shift"),
                            help="0 is the current bar. History only: it never moves a signal forward.")
    if shift:
        new["shift"] = int(shift)
    if new != ref:
        set_value(path, new)


# ---------------------------------------------------------------------------
# Rule builder
# ---------------------------------------------------------------------------

def new_leaf(kind, features):
    first = next(iter(features), None)
    if kind == "pattern":
        return {"pattern": "hammer", "module": "candle", "side": "bullish", "params": {}}
    if kind == "filter":
        return {"filter": "weekday", "days": [0, 1, 2, 3, 4]}
    return {"left": {"feature": first} if first else {"price": "close"}, "op": ">", "right": {"const": 0}}


@st.cache_data(show_spinner=False)
def pattern_catalogue():
    from .blocks import pattern_catalogue as catalogue
    return catalogue()


def _pattern_editor(path, node):
    columns = st.columns(3)
    module = columns[0].selectbox("Family", ["candle", "chart"], index=["candle", "chart"].index(node.get("module", "candle")),
                                  format_func={"candle": "Candlestick", "chart": "Chart pattern"}.get,
                                  key=key(path, ":module"))
    catalogue = [row for row in pattern_catalogue() if row["module"] == module]
    names = [row["name"] for row in catalogue]
    name = columns[1].selectbox("Pattern", names, index=names.index(node["pattern"]) if node["pattern"] in names else 0,
                                key=key(path, ":pattern"))
    side = columns[2].selectbox("Side", ["bullish", "bearish", "either"],
                                index=["bullish", "bearish", "either"].index(node.get("side", "either")),
                                key=key(path, ":side"))
    row = next(item for item in catalogue if item["name"] == name)
    if (module, name, side) != (node.get("module"), node["pattern"], node.get("side")):
        set_value(path, {"pattern": name, "module": module, "side": side,
                         "params": {} if name != node["pattern"] else node.get("params", {})})
        st.rerun()
    if row["direction"] in ("bullish", "bearish") and side not in (row["direction"], "either"):
        st.warning(f"{name} is a {row['direction']} pattern; as {side} it will never fire.")
    with st.expander("Detector settings"):
        for keyword, default in row["params"].items():
            if isinstance(default, bool):
                declaration = {"type": "bool", "default": default}
            elif isinstance(default, int):
                declaration = {"type": "int", "default": default}
            elif isinstance(default, float):
                declaration = {"type": "float", "default": default}
            elif isinstance(default, str):
                declaration = {"type": "str", "default": default}
            else:
                continue
            param_input(f"{path}.params.{keyword}", declaration, label=keyword.replace("_", " "))


def _comparison_editor(path, node, features):
    columns = st.columns([3, 2, 3])
    with columns[0]:
        ref_input(f"{path}.left", features, "When")
    with columns[1]:
        op = st.selectbox("Condition", COMPARISON_OPS, index=COMPARISON_OPS.index(node.get("op", ">")),
                          format_func=OP_LABELS.get, key=key(path, ":op"))
        if op != node.get("op"):
            # Keep the leaf's own keys (a higher-timeframe filter keeps "filter" and
            # "timeframe"); swap the operands for the new operator's.
            leaf = {k: v for k, v in node.items() if k not in ("right", "lower", "upper", "bars")}
            leaf["op"] = op
            if op == "between":
                leaf.update(lower={"const": 0}, upper={"const": 1})
            elif op in ("rising", "falling"):
                leaf["bars"] = 1
            else:
                leaf["right"] = node.get("right", {"const": 0})
            set_value(path, leaf)
            st.rerun()
    with columns[2]:
        if op == "between":
            ref_input(f"{path}.lower", features, "Lower bound")
            ref_input(f"{path}.upper", features, "Upper bound")
        elif op in ("rising", "falling"):
            param_input(f"{path}.bars", {"type": "int", "default": 1, "min": 1}, label="Bars")
        else:
            ref_input(f"{path}.right", features, "Than")


def _filter_editor(path, node, features, timeframe):
    kinds = ["weekday", "month", "session", "higher_timeframe"]
    if TIMEFRAME_MINUTES.get(timeframe, 0) >= 1440:
        kinds.remove("session")          # time-of-day means nothing on daily bars
    labels = {"weekday": "Weekdays", "month": "Months", "session": "Trading session (UTC)",
              "higher_timeframe": "Higher timeframe condition"}
    kind = st.selectbox("Filter", kinds, index=kinds.index(node["filter"]) if node["filter"] in kinds else 0,
                        format_func=labels.get, key=key(path, ":filter"))
    if kind != node["filter"]:
        defaults = {"weekday": {"days": [0, 1, 2, 3, 4]}, "month": {"months": list(range(1, 13))},
                    "session": {"from": "07:00", "to": "16:00"},
                    "higher_timeframe": {"timeframe": "1w", "left": {"price": "close"}, "op": ">",
                                         "right": {"price": "open"}}}
        set_value(path, {"filter": kind, **defaults[kind]})
        st.rerun()
    if kind == "weekday":
        days = st.multiselect("Trade on", list(range(7)), default=node.get("days", []), format_func=DAY_NAMES.__getitem__,
                              key=key(path, ":days"))
        if days != node.get("days"):
            set_value(f"{path}.days", days)
    elif kind == "month":
        months = st.multiselect("Trade in", list(range(1, 13)), default=node.get("months", []),
                                format_func=lambda month: MONTH_NAMES[month - 1], key=key(path, ":months"))
        if months != node.get("months"):
            set_value(f"{path}.months", months)
    elif kind == "session":
        left, right = st.columns(2)
        start = left.text_input("From (HH:MM)", value=node.get("from", "07:00"), key=key(path, ":from"))
        end = right.text_input("To (HH:MM)", value=node.get("to", "16:00"), key=key(path, ":to"))
        if (start, end) != (node.get("from"), node.get("to")):
            set_value(f"{path}.from", start)
            set_value(f"{path}.to", end)
    else:
        higher = [tf for tf in ["1h", "4h", "1d", "1w", "1M"] if TIMEFRAME_MINUTES[tf] > TIMEFRAME_MINUTES.get(timeframe, 0)]
        chosen = st.selectbox("Timeframe", higher, index=higher.index(node["timeframe"]) if node.get("timeframe") in higher else 0,
                              key=key(path, ":tf"), help="Computed on completed higher bars only, so it never peeks ahead.")
        if chosen != node.get("timeframe"):
            set_value(f"{path}.timeframe", chosen)
        _comparison_editor(path, node, features)


def rule_editor(path, features, timeframe, entry, depth=0):
    """A nested rule builder: groups joined by all / any / not, up to three levels deep.

    One recursive function. Widget keys come from the node's path in the tree, so
    they stay unique and stable."""
    node = _get(path, None)
    with st.container(border=True):
        if is_choice(node):
            _choice_editor(path, node, features, timeframe, entry, depth)
            return
        if node is not None and ("all" in node or "any" in node or "not" in node):
            joiner = "all" if "all" in node else "any" if "any" in node else "not"
            labels = {"all": "All of these are true", "any": "Any of these is true", "not": "This is NOT true"}
            top_left, top_right = st.columns([4, 1])
            chosen = top_left.selectbox("Group", ["all", "any", "not"], index=["all", "any", "not"].index(joiner),
                                        format_func=labels.get, key=key(path, ":group"), label_visibility="collapsed")
            children = node["not"] if joiner == "not" else node[joiner]
            children = [children] if joiner == "not" else children
            if chosen != joiner:
                set_value(path, {"not": children[0]} if chosen == "not" else {chosen: children})
                st.rerun()
            if depth and top_right.button("Remove group", key=key(path, ":remove")):
                _remove(path)
            for index in range(len(children)):
                child = f"{path}.not" if joiner == "not" else f"{path}.{joiner}.{index}"
                rule_editor(child, features, timeframe, entry, depth + 1)
            if joiner != "not":
                buttons = st.columns(4)
                kinds = [("comparison", "Add condition"), ("pattern", "Add pattern")]
                if entry:
                    kinds.append(("filter", "Add filter"))
                for column, (kind, label) in zip(buttons, kinds):
                    if column.button(label, key=key(path, f":add:{kind}")):
                        node[joiner].append(new_leaf(kind, features))
                        save_draft()
                        st.rerun()
                if depth < 2 and buttons[3].button("Add group", key=key(path, ":add:group")):
                    node[joiner].append({"any": [new_leaf("comparison", features)]})
                    save_draft()
                    st.rerun()
            if depth < 2 and st.button("Turn into alternatives", key=key(path, ":alternatives"),
                                       help="Keep this rule as one option and add others; the optimiser "
                                            "(BUTChC or Optuna) can then choose between them."):
                set_value(path, {"choose": "a", "options": {"a": copy.deepcopy(node)}})
                st.rerun()
            return
        if node is None:
            return
        header, remove = st.columns([4, 1])
        kind = "pattern" if "pattern" in node else "filter" if "filter" in node else "comparison"
        header.caption({"pattern": "Pattern", "filter": "Filter", "comparison": "Condition"}[kind])
        if depth and remove.button("Remove", key=key(path, ":remove")):
            _remove(path)
        if kind == "pattern":
            _pattern_editor(path, node)
        elif kind == "filter":
            _filter_editor(path, node, features, timeframe)
        else:
            _comparison_editor(path, node, features)


def _choice_editor(path, node, features, timeframe, entry, depth):
    """Structural alternatives: one rule tree per option, and which one is used now."""
    labels = list(node["options"])
    st.caption("Alternatives: the strategy uses one of these. Turn on Optimise to let BUTChC or Optuna choose.")
    param_input(f"{path}.choose", {"type": "choice", "options": labels, "default": labels[0]}, label="Use")
    for label in labels:
        with st.expander(f"Alternative {label}", expanded=label == value_of(node["choose"])):
            rule_editor(f"{path}.options.{label}", features, timeframe, entry, depth + 1)
            if len(labels) > 1 and st.button("Remove this alternative", key=key(path, f":drop:{label}")):
                del node["options"][label]
                remaining = list(node["options"])
                selector = node["choose"]
                if is_opt(selector):
                    selector["opt"]["values"] = [value for value in selector["opt"]["values"] if value in remaining]
                    if selector["value"] not in remaining:
                        selector["value"] = remaining[0]
                elif selector not in remaining:
                    node["choose"] = remaining[0]
                save_draft()
                st.rerun()
    left, middle, right = st.columns([2, 1, 2])
    new_label = left.text_input("New alternative", value=f"option_{len(labels) + 1}", key=key(path, ":new_label"),
                                label_visibility="collapsed")
    if middle.button("Add", key=key(path, ":add_alternative")):
        if not new_label.isidentifier() or new_label in labels:
            st.error("Name the alternative with letters, digits and underscores, different from the others.")
        else:
            node["options"][new_label] = copy.deepcopy(node["options"][value_of(node["choose"])])
            if is_opt(node["choose"]):
                node["choose"]["opt"]["values"].append(new_label)
            save_draft()
            st.rerun()
    if right.button("Keep only the one in use", key=key(path, ":collapse")):
        set_value(path, copy.deepcopy(node["options"][value_of(node["choose"])]))
        st.rerun()


def _remove(path):
    parent, _, last = path.rpartition(".")
    container = get_path(current(), parent)
    if isinstance(container, list):
        del container[int(last)]
    else:
        # The only child of a "not": the parent group goes with it.
        grand, _, slot = parent.rpartition(".")
        holder = get_path(current(), grand)
        if isinstance(holder, list):
            del holder[int(slot)]
    save_draft()
    st.rerun()


# ---------------------------------------------------------------------------
# Data helpers for pages
# ---------------------------------------------------------------------------

def data_mtimes(spec, data_dir):
    stamps = []
    for symbol in spec["data"]["symbols"]:
        path = data_module.price_path(data_dir, symbol, spec["data"]["timeframe"])
        stamps.append(path.stat().st_mtime if path.exists() else None)
    return tuple(stamps)


@st.cache_data(max_entries=16, show_spinner=False)
def _cached_frames(data_json, instruments_json, fx_json, data_dir, mtimes):
    spec = {"data": json.loads(data_json), "instruments": json.loads(instruments_json),
            "account": {"fx": json.loads(fx_json)}}
    return data_module.load_for_spec(spec, Path(data_dir))


def load_frames_for(spec, data_dir):
    """The run's data, cached on the data section, instruments and the parquet files' modification times."""
    return _cached_frames(json.dumps(spec["data"], sort_keys=True), json.dumps(spec.get("instruments") or {}, sort_keys=True),
                          json.dumps((spec.get("account") or {}).get("fx"), sort_keys=True), str(data_dir),
                          data_mtimes(spec, data_dir))


def missing_data(spec, data_dir):
    """Symbols the spec needs that are not stored at its timeframe or any finer one."""
    from .spec import TIMEFRAMES
    timeframe = spec["data"]["timeframe"]
    needed = list(spec["data"]["symbols"])
    fx = (spec.get("account") or {}).get("fx")
    if fx and fx.get("symbol"):
        needed.append(fx["symbol"])
    usable = [tf for tf in TIMEFRAMES if TIMEFRAME_MINUTES[tf] <= TIMEFRAME_MINUTES[timeframe]]
    return [symbol for symbol in needed
            if not any(data_module.price_path(data_dir, symbol, tf).exists() for tf in usable)]


def stored_symbols(data_dir):
    from . import store
    catalogue = store.list_datasets(data_dir)
    return sorted(catalogue["symbol"].unique()) if not catalogue.empty else []


def spec_copy():
    return copy.deepcopy(current())
