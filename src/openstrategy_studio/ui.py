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
import html
import json
import re
import traceback
from importlib import resources
from pathlib import Path

import streamlit as st

from . import config, data as data_module, indicators, style
from .expressions import LOGIC_WORDS, append_letter, condition_text, next_letter, parse_logic, remove_letter, summary
from .spec import (COMPARISON_OPS, PATTERN_SIDES, PRICE_FIELDS, TIMEFRAME_MINUTES, default_spec, get_path,
                   is_choice, set_path)

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
    """Call first on every page. Returns (settings, data_dir).

    Also dresses the page in its theme (style.py): the stylesheet, the sidebar's logo,
    and the "page" chart theme, all from the theme file Streamlit itself started with.
    """
    st.set_page_config(page_title=f"{title} · OpenStrategy Studio", page_icon=":material/candlestick_chart:",
                       layout="wide")
    if "settings" not in st.session_state:
        st.session_state["settings"] = config.load_settings()
    settings = st.session_state["settings"]
    _apply_settings_once(json.dumps(settings, sort_keys=True))
    look, problems = style.active_theme(settings)
    # Style tags alone: Streamlit puts them where they take no space on the page.
    st.html(f"<style>{style.stylesheet(look)}</style>")
    st.logo(style.logo_svg(look), icon_image=style.logo_svg(look, mark_only=True), size="large")
    from . import analysis
    analysis.use_page_theme(look)
    if problems:
        st.warning("Page theme: " + " ".join(problems))
    data_dir = config.data_dir(settings)
    if "spec" not in st.session_state:
        st.session_state["spec"] = default_spec()
        st.session_state["spec_rev"] = 0
    return settings, data_dir


def page_help(page):
    """The sidebar's guide to this page, from help/<page>.md (packaged, so easy to edit)."""
    entry = resources.files("openstrategy_studio") / "help" / f"{page}.md"
    with st.sidebar.expander("How to use this page", expanded=False, icon=":material/help:"):
        st.markdown(entry.read_text(encoding="utf-8"))


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
    return any(feature.get("indicator") == "studio:python" for feature in (spec.get("features") or {}).values())


def _get(path, default):
    try:
        return get_path(current(), path)
    except (KeyError, IndexError, TypeError):
        return default


def renew_widgets():
    """New keys for every spec-bound widget, so each re-seeds from the spec on the next run.

    ⚠️ Needed after any structural change (a condition added or removed, a switch of
    kind): Streamlit keeps a keyed widget's own state and ignores its new value, so the
    logic line would keep its old text, and a reused letter would show a deleted
    condition's settings, and write them back.
    """
    st.session_state["spec_rev"] = st.session_state.get("spec_rev", 0) + 1


def rerun_after(change):
    """Make a structural change to the spec (add, remove, switch), save it and redraw."""
    change()
    save_draft()
    renew_widgets()
    st.rerun()


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


def glance(spec, problems):
    """The strategy at a glance: one compact, always-visible summary of every step, then the problems."""
    rows = []
    for label, text in summary(spec):
        indent = " os-glance-sub" if label.startswith(" ") else ""
        rows.append(f'<div class="os-glance-row{indent}"><span class="os-glance-label">{html.escape(label.strip())}'
                    f'</span><span class="os-glance-text">{html.escape(text)}</span></div>')
    status = (f'<div class="os-glance-ok">Ready to backtest</div>' if not problems else
              '<div class="os-glance-problems">' + "".join(f"<div>{html.escape(problem)}</div>"
                                                           for problem in problems[:8])
              + (f"<div>... and {len(problems) - 8} more</div>" if len(problems) > 8 else "") + "</div>")
    st.html(f'<div class="os-glance"><div class="os-glance-title">At a glance</div>{"".join(rows)}{status}</div>')


def chart_theme_options():
    """(chart theme names, {name: label}) in the selector's groups, for the pickers and Settings."""
    from . import analysis

    groups = analysis.theme_groups()
    options = [theme for names in groups.values() for theme in names]
    labels = {theme: f"{group}: {theme.replace('_', ' ')}" for group, names in groups.items() for theme in names}
    labels[analysis.PAGE_THEME] = "Match the page theme"
    return options, labels


def theme_picker(settings, prefix):
    """Chart theme selector in its groups, plus the glow checkbox for line figures."""
    from . import analysis

    options, labels = chart_theme_options()
    default = settings.get("default_theme", analysis.PAGE_THEME)
    left, right = st.columns([3, 1])
    theme = left.selectbox("Chart theme", options, index=options.index(default) if default in options else 0,
                           format_func=labels.get, key=f"{prefix}:theme")
    glow = right.checkbox("Glow", value=theme in analysis.GLOW_THEMES, key=f"{prefix}:glow",
                          help="Adds a glow to line charts; suits star wars, cyberpunk and synthwave.")
    return theme, glow


# ---------------------------------------------------------------------------
# Spec-bound inputs
# ---------------------------------------------------------------------------

def param_input(path, declaration, label=None, help=None, container=None):
    """One value bound to the spec: a number, a choice, a switch, text or code.

    A declaration with no default (pandas-ta leaves many to the library) shows as a
    blank box, and blank means "the library's own default": nothing is written.
    Optimisation ranges live on the Optimise page, not here.
    """
    where = container or st
    default = declaration.get("default")
    value = _get(path, default)
    kind = declaration["type"]
    label = label or declaration.get("label") or path.rsplit(".", 1)[-1].replace("_", " ").capitalize()
    if kind in ("int", "float") and value is None:
        text = where.text_input(label, value="", placeholder="the indicator's default", help=help, key=key(path))
        new = None
        if text.strip():
            try:
                new = int(text) if kind == "int" else float(text)
            except ValueError:
                where.error(f"{label}: type a {'whole ' if kind == 'int' else ''}number, or leave it blank.")
                return value
    elif kind == "int":
        new = int(where.number_input(label, value=int(value), step=1, help=help, key=key(path)))
    elif kind == "float":
        new = float(where.number_input(label, value=float(value), format="%g", help=help, key=key(path)))
    elif kind == "bool":
        new = bool(where.checkbox(label, value=bool(value), help=help, key=key(path)))
    elif kind == "choice":
        options = list(declaration["options"])
        if value is None:
            options = [None] + options
        new = where.selectbox(label, options, index=options.index(value) if value in options else 0, help=help,
                              key=key(path), format_func=lambda option: "the indicator's default" if option is None
                              else str(option))
    elif kind == "code":
        new = where.text_area(label, value=value or "", height=160, help=help, key=key(path))
    else:
        text = where.text_input(label, value="" if value is None else str(value), help=help, key=key(path))
        new = text if text != "" or value is not None else None
    if new != value and not (new is None and value is None):
        set_value(path, new)
    return new


def _output_options(features, exclude=None):
    """Every feature output as "name.output", for pickers."""
    options = []
    for name, feature in features.items():
        if name == exclude or feature.get("indicator") not in indicators.catalogue():
            continue
        options += [f"{name}.{output}" for output in indicators.outputs(feature["indicator"])]
    return options


def _output_label(features, option):
    """How an output reads in a picker: a feature with one output is just its name ("fast", not "fast.real")."""
    name, _, output = option.partition(".")
    if name in features and len(indicators.outputs(features[name]["indicator"])) == 1:
        return name
    return option


def input_select(path, features, own_name, label):
    """A feature input: a price field or another feature's output."""
    node = _get(path, "close")
    options = list(PRICE_FIELDS) + _output_options(features, exclude=own_name)
    if isinstance(node, dict):
        output = node.get("output") or (indicators.outputs(features[node["feature"]]["indicator"])[0]
                                        if node.get("feature") in features else "")
        current_label = f"{node['feature']}.{output}"
    else:
        current_label = node
    chosen = st.selectbox(label, options, index=options.index(current_label) if current_label in options else 0,
                          key=key(path), format_func=lambda option: _output_label(features, option))
    new = chosen if chosen in PRICE_FIELDS else dict(zip(("feature", "output"), chosen.split(".", 1)))
    if new != node:
        set_value(path, new)


def ref_input(path, features, label, allow_const=True):
    """One side of a comparison: a feature output, a price, or a number."""
    ref = _get(path, {"price": "close"})
    kinds = (["feature"] if features else []) + ["price"] + (["const"] if allow_const else [])
    names = {"feature": "Feature", "price": "Price", "const": "Number"}
    kind = next((name for name in ("feature", "price", "const") if name in ref), "price")
    chosen = st.selectbox(label, kinds, index=kinds.index(kind) if kind in kinds else 0, format_func=names.get,
                          key=key(path, ":kind"))
    if chosen != kind:
        first = _output_options(features)[:1]
        set_value(path, dict(zip(("feature", "output"), first[0].split(".", 1))) if chosen == "feature" and first
                  else {"price": "close"} if chosen == "price" else {"const": 0.0})
        renew_widgets()
        st.rerun()
    if chosen == "feature":
        options = _output_options(features)
        output = ref.get("output") or (indicators.outputs(features[ref["feature"]]["indicator"])[0]
                                       if ref.get("feature") in features else None)
        current_label = f"{ref.get('feature')}.{output}"
        picked = st.selectbox("Which", options, index=options.index(current_label) if current_label in options else 0,
                              key=key(path, ":feature"), label_visibility="collapsed",
                              format_func=lambda option: _output_label(features, option))
        name, output = picked.split(".", 1)
        new = {"feature": name}
        if len(indicators.outputs(features[name]["indicator"])) > 1:
            new["output"] = output
    elif chosen == "price":
        new = {"price": st.selectbox("Which", PRICE_FIELDS, index=PRICE_FIELDS.index(ref.get("price", "close")),
                                     key=key(path, ":price"), label_visibility="collapsed")}
    else:
        param_input(f"{path}.const", {"type": "float", "default": 0.0}, label="Value")
        return
    shift = st.number_input("Bars ago", min_value=0, step=1, value=int(ref.get("shift", 0) or 0),
                            key=key(path, ":shift"), help="0 is the current bar. History only: it never moves a "
                                                          "signal forward.")
    if shift:
        new["shift"] = int(shift)
    if new != ref:
        set_value(path, new)


# ---------------------------------------------------------------------------
# Features: the nested picker and each feature's settings
# ---------------------------------------------------------------------------

def feature_picker(features):
    """Add a feature: category, then family, then the indicator or pattern, with its outputs shown first."""
    menu = indicators.menu()
    columns = st.columns(3)
    # Plain keys, not spec-bound: the picker remembers where you were between adds.
    category = columns[0].selectbox("Category", list(menu), key="pick:category")
    families = menu[category]
    family = columns[1].selectbox("Family", list(families), key=f"pick:family:{category}",
                                  format_func=lambda name: f"{name} ({len(families[name])})")
    ids = families[family]
    picked = columns[2].selectbox("Indicator" if category in ("Indicator", "Statistical") else "Item", ids,
                                  format_func=lambda item_id: indicators.entry(item_id)["label"],
                                  key=f"pick:item:{category}:{family}")
    item = indicators.entry(picked)
    with st.container(border=True):
        st.markdown(f"**{item['label']}**  \n" + (item.get("description") or "")[:240])
        st.caption("Outputs: " + ", ".join(item["outputs"]) + (f"  ·  reads: {', '.join(item['inputs'])}"
                                                               if item["inputs"] else ""))
        stem = re.sub(r"[^a-z0-9]+", "_", item["function"].split(":")[-1].lower()).strip("_") or "feature"
        if not stem[0].isalpha():
            stem = "f_" + stem
        suggestion, number = stem, 2
        while suggestion in features:
            suggestion, number = f"{stem}_{number}", number + 1
        left, right = st.columns([3, 1])
        # Keyed on the suggestion too, so it moves on (sma, sma_2 ...) once a name is taken.
        name = left.text_input("Name it", value=suggestion, key=f"pick:name:{picked}:{suggestion}")
        right.write("")
        if right.button("Add feature", type="primary", key="pick:add"):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in PRICE_FIELDS:
                st.error("Use letters, digits and underscores for the name, starting with a letter, and not a price "
                         "name such as close.")
            elif name in features:
                st.error(f"There is already a feature called {name}.")
            else:
                rerun_after(lambda: features.__setitem__(name, indicators.new_feature(picked)))


def feature_editor(name, feature, features):
    """One feature's inputs and settings, its outputs, and a delete button."""
    item = indicators.catalogue().get(feature.get("indicator"))
    if item is None:
        st.error(f"{name} uses an unknown indicator ({feature.get('indicator')}). Delete it and add it again.")
    else:
        if item["id"] == "studio:python":
            st.warning("Python code runs on your machine. Only use code you trust.")
        if item.get("description"):
            st.caption(item["description"][:300])
        columns = st.columns(3)
        for index, input_name in enumerate(item["inputs"]):
            with columns[index % 3]:
                input_select(f"features.{name}.inputs.{input_name}", features, name,
                             "Applied to" if len(item["inputs"]) == 1 else f"Input: {input_name}")
        params = list(item["params"].items())
        columns = st.columns(3) if params else []
        for index, (param, declaration) in enumerate(params):
            with columns[index % 3]:
                param_input(f"features.{name}.params.{param}", declaration, label=param.replace("_", " "))
        st.caption(f"Outputs: {', '.join(f'{name}.{output}' for output in item['outputs'])}")
    if st.button("Delete feature", key=key(f"features.{name}", ":delete")):
        rerun_after(lambda: features.pop(name))


# ---------------------------------------------------------------------------
# Rules: lettered conditions and a logic line
# ---------------------------------------------------------------------------

def new_condition(kind, features):
    if kind == "pattern":
        patterns = [name for name, feature in features.items() if str(feature.get("indicator")).startswith("pattern:")]
        return {"pattern": patterns[0] if patterns else "", "side": "bullish"}
    if kind == "filter":
        return {"filter": "weekday", "days": [0, 1, 2, 3, 4]}
    first = _output_options(features)[:1]
    left = dict(zip(("feature", "output"), first[0].split(".", 1))) if first else {"price": "close"}
    if first and len(indicators.outputs(features[left["feature"]]["indicator"])) == 1:
        left.pop("output", None)
    return {"left": left, "op": ">", "right": {"const": 0.0}}


def new_rule(features):
    return {"conditions": {"A": new_condition("comparison", features)}, "logic": "A"}


def _pattern_condition(path, node, features):
    patterns = [name for name, feature in features.items() if str(feature.get("indicator")).startswith("pattern:")]
    if not patterns:
        st.info("Add a candlestick or chart pattern on the Features step first.")
        return
    left, right = st.columns(2)
    name = left.selectbox("Pattern", patterns, index=patterns.index(node.get("pattern")) if node.get("pattern")
                          in patterns else 0, key=key(path, ":pattern"))
    side = right.selectbox("Side", PATTERN_SIDES, index=PATTERN_SIDES.index(node.get("side", "either")),
                           key=key(path, ":side"))
    if (name, side) != (node.get("pattern"), node.get("side")):
        set_value(path, {"pattern": name, "side": side})
    item = indicators.entry(features[name]["indicator"])
    if item.get("direction") in ("bullish", "bearish") and side not in (item["direction"], "either"):
        st.warning(f"{item['label']} is a {item['direction']} pattern; as {side} it will never fire.")


def _comparison_condition(path, node, features):
    columns = st.columns([3, 2, 3])
    with columns[0]:
        ref_input(f"{path}.left", features, "When")
    with columns[1]:
        op = st.selectbox("Condition", COMPARISON_OPS, index=COMPARISON_OPS.index(node.get("op", ">")),
                          format_func=OP_LABELS.get, key=key(path, ":op"))
        if op != node.get("op"):
            # Keep the condition's own keys (a higher-timeframe filter keeps "filter" and
            # "timeframe"); swap the operands for the new operator's.
            changed = {name: value for name, value in node.items() if name not in ("right", "lower", "upper", "bars")}
            changed["op"] = op
            if op == "between":
                changed.update(lower={"const": 0.0}, upper={"const": 1.0})
            elif op in ("rising", "falling"):
                changed["bars"] = 1
            else:
                changed["right"] = node.get("right", {"const": 0.0})
            set_value(path, changed)
            renew_widgets()
            st.rerun()
    with columns[2]:
        if op == "between":
            ref_input(f"{path}.lower", features, "Lower bound")
            ref_input(f"{path}.upper", features, "Upper bound")
        elif op in ("rising", "falling"):
            param_input(f"{path}.bars", {"type": "int", "default": 1}, label="Bars")
        else:
            ref_input(f"{path}.right", features, "Than")


def _filter_condition(path, node, features, timeframe):
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
        renew_widgets()
        st.rerun()
    if kind == "weekday":
        days = st.multiselect("Trade on", list(range(7)), default=node.get("days", []),
                              format_func=DAY_NAMES.__getitem__, key=key(path, ":days"))
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
        chosen = st.selectbox("Timeframe", higher, index=higher.index(node["timeframe"]) if node.get("timeframe")
                              in higher else 0, key=key(path, ":tf"),
                              help="Computed on completed higher bars only, so it never peeks ahead.")
        if chosen != node.get("timeframe"):
            set_value(f"{path}.timeframe", chosen)
        _comparison_condition(path, node, features)


def condition_editor(path, letter, node, features, timeframe, entry, on_remove):
    """One lettered condition: what kind it is, its settings, and a remove button."""
    with st.container(border=True):
        kinds = ["comparison", "pattern"] + (["filter"] if entry else [])
        kind = "pattern" if "pattern" in node else "filter" if "filter" in node else "comparison"
        names = {"comparison": "Compare two lines", "pattern": "A pattern appears", "filter": "Only when (filter)"}
        head = st.columns([1, 5, 2])
        head[0].markdown(f"### {letter}")
        chosen = head[1].selectbox("Kind", kinds, index=kinds.index(kind) if kind in kinds else 0,
                                   format_func=names.get, key=key(path, ":kind"), label_visibility="collapsed")
        head[2].button("Remove", key=key(path, ":remove"), on_click=on_remove)
        if chosen != kind:
            set_value(path, new_condition(chosen, features))
            renew_widgets()
            st.rerun()
        if kind == "pattern":
            _pattern_condition(path, node, features)
        elif kind == "filter":
            _filter_condition(path, node, features, timeframe)
        else:
            _comparison_condition(path, node, features)
        st.caption(condition_text(_get(path, node)))


def _rule_body(path, rule, features, timeframe, entry):
    """The conditions and the logic line of one rule (one version of it)."""
    conditions = rule.setdefault("conditions", {})
    for letter in list(conditions):
        def remove(letter=letter):
            conditions.pop(letter, None)
            rule["logic"] = remove_letter(rule.get("logic", ""), letter)
            save_draft()
            renew_widgets()
        condition_editor(f"{path}.conditions.{letter}", letter, conditions[letter], features, timeframe, entry, remove)
    buttons = st.columns(4)
    adds = [("comparison", "Add a comparison"), ("pattern", "Add a pattern")] + ([("filter", "Add a filter")]
                                                                                  if entry else [])
    joiner = buttons[3].selectbox("Join new with", ["and", "or"], key=key(path, ":join"),
                                  label_visibility="collapsed", format_func=lambda op: f"join with {op}")
    for column, (kind, label) in zip(buttons, adds):
        if column.button(label, key=key(path, f":add:{kind}")):
            def add(kind=kind):
                letter = next_letter(set(conditions))
                conditions[letter] = new_condition(kind, features)
                rule["logic"] = append_letter(rule.get("logic", ""), letter, joiner)
            rerun_after(add)
    logic = st.text_input("Logic", value=rule.get("logic", ""), key=key(path, ":logic"),
                          help="Combine the letters with and, or, xor, nand, nor, xnor and not, using brackets: "
                               "(A and (B or C)) and not D. Brackets are needed wherever two different words meet.")
    if logic != rule.get("logic"):
        set_value(f"{path}.logic", logic)
    try:
        tree = parse_logic(logic)
    except ValueError as error:
        st.error(f"Logic: {error}")
    else:
        if isinstance(tree, dict) and "op" in tree:
            st.caption(f"Fires when {LOGIC_WORDS[tree['op']]} of its parts.")


def rule_editor(path, features, timeframe, entry):
    """A rule slot: its conditions and logic line, or several versions the optimiser can choose between."""
    node = _get(path, None)
    if node is None:
        return
    if is_choice(node):
        labels = list(node["options"])
        top = st.columns([2, 2, 2])
        chosen = top[0].selectbox("Version in use", labels, index=labels.index(node["choose"]),
                                  key=key(path, ":choose"),
                                  help="The optimiser can choose between versions: add rules." + path.split(".")[-1]
                                       + ".choose to the variables on the Optimise page.")
        if chosen != node["choose"]:
            set_value(f"{path}.choose", chosen)
            renew_widgets()
            st.rerun()
        if top[1].button("Add a version (a copy of this one)", key=key(path, ":copy")):
            def add_version():
                label = next_letter(set(labels)).lower()
                node["options"][label] = copy.deepcopy(node["options"][chosen])
                node["choose"] = label
            rerun_after(add_version)
        if len(labels) > 1 and top[2].button("Delete this version", key=key(path, ":drop")):
            def drop_version():
                del node["options"][chosen]
                node["choose"] = next(iter(node["options"]))
            rerun_after(drop_version)
        if len(labels) == 1 and top[2].button("Back to a single rule", key=key(path, ":single")):
            rerun_after(lambda: set_path(current(), path, node["options"][chosen]))
        _rule_body(f"{path}.options.{chosen}", node["options"][chosen], features, timeframe, entry)
        return
    _rule_body(path, node, features, timeframe, entry)
    if st.button("Make versions of this rule", key=key(path, ":versions"),
                 help="Keep this rule as version a and add others; the optimiser (BUTChC or Optuna) can then choose "
                      "between them."):
        rerun_after(lambda: set_path(current(), path, {"choose": "a", "options": {"a": copy.deepcopy(node)}}))


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
def _cached_frames(data_json, instruments_json, fx_json, setup_json, data_dir, mtimes):
    spec = {"data": json.loads(data_json), "instruments": json.loads(instruments_json),
            "account": {"fx": json.loads(fx_json)}, "setup": json.loads(setup_json)}
    return data_module.load_for_spec(spec, Path(data_dir))


def load_frames_for(spec, data_dir):
    """The run's data, cached on the data section, instruments and the parquet files' modification times."""
    return _cached_frames(json.dumps(spec["data"], sort_keys=True), json.dumps(spec.get("instruments") or {}, sort_keys=True),
                          json.dumps((spec.get("account") or {}).get("fx"), sort_keys=True),
                          json.dumps(spec.get("setup") or {}, sort_keys=True), str(data_dir),
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
