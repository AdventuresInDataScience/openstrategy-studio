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
import os
import re
import threading
import traceback
from importlib import resources
from pathlib import Path

import streamlit as st

from . import config, data as data_module, indicators, style
from .expressions import (LOGIC_WORDS, append_letter, condition_name_problem, condition_text, next_letter, parse_logic,
                          remove_letter, summary)
from .spec import (COMPARISON_OPS, HIGHER_TIMEFRAMES, PATTERN_SIDES, PRICE_FIELDS, TIMEFRAME_MINUTES, TIMING_KEYS,
                   default_spec, default_variable, feature_name_problem, get_path, is_choice, rename_condition,
                   rename_feature, set_path, tunable_inputs)

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


# The start-up warm-up's state, once it has been started: {"done": threading.Event, "error": text or None}.
WARM_UP = {}


@st.cache_resource(show_spinner=False)
def _start_warm_up():
    """Compile numba's code in a background thread, once per Studio start (engine.warm_up).

    The thread never touches Streamlit. Its own try/except is the exception to "one per
    page": it runs outside any page, and a failure there costs nothing but the head start
    (the first real run compiles what it needs anyway), so it is recorded and shown as a
    note, never raised.
    """
    state = {"done": threading.Event(), "error": None}

    def work():
        from . import engine
        try:
            engine.warm_up()
        except Exception as error:   # noqa: BLE001  (best effort; see the docstring)
            state["error"] = f"{type(error).__name__}: {error}"
        finally:
            state["done"].set()

    threading.Thread(target=work, name="openstrategy-warm-up", daemon=True).start()
    WARM_UP.update(state)
    return True


def warming_up():
    """True while the start-up warm-up is still compiling."""
    return bool(WARM_UP) and not WARM_UP["done"].is_set()


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
    # OPENSTRATEGY_STUDIO_WARM_UP=0 turns it off for one process (the tests set it).
    if settings.get("warm_up", True) and os.environ.get("OPENSTRATEGY_STUDIO_WARM_UP", "1") != "0":
        _start_warm_up()
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


def prune_variables():
    """Drop optimise variables whose value the strategy no longer has.

    Called by any change that removes a value: a rule or condition deleted, a condition's
    kind or comparison switched, a side switched from a number to a feature, a stop turned
    into another kind. Without it the variable would stay behind, and validate_spec would
    say so until it was removed by hand.
    """
    known = {row["path"] for row in tunable_inputs(current())}
    found = (current().get("optimise") or {}).get("variables") or {}
    for path in [path for path in found if path not in known]:
        del found[path]


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
                           format_func=labels.get, key=f"{prefix}:theme",
                           help="The colours and fonts of the charts on this page. Match the page theme follows the "
                                "page's own colours; the default is set on the Settings page.")
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
    """A feature input: a price field or another feature's output.

    Its tooltip says what can be picked; the entry's own description sits above the inputs.
    """
    node = _get(path, "close")
    options = list(PRICE_FIELDS) + _output_options(features, exclude=own_name)
    if isinstance(node, dict):
        output = node.get("output") or (indicators.outputs(features[node["feature"]]["indicator"])[0]
                                        if node.get("feature") in features else "")
        current_label = f"{node['feature']}.{output}"
    else:
        current_label = node
    chosen = st.selectbox(label, options, index=options.index(current_label) if current_label in options else 0,
                          key=key(path), format_func=lambda option: _output_label(features, option),
                          help="The line this is computed on: a price, or another feature's output, so features "
                               "can be chained (an RSI of an EMA).")
    new = chosen if chosen in PRICE_FIELDS else dict(zip(("feature", "output"), chosen.split(".", 1)))
    if new != node:
        set_value(path, new)


def ref_input(path, features, label, allow_const=True, help=None):
    """One side of a comparison: a feature output, a price, or a number.

    `help` is the tooltip on the first box, which says what this side is for; it defaults
    to what the three kinds are.
    """
    ref = _get(path, {"price": "close"})
    kinds = (["feature"] if features else []) + ["price"] + (["const"] if allow_const else [])
    names = {"feature": "Feature", "price": "Price", "const": "Number"}
    kind = next((name for name in ("feature", "price", "const") if name in ref), "price")
    chosen = st.selectbox(label, kinds, index=kinds.index(kind) if kind in kinds else 0, format_func=names.get,
                          key=key(path, ":kind"),
                          help=help or "A feature's output, a price, or a fixed number. Pick which, then which one "
                                       "in the box below.")
    if chosen != kind:
        first = _output_options(features)[:1]
        set_value(path, dict(zip(("feature", "output"), first[0].split(".", 1))) if chosen == "feature" and first
                  else {"price": "close"} if chosen == "price" else {"const": 0.0})
        # A number's variable goes when the side reads a line instead, and bars ago's the other way.
        prune_variables()
        save_draft()
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
        param_input(f"{path}.const", {"type": "float", "default": 0.0}, label="Value",
                    help="A fixed number, such as 30 for an RSI level. It can be optimised on the Optimise page.")
        return
    shift = st.number_input("Bars ago", min_value=0, step=1, value=int(ref.get("shift", 0) or 0),
                            key=key(path, ":shift"), help="0 is the current bar. History only: it never moves a "
                                                          "signal forward. It can be optimised, from 0 up, with the "
                                                          "condition's Optimise list.")
    # A shift of 0 that a trial wrote (Apply to the strategy) stays: rendering must not change the spec.
    if shift or "shift" in ref:
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
    category = columns[0].selectbox("Category", list(menu), key="pick:category",
                                    help="Indicators, statistics, price levels, candlestick and chart patterns, "
                                         "transforms of other lines, and your own Python code.")
    families = menu[category]
    family = columns[1].selectbox("Family", list(families), key=f"pick:family:{category}",
                                  format_func=lambda name: f"{name} ({len(families[name])})",
                                  help="A group within the category. The number is how many it holds.")
    ids = families[family]
    picked = columns[2].selectbox("Indicator" if category in ("Indicator", "Statistical") else "Item", ids,
                                  format_func=lambda item_id: indicators.entry(item_id)["label"],
                                  key=f"pick:item:{category}:{family}",
                                  help="The one to add. Its description, outputs and the prices it reads show "
                                       "below; its settings can be changed once it is added.")
    item = indicators.entry(picked)
    with st.container(border=True):
        st.markdown(f"**{item['label']}**  \n" + (item.get("description") or "")[:400])
        # Its settings and their defaults, so "over n bars" shows its n before it is added;
        # each can be changed on the feature's card once added.
        settings = []
        for name, declaration in item["params"].items():
            if declaration["type"] == "code":
                continue
            default = declaration.get("default")
            settings.append(f"{name} {'(library default)' if default is None else default}")
        reads = [f"{name} (default {item['defaults'].get(name) or 'close'})" for name in item["inputs"]]
        st.caption("Outputs: " + ", ".join(item["outputs"])
                   + (f"  ·  reads: {', '.join(reads)}" if reads else "")
                   + (f"  ·  settings: {', '.join(settings)}" if settings else "  ·  no settings"))
        stem = re.sub(r"[^a-z0-9]+", "_", item["function"].split(":")[-1].lower()).strip("_") or "feature"
        if not stem[0].isalpha():
            stem = "f_" + stem
        suggestion, number = stem, 2
        while suggestion in features:
            suggestion, number = f"{stem}_{number}", number + 1
        left, right = st.columns([3, 1])
        # Keyed on the suggestion too, so it moves on (sma, sma_2 ...) once a name is taken.
        name = left.text_input("Name it", value=suggestion, key=f"pick:name:{picked}:{suggestion}",
                               help="How the rules refer to it, such as fast or rsi. Letters, digits and "
                                    "underscores, starting with a letter.")
        right.write("")
        if right.button("Add feature", type="primary", key="pick:add"):
            problem = feature_name_problem(name, set(features))
            if problem:
                st.error(problem)
            else:
                rerun_after(lambda: features.__setitem__(name, indicators.new_feature(picked)))


def feature_editor(name, feature, features):
    """One feature's name, inputs and settings, its outputs, and a delete button."""
    typed = st.columns([1, 2])[0].text_input(
        "Name", value=name, key=key(f"features.{name}", ":name"),
        help="How the rules and other features refer to it, such as fast or rsi: letters, digits and underscores, "
             "starting with a letter. Renaming it renames every use too: the rules, features that read it, the "
             "selection score, the entry priority and the Optimise page's variables.")
    if typed.strip() != name:
        problem = feature_name_problem(typed.strip(), set(features) - {name})
        if problem:
            # A refused name stays in the box with the reason under it; nothing changes.
            st.error(problem)
        else:
            def rename():
                rename_feature(current(), name, typed.strip())
                # Its panel is keyed by name, so it would come back closed: the Build page opens it.
                st.session_state["renamed_feature"] = typed.strip()
            # A structural change: every widget key under the old name must be renewed.
            rerun_after(rename)
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
                param_input(f"features.{name}.params.{param}", declaration, label=param.replace("_", " "),
                            help=indicators.param_help(item["id"], param))
        st.caption(f"Outputs: {', '.join(f'{name}.{output}' for output in item['outputs'])}")
        # A single-type average (SMA, EMA ...) can become the Moving average entry at the same
        # type, which gives the same numbers, so its type can be optimised like any setting.
        kind = indicators.SAME_AS_MA.get(item["id"])
        if kind:
            tries = [kind] + [other for other in ("SMA", "EMA") if other != kind]
            if st.button("Let the optimiser pick the type", key=key(f"features.{name}", ":any_type"),
                         help=f"Swaps this {kind} for the Moving average entry set to {kind}, which gives exactly "
                              "the same numbers, and adds its type to the Optimise page's variables with "
                              f"{' and '.join(tries)} to try (add others there). The period, the inputs and the "
                              "rules stay as they are."):
                def swap():
                    feature["indicator"] = "talib:MA"
                    feature.setdefault("params", {})["matype"] = kind
                    # Written directly, not through set_path: the variable's key holds dots.
                    found = current().setdefault("optimise", {}).setdefault("variables", {})
                    found.setdefault(f"features.{name}.params.matype", {"values": tries})
                rerun_after(swap)
    if st.button("Delete feature", key=key(f"features.{name}", ":delete")):
        rerun_after(lambda: features.pop(name))


# ---------------------------------------------------------------------------
# Rules: lettered conditions and a logic line
# ---------------------------------------------------------------------------

# The kinds a condition can be, in the Kind list of every rule (exits included: see compile_signals).
# Calendar and higher-timeframe conditions are both {"filter": ...} in the spec.
CONDITION_KINDS = {"comparison": "Compare two lines", "pattern": "A pattern appears",
                   "calendar": "Day, month or time of day", "higher": "On a higher timeframe"}
TIMING_LABELS = {"now": "On this bar", "for_bars": "On each of the last N bars", "within_bars": "On any of the last N bars",
                 "bars_ago": "N bars ago"}


def condition_kind(node):
    """Which entry of CONDITION_KINDS a condition is."""
    if "pattern" in node:
        return "pattern"
    if node.get("filter") == "higher_timeframe":
        return "higher"
    if "filter" in node:
        return "calendar"
    return "comparison"


def _higher_than(timeframe):
    return [tf for tf in HIGHER_TIMEFRAMES if TIMEFRAME_MINUTES[tf] > TIMEFRAME_MINUTES.get(timeframe, 0)]


def new_condition(kind, features, timeframe="1d"):
    if kind == "pattern":
        patterns = [name for name, feature in features.items() if str(feature.get("indicator")).startswith("pattern:")]
        return {"pattern": patterns[0] if patterns else "", "side": "bullish"}
    if kind == "calendar":
        return {"filter": "weekday", "days": [0, 1, 2, 3, 4]}
    if kind == "higher":
        higher = _higher_than(timeframe) or ["1M"]
        return {"filter": "higher_timeframe", "timeframe": higher[0], "left": {"price": "close"}, "op": ">",
                "right": {"price": "open"}}
    first = _output_options(features)[:1]
    left = dict(zip(("feature", "output"), first[0].split(".", 1))) if first else {"price": "close"}
    if first and len(indicators.outputs(features[left["feature"]]["indicator"])) == 1:
        left.pop("output", None)
    return {"left": left, "op": ">", "right": {"const": 0.0}}


def new_rule(features):
    return {"conditions": {"A": new_condition("comparison", features)}, "logic": "A"}


def open_panel(path):
    """Have a condition's panel open the next time it is drawn (a new or renamed condition)."""
    st.session_state.setdefault("open_panels", set()).add(path)


def _pattern_condition(path, node, features):
    patterns = [name for name, feature in features.items() if str(feature.get("indicator")).startswith("pattern:")]
    if not patterns:
        st.info("Add a candlestick or chart pattern on the Features step first.")
        return
    left, right = st.columns(2)
    name = left.selectbox("Pattern", patterns, index=patterns.index(node.get("pattern")) if node.get("pattern")
                          in patterns else 0, key=key(path, ":pattern"),
                          help="One of the pattern features added on the Features tab. It counts on the bar that "
                               "completes the pattern; the trade fills at the next bar's open.")
    side = right.selectbox("Side", PATTERN_SIDES, index=PATTERN_SIDES.index(node.get("side", "either")),
                           key=key(path, ":side"),
                           help="bullish: only when the pattern points up. bearish: only when it points down. "
                                "either: whichever way it points.")
    if (name, side) != (node.get("pattern"), node.get("side")):
        set_value(path, {"pattern": name, "side": side})
    item = indicators.entry(features[name]["indicator"])
    if item.get("direction") in ("bullish", "bearish") and side not in (item["direction"], "either"):
        st.warning(f"{item['label']} is a {item['direction']} pattern; as {side} it will never fire.")


def _comparison_condition(path, node, features):
    columns = st.columns([3, 2, 3])
    with columns[0]:
        ref_input(f"{path}.left", features, "When",
                  help="The line being tested: a feature's output or a price.")
    with columns[1]:
        op = st.selectbox("Condition", COMPARISON_OPS, index=COMPARISON_OPS.index(node.get("op", ">")),
                          format_func=OP_LABELS.get, key=key(path, ":op"),
                          help="Above and below hold on every bar they are true. Crosses fires only on the bar "
                               "the line moves from one side to the other. Between needs two bounds. Rising and "
                               "falling compare the line with itself some bars ago.")
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
            prune_variables()
            save_draft()
            renew_widgets()
            st.rerun()
    with columns[2]:
        if op == "between":
            ref_input(f"{path}.lower", features, "Lower bound", help="The line must be at or above this.")
            ref_input(f"{path}.upper", features, "Upper bound", help="The line must be at or below this.")
        elif op in ("rising", "falling"):
            param_input(f"{path}.bars", {"type": "int", "default": 1}, label="Bars",
                        help="Compared with its value this many bars ago: rising means higher than then.")
        else:
            ref_input(f"{path}.right", features, "Than",
                      help="What the line is compared with: another feature, a price, or a fixed number.")


def _calendar_condition(path, node, timeframe):
    """A weekday, month or time-of-day condition: the same on every bar of its day, so no timing."""
    kinds = ["weekday", "month", "session"]
    if TIMEFRAME_MINUTES.get(timeframe, 0) >= 1440:
        kinds.remove("session")          # time-of-day means nothing on daily bars
    labels = {"weekday": "On chosen weekdays", "month": "In chosen months", "session": "In a time-of-day session (UTC)"}
    kind = st.selectbox("Which", kinds, index=kinds.index(node["filter"]) if node["filter"] in kinds else 0,
                        format_func=labels.get, key=key(path, ":filter"),
                        help="True on the weekdays or in the months you pick, or inside a time-of-day session "
                             "(intraday bars only). In an entry rule joined with and, it allows entries only then; "
                             "in an exit rule joined with or, it closes a position then (a Friday exit).")
    if kind != node["filter"]:
        defaults = {"weekday": {"days": [0, 1, 2, 3, 4]}, "month": {"months": list(range(1, 13))},
                    "session": {"from": "07:00", "to": "16:00"}}
        set_value(path, {"filter": kind, **defaults[kind]})
        renew_widgets()
        st.rerun()
    if kind == "weekday":
        days = st.multiselect("Weekdays", list(range(7)), default=node.get("days", []),
                              format_func=DAY_NAMES.__getitem__, key=key(path, ":days"),
                              help="The weekday of the bar the condition is read on (UTC). The order fills at the "
                                   "next bar's open, which may fall on the next day.")
        if days != node.get("days"):
            set_value(f"{path}.days", days)
    elif kind == "month":
        months = st.multiselect("Months", list(range(1, 13)), default=node.get("months", []),
                                format_func=lambda month: MONTH_NAMES[month - 1], key=key(path, ":months"),
                                help="The month of the bar the condition is read on.")
        if months != node.get("months"):
            set_value(f"{path}.months", months)
    else:
        left, right = st.columns(2)
        start = left.text_input("From (HH:MM)", value=node.get("from", "07:00"), key=key(path, ":from"),
                                help="The session's start, in UTC, 24-hour clock: a bar that starts at or after it "
                                     "counts.")
        end = right.text_input("To (HH:MM)", value=node.get("to", "16:00"), key=key(path, ":to"),
                               help="The session's end, in UTC, 24-hour clock: a bar that starts at it no longer "
                                    "counts. It may wrap past midnight (22:00 to 02:00).")
        if (start, end) != (node.get("from"), node.get("to")):
            set_value(f"{path}.from", start)
            set_value(f"{path}.to", end)


def _higher_condition(path, node, features, timeframe):
    higher = _higher_than(timeframe)
    chosen = st.selectbox("Timeframe", higher, index=higher.index(node["timeframe"]) if node.get("timeframe")
                          in higher else 0, key=key(path, ":tf"),
                          help="The comparison below is computed on these bars, completed ones only, so it never "
                               "peeks ahead: on a Wednesday a weekly condition reads last week.")
    if chosen != node.get("timeframe"):
        set_value(f"{path}.timeframe", chosen)
    _comparison_condition(path, node, features)


def _timing_input(path, node, higher=False):
    """When the condition must have held: now, on each of the last N bars, on any of them, or N bars ago.

    Bound to the condition's own TIMING_KEYS (one at most), so the bars can be optimised
    like any other number.
    """
    current = next((name for name in TIMING_KEYS if node.get(name)), "now")
    options = ["now"] + TIMING_KEYS
    columns = st.columns([3, 1])
    unit = "higher-timeframe bars" if higher else "bars"
    chosen = columns[0].selectbox(
        "Must hold", options, index=options.index(current), format_func=TIMING_LABELS.get, key=key(path, ":timing"),
        help=f"On this bar: as it is now. On each of the last N {unit}: true on this bar and the N - 1 before it, "
             f"such as RSI above 50 for 3 bars in a row. On any of the last N {unit}: true at least once in that "
             f"window, this bar included, such as a cross within the last 5 bars. N {unit} ago: the whole condition "
             "as it was then, such as a cross 1 bar ago. Each reads past bars only. To let the optimiser search N, "
             "pick one of the last three and add N in the Optimise list below: on each of, or any of, the last 1 "
             "bar is the same as on this bar, so a search from 1 includes it.")
    if chosen != current:
        changed = {name: value for name, value in node.items() if name not in TIMING_KEYS}
        if chosen != "now":
            changed[chosen] = int(node.get(current) or (1 if chosen == "bars_ago" else 3))
        set_value(path, changed)
        prune_variables()
        save_draft()
        renew_widgets()
        st.rerun()
    if chosen != "now":
        bars = columns[1].number_input("N", min_value=1, step=1, value=int(node[chosen]), key=key(f"{path}.{chosen}"),
                                       help="How many bars. It can be optimised: add it in the Optimise list below.")
        if int(bars) != node[chosen]:
            set_value(f"{path}.{chosen}", int(bars))


def condition_editor(path, name, node, features, timeframe, on_remove, on_rename):
    """One named condition in its own panel: its name, kind, settings, timing and a remove button.

    The panel's title is the condition in words, so a closed panel still says what it does.
    It is keyed by path, not by revision, so it stays open or closed across edits; a new or
    renamed one opens (open_panel). on_rename(new) renames it (and reruns), or returns why it
    cannot.
    """
    kind = condition_kind(node)
    opened = path in st.session_state.get("open_panels", set())
    with st.expander(f"{name} · {condition_text(node)}", expanded=opened, key=f"condition-panel:{path}"):
        head = st.columns([2, 4, 2], vertical_alignment="bottom")
        typed = head[0].text_input("Name", value=name, key=key(path, ":name"),
                                   help="What the logic line calls this condition, such as trend or rsi_low: "
                                        "letters, digits and underscores, starting with a letter. Renaming it "
                                        "renames it in the logic line and in the Optimise page's variables too.")
        kinds = list(CONDITION_KINDS)
        chosen = head[1].selectbox("Kind", kinds, index=kinds.index(kind), format_func=CONDITION_KINDS.get,
                                   key=key(path, ":kind"),
                                   help="Compare two lines (fast crosses above slow), a pattern appearing, the day, "
                                        "month or time of day, or a comparison on a higher timeframe (the weekly "
                                        "close above its open). Every rule takes every kind.")
        head[2].button("Delete condition", key=key(path, ":remove"), on_click=on_remove,
                       help="Removes this condition, takes its name out of the logic line, and takes its values "
                            "off the Optimise page.")
        if typed.strip() != name:
            # A refused name stays in the box with the reason under it, as a broken logic line does.
            st.error(on_rename(typed.strip()))
        if chosen != kind:
            set_value(path, new_condition(chosen, features, timeframe))
            prune_variables()
            save_draft()
            renew_widgets()
            st.rerun()
        if kind == "pattern":
            _pattern_condition(path, node, features)
        elif kind == "calendar":
            _calendar_condition(path, node, timeframe)
        elif kind == "higher":
            _higher_condition(path, node, features, timeframe)
        else:
            _comparison_condition(path, node, features)
        if kind != "calendar":
            _timing_input(path, _get(path, node), higher=kind == "higher")
        _optimise_picker(path)


def _optimise_picker(path):
    """Which of one condition's values the optimiser searches, picked where the condition is built.

    The list is the condition's rows of tunable_inputs, so it offers exactly what the Optimise
    page does: the comparison, each number, each side's bars ago (0 included), the timing's N,
    the bars of rising and falling, a pattern's side, a higher timeframe. Picking one adds it
    to optimise.variables with a starting range (spec.default_variable), as the Optimise
    page's Add does; the range is changed there. Taking it out of the list removes it.
    """
    spec = current()
    rows = {row["path"]: row for row in tunable_inputs(spec) if row["path"].startswith(path + ".")}
    if not rows:
        return
    found = (spec.get("optimise") or {}).get("variables") or {}

    def shown(row, value):
        return (row.get("labels") or {}).get(value, value)

    def label(item):
        row = rows[item]
        return f"{row['label'].rsplit(' › ', 1)[-1]} (now {shown(row, row['value'])})"

    searched = [item for item in rows if item in found]
    chosen = st.multiselect("Optimise", list(rows), default=searched, format_func=label, key=key(path, ":optimise"),
                            placeholder="Nothing in this condition is searched",
                            help="Let the optimiser search these values of the condition. Each is added to the "
                                 "Optimise page with a starting range (half to double its value; a comparison "
                                 "and its mirror; every side of a pattern), where its range or list is set. "
                                 "Taking one out of this list stops the search on it.")
    if set(chosen) != set(searched):
        variables = spec.setdefault("optimise", {}).setdefault("variables", {})
        for item, row in rows.items():
            if item in chosen and item not in variables:
                variables[item] = default_variable(row)
            elif item not in chosen and item in variables:
                del variables[item]
        save_draft()
        found = variables
    # What each searched value will try, so the range is in sight without leaving the page.
    parts = []
    for item in chosen:
        variable, row = found.get(item) or {}, rows[item]
        if "values" in variable:
            tries = ", ".join(str(shown(row, value)) for value in variable["values"])
        elif "start" in variable:
            tries = f"{variable['start']:g} to {variable['stop']:g}" + (f" by {variable['step']:g}"
                                                                         if variable.get("step") else "")
        else:
            continue
        parts.append(f"{row['label'].rsplit(' › ', 1)[-1]}: {tries}")
    if parts:
        st.caption("Searched: " + "; ".join(parts) + ". Change the ranges on the Optimise page.")


def _rule_body(path, rule, features, timeframe):
    """The conditions and the logic line of one rule (one version of it)."""
    conditions = rule.setdefault("conditions", {})
    for letter in list(conditions):
        def remove(letter=letter):
            conditions.pop(letter, None)
            rule["logic"] = remove_letter(rule.get("logic", ""), letter)
            prune_variables()
            save_draft()
            renew_widgets()

        def rename(new, letter=letter):
            problem = condition_name_problem(new, set(conditions) - {letter})
            if problem:
                return problem
            open_panel(f"{path}.conditions.{new}")
            # A structural change: every widget key under the old name must be renewed.
            rerun_after(lambda: rename_condition(current(), path, letter, new))
        described = condition_text(conditions[letter])
        condition_editor(f"{path}.conditions.{letter}", letter, conditions[letter], features, timeframe, remove,
                         rename)
        # The panel's title is drawn before the settings inside write their new values, so after
        # an edit it would show the old words until the next click: draw the page again, once
        # (the second pass finds them equal), as the Features tab does.
        if letter in conditions and condition_text(conditions[letter]) != described:
            st.rerun()
    buttons = st.columns([2, 2, 4])
    joiner = buttons[1].selectbox("Join new with", ["and", "or"], key=key(path, ":join"),
                                  label_visibility="collapsed", format_func=lambda op: f"join with {op}")
    if buttons[0].button("Add condition", key=key(path, ":add"),
                         help="Adds a condition comparing two lines, named with the next free letter, and joins it "
                              "to the logic line with the word on the right (and, or). Pick its kind in its Kind "
                              "list: a comparison, a pattern, the day or time, or a higher timeframe. Edit the logic "
                              "line for anything else."):
        def add():
            letter = next_letter(set(conditions))
            conditions[letter] = new_condition("comparison", features, timeframe)
            rule["logic"] = append_letter(rule.get("logic", ""), letter, joiner)
            open_panel(f"{path}.conditions.{letter}")
        rerun_after(add)
    logic = st.text_input("Logic", value=rule.get("logic", ""), key=key(path, ":logic"),
                          help="Combine the conditions by name with and, or, xor, nand, nor, xnor and not, using "
                               "brackets: (A and (B or C)) and not D, or trend and not overbought. Names match "
                               "exactly, capitals included. Brackets are needed wherever two different words meet.")
    if logic != rule.get("logic"):
        set_value(f"{path}.logic", logic)
    try:
        tree = parse_logic(logic)
    except ValueError as error:
        st.error(f"Logic: {error}")
    else:
        if isinstance(tree, dict) and "op" in tree:
            st.caption(f"Fires when {LOGIC_WORDS[tree['op']]} of its parts.")


def rule_editor(path, features, timeframe):
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
                prune_variables()
            rerun_after(drop_version)
        if len(labels) == 1 and top[2].button("Back to a single rule", key=key(path, ":single")):
            def single():
                set_path(current(), path, node["options"][chosen])
                prune_variables()
            rerun_after(single)
        _rule_body(f"{path}.options.{chosen}", node["options"][chosen], features, timeframe)
        return
    _rule_body(path, node, features, timeframe)
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
