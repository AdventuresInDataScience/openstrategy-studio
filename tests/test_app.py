"""Test 14: every page loads with every template, with no uncaught exception.

Uses streamlit.testing.v1.AppTest (VERIFY the current API on the installed version).
"""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import openstrategy_studio
from openstrategy_studio.spec import blank_spec, list_templates, load_template, validate_spec

APP = Path(openstrategy_studio.__file__).parent / "app"

# Every widget kind that holds a setting (buttons are actions, not settings).
SETTING_WIDGETS = ("text_input", "number_input", "selectbox", "multiselect", "radio", "checkbox", "toggle", "slider",
                   "select_slider", "date_input", "text_area")


def settings_without_a_tooltip(app):
    """The visible settings on a rendered page that have no ⓘ (help=) beside them.

    A widget whose label is collapsed has nowhere to show an ⓘ, so it is skipped; the
    heading or name beside it carries one instead (proto LabelVisibility: 0 is VISIBLE).
    """
    missing = []
    for kind in SETTING_WIDGETS:
        for widget in getattr(app, kind):
            if widget.proto.label_visibility.value == 0 and not (widget.help or "").strip():
                missing.append(f"{kind} {widget.label!r}")
    return missing


def test_every_page_loads_with_every_template(tmp_path, monkeypatch):
    """Every page renders with every template and with a blank strategy, leaves the spec as it was,
    and explains every setting."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    pages = [APP / "Home.py"] + sorted((APP / "pages").glob("*.py"))
    starts = {name: load_template(name) for name in list_templates()}
    starts["blank"] = blank_spec()
    for name, spec in starts.items():
        for page in pages:
            app = testing.AppTest.from_file(str(page), default_timeout=180)
            app.session_state["spec"] = copy.deepcopy(spec)
            app.session_state["spec_rev"] = 0
            app.run()
            assert not app.exception, (name, page.name, [item.value for item in app.exception])
            # Rendering must not change the spec (CLAUDE.md): every widget is seeded from it.
            assert app.session_state["spec"] == spec, (name, page.name)
            # Every setting has an ⓘ tooltip (CLAUDE.md).
            assert not settings_without_a_tooltip(app), (name, page.name, settings_without_a_tooltip(app))


def test_a_new_session_starts_blank(tmp_path, monkeypatch):
    """No template and no draft is loaded on its own (the owner's call, 19 November 2026); Home
    offers last time's draft, and Start blank and the Build page's Start from give a blank one again."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    spec = app.session_state["spec"]
    assert spec == blank_spec()
    assert spec["data"]["symbols"] == [] and spec["features"] == {} and not any(spec["rules"].values())
    # What is still missing is the At a glance panel's to-do list, not an error.
    assert "Pick at least one symbol." in validate_spec(spec) and "Add a long or a short entry rule." in validate_spec(spec)
    # The Build page's Start from gives a blank strategy over a template.
    app.session_state["spec"] = load_template("ma_cross")
    app.run()
    next(box for box in app.selectbox if box.label == "Start from").set_value("blank").run()
    next(button for button in app.button if button.label == "Start from this").click().run()
    assert app.session_state["spec"] == blank_spec()
    # Home: last time's draft is offered, not loaded.
    home = testing.AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    (tmp_path / "data" / "drafts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "drafts" / "current.json").write_text(json.dumps(load_template("ma_cross")), encoding="utf-8")
    home.run()
    assert not home.exception, [item.value for item in home.exception]
    assert home.session_state["spec"] == blank_spec()
    assert any(button.label == "Restore it" for button in home.button)
    next(button for button in home.button if button.label == "Start blank").click().run()
    assert home.session_state["spec"] == blank_spec()


def test_every_account_choice_renders_without_changing_the_spec(tmp_path, monkeypatch):
    """The Account tab's cash section, in each of its branches."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    base = load_template("shared_account_ma_cross")
    variants = [{"init_cash": 10000, "freq": "D"},
                {"init_cash": 10000, "freq": "D", "cash": "per_instrument"},
                dict(base["account"], entry_priority={"by": "symbols", "order": ["NVDA", "AAPL"]}),
                dict(base["account"], entry_priority={"by": "symbols"}, when_cash_short="partial"),
                dict(base["account"], entry_priority={"by": "random"})]
    for account in variants:
        spec = copy.deepcopy(base)
        spec["account"] = account
        spec["engine_extra"] = {"seed": 3}
        app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
        app.session_state["spec"] = copy.deepcopy(spec)
        app.session_state["spec_rev"] = 0
        app.run()
        assert not app.exception, (account, [item.value for item in app.exception])
        assert app.session_state["spec"] == spec, account
        assert not settings_without_a_tooltip(app), (account, settings_without_a_tooltip(app))


def test_backtest_page_says_how_the_run_was_funded(tmp_path, monkeypatch):
    """A finished run, handed to the page: separate accounts are named as such, with the per-symbol table."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from conftest import make_ohlcv
    from openstrategy_studio import analysis, engine

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    data = make_ohlcv(400, ("AAPL", "MSFT", "NVDA"), seed=3, base=100.0)
    shared = load_template("shared_account_ma_cross")
    separate = copy.deepcopy(shared)
    separate["account"] = {"init_cash": 10000, "freq": "D"}
    for spec, expected in ((separate, "3 separate accounts"), (shared, "One account of 10,000")):
        run = engine.run_backtest(spec, data)
        app = testing.AppTest.from_file(str(APP / "pages" / "3_Backtest_results.py"), default_timeout=180)
        app.session_state["spec"] = spec
        app.session_state["spec_rev"] = 0
        app.session_state["last_run"] = {"results": analysis.extract_results(run), "spec": run["spec"],
                                         "fingerprint": {}, "saved": None, "id": "test"}
        app.run()
        assert not app.exception, [item.value for item in app.exception]
        texts = [item.value for item in list(app.info) + list(app.caption)]
        assert any(expected in text for text in texts), (expected, texts)
        assert any(item.value == "**Each symbol**" for item in app.markdown)
        # The results half of the page (metrics, views, chart theme) explains its settings too.
        assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_the_backtest_page_shows_what_vectorbtx_flagged(tmp_path, monkeypatch):
    """vectorbtX reports what it could not price (a short borrowed for free, a carry leg
    with no currency) as Python warnings, which never reach a page on their own."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from conftest import make_ohlcv
    from openstrategy_studio import analysis, engine

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = load_template("shared_account_ma_cross")
    run = engine.run_backtest(spec, make_ohlcv(400, ("AAPL", "MSFT", "NVDA"), seed=3, base=100.0))
    results = analysis.extract_results(run)
    results["meta"]["warnings"] = ["Short positions in ['AAPL'] borrowed the securities for free."]
    app = testing.AppTest.from_file(str(APP / "pages" / "3_Backtest_results.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    app.session_state["last_run"] = {"results": results, "spec": run["spec"], "fingerprint": {},
                                     "saved": None, "id": "test"}
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert any("borrowed the securities for free" in item.value for item in app.warning)


def test_account_choices_write_the_spec(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = load_template("shared_account_ma_cross")
    spec["account"] = {"init_cash": 10000, "freq": "D"}
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    app.run()
    from openstrategy_studio.spec import when_cash_short

    app.radio(key=current_key(app, "account.cash")).set_value("shared").run()
    account = app.session_state["spec"]["account"]
    # Left unset, a shared account skips an entry it cannot fund; the Risk tab's radio shows it.
    assert account["cash"] == "shared" and when_cash_short(app.session_state["spec"]) == "skip"
    assert app.radio(key=current_key(app, "account.when_cash_short")).value == "skip"
    app.radio(key=current_key(app, "account.entry_priority:by")).set_value("feature").run()
    priority = app.session_state["spec"]["account"]["entry_priority"]
    assert priority == {"by": "feature", "highest_first": True, "score": {"feature": "fast"}}
    # A choice made on the Risk tab stays when the account changes; the priority does not.
    app.radio(key=current_key(app, "account.when_cash_short")).set_value("partial").run()
    app.radio(key=current_key(app, "account.cash")).set_value("per_instrument").run()
    account = app.session_state["spec"]["account"]
    assert account["cash"] == "per_instrument" and "entry_priority" not in account
    assert account["when_cash_short"] == "partial"
    assert not app.exception


def current_key(app, suffix):
    """A spec-bound widget's key: structural changes renew them (ui.renew_widgets), so read the revision."""
    return f"r{app.session_state['spec_rev']}:{suffix}"


def open_cards(app, *paths):
    """Open feature and condition cards before the next run (ui.is_open): closed, a card shows its title only."""
    panels = dict(app.session_state["panels"]) if "panels" in app.session_state else {}
    panels.update({path: True for path in paths})
    app.session_state["panels"] = panels


def card_titles(app):
    """The text of every card's title row (ui.card_header), tags removed: a folding card's title is
    its fold button's label, followed by the details beside it."""
    import re
    titles = []

    def walk(node):
        children = getattr(node, "children", None)
        if not isinstance(children, dict):
            return
        parts, is_title_row = [], False
        for child in children.values():
            if child.type == "button" and str(child.key or "").endswith(":open"):
                parts.append(child.label)
                is_title_row = True
            elif child.type == "markdown" and ("os-title" in child.value or "os-code" in child.value
                                               or "os-meta" in child.value or "os-off" in child.value):
                parts.append(re.sub(r"<[^>]+>", " ", child.value))
                is_title_row = True
        if is_title_row:
            titles.append(" " + " ".join(parts) + " ")
        for child in children.values():
            walk(child)

    walk(app.main)
    return titles


def test_build_steps_edit_the_spec(tmp_path, monkeypatch):
    """The Build page as a user drives it: the nested picker, a condition and the logic line, the two Setup questions."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    # Features: Indicator > Bands and channels > Keltner Channels, added under its own name.
    app.selectbox(key="pick:category").set_value("Indicator").run()
    app.selectbox(key="pick:family:Indicator").set_value("Bands and channels").run()
    app.selectbox(key="pick:item:Indicator:Bands and channels").set_value("talib:KC").run()
    app.button(key="pick:add").click().run()
    assert not app.exception, [item.value for item in app.exception]
    kc = app.session_state["spec"]["features"]["kc"]
    assert kc["indicator"] == "talib:KC" and kc["params"] == {"timeperiod": 20, "atrperiod": 10, "nbdev": 2.0}
    # It opens at once with its settings showing; the others stay closed.
    assert app.number_input(key=current_key(app, "features.kc.params.timeperiod")).value == 20
    assert not [box for box in app.number_input if box.key == current_key(app, "features.fast.params.timeperiod")]
    # The picker remembers where it was; the suggested name moves on.
    assert app.selectbox(key="pick:family:Indicator").value == "Bands and channels"
    # Rules: a second condition joins the logic line with "and", and the box shows it; a broken
    # line says what is wrong.
    app.button(key=current_key(app, "rules.long_entry:add")).click().run()
    rule = app.session_state["spec"]["rules"]["long_entry"]
    # A new condition is named condition1, not a letter (the owner's call, 20 November 2026).
    assert set(rule["conditions"]) == {"A", "condition1"} and rule["logic"] == "A and condition1"
    assert app.text_input(key=current_key(app, "rules.long_entry:logic")).value == "A and condition1"
    app.text_input(key=current_key(app, "rules.long_entry:logic")).set_value("A or condition1 or").run()
    assert any("ends too soon" in item.value for item in app.error)
    app.text_input(key=current_key(app, "rules.long_entry:logic")).set_value("A or not condition1").run()
    assert app.session_state["spec"]["rules"]["long_entry"]["logic"] == "A or not condition1"
    # Its bin, on its title row, takes it out of the logic line too.
    app.button(key=current_key(app, "rules.long_entry.conditions.condition1:remove")).click().run()
    rule = app.session_state["spec"]["rules"]["long_entry"]
    assert set(rule["conditions"]) == {"A"} and rule["logic"] == "A"
    assert not app.exception
    # Setup: how it is traded drops the fields that way of trading rejects; a portfolio replaces the rules.
    app.radio(key=current_key(app, "setup.trading")).set_value("equity").run()
    assert app.session_state["spec"]["setup"]["trading"] == "equity"
    app.radio(key=current_key(app, "setup.kind")).set_value("portfolio").run()
    spec = app.session_state["spec"]
    assert spec["setup"]["kind"] == "portfolio" and "rules" not in spec
    assert spec["schedule"]["rebalance"] == "never" and spec["weighting"]["scheme"] == "equal"
    assert not any(path.startswith("rules.") for path in spec["optimise"]["variables"])
    assert not app.exception, [item.value for item in app.exception]


def test_optimise_variables_are_added_and_changed(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    picker = app.selectbox(key=current_key(app, "optimise:add"))
    stop = next(option for option in picker.options if "Stop distance" in option)
    picker.select_index(picker.options.index(stop)).run()
    app.button(key=current_key(app, "optimise:add:go")).click().run()
    variables = app.session_state["spec"]["optimise"]["variables"]
    assert variables["risk.stop.points"] == {"start": 30.0, "stop": 120.0, "step": 9.0}
    app.radio(key=current_key(app, "opt:risk.stop.points:how")).set_value("list").run()
    assert app.session_state["spec"]["optimise"]["variables"]["risk.stop.points"]["values"][:2] == [30.0, 39.0]
    assert not app.exception, [item.value for item in app.exception]


def test_start_checks_the_ranges_first_and_holds_a_job_no_trial_can_pass(tmp_path, monkeypatch):
    """The owner's MA job (20 November 2026): 5,000 trials, none valid. Start now runs the strategy at its own
    settings and each setting at the ends of its range first; when none reaches the minimum trades the job is
    not started, the page says why and offers Start anyway."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import config, jobs

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    spec["instruments"]["SPY"]["min_size"] = 100.0          # 500 over a 100-point stop is a stake of 5 a point
    spec["optimise"] = {"variables": {"features.fast.params.timeperiod": {"start": 5, "stop": 15, "step": 5}}}
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    button = lambda label: next(item for item in app.button if item.label == label)
    next(item for item in app.number_input if item.label == "Minimum trades").set_value(5)
    button("Start optimisation").click().run()
    assert not app.exception, [item.value for item in app.exception]
    (error,) = [item.value for item in app.error if item.value.startswith("No trial is likely to be valid")]
    assert "makes 0 trades" in error and "minimum tradeable size" in error, error
    assert [item for item in app.button if item.label == "Start anyway"]
    assert not [item for item in app.success if item.value.startswith("Started")]
    assert jobs.list_jobs(config.data_dir(config.load_settings())).empty
    assert app.session_state["spec"] == spec                            # the check changes nothing
    # A strategy that trades: no hold, and Check the ranges starts nothing.
    spec["instruments"]["SPY"]["min_size"] = 0.01
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 5
    app.run()
    next(item for item in app.number_input if item.label == "Minimum trades").set_value(5)
    button("Check the ranges").click().run()
    assert not app.exception, [item.value for item in app.exception]
    held = [item.value for item in app.error if item.value.startswith("No trial is likely to be valid")]
    assert not held, held
    assert not [item for item in app.button if item.label == "Start anyway"]
    assert jobs.list_jobs(config.data_dir(config.load_settings())).empty
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_the_stop_and_target_switch_off_and_keep_their_distance_for_the_optimiser(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("rsi_reversion")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    app.radio(key=current_key(app, "risk.stop:kind")).set_value("trailing").run()
    assert app.session_state["spec"]["risk"]["stop"] == {"kind": "trailing", "unit": "points", "points": 50, "pct": 0.02}
    # Measured in % of the price: the box shows the fraction, and both distances stay.
    app.radio(key=current_key(app, "risk.stop:unit")).set_value("pct").run()
    assert app.session_state["spec"]["risk"]["stop"]["unit"] == "pct"
    app.number_input(key=current_key(app, "risk.stop.pct")).set_value(0.03).run()
    assert app.session_state["spec"]["risk"]["stop"] == {"kind": "trailing", "unit": "pct", "points": 50, "pct": 0.03}
    assert any("3.00% of the best price since the entry" in item.value for item in app.caption)
    app.radio(key=current_key(app, "risk.stop:kind")).set_value("none").run()
    assert app.session_state["spec"]["risk"]["stop"] == {"kind": "none", "unit": "pct", "points": 50, "pct": 0.03}
    app.checkbox(key=current_key(app, "risk.target:on")).check().run()
    assert app.session_state["spec"]["risk"]["target"] == {"kind": "fixed", "unit": "points", "points": 100, "pct": 0.04}
    app.checkbox(key=current_key(app, "risk.target:on")).uncheck().run()
    assert app.session_state["spec"]["risk"]["target"]["kind"] == "none"
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app)
    # Both are on the Optimise page's list, the stop's kind in words.
    spec = app.session_state["spec"]
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    app.run()
    picker = app.selectbox(key=current_key(app, "optimise:add"))
    assert any(option.startswith("Risk: Stop: none, fixed or trailing (now no stop)") for option in picker.options)
    assert any(option.startswith("Risk: Take profit: on or off") for option in picker.options)
    assert not app.exception, [item.value for item in app.exception]


def test_an_average_can_hand_its_type_to_the_optimiser(tmp_path, monkeypatch):
    """The feature card's button swaps an SMA for the Moving average entry at SMA and adds its type to
    the variables; the period, its variable and the rules are untouched."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio.spec import validate_spec

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    before = load_template("ma_cross")
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(before)
    app.session_state["spec_rev"] = 0
    open_cards(app, "features.fast")
    app.run()
    app.button(key=current_key(app, "features.fast:any_type")).click().run()
    assert not app.exception, [item.value for item in app.exception]
    spec = app.session_state["spec"]
    assert spec["features"]["fast"] == {"indicator": "talib:MA", "inputs": {"close": "close"},
                                        "params": {"timeperiod": 50, "matype": "SMA"}}
    assert spec["features"]["slow"] == before["features"]["slow"] and spec["rules"] == before["rules"]
    variables = spec["optimise"]["variables"]
    assert variables["features.fast.params.matype"] == {"values": ["SMA", "EMA"]}
    assert variables["features.fast.params.timeperiod"] == before["optimise"]["variables"]["features.fast.params.timeperiod"]
    assert validate_spec(spec) == []
    # The Moving average entry has no such button: its type is already a setting.
    assert not [button for button in app.button if button.key == current_key(app, "features.fast:any_type")]


def test_a_comparison_is_added_as_a_variable_in_words(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    picker = app.selectbox(key=current_key(app, "optimise:add"))
    wanted = "Rules: Long entry › A › the comparison (now crosses above)"
    assert wanted in picker.options, picker.options
    picker.select_index(picker.options.index(wanted)).run()
    app.button(key=current_key(app, "optimise:add:go")).click().run()
    path = "rules.long_entry.conditions.A.op"
    assert app.session_state["spec"]["optimise"]["variables"][path] == {"values": ["crosses_above", "crosses_below"]}
    values = app.multiselect(key=current_key(app, f"opt:{path}:values"))
    assert "crosses below" in values.options and "is at or above" in values.options
    assert values.value == ["crosses_above", "crosses_below"]
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_condition_is_renamed_in_its_name_box(tmp_path, monkeypatch):
    """A good name renames the condition, its logic line and its optimise variables; a bad one
    stays in the box with the reason, and the spec is left alone."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = load_template("ma_cross")
    spec["optimise"]["variables"]["rules.long_entry.conditions.A.op"] = {"values": ["crosses_above", "crosses_below"]}
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    open_cards(app, "rules.long_entry.conditions.A")
    app.run()
    app.text_input(key=current_key(app, "rules.long_entry.conditions.A:name")).set_value("trend").run()
    assert not app.exception, [item.value for item in app.exception]
    rule = app.session_state["spec"]["rules"]["long_entry"]
    assert list(rule["conditions"]) == ["trend"] and rule["logic"] == "trend"
    assert app.text_input(key=current_key(app, "rules.long_entry:logic")).value == "trend"
    assert "rules.long_entry.conditions.trend.op" in app.session_state["spec"]["optimise"]["variables"]
    assert app.session_state["spec"]["rules"]["long_exit"]["logic"] == "A"      # another rule keeps its own A
    # An operator word, or a name the rule already has, is refused with the reason.
    settled = copy.deepcopy(app.session_state["spec"])
    app.text_input(key=current_key(app, "rules.long_entry.conditions.trend:name")).set_value("and").run()
    assert any("would read as an operator" in item.value for item in app.error)
    assert app.session_state["spec"] == settled
    app.text_input(key=current_key(app, "rules.long_entry.conditions.trend:name")).set_value("trend").run()
    app.button(key=current_key(app, "rules.long_entry:add")).click().run()
    assert app.session_state["spec"]["rules"]["long_entry"]["logic"] == "trend and condition1"
    app.text_input(key=current_key(app, "rules.long_entry.conditions.condition1:name")).set_value("trend").run()
    assert any("already has a condition called trend" in item.value for item in app.error)
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_feature_is_renamed_in_its_name_box(tmp_path, monkeypatch):
    """A good name renames the feature, the rules that read it and its optimise variables; a bad one
    stays in the box with the reason, and the spec is left alone."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    open_cards(app, "features.fast")
    app.run()
    app.text_input(key=current_key(app, "features.fast:name")).set_value("quick").run()
    assert not app.exception, [item.value for item in app.exception]
    spec = app.session_state["spec"]
    assert list(spec["features"]) == ["quick", "slow"]
    assert spec["rules"]["long_entry"]["conditions"]["A"]["left"] == {"feature": "quick"}
    assert spec["rules"]["long_exit"]["conditions"]["A"]["left"] == {"feature": "quick"}
    assert list(spec["optimise"]["variables"]) == ["features.quick.params.timeperiod", "features.slow.params.timeperiod"]
    # The rule editor reads the new name too.
    assert app.text_input(key=current_key(app, "features.quick:name")).value == "quick"
    settled = copy.deepcopy(spec)
    for bad, reason in (("close", "cannot name a feature"), ("slow", "already a feature called slow")):
        app.text_input(key=current_key(app, "features.quick:name")).set_value(bad).run()
        assert any(reason in item.value for item in app.error), (bad, [item.value for item in app.error])
        assert app.session_state["spec"] == settled
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_feature_panel_title_shows_an_edit_at_once(tmp_path, monkeypatch):
    """The panel's title is drawn before the settings inside save their new values; after an edit it
    must show the new value straight away, not one click later."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    open_cards(app, "features.fast", "features.slow")
    app.run()
    app.number_input(key=current_key(app, "features.fast.params.timeperiod")).set_value(60).run()
    assert not app.exception, [item.value for item in app.exception]
    titles = card_titles(app)
    assert any(" fast " in title and "SMA(close, 60)" in title for title in titles), titles
    # An input too: the slow average applied to the open instead of the close.
    app.selectbox(key=current_key(app, "features.slow.inputs.close")).set_value("open").run()
    titles = card_titles(app)
    assert any(" slow " in title and "SMA(open, 200)" in title for title in titles), titles
    assert app.session_state["spec"]["features"]["slow"]["inputs"]["close"] == "open"
    assert not app.exception, [item.value for item in app.exception]


def test_conditions_take_every_kind_and_a_timing_in_any_rule(tmp_path, monkeypatch):
    """One Add condition button; an exit rule takes a calendar condition like any other kind; a
    condition can hold "on any of the last N bars", and its panel's title says so at once."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio.spec import validate_spec

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    open_cards(app, "rules.long_entry.conditions.A")
    app.run()
    app.button(key=current_key(app, "rules.long_exit:add")).click().run()
    rule = app.session_state["spec"]["rules"]["long_exit"]
    assert set(rule["conditions"]) == {"A", "condition1"} and rule["logic"] == "A and condition1"
    new = "rules.long_exit.conditions.condition1"
    app.selectbox(key=current_key(app, f"{new}:kind")).set_value("calendar").run()
    assert app.session_state["spec"]["rules"]["long_exit"]["conditions"]["condition1"] == {
        "filter": "weekday", "days": [0, 1, 2, 3, 4]}
    app.selectbox(key=current_key(app, f"{new}:kind")).set_value("higher").run()
    higher = app.session_state["spec"]["rules"]["long_exit"]["conditions"]["condition1"]
    assert higher["filter"] == "higher_timeframe" and higher["timeframe"] == "1w"     # the first above 1d
    assert validate_spec(app.session_state["spec"]) == []
    # The cross may have happened on any of the last 5 bars.
    app.selectbox(key=current_key(app, "rules.long_entry.conditions.A:timing")).set_value("within_bars").run()
    assert app.session_state["spec"]["rules"]["long_entry"]["conditions"]["A"]["within_bars"] == 3
    app.number_input(key=current_key(app, "rules.long_entry.conditions.A.within_bars")).set_value(5).run()
    condition = app.session_state["spec"]["rules"]["long_entry"]["conditions"]["A"]
    assert condition["within_bars"] == 5 and condition["op"] == "crosses_above"
    titles = card_titles(app)
    assert any(" A " in title and "fast crosses above slow, on any of the last 5 bars" in title for title in titles), \
        titles
    assert any("Long entry" in title and " A " in title for title in titles), titles
    assert any("Short entry" in title and "not used" in title for title in titles), titles
    app.selectbox(key=current_key(app, "rules.long_entry.conditions.A:timing")).set_value("now").run()
    assert "within_bars" not in app.session_state["spec"]["rules"]["long_entry"]["conditions"]["A"]
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_rule_is_added_and_deleted_with_its_variables(tmp_path, monkeypatch):
    """Add a … rule starts one; Delete the … rule removes it and its values on the Optimise page
    (a "use this rule" switch did not read as a way to delete one: the owner, 19 November 2026)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    app.button(key=current_key(app, "rules.short_entry:add_rule")).click().run()
    assert app.session_state["spec"]["rules"]["short_entry"]["logic"] == "condition1"
    # Its comparison searched (added on the Optimise page, the one place a variable is added).
    spec = app.session_state["spec"]
    spec["optimise"]["variables"]["rules.short_entry.conditions.condition1.op"] = {"values": ["<", ">"]}
    app.session_state["spec"] = spec
    app.run()
    app.button(key=current_key(app, "rules.short_entry:delete_rule")).click().run()
    spec = app.session_state["spec"]
    assert spec["rules"]["short_entry"] is None
    assert not any(path.startswith("rules.short_entry") for path in spec["optimise"]["variables"])
    # The template's own variables stay.
    assert "features.fast.params.timeperiod" in spec["optimise"]["variables"]
    app.button(key=current_key(app, "rules.long_exit:delete_rule")).click().run()
    assert app.session_state["spec"]["rules"]["long_exit"] is None
    assert not app.exception, [item.value for item in app.exception]


def test_a_conditions_searched_values_show_in_its_card(tmp_path, monkeypatch):
    """The Optimise page is the one place a value is added to the search (the owner's call, 20
    November 2026); each condition's card only says what is searched, open or closed, and deleting
    the condition takes its variables with it."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("rsi_reversion")
    app.session_state["spec_rev"] = 0
    app.run()
    base = "rules.long_entry.conditions.A"
    # The template searches the number (15 to 40); the closed card says so, and offers no picker.
    assert any("Optimised: the number (right) 15 to 40. Set on the Optimise page." in item.value
               for item in app.caption), [item.value for item in app.caption]
    assert not [box for box in app.multiselect if box.label == "Optimise"]
    # A trial's bars ago of 0, written back by Apply to the strategy, is left as it is.
    open_cards(app, base)
    spec = app.session_state["spec"]
    spec["rules"]["long_entry"]["conditions"]["A"]["left"]["shift"] = 0
    app.session_state["spec"] = spec
    before = copy.deepcopy(spec)
    app.run()
    assert app.session_state["spec"] == before
    # Deleting the condition takes its variables with it.
    app.button(key=current_key(app, "rules.long_entry.conditions.B:remove")).click().run()
    app.button(key=current_key(app, f"{base}:remove")).click().run()
    assert not any(path.startswith(base) for path in app.session_state["spec"]["optimise"]["variables"])
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_the_backtest_results_page_shows_the_chart_before_every_metric(tmp_path, monkeypatch):
    """No choice of metric sets: every metric, under the chart (the owner's call, 19 November 2026)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from conftest import make_ohlcv
    from openstrategy_studio import analysis, engine

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = load_template("ma_cross")
    run = engine.run_backtest(spec, make_ohlcv(400, ("EURUSD=X",), seed=3, base=1.1))
    app = testing.AppTest.from_file(str(APP / "pages" / "3_Backtest_results.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    app.session_state["last_run"] = {"results": analysis.extract_results(run), "spec": run["spec"],
                                     "fingerprint": {}, "saved": None, "id": "test"}
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert not any(radio.label == "Metrics" for radio in app.radio)

    def walk(node):
        """Every element on the page, in the order it is drawn."""
        yield node
        for child in (getattr(node, "children", None) or {}).values():
            yield from walk(child)

    # The View box (the chart's) is drawn before the Portfolio table's heading.
    seen = []
    for node in walk(app.main):
        if getattr(node, "type", None) == "selectbox" and node.label == "View":
            seen.append("View")
        if getattr(node, "type", None) == "markdown" and node.value == "**Portfolio**":
            seen.append("Portfolio")
    assert seen == ["View", "Portfolio"], seen


def spread_bet_on_spy(tmp_path):
    """The owner's run (17 November 2026): SPY as a spread bet, 500 (5% of the cash) risked over a
    100-point stop, on stored daily prices near 600."""
    from openstrategy_studio import config, data as data_module

    data_dir = config.data_dir(config.load_settings())
    synthetic = data_module.synthetic_ohlcv(["SPY"], n=600, seed=2)
    frame = pd.DataFrame({field: synthetic[field]["SPY"] * (6 if field != "volume" else 1)
                          for field in ("open", "high", "low", "close", "volume")})
    data_module.store_dataset(data_dir, "SPY", "1d", frame, "yahoo")
    spec = load_template("ma_cross")
    spec.pop("optimise")
    spec["data"].update(symbols=["SPY"], start="2018-01-01", end="2021-12-31")
    spec["features"]["fast"]["params"]["timeperiod"] = 10
    spec["features"]["slow"]["params"]["timeperiod"] = 30
    spec["instruments"] = {"SPY": {"point_size": 0.01}}
    spec["account"].pop("rates")
    spec["risk"] = {"sizing": {"risk_amount": 500.0}, "stop": {"kind": "fixed", "points": 100.0}, "target": None}
    return spec


def test_the_risk_tab_shows_leverage_and_checks_the_trade_size(tmp_path, monkeypatch):
    """What the risk is as a share, the leverage a blank box gives, and the trade the run would cut down."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["spec"] == spec
    assert any(item.value.startswith("500.00 lost on each trade that hits its stop, the same on every trade: 5.00% of "
                                     "the starting cash of 10,000") for item in app.caption)
    warned = [item.value for item in app.warning if "leverage funds" in item.value]
    assert warned and "5x leverage funds 50,000" in warned[0] and "cut it down to 50,000" in warned[0], warned
    # The stop box says what its points are in price, in the market's words ...
    assert any("on SPY, 100 points is a price move of 1: a point is 1 cent (0.01)" in item.value for item in app.caption)
    # ... and a 100-point stop on SPY is a 1.00 move, inside an ordinary bar's range.
    assert any("will be stopped out within a bar or two" in item.value for item in app.warning)
    assert app.radio(key=current_key(app, "account.when_cash_short")).value == "partial"
    # Skipping instead: the warning says so.
    app.radio(key=current_key(app, "account.when_cash_short")).set_value("skip").run()
    assert app.session_state["spec"]["account"]["when_cash_short"] == "skip"
    assert any("The run will skip it" in item.value for item in app.warning)
    # Enough leverage, and the check is quiet.
    spec["instruments"]["SPY"]["leverage"] = 50.0
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 7
    app.run()
    assert not [item.value for item in app.warning if "leverage funds" in item.value]
    assert app.session_state["spec"] == spec
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_slippage_typed_as_points_is_flagged_where_it_is_typed(tmp_path, monkeypatch):
    """The owner's MA job (20 November 2026): slippage 0.2 beside a 1.5-point spread is 20% a fill,
    so one trade emptied the account and every trial made 0 or 1 trades, with nothing flagged."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    spec["instruments"]["SPY"].update(spread_points=1.5, slippage=0.2)
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["spec"] == spec
    warned = [item.value for item in app.warning if item.value.startswith("Slippage is in %")]
    assert warned and "SPY's 20% a fill is about 40% of the price a round trip" in warned[0], warned
    # A usual slippage is quiet.
    spec["instruments"]["SPY"]["slippage"] = 0.0005
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 3
    app.run()
    assert not [item.value for item in app.warning if item.value.startswith("Slippage is in %")]
    assert app.session_state["spec"] == spec


def test_slippage_is_typed_in_percent_and_kept_as_a_fraction(tmp_path, monkeypatch):
    """The grid shows slippage in % and writes vectorbtX's fraction back; a new symbol starts at the
    usual value for its market, and a blank runs as that value."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import engine

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    spec["instruments"]["SPY"]["slippage"] = 0.0002
    assert engine.usual_slippage("spreadbet", "stocks") == 0.0002
    assert engine.usual_slippage("fx_spot") == engine.usual_slippage("cfd", "fx") == 0.0001
    # A blank is named as the usual one, so the plan (and the export) always says it.
    blank = copy.deepcopy(spec)
    blank["instruments"]["SPY"].pop("slippage")
    assert engine.instrument_fields(blank)["SPY"]["slippage"] == 0.0002
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["spec"] == spec                       # nothing written back on render
    # The Costs grid (a data editor, which AppTest reads but cannot edit) shows it in %.
    costs = [frame for frame in app.dataframe if frame.proto.editing_mode and "slippage" in frame.value.columns]
    assert len(costs) == 1 and costs[0].value.at["SPY", "slippage"] == "0.02", [frame.value for frame in costs]
    assert json.loads(costs[0].proto.columns)["slippage"]["label"] == "Slippage (% a fill)"


def test_the_data_page_names_an_import_and_warns_before_replacing_another_source(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    from conftest import make_ohlcv
    from openstrategy_studio import config, data as data_module, store

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    data_dir = config.data_dir(config.load_settings())
    bars = make_ohlcv(300, ("SPY",), base=100.0)
    frame = pd.DataFrame({field: bars[field]["SPY"] for field in ("open", "high", "low", "close", "volume")})
    data_module.store_dataset(data_dir, "SPY", "1d", frame, "file:spy.csv")
    clean, info = data_module.normalise_ohlcv(frame, "1d")
    item = {"id": "one", "name": "SPY", "ticker": "SPY", "timeframe": "1d", "frame": clean, "info": info,
            "found": data_module.validate_ohlcv(clean, "1d"), "source": "yahoo"}
    app = testing.AppTest.from_file(str(APP / "pages" / "1_Data.py"), default_timeout=180)
    app.session_state["pending_imports"] = [item]
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert any("already stored at 1d from another source (file:spy.csv" in item.value for item in app.warning)
    assert app.button(key="pending:one:store").disabled
    app.text_input(key="pending:one:name").set_value("spy").run()
    assert any("only in capitals" in item.value for item in app.error)
    app.text_input(key="pending:one:name").set_value("SPY_yahoo").run()
    app.button(key="pending:one:store").click().run()
    stored = store.list_datasets(data_dir)
    assert sorted(zip(stored["symbol"], stored["source"], stored["ticker"])) == [
        ("SPY", "file:spy.csv", "SPY"), ("SPY_yahoo", "yahoo", "SPY")]
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_leverage_follows_the_way_of_trading_unless_typed_in(tmp_path, monkeypatch):
    """The usual leverage fills in (1x bought outright, 5x for a stock spread bet, 30x for FX);
    one typed in by hand stays when the way of trading changes."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    app.radio(key=current_key(app, "setup.trading")).set_value("equity").run()
    assert app.session_state["spec"]["instruments"]["SPY"]["leverage"] == 1.0
    app.radio(key=current_key(app, "setup.trading")).set_value("spreadbet").run()
    assert app.session_state["spec"]["instruments"]["SPY"]["leverage"] == 5.0
    edited = copy.deepcopy(app.session_state["spec"])
    edited["instruments"]["SPY"].update(leverage=3.0, asset_class="index")
    app.session_state["spec"] = edited
    app.session_state["spec_rev"] = 20
    app.run()
    app.radio(key=current_key(app, "setup.trading")).set_value("cfd").run()
    assert app.session_state["spec"]["instruments"]["SPY"]["leverage"] == 3.0
    # The Risk tab says where it is set, and what it funds.
    assert any("leverage column of the instrument grid" in item.value for item in app.caption)
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_portfolio_is_unlevered_unless_typed_in(tmp_path, monkeypatch):
    """A stock spread bet traded on signals is 5x; held in a portfolio, the usual leverage is 1x."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = spread_bet_on_spy(tmp_path)
    spec["instruments"]["SPY"]["leverage"] = 5.0
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    app.radio(key=current_key(app, "setup.kind")).set_value("portfolio").run()
    assert app.session_state["spec"]["instruments"]["SPY"]["leverage"] == 1.0
    app.radio(key=current_key(app, "setup.kind")).set_value("signals").run()
    assert app.session_state["spec"]["instruments"]["SPY"]["leverage"] == 5.0
    assert not app.exception, [item.value for item in app.exception]


def test_the_usual_terms_follow_the_way_of_trading(tmp_path, monkeypatch):
    """EURUSD as a spread bet (30x, a pip) bought as FX spot is 1x and still a pip; back as a CFD it is
    30x again, still FX though FX spot never named its asset class."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    spec = load_template("ma_cross")
    del spec["instruments"]["EURUSD=X"]["asset_class"]
    spec["setup"]["trading"] = "fx_spot"
    spec["instruments"]["EURUSD=X"].update(leverage=1.0, contract_size=100000.0)
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = copy.deepcopy(spec)
    app.session_state["spec_rev"] = 0
    app.run()
    app.radio(key=current_key(app, "setup.trading")).set_value("cfd").run()
    fields = app.session_state["spec"]["instruments"]["EURUSD=X"]
    assert fields["asset_class"] == "fx" and fields["leverage"] == 30.0 and fields["point_size"] == 0.0001
    app.radio(key=current_key(app, "setup.trading")).set_value("equity").run()
    fields = app.session_state["spec"]["instruments"]["EURUSD=X"]
    assert fields["leverage"] == 1.0 and fields["point_size"] == 0.0001
    assert any("on EURUSD=X, 1 pip (0.0001)" in item.value for item in app.caption)
    assert not app.exception, [item.value for item in app.exception]


def test_the_risk_tab_sizes_from_the_account_and_measures_a_stop_from_a_line(tmp_path, monkeypatch):
    """Position size as two questions (what sets it, how much: the owner's call, 20 November 2026), a
    share of the balance or equity (issues 23 and 30), risk sizing with no stop (28), and a stop as a
    multiple of a line or at its level (26 and 31)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert app.session_state["spec"]["risk"]["sizing"] == {"risk_amount": 100.0}
    # A fixed risk of 100 becomes the share it is of the starting cash, of the equity.
    app.radio(key=current_key(app, "risk.sizing:amount_of")).set_value("equity").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"risk_pct": 0.01, "risk_of": "equity"}
    assert any("read at the close of each signal bar" in item.value for item in app.caption)
    # Sized by its value instead, still a share of the equity; then a fixed value, the same money.
    app.radio(key=current_key(app, "risk.sizing:by")).set_value("value").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"notional_pct": 0.1, "risk_of": "equity"}
    app.radio(key=current_key(app, "risk.sizing:amount_of")).set_value("fixed").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"notional": 1000.0}
    # A fixed size counts the way of trading's own unit: money a point on a spread bet.
    app.radio(key=current_key(app, "risk.sizing:by")).set_value("size").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"stake": 1.0}
    assert app.number_input(key=current_key(app, "risk.sizing.stake")).label.startswith("Stake per point")
    # A stop as a multiple of a line: the first feature, three times.
    app.radio(key=current_key(app, "risk.stop:unit")).set_value("multiple").run()
    stop = app.session_state["spec"]["risk"]["stop"]
    assert stop["unit"] == "multiple" and stop["multiple"] == 3.0 and stop["line"] == {"feature": "fast"}
    assert any("3 x the line from the price the trade fills at" in item.value for item in app.caption)
    # At the line's level, read on the signal bar; one on the wrong side of the fill is rejected.
    app.radio(key=current_key(app, "risk.stop:unit")).set_value("level").run()
    assert app.session_state["spec"]["risk"]["stop"]["unit"] == "level"
    assert any("on the wrong side is an order a broker rejects" in item.value for item in app.caption)
    assert validate_spec(app.session_state["spec"]) == []
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)
    # Risk sizing with no stop: sized from the leverage, and the tab says what that means.
    app.radio(key=current_key(app, "risk.sizing:by")).set_value("risk").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"risk_amount": 100.0}
    app.radio(key=current_key(app, "risk.sizing:amount_of")).set_value("balance").run()
    app.radio(key=current_key(app, "risk.stop:kind")).set_value("none").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"risk_pct": 0.01, "risk_of": "balance"}
    assert any("as if its margin were the risk" in item.value and "30x" in item.value for item in app.caption)
    assert validate_spec(app.session_state["spec"]) == []
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_a_fixed_size_is_counted_in_each_markets_unit(tmp_path, monkeypatch):
    """Shares bought outright, money a point on a spread bet, lots on FX, contracts on other CFDs."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    for trading, asset_class, wanted in (("equity", "stocks", "Shares per trade"),
                                         ("spreadbet", "fx", "Stake per point"),
                                         ("fx_spot", "fx", "Lots per trade"),
                                         ("cfd", "fx", "Lots per trade"),
                                         ("cfd", "index", "Contracts per trade")):
        spec = load_template("ma_cross")
        spec["setup"]["trading"] = trading
        spec["instruments"]["EURUSD=X"]["asset_class"] = asset_class
        if trading in ("cfd", "fx_spot"):
            spec["instruments"]["EURUSD=X"]["contract_size"] = 100000.0 if asset_class == "fx" else 1.0
        spec["risk"]["sizing"] = {"stake": 1.0}
        app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
        app.session_state["spec"] = spec
        app.session_state["spec_rev"] = 0
        app.run()
        assert not app.exception, (trading, [item.value for item in app.exception])
        assert app.number_input(key=current_key(app, "risk.sizing.stake")).label.startswith(wanted), trading


def test_a_new_symbol_starts_with_every_value_filled_in(tmp_path, monkeypatch):
    """No blank cells (the owner's call, 20 November 2026): a symbol added gets its market's usual values,
    and an older strategy's blank cells fill in on one press, leaving typed values alone."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import config, data as data_module

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    data_dir = config.data_dir(config.load_settings())
    synthetic = data_module.synthetic_ohlcv(["EURUSD=X"], n=300, seed=4)
    frame = pd.DataFrame({field: synthetic[field]["EURUSD=X"] for field in ("open", "high", "low", "close", "volume")})
    data_module.store_dataset(data_dir, "EURUSD=X", "1d", frame, "yahoo")
    spec = blank_spec()
    spec["setup"]["trading"] = "cfd"
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 0
    app.run()
    app.multiselect(key=current_key(app, "data.symbols")).select("EURUSD=X").run()
    assert not app.exception, [item.value for item in app.exception]
    fields = app.session_state["spec"]["instruments"]["EURUSD=X"]
    # An FX CFD: the pair read from its ticker, a lot of 100,000, 30x, a pip, and every cost written out.
    assert fields["asset_class"] == "fx" and fields["base_currency"] == "EUR" and fields["quote_currency"] == "USD"
    assert fields["contract_size"] == 100000.0 and fields["leverage"] == 30.0 and fields["point_size"] == 0.0001
    assert fields["commission_flat"] == 0.0 and fields["slippage"] == 0.0001 and fields["rounding"] == "floor"
    assert not [button for button in app.button if button.key == current_key(app, "instruments:fill")]
    # An older strategy with blanks: one press fills them, a typed spread stays.
    spec = app.session_state["spec"]
    spec["instruments"]["EURUSD=X"] = {"spread_points": 0.6, "point_size": 0.0001}
    app.session_state["spec"] = spec
    app.session_state["spec_rev"] = 30
    app.run()
    app.button(key=current_key(app, "instruments:fill")).click().run()
    fields = app.session_state["spec"]["instruments"]["EURUSD=X"]
    assert fields["spread_points"] == 0.6 and fields["contract_size"] == 100000.0 and fields["leverage"] == 30.0
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)


def test_every_card_folds_by_its_title(tmp_path, monkeypatch):
    """A rule card folds and opens by its name, like an expander (the owner's call, 20 November 2026: a
    small icon at the right was not found); folded, its title still shows its logic line."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    spec = copy.deepcopy(app.session_state["spec"])
    logic = lambda: [box for box in app.text_input if box.key == current_key(app, "rules.long_entry:logic")]
    title = app.button(key=current_key(app, "rules.long_entry:open"))
    assert title.label == "Long entry" and logic()                     # rules start open
    title.click().run()
    assert not logic() and any("Long entry" in text and " A " in text for text in card_titles(app))
    app.button(key=current_key(app, "rules.long_entry:open")).click().run()
    assert logic()
    # A feature's card the same way, closed to start with.
    assert not [box for box in app.number_input if box.key == current_key(app, "features.fast.params.timeperiod")]
    app.button(key=current_key(app, "features.fast:open")).click().run()
    assert app.number_input(key=current_key(app, "features.fast.params.timeperiod")).value == 50
    assert app.session_state["spec"] == spec                            # folding changes nothing in the strategy
    assert not app.exception, [item.value for item in app.exception]


def finished_job(tmp_path, held_back=0.3, constraints=None, instrument=None, folds=None, budget=6):
    """A small real optimisation (the MA cross on synthetic prices, six random trials, 30% held back), run in
    this process: the worker's own main, on a job folder written as jobs.start_job writes one. `instrument`:
    fields to change on the pair; `folds`: the job's folds, as the Optimise page sets them."""
    import pickle
    from openstrategy_studio import config, jobs, optimise, store, worker
    from conftest import make_ohlcv

    data_dir = config.data_dir(config.load_settings())
    spec = load_template("ma_cross")
    spec["account"].pop("rates")
    spec["features"]["fast"]["params"]["timeperiod"] = 5
    spec["features"]["slow"]["params"]["timeperiod"] = 20
    spec["instruments"]["EURUSD=X"].update(instrument or {})
    spec["optimise"]["variables"] = {"features.fast.params.timeperiod": {"start": 3, "stop": 8, "step": 1},
                                     "features.slow.params.timeperiod": {"start": 15, "stop": 30, "step": 5}}
    job = {"name": "MA cross: random", "kind": "optimise", "spec": spec, "space": optimise.build_space(spec),
           "compare": None, "backend": "random", "backend_options": {"batch": 1}, "budget": budget, "metric": "sharpe",
           "direction": "maximize", "seed": 0, "workers": 1, "settings": config.load_settings(),
           "constraints": constraints or {}, "split": {"train_fraction": 1 - held_back} if held_back else None,
           "walk_forward": None, "stability": False, "folds": folds, "job_id": "test-job", "created_at": jobs.now(),
           "data_dir": str(data_dir)}
    job_dir = data_dir / "jobs" / "test-job"
    job_dir.mkdir(parents=True)
    jobs.write_json(job_dir / "job.json", job)
    with open(job_dir / "frames.pkl", "wb") as handle:
        pickle.dump(make_ohlcv(500, ("EURUSD=X",)), handle)
    jobs.write_json(job_dir / "progress.json", {"status": "starting", "done": 0, "budget": budget,
                                                "updated_at": jobs.now()})
    store.add_job(data_dir, "test-job", job["name"], "optimise", "random", job["created_at"], job_dir)
    assert worker.main(job_dir) == "done", jobs.read_progress(job_dir).get("error")
    return job_dir


def test_the_optimiser_results_lead_with_the_best_settings_in_and_out_of_sample(tmp_path, monkeypatch):
    """The answer first (the owner's call, 20 November 2026): the best settings named as the strategy names
    them, the objective in sample and out of sample, and the best trial's backtest with its numbers in each
    part; the trials table names the settings the same way."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis, jobs

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    job_dir = finished_job(tmp_path)
    summary = jobs.read_summary(job_dir)
    # The job saved its best trial's backtest as it finished, so the page opens it without a run (the owner,
    # 25 November 2026).
    assert (job_dir / "best_backtest.pkl").exists() and "best_backtest_error" not in summary
    reruns = []
    real = analysis.split_backtest
    monkeypatch.setattr(analysis, "split_backtest", lambda *args, **kwargs: reruns.append(1) or real(*args, **kwargs))
    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    metrics = {item.label: item.value for item in app.metric}
    assert summary["held_out"]["n_trades"] > 0
    assert metrics["Sharpe ratio, out of sample"] == f"{summary['held_out']['metric']:.4g}"
    assert "Sharpe ratio, in sample" in metrics
    # The job's number and objective in the picker, its method and objective under it.
    picker = next(item for item in app.selectbox if item.label == "Job")
    assert picker.options[0].startswith("#1 MA cross: random · Sharpe ratio, maximised · "), picker.options
    assert "Random search · Sobol sequence · Sharpe ratio, maximised" in [item.value for item in app.caption]
    settings_table = app.dataframe[0].value
    assert list(settings_table.columns) == ["Setting", "Best", "When the job ran", "Searched"]
    assert set(settings_table["Setting"]) == {"fast › timeperiod", "slow › timeperiod"}
    assert set(settings_table["Searched"]) == {"3 to 8 by 1", "15 to 30 by 5"}
    best = summary["best_params"]
    assert settings_table.set_index("Setting").loc["fast › timeperiod", "Best"] == str(best["features.fast.params.timeperiod"])
    # The trials table names the settings too, never as param: paths. It is in the other view.
    view = next(item for item in app.get("button_group") if item.label == "Show")
    view.set_value("How the search went").run()
    assert not app.exception, [item.value for item in app.exception]
    assert not [item for item in app.get("button_group") if item.label == "Tear sheet of"]
    trials_table = app.dataframe[-1].value
    assert "fast › timeperiod" in trials_table.columns and not any(str(c).startswith("param:") for c in trials_table.columns)
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)
    next(item for item in app.get("button_group") if item.label == "Show").set_value("The best trial's backtest").run()
    assert not app.exception, [item.value for item in app.exception]
    # The best trial's backtest opens on its own, with no button (the owner, 26 November 2026): its out-of-sample
    # objective is the job's held-back number. The search's tables are the other view.
    assert not [button for button in app.button if button.label == "Show its backtest"]
    assert not reruns, "the stored backtest was run again"
    shown = app.session_state["best_backtest"]
    assert shown["split"] is not None
    table = shown["table"]
    assert list(table.columns) == ["In sample (searched)", "Out of sample (held back)", "Whole period",
                                   "Held back, inside the whole run"]
    assert table.loc["Sharpe ratio (the objective)", "Out of sample (held back)"] == f"{summary['held_out']['metric']:.4g}"
    assert "Sharpe ratio" not in table.index          # the objective is not listed twice
    # Each part's own run from the starting cash, and the whole run's held-back part from what it had grown to.
    assert table.loc["Starting value", "Out of sample (held back)"] == "10,000"
    assert table.loc["From", "Held back, inside the whole run"] == table.loc["From", "Out of sample (held back)"]
    # A whole tear sheet for each part (the owner, 25 November 2026), the whole period's first.
    assert list(shown["windows"]) == ["In sample (searched)", "Out of sample (held back)", "Whole period"]
    assert any(item.value == "**Portfolio**" for item in app.markdown)
    part = next(item for item in app.get("button_group") if item.label == "Tear sheet of")
    part.set_value("Out of sample (held back)").run()
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)
    # The equity curve on a log scale, from the checkbox beside the view.
    next(box for box in app.checkbox if box.label == "Log scale").check().run()
    assert not app.exception, [item.value for item in app.exception]
    # A job from before the backtest was saved runs it, with the same numbers, and keeps it, so it runs once.
    (job_dir / "best_backtest.pkl").unlink()
    del app.session_state["best_backtest"]
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert reruns == [1]
    assert app.session_state["best_backtest"]["table"].equals(table)
    assert (job_dir / "best_backtest.pkl").exists()
    del app.session_state["best_backtest"]
    app.run()
    assert reruns == [1]


def test_a_job_is_saved_renamed_and_deleted(tmp_path, monkeypatch):
    """The owner (26 November 2026): the list of jobs was just always there. Beside the results page's picker a job
    is saved (★) or renamed, keeping it picked, and deleted after a confirmation, with its folder; the Optimise
    page deletes every finished job not saved; a running job is never deleted."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    import os
    import shutil
    from openstrategy_studio import config, jobs, store

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    job_dir = finished_job(tmp_path)
    data_dir = config.data_dir(config.load_settings())
    copy_dir = job_dir.parent / "copy-job"                  # a second finished job, newer than the first
    shutil.copytree(job_dir, copy_dir)
    store.add_job(data_dir, "copy-job", "Copy", "optimise", "random", jobs.now(), copy_dir)
    listed = lambda: jobs.list_jobs(data_dir).set_index("job_id")
    picker = lambda app: next(item for item in app.selectbox if item.label == "Job")
    button = lambda app, label: next(item for item in app.button if item.label == label)

    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=180)
    app.run()
    assert picker(app).value == "copy-job"                  # the newest first
    picker(app).set_value("test-job").run()
    button(app, "Save").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert listed().loc["test-job", "saved"] and not listed().loc["copy-job", "saved"]
    assert picker(app).value == "test-job" and any(option.startswith("#1 MA cross: random ★") for option in
                                                     picker(app).options)
    next(box for box in app.text_input if box.label == "New name").input("Best MA").run()
    button(app, "Rename").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert listed().loc["test-job", "name"] == "Best MA" and picker(app).value == "test-job"
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)

    # The Optimise page lists which are saved, and deletes the rest.
    page = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    page.session_state["spec"] = load_template("ma_cross")
    page.session_state["spec_rev"] = 0
    page.run()
    table = next(frame.value for frame in page.dataframe if "Saved" in frame.value.columns)
    assert table.set_index("#")["Saved"].to_dict() == {1: True, 2: False}
    button(page, "Delete them for good").click().run()
    assert not page.exception, [item.value for item in page.exception]
    assert not copy_dir.exists() and job_dir.exists() and list(listed().index) == ["test-job"]

    # A running job is never deleted: its worker would write into a folder that is gone.
    progress = jobs.read_json(job_dir / "progress.json")
    jobs.write_json(job_dir / "progress.json", dict(progress, status="running", pid=os.getpid(), updated_at=jobs.now()))
    (job_dir / "heartbeat").touch()
    with pytest.raises(ValueError, match="still running"):
        jobs.delete_job(data_dir, "test-job")
    jobs.write_json(job_dir / "progress.json", progress)

    # Unsaved, then deleted from the results page, after the confirmation.
    app.run()
    button(app, "Unsave").click().run()
    assert not listed().loc["test-job", "saved"]
    button(app, "Delete for good").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert not job_dir.exists() and store.list_jobs(data_dir).empty
    assert any("No optimisations yet" in item.value for item in app.markdown)


def test_the_best_trials_chart_marks_where_the_held_back_part_starts():
    """A dashed line at the first held-back bar, that part shaded, both sides labelled, on every view."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis, engine
    from conftest import make_ohlcv

    spec = load_template("ma_cross")
    spec["account"].pop("rates")
    frames = make_ohlcv(300, ("EURUSD=X",))
    results = analysis.extract_results(engine.run_backtest(spec, frames), benchmark=True)
    split = frames["close"].index[200]
    for view in ("equity", "drawdown", "price", "rolling", "trades"):
        fig = analysis.fig_split(results, split, "page", view=view)
        kinds = sorted(shape.type for shape in fig.layout.shapes if shape.type in ("line", "rect"))
        assert kinds == ["line", "rect"], (view, kinds)
        assert [shape.line.dash for shape in fig.layout.shapes if shape.type == "line"] == ["dash"]
        assert any("Out of sample" in note.text for note in fig.layout.annotations), view
    # A chart that does not run along the dates (the monthly heatmap) is not marked, and with nothing held back
    # nothing is.
    assert not analysis.FIGURES["monthly"].get("dates")
    assert not [shape for shape in analysis.fig_split(results, split, "page", view="monthly").layout.shapes
                if shape.type == "rect"]
    assert not analysis.fig_split(results, None, "page").layout.shapes
    # A log scale where the figure can take one (a value or a price), and nowhere else (a drawdown is below zero).
    for view in ("equity", "price"):
        assert analysis.fig_split(results, split, "page", view=view, log=True).layout.yaxis.type == "log", view
    assert analysis.fig_split(results, split, "page", view="drawdown", log=True).layout.yaxis.type != "log"
    assert analysis.fig_split(results, split, "page", view="equity").layout.yaxis.type != "log"


def test_the_held_back_part_inside_the_whole_run_is_the_charts():
    """continuing_extras: the whole run's held-back part, from what the account was worth at the split. From the
    first bar it is the whole run; from a later one, its trades are those entered from there."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis, engine, optimise
    from conftest import make_ohlcv

    spec = load_template("ma_cross")
    spec["account"].pop("rates")
    spec["features"]["fast"]["params"]["timeperiod"] = 5
    spec["features"]["slow"]["params"]["timeperiod"] = 20
    pf = engine.run_backtest(spec, make_ohlcv(500, ("EURUSD=X",)))["pf"]
    index = pf.wrapper.index
    whole = optimise.trial_extras(pf)
    start = analysis.continuing_extras(pf, index[0])
    assert start["start_value"] == 10000 and start["n_trades"] == whole["n_trades"]
    assert start["total_return"] == pytest.approx(whole["total_return"])
    assert start["max_drawdown"] == pytest.approx(whole["max_drawdown"])
    later = analysis.continuing_extras(pf, index[300])
    entered = pf.trades.records["entry_idx"]
    assert later["n_trades"] == int((entered >= 300).sum()) and 0 < later["n_trades"] < whole["n_trades"]
    value = analysis._total(pf.value())            # one column: an FX pair runs through PortfolioBacktest
    assert later["start_value"] == pytest.approx(float(value.iloc[299]))
    assert later["total_return"] == pytest.approx(float(value.iloc[-1] / value.iloc[299] - 1))
    # A run with no trades has no Sharpe ratio to show, whatever vectorbt says (+inf, docs/UPSTREAM.md 39).
    assert analysis.shown_number(float("inf"), "ratio") == "—"
    assert analysis.shown_number(float("inf"), "factor") == "no losing trades"
    assert analysis.shown_number(0.2, "pct") == "20.0%"


def test_a_held_back_test_with_no_trades_says_why(tmp_path, monkeypatch):
    """The owner (25 November 2026): no trade out of sample and a Sharpe ratio of inf looked like a bug. The page
    says the test made no trades, from the starting cash, in the backtest's words with the Studio's hint, and
    shows no score for it."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import jobs

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    job_dir = finished_job(tmp_path, constraints={"min_trades": 2})
    summary = jobs.read_summary(job_dir)
    assert {"notes", "error"} <= set(summary["held_out"])             # what the worker records now
    note = ("EURUSD=X: 78 of 78 entries fall below the minimum tradeable size (1.0 £/point) and were rounded to "
            "zero, so NO ORDER is placed on those bars.")
    summary["held_out"].update(n_trades=0, metric=float("inf"), sharpe=float("inf"), total_return=0.0, notes=note)
    jobs.write_json(job_dir / "summary.json", summary)
    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Sharpe ratio, out of sample"] == "—"
    (warning,) = [item.value for item in app.warning if "held-back test" in item.value]
    assert "made no trades" in warning and "starting cash (10,000)" in warning and "minimum tradeable size" in warning
    assert any(item.value.startswith("In the Studio: raise the risk or the amount") for item in app.caption)


def test_the_best_settings_over_random_periods(tmp_path, monkeypatch):
    """The owner (25 November 2026): the spread of each number over periods drawn at random, each from the
    starting cash, as a box chart; the same seed draws the same periods, and the held-back history's are in it."""
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import analysis, config, jobs, optimise

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    job_dir = finished_job(tmp_path)
    job = jobs.read_job(job_dir)
    with open(job_dir / "frames.pkl", "rb") as handle:
        import pickle
        frames = pickle.load(handle)
    params = jobs.read_summary(job_dir)["best_params"]
    seen = []
    out = analysis.random_periods(job, frames, params, config.load_settings(), count=6, length=0.3, seed=3,
                                  progress=lambda done, total: seen.append((done, total)))
    periods = out["periods"]
    assert len(periods) == 6 and seen[-1] == (6, 6) and out["bars"] == 150
    index = frames["close"].index
    for start, end in zip(periods["from"], periods["to"]):
        assert index.get_loc(end) - index.get_loc(start) + 1 == 150
    again = analysis.random_periods(job, frames, params, config.load_settings(), count=6, length=0.3, seed=3)
    assert again["periods"]["from"].tolist() == periods["from"].tolist()
    held = analysis.random_periods(job, frames, params, config.load_settings(), count=4, length=0.5, seed=3,
                                   within="held_back")["periods"]
    first_held = optimise.split_window(index, job["split"]["train_fraction"])[1][0]
    assert (held["from"] >= first_held).all()
    assert {"annualised_return", "max_drawdown", "sharpe", "n_trades", "metric", "notes"} <= set(periods.columns)
    fig = analysis.fig_periods(periods, out["whole"], "page")
    assert [title.text for title in fig.layout.annotations[:4]] == ["Annualised return", "Max drawdown",
                                                                   "Sharpe ratio", "Trades"]
    assert len({trace.marker.color for trace in fig.data}) == 1           # one series, one colour


def test_the_optimise_page_has_no_job_runner_check(tmp_path, monkeypatch):
    """The self-test job was for checking an install, not for a user (the owner's call, 20 November 2026)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    assert not [panel for panel in app.expander if "job runner" in panel.label]
    assert not [button for button in app.button if "self-test" in button.label.lower()]


def test_a_running_job_shows_its_stage_progress_and_time_left(tmp_path, monkeypatch):
    """A running job is a card: what it is doing in words, a bar of trials done, the best so far, how long it
    has run and how long is left at the pace so far (the owner's call, 20 November 2026)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    import os
    from openstrategy_studio import config, jobs, store

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    data_dir = config.data_dir(config.load_settings())
    job_dir = data_dir / "jobs" / "running-job"
    job_dir.mkdir(parents=True)
    minute_ago = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=60)).isoformat()
    jobs.write_json(job_dir / "progress.json", {
        "status": "running", "done": 10, "budget": 40, "best_value": 0.71, "pid": os.getpid(),
        "started_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=150)).isoformat(),
        "searching_since": minute_ago, "updated_at": jobs.now(), "stage": "Searching"})
    (job_dir / "heartbeat").touch()
    store.add_job(data_dir, "running-job", "MA cross: optuna", "optimise", "optuna", jobs.now(), job_dir)
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    (bar,) = app.get("progress")
    text = bar.proto.text
    # 30 trials left at 10 a minute: about 3 minutes.
    assert "10 of 40 trials" in text and "best so far 0.71" in text and "about 3 min left" in text, text
    assert any("Searching" in item.value for item in app.markdown if "os-meta" in item.value)
    assert [button for button in app.button if button.key == "cancel:running-job"]


def test_every_backend_shows_its_options_named_in_words(tmp_path, monkeypatch):
    """Every backend is asked the same two things, a backend and one of its methods (the owner, 25 November 2026:
    Optuna asked for a sampler, SciPy for a method, Nevergrad for an optimiser). Each method's options render,
    each with a tooltip; the job is named by both; BUTChC's batch starts at the worker count."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import config, optimise

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    picker = next(item for item in app.selectbox if item.label == "Backend")
    assert picker.options == [optimise.BACKENDS[name]["label"] for name in optimise.BACKENDS]
    for name in optimise.BACKENDS:
        next(item for item in app.selectbox if item.label == "Backend").set_value(name).run()
        for method in optimise.METHODS[name]:
            box = next(item for item in app.selectbox if item.label == "Method")
            assert box.options == [item["label"] for item in optimise.METHODS[name].values()]
            assert box.disabled == (len(optimise.METHODS[name]) == 1)
            if not box.disabled:
                box.set_value(method).run()
            assert not app.exception, (name, method, [item.value for item in app.exception])
            assert not settings_without_a_tooltip(app), (name, method, settings_without_a_tooltip(app))
            assert next(item for item in app.text_input if item.label == "Job name").value.endswith(
                optimise.search_label(name, {"method": method}))
        if name == "butchc":
            batch = next(item for item in app.slider if item.label == "Batch")
            assert batch.value == min(16, int(config.load_settings()["workers"]))
    assert not [item for item in app.warning if "optional extra" in item.value]
    # Old labels are gone: one word, Method, for every backend.
    assert not [item for item in app.selectbox if item.label in ("Sampler", "Optimiser")]
    # A comparison picks searches, each a backend and a method.
    next(item for item in app.radio if item.label == "Run").set_value("compare").run()
    assert not app.exception, [item.value for item in app.exception]
    searches = next(item for item in app.multiselect if item.label == "Searches to compare")
    assert "Optuna · CMA-ES" in searches.options and "Grid search · Many trials per backtest" not in searches.options
    assert not [item for item in app.selectbox if item.label == "Method"]


def test_the_optimise_page_folds_by_section(tmp_path, monkeypatch):
    """The owner (25 November 2026): with many variables the page grew too long. Each part is a section that folds."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    labels = [panel.label for panel in app.expander]
    assert labels[:5] == ["Variables to optimise (2)", "Search", "Objective and constraints", "Robustness", "Jobs"]


def test_the_optimise_settings_last_across_pages(tmp_path, monkeypatch):
    """The owner (25 November 2026): leaving the Optimise page and coming back put every search setting back to
    its default, which also made Start check the ranges again. They are kept for the session (ui.kept)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    app.switch_page("pages/4_Optimise.py").run()
    assert not app.exception, [item.value for item in app.exception]
    widget = lambda kind, label: next(item for item in getattr(app, kind) if item.label == label)
    widget("selectbox", "Backend").set_value("nevergrad").run()
    widget("selectbox", "Method").set_value("PSO").run()
    widget("number_input", "Budget (trials)").set_value(120).run()
    widget("number_input", "Minimum trades").set_value(7).run()
    widget("selectbox", "Metric").set_value("sortino").run()
    widget("slider", "Hold back the last part of the history for testing").set_value(20).run()
    spec = copy.deepcopy(app.session_state["spec"])
    app.switch_page("pages/2_Build.py").run()
    assert not app.exception, [item.value for item in app.exception]
    app.switch_page("pages/4_Optimise.py").run()
    assert not app.exception, [item.value for item in app.exception]
    assert widget("selectbox", "Backend").value == "nevergrad"
    assert widget("selectbox", "Method").value == "PSO"
    assert widget("number_input", "Budget (trials)").value == 120
    assert widget("number_input", "Minimum trades").value == 7
    assert widget("selectbox", "Metric").value == "sortino"
    assert widget("slider", "Hold back the last part of the history for testing").value == 20
    assert app.session_state["spec"] == spec            # they are the page's, not the strategy's
    # A budget is kept for each backend; a backend's own default is where one starts.
    widget("selectbox", "Backend").set_value("optuna").run()
    assert widget("number_input", "Budget (trials)").value == 200
    assert not [item for item in app.warning if "Session State API" in item.value]


def test_a_stopped_job_is_reported_until_another_is_started(tmp_path, monkeypatch):
    """The owner (25 November 2026): a job that died on 19 November still headed the Optimise page under every
    later run. Its warning shows while it is the newest job; once another is started its row says it stopped."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    from openstrategy_studio import config, jobs, store

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    data_dir = config.data_dir(config.load_settings())

    def add(job_id, status, created_at, backend="optuna"):
        job_dir = data_dir / "jobs" / job_id
        job_dir.mkdir(parents=True)
        jobs.write_json(job_dir / "progress.json", {
            "status": status, "done": 0, "budget": 200, "updated_at": created_at,
            "error": "The job stopped without finishing, after 0 of 200 trials." if status == "stopped" else None})
        store.add_job(data_dir, job_id, f"MA cross: {job_id}", "optimise", backend, created_at, job_dir)

    def render():
        app = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
        app.session_state["spec"] = load_template("ma_cross")
        app.session_state["spec_rev"] = 0
        app.run()
        assert not app.exception, [item.value for item in app.exception]
        return app, [item.value for item in app.warning if "stopped without finishing" in item.value]

    add("old", "stopped", "2026-11-19T13:21:52+00:00")
    _, said = render()
    assert said == ["**MA cross: old**: The job stopped without finishing, after 0 of 200 trials."]
    add("new", "done", "2026-11-25T09:51:20+00:00", backend="butchc")
    app, said = render()
    assert not said, said
    table = app.dataframe[-1].value.set_index("Job")
    # One trials cell says how a job ended (no Status or Budget column), and the search names its method too.
    assert list(table.columns) == ["#", "Saved", "Search", "Objective", "Best", "Trials", "Started"]
    assert table.loc["MA cross: old", "Trials"] == "0 of 200, stopped from outside before it finished"
    assert table.loc["MA cross: new", "Trials"] == "0 of 200"
    assert table.loc["MA cross: new", "Search"] == "BUTChC · Bayesian update tree"
    assert table["#"].tolist() == [2, 1]


def test_a_job_scored_across_folds_shows_them(tmp_path, monkeypatch):
    """The owner (26 November 2026): the best trial's numbers in sample should carry a range from the folds. A real
    job scored across six folds: the card sets the whole searched history beside the held-back test, then the
    folds combined and the probability of backtest overfitting; the backtest's table has a column across the
    folds, with a box per number under it."""
    testing = pytest.importorskip("streamlit.testing.v1")
    from conftest import windows_or_skip

    windows_or_skip()
    from openstrategy_studio import jobs

    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    job_dir = finished_job(tmp_path, folds={"count": 6, "summary": "mean_less_sd", "min_profitable": None},
                           budget=16)
    summary = jobs.read_summary(job_dir)
    assert len(summary["folds"]["windows"]) == 6 and summary["in_sample"]["n_trades"] is not None
    trials = jobs.read_trials(job_dir)
    assert trials[trials["phase"] == "search"]["folds_json"].notna().all()
    assert set(trials["phase"]) >= {"search", "in_sample", "held_out"}
    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=300)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    labels = [item.label for item in app.metric]
    assert "Sharpe ratio, across 6 folds" in labels and "Probability of backtest overfitting" in labels, labels
    assert any(item.value.startswith("Each fold's Sharpe ratio: ") for item in app.caption)
    table = next(frame.value for frame in app.dataframe if "In sample, across its 6 folds" in frame.value.columns)
    assert ", median " in table.set_index("")["In sample, across its 6 folds"]["Trades"]
    assert any("The 6 folds" in item.value for item in app.markdown)
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)
    # The Optimise page: folds for a rule strategy, with what each choice does; never for a portfolio.
    page = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    page.session_state["spec"] = load_template("ma_cross")
    page.session_state["spec_rev"] = 0
    page.run()
    toggle = next(item for item in page.toggle if item.label == "Score each trial across folds")
    toggle.set_value(True).run()
    assert not page.exception, [item.value for item in page.exception]
    assert {"Folds", "Folds that must be profitable (0 = off)"} <= {item.label for item in page.number_input}
    assert "Combine the folds by" in [item.label for item in page.selectbox]
    assert not settings_without_a_tooltip(page), settings_without_a_tooltip(page)
    portfolio = testing.AppTest.from_file(str(APP / "pages" / "4_Optimise.py"), default_timeout=180)
    portfolio.session_state["spec"] = load_template("sixty_forty")
    portfolio.session_state["spec_rev"] = 0
    portfolio.run()
    assert next(item for item in portfolio.toggle if item.label == "Score each trial across folds").disabled


def test_a_job_with_no_valid_trial_says_why(tmp_path, monkeypatch):
    """Every trial under the minimum trades: the page says so, with the commonest reason, not an empty page. Here a
    minimum no trial's entry signals reach, so every trial is ruled out from its signals, with no backtest
    (optimise.screen), and the page says that, and how many it counted."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    finished_job(tmp_path, constraints={"min_trades": 100000})
    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    (warning,) = [item.value for item in app.warning if "No trial was valid" in item.value]
    assert "Most often: at most N trades: only N entry signals (not run) (6 of 6 trials)" in warning, warning
    assert "The most entry signals any trial had was" in warning, warning
    assert any(item.value.startswith("6 of 6 trials were ruled out from their signals alone") for item in app.caption)
    assert not [item for item in app.metric if "in sample" in item.label]


def test_a_job_with_no_valid_trial_says_what_the_backtests_said(tmp_path, monkeypatch):
    """The owner's MA job (20 November 2026): every trial said only "only 1 trades". Now the page names what
    vectorbtX said most often about the trials' runs, with the Studio's hint, and each trial's notes are in
    the table. Here every entry is under the minimum stake (risk 100 over 60 points is a stake of 1.67). A minimum
    of one trade, which every trial's signals reach, so each is run (a higher one is ruled out from the signals
    before vectorbtX can say anything: optimise.screen)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    finished_job(tmp_path, constraints={"min_trades": 1}, instrument={"min_size": 5.0})
    app = testing.AppTest.from_file(str(APP / "pages" / "5_Optimiser_results.py"), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    (warning,) = [item.value for item in app.warning if "No trial was valid" in item.value]
    assert "Most often: only N trades (6 of 6 trials)" in warning, warning
    assert "The backtests said, in 6 of 6 trials, for example: \"EURUSD=X: " in warning, warning
    assert "minimum tradeable size" in warning, warning
    assert any(item.value.startswith("In the Studio: raise the risk or the amount") for item in app.caption)
    tables = [frame.value for frame in app.dataframe if "What the backtest said" in frame.value.columns]
    assert tables and tables[0]["What the backtest said"].str.contains("minimum tradeable size").all()
