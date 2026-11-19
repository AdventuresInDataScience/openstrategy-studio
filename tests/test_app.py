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
    # The picker remembers where it was; the suggested name moves on.
    assert app.selectbox(key="pick:family:Indicator").value == "Bands and channels"
    # Rules: a second condition joins the logic line with "and", and the box shows it; a broken
    # line says what is wrong.
    app.button(key=current_key(app, "rules.long_entry:add")).click().run()
    rule = app.session_state["spec"]["rules"]["long_entry"]
    assert set(rule["conditions"]) == {"A", "B"} and rule["logic"] == "A and B"
    assert app.text_input(key=current_key(app, "rules.long_entry:logic")).value == "A and B"
    app.text_input(key=current_key(app, "rules.long_entry:logic")).set_value("A or B or").run()
    assert any("ends too soon" in item.value for item in app.error)
    app.text_input(key=current_key(app, "rules.long_entry:logic")).set_value("A or not B").run()
    assert app.session_state["spec"]["rules"]["long_entry"]["logic"] == "A or not B"
    # Removing B takes it out of the logic line too.
    app.button(key=current_key(app, "rules.long_entry.conditions.B:remove")).click().run()
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
    assert app.session_state["spec"]["rules"]["long_entry"]["logic"] == "trend and A"
    app.text_input(key=current_key(app, "rules.long_entry.conditions.A:name")).set_value("trend").run()
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
    app.run()
    app.number_input(key=current_key(app, "features.fast.params.timeperiod")).set_value(60).run()
    assert not app.exception, [item.value for item in app.exception]
    labels = [panel.label for panel in app.expander]
    assert "fast: Simple Moving Average · SMA(close, 60)" in labels, labels
    # An input too: the slow average applied to the open instead of the close.
    app.selectbox(key=current_key(app, "features.slow.inputs.close")).set_value("open").run()
    labels = [panel.label for panel in app.expander]
    assert "slow: Simple Moving Average · SMA(open, 200)" in labels, labels
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
    app.run()
    app.button(key=current_key(app, "rules.long_exit:add")).click().run()
    rule = app.session_state["spec"]["rules"]["long_exit"]
    assert set(rule["conditions"]) == {"A", "B"} and rule["logic"] == "A and B"
    app.selectbox(key=current_key(app, "rules.long_exit.conditions.B:kind")).set_value("calendar").run()
    assert app.session_state["spec"]["rules"]["long_exit"]["conditions"]["B"] == {"filter": "weekday",
                                                                                   "days": [0, 1, 2, 3, 4]}
    app.selectbox(key=current_key(app, "rules.long_exit.conditions.B:kind")).set_value("higher").run()
    higher = app.session_state["spec"]["rules"]["long_exit"]["conditions"]["B"]
    assert higher["filter"] == "higher_timeframe" and higher["timeframe"] == "1w"     # the first above 1d
    assert validate_spec(app.session_state["spec"]) == []
    # The cross may have happened on any of the last 5 bars.
    app.selectbox(key=current_key(app, "rules.long_entry.conditions.A:timing")).set_value("within_bars").run()
    assert app.session_state["spec"]["rules"]["long_entry"]["conditions"]["A"]["within_bars"] == 3
    app.number_input(key=current_key(app, "rules.long_entry.conditions.A.within_bars")).set_value(5).run()
    condition = app.session_state["spec"]["rules"]["long_entry"]["conditions"]["A"]
    assert condition["within_bars"] == 5 and condition["op"] == "crosses_above"
    labels = [panel.label for panel in app.expander]
    assert "A · fast crosses above slow, on any of the last 5 bars" in labels, labels
    assert "Long entry · A" in labels and "Short entry · not used" in labels, labels
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
    assert app.session_state["spec"]["rules"]["short_entry"]["logic"] == "A"
    app.multiselect(key=current_key(app, "rules.short_entry.conditions.A:optimise")).select(
        "rules.short_entry.conditions.A.op").run()
    assert "rules.short_entry.conditions.A.op" in app.session_state["spec"]["optimise"]["variables"]
    app.button(key=current_key(app, "rules.short_entry:delete_rule")).click().run()
    spec = app.session_state["spec"]
    assert spec["rules"]["short_entry"] is None
    assert not any(path.startswith("rules.short_entry") for path in spec["optimise"]["variables"])
    # The template's own variables stay.
    assert "features.fast.params.timeperiod" in spec["optimise"]["variables"]
    app.button(key=current_key(app, "rules.long_exit:delete_rule")).click().run()
    assert app.session_state["spec"]["rules"]["long_exit"] is None
    assert not app.exception, [item.value for item in app.exception]


def test_a_conditions_values_are_optimised_from_its_panel(tmp_path, monkeypatch):
    """Each condition's Optimise list adds its values to the Optimise page with a starting range,
    bars ago at 0 included; taking one out, or deleting the condition, removes it."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("rsi_reversion")
    app.session_state["spec_rev"] = 0
    app.run()
    base = "rules.long_entry.conditions.A"
    picker = app.multiselect(key=current_key(app, f"{base}:optimise"))
    # Shown in words, each with its value now (AppTest lists the shown text).
    assert picker.options == ["the comparison (now crosses above)", "bars ago (left) (now 0)",
                              "the number (right) (now 30.0)"], picker.options
    # The template already searches the number (15 to 40), so the list opens with it.
    assert picker.value == [f"{base}.right.const"]
    picker.select(f"{base}.left.shift").run()
    found = app.session_state["spec"]["optimise"]["variables"]
    assert found[f"{base}.right.const"] == {"start": 15, "stop": 40}
    assert found[f"{base}.left.shift"] == {"start": 0, "stop": 2, "step": 1}         # 0, 1 and 2 bars ago
    assert any("Searched: the number (right): 15 to 40; bars ago (left): 0 to 2 by 1" in item.value
               for item in app.caption), [item.value for item in app.caption]
    app.multiselect(key=current_key(app, f"{base}:optimise")).unselect(f"{base}.right.const").run()
    found = app.session_state["spec"]["optimise"]["variables"]
    assert f"{base}.right.const" not in found and f"{base}.left.shift" in found
    # A trial's bars ago of 0, written back by Apply to the strategy, is left as it is.
    spec = app.session_state["spec"]
    spec["rules"]["long_entry"]["conditions"]["A"]["left"]["shift"] = 0
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
    """The owner's run (17 November 2026): SPY as a spread bet, 0.05 of the cash risked over a
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
    spec["risk"] = {"sizing": {"risk_pct": 0.05}, "stop": {"kind": "fixed", "points": 100.0}, "target": None}
    return spec


def test_the_risk_tab_shows_leverage_and_checks_the_trade_size(tmp_path, monkeypatch):
    """What 0.05 means in money, the leverage a blank box gives, and the trade the run would cut down."""
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
    assert any(item.value.startswith("0.05 is 5.00% of the starting cash of 10,000: 500.00 lost") for item in app.caption)
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
    """vectorbtX 2b2d090 on the Risk tab: a share of the balance or equity (issues 23 and 30), risk
    sizing with no stop (28), and a stop as a multiple of a line or at its level (26 and 31)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    app = testing.AppTest.from_file(str(APP / "pages" / "2_Build.py"), default_timeout=180)
    app.session_state["spec"] = load_template("ma_cross")
    app.session_state["spec_rev"] = 0
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    # A share of the equity, then a position worth a share of it: the basis goes with the share.
    app.radio(key=current_key(app, "risk.sizing.risk_of")).set_value("equity").run()
    assert app.session_state["spec"]["risk"]["sizing"]["risk_of"] == "equity"
    assert any("read at the close of each signal bar" in item.value for item in app.caption)
    app.radio(key=current_key(app, "risk.sizing:kind")).set_value("notional_pct").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"notional_pct": 0.1, "risk_of": "equity"}
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
    app.radio(key=current_key(app, "risk.sizing:kind")).set_value("risk_pct").run()
    app.radio(key=current_key(app, "risk.stop:kind")).set_value("none").run()
    assert app.session_state["spec"]["risk"]["sizing"] == {"risk_pct": 0.01, "risk_of": "equity"}
    assert any("as if its margin were the risk" in item.value and "30x" in item.value for item in app.caption)
    assert validate_spec(app.session_state["spec"]) == []
    assert not app.exception, [item.value for item in app.exception]
    assert not settings_without_a_tooltip(app), settings_without_a_tooltip(app)
