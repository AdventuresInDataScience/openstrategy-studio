"""Test 14: every page loads with every template, with no uncaught exception.

Uses streamlit.testing.v1.AppTest (VERIFY the current API on the installed version).
"""

import copy
from pathlib import Path

import pytest

import openstrategy_studio
from openstrategy_studio.spec import list_templates, load_template

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
    """Every page renders with every template, leaves the spec as it was, and explains every setting."""
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    pages = [APP / "Home.py"] + sorted((APP / "pages").glob("*.py"))
    for name in list_templates():
        for page in pages:
            app = testing.AppTest.from_file(str(page), default_timeout=180)
            app.session_state["spec"] = load_template(name)
            app.session_state["spec_rev"] = 0
            app.run()
            assert not app.exception, (name, page.name, [item.value for item in app.exception])
            # Rendering must not change the spec (CLAUDE.md): every widget is seeded from it.
            assert app.session_state["spec"] == load_template(name), (name, page.name)
            # Every setting has an ⓘ tooltip (CLAUDE.md).
            assert not settings_without_a_tooltip(app), (name, page.name, settings_without_a_tooltip(app))


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
        app = testing.AppTest.from_file(str(APP / "pages" / "3_Backtest.py"), default_timeout=180)
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
    app = testing.AppTest.from_file(str(APP / "pages" / "3_Backtest.py"), default_timeout=180)
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
    app.radio(key=current_key(app, "account.cash")).set_value("shared").run()
    account = app.session_state["spec"]["account"]
    assert account["cash"] == "shared" and account["when_cash_short"] == "skip"
    app.radio(key=current_key(app, "account.entry_priority:by")).set_value("feature").run()
    priority = app.session_state["spec"]["account"]["entry_priority"]
    assert priority == {"by": "feature", "highest_first": True, "score": {"feature": "fast"}}
    app.radio(key=current_key(app, "account.cash")).set_value("per_instrument").run()
    account = app.session_state["spec"]["account"]
    assert account["cash"] == "per_instrument" and "entry_priority" not in account and "when_cash_short" not in account
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
    app.button(key=current_key(app, "rules.long_entry:add:comparison")).click().run()
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
    app.button(key=current_key(app, "rules.long_entry:add:comparison")).click().run()
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
