"""Test 14: every page loads with every template, with no uncaught exception.

Uses streamlit.testing.v1.AppTest (VERIFY the current API on the installed version).
"""

import copy
from pathlib import Path

import pytest

import openstrategy_studio
from openstrategy_studio.spec import list_templates, load_template

APP = Path(openstrategy_studio.__file__).parent / "app"


def test_every_page_loads_with_every_template(tmp_path, monkeypatch):
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
    app.radio(key="r0:account.cash").set_value("shared").run()
    account = app.session_state["spec"]["account"]
    assert account["cash"] == "shared" and account["when_cash_short"] == "skip"
    app.radio(key="r0:account.entry_priority:by").set_value("feature").run()
    priority = app.session_state["spec"]["account"]["entry_priority"]
    assert priority == {"by": "feature", "highest_first": True, "score": {"feature": "fast"}}
    app.radio(key="r0:account.cash").set_value("per_instrument").run()
    account = app.session_state["spec"]["account"]
    assert account["cash"] == "per_instrument" and "entry_priority" not in account and "when_cash_short" not in account
    assert not app.exception
