"""The look: page theme files, the stylesheet's tokens, the chart theme that matches the page,
and how the server is started with them (style.py)."""

import os
import re
import sys
from pathlib import Path

import pytest

from openstrategy_studio import cli, config, style
from openstrategy_studio.spec import load_template

MINIMAL = """
[theme]
base = "dark"
primaryColor = "#FF6600"
backgroundColor = "#101010"
secondaryBackgroundColor = "#1A1A1A"
textColor = "#EEEEEE"
borderColor = "#333333"
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty Studio home, so the user's own themes folder starts empty."""
    monkeypatch.setenv("OPENSTRATEGY_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv(style.ACTIVE_ENV, raising=False)
    return tmp_path


def packaged():
    return {path.stem: path for path in sorted(style.PACKAGED.glob("*.toml"))}


def test_every_packaged_theme_is_usable(home):
    assert style.DEFAULT_THEME in packaged()
    for name, path in packaged().items():
        assert style.theme_problems(path) == [], name
        look = style.load_theme(path)
        assert look["studio"]["label"] and look["studio"]["description"], name
        # Chart colours are what the "page" chart theme draws with.
        assert len(look["theme"]["chartCategoricalColors"]) >= 6, name


def test_packaged_themes_use_only_options_streamlit_knows(home):
    """Streamlit drops an unknown [theme] option with only a terminal warning, so a typo would go unseen."""
    pytest.importorskip("streamlit")
    from streamlit import config as streamlit_config

    for name, path in packaged().items():
        table = style.load_theme(path)["theme"]
        known = streamlit_config.get_options_for_section("theme")
        for key, value in table.items():
            if isinstance(value, dict):
                section = streamlit_config.get_options_for_section(f"theme.{key}")
                assert section, (name, key)
                assert set(value) <= set(section), (name, key, set(value) - set(section))
            else:
                assert key in known, (name, key)


def test_the_stylesheet_uses_exactly_the_tokens(home):
    css = (style.PACKAGED / "studio.css").read_text(encoding="utf-8")
    used = set(re.findall(r"var\((--os-[a-z0-9-]+)", css))
    for name, path in packaged().items():
        defined = set(re.findall(r"(--os-[a-z0-9-]+):", style.css_variables(style.load_theme(path))))
        assert used == defined, (name, used - defined, defined - used)


def test_the_stylesheet_never_holds_a_tag(home):
    """DOMPurify (st.html) silently drops a <style> whose text has '<' and a letter, even in a comment."""
    assert "<" not in (style.PACKAGED / "studio.css").read_text(encoding="utf-8")
    look = style.load_theme(style.default_file())
    look["studio"]["css"] = '.x::after { content: "<b>"; } /* </style> */'
    sheet = style.stylesheet(look)
    assert "<" not in sheet and '"\\3c b>"' in sheet


def test_a_short_theme_file_gets_every_token(home):
    folder = style.user_themes_dir()
    folder.mkdir(parents=True)
    (folder / "short.toml").write_text(MINIMAL, encoding="utf-8")
    path, problems = style.choose_theme_file({"page_theme": "short"})
    assert path == folder / "short.toml" and problems == []
    look = style.load_theme(path)
    assert look["studio"]["label"] == "Short" and look["studio"]["chart_theme"] == "page"
    css = (style.PACKAGED / "studio.css").read_text(encoding="utf-8")
    assert set(re.findall(r"var\((--os-[a-z0-9-]+)", css)) <= set(re.findall(r"(--os-[a-z0-9-]+):", style.css_variables(look)))
    assert "#FF6600" in style.logo_svg(look) and "#FF6600" in style.preview_html(look)


def test_a_missing_or_broken_choice_falls_back_to_the_default(home):
    path, problems = style.choose_theme_file({"page_theme": "nowhere"})
    assert path == style.default_file() and "nowhere" in problems[0]
    folder = style.user_themes_dir()
    folder.mkdir(parents=True)
    (folder / "broken.toml").write_text("[theme\nbase = 'light'", encoding="utf-8")
    (folder / "pink.toml").write_text(MINIMAL.replace('"#FF6600"', '"hotpink"'), encoding="utf-8")
    (folder / "lost.toml").write_text(MINIMAL + '\n[studio]\nchart_theme = "no_such_theme"\n', encoding="utf-8")
    for name, says in (("broken", "cannot be read"), ("pink", "hex colour"), ("lost", "no chart theme")):
        path, problems = style.choose_theme_file({"page_theme": name})
        assert path == style.default_file() and says in problems[0], (name, problems)
    usable, broken = style.catalogue()
    assert set(usable) == set(packaged()) and len(broken) == 3


def test_a_running_server_keeps_the_theme_it_started_with(home, monkeypatch):
    started = packaged()["nord"]
    monkeypatch.setenv(style.ACTIVE_ENV, str(started))
    look, problems = style.active_theme({"page_theme": "paper"})       # picked since, applies at the next start
    assert look["name"] == "nord" and problems == []


def test_a_copy_lands_in_the_user_folder_and_can_be_chosen(home):
    first = style.copy_theme("midnight")
    second = style.copy_theme("midnight")
    assert first.parent == style.user_themes_dir() and first != second
    assert style.load_theme(first)["studio"]["label"] == "Midnight (copy)"
    assert style.load_theme(second)["studio"]["label"] == "Midnight (copy 2)"
    assert style.choose_theme_file({"page_theme": first.stem}) == (first, [])


def test_the_server_starts_with_the_chosen_theme(home, monkeypatch):
    """cli.run_app hands Streamlit the file as --theme.base and tells the pages which file it was."""
    pytest.importorskip("streamlit")
    from streamlit.web import cli as streamlit_cli

    settings = config.load_settings()
    settings["page_theme"] = "terminal"
    config.save_settings(settings)
    captured = {}
    monkeypatch.setattr(streamlit_cli, "main", lambda: captured.setdefault("argv", list(sys.argv)) and 0)
    monkeypatch.setattr(sys, "argv", ["openstrategy-studio"])
    # Registered so monkeypatch restores them: run_app sets both for the life of the process.
    monkeypatch.setenv(style.ACTIVE_ENV, "unset")
    monkeypatch.setenv("NUMBA_CACHE_DIR", os.environ.get("NUMBA_CACHE_DIR", str(home / "numba")))
    cli.run_app(headless=True)
    argv = captured["argv"]
    theme_file = Path(argv[argv.index("--theme.base") + 1])
    assert theme_file == packaged()["terminal"] and os.environ[style.ACTIVE_ENV] == str(theme_file)
    assert argv[argv.index("--client.toolbarMode") + 1] == "viewer"


def test_the_page_chart_theme_follows_the_page(home):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go
    import plotly.io as pio
    from openstrategy_studio import analysis

    midnight = style.load_theme(packaged()["midnight"])
    template = analysis.page_template(midnight)
    assert template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert list(template.layout.colorway) == midnight["theme"]["chartCategoricalColors"]
    assert template.layout.font.color == midnight["theme"]["textColor"]
    assert template.layout.colorscale.diverging[0][1] == midnight["studio"]["negative"]
    # A theme that names a chart theme of its own uses it as it is.
    assert analysis.page_template(style.load_theme(packaged()["nord"])) is pio.templates["nord"]
    analysis.use_page_theme(midnight)
    try:
        figure = analysis.finish(go.Figure(go.Scatter(y=[1, 2, 3])), analysis.PAGE_THEME)
        assert figure.layout.template.layout.font.color == midnight["theme"]["textColor"]
        assert analysis.PAGE_THEME in analysis.theme_groups()["Page"]
    finally:
        analysis.use_page_theme(style.load_theme(style.default_file()))


def test_the_pipeline_takes_the_page_colours(home):
    pytest.importorskip("plotly")
    from openstrategy_studio import analysis

    spec = load_template("ma_cross")
    assert 'fillcolor="white"' in analysis.pipeline_dot(spec)               # unchanged without a palette
    look = style.load_theme(packaged()["midnight"])
    palette = style.pipeline_palette(look)
    dot = analysis.pipeline_dot(spec, palette)
    assert 'bgcolor="transparent"' in dot and f'fontcolor="{look["theme"]["textColor"]}"' in dot
    assert f'fillcolor="{palette["rule"]}"' in dot and f'fillcolor="{palette["risk"]}"' in dot
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}|transparent", value) for key, value in palette.items() if key != "font")
    # Graphviz sizes boxes from the font it is told: a fixed-width page needs Courier, or labels overflow.
    assert palette["font"] == "Helvetica"
    terminal = style.load_theme(packaged()["terminal"])
    assert 'fontname="Courier"' in analysis.pipeline_dot(spec, style.pipeline_palette(terminal))
    assert 'viewBox="0 0 280 40"' in style.logo_svg(terminal) and 'viewBox="0 0 250 40"' in style.logo_svg(look)


def test_settings_page_offers_the_themes(home):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("vectorbtx")
    page = Path(style.__file__).parent / "app" / "pages" / "6_Settings.py"
    app = testing.AppTest.from_file(str(page), default_timeout=180)
    app.run()
    assert not app.exception, [item.value for item in app.exception]
    picker = next(box for box in app.selectbox if box.label == "Page theme")
    assert set(picker.options) >= {"Daylight", "Midnight", "Paper", "Nord", "Terminal"}
    assert not any("restart the Studio" in item.value for item in app.info)
    picker.set_value("midnight").run()
    assert any("restart the Studio" in item.value for item in app.info)
    next(button for button in app.button if button.label == "Make an editable copy").click().run()
    assert not app.exception, [item.value for item in app.exception]
    assert (style.user_themes_dir() / "midnight_copy.toml").exists()
