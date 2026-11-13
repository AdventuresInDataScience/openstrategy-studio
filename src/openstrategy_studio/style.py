"""The look: page themes and the stylesheet, in one place.

A page theme is one TOML file, in themes/ beside this module or in the user's own
folder (user_themes_dir). It has two tables:

* [theme] is Streamlit's own theme table. cli.py starts Streamlit with
  `--theme.base <the file>`, and Streamlit (1.67) reads the table itself: colours,
  fonts, corner radius, heading sizes, the sidebar's own colours, the chart colours.
  Every option `streamlit config show` lists under [theme] works; an unknown one
  is dropped with a warning in the terminal (tests/test_style.py checks the
  packaged files against the installed Streamlit).
* [studio] is the Studio's own, and Streamlit ignores it: a second accent, the
  gradient, card surfaces and shadows, the page glow, and which chart theme
  matches the page. Its tokens become CSS variables (--os-*) in front of the
  packaged stylesheet, themes/studio.css, which ui.boot puts on every page. They
  also colour the sidebar's logo and the pipeline diagram.

⚠️ The theme is fixed for the life of the server. Streamlit reads its theme options
when it starts, and st.set_option cannot change them (1.67 allows client.* options
only). So cli.py records the file it started with in ACTIVE_ENV, and everything
else reads that same file, so the stylesheet always matches Streamlit's own
colours. A theme picked on the Settings page applies at the next start.

Not .streamlit/config.toml: Streamlit reads that from the working directory, which
the Studio does not control (and a frozen bundle has none). A user's own
~/.streamlit/config.toml still overrides single [theme] options, as Streamlit
documents for theme files.
"""

import html
import os
import re
import tomllib
from pathlib import Path

from . import config

# Streamlit needs a real file path for --theme.base, so this is the folder on disk
# (package data, collected by hatch and by the PyInstaller spec's collect_all).
PACKAGED = Path(__file__).with_name("themes")
DEFAULT_THEME = "daylight"
# The theme file the running server was started with (set by cli.run_app).
ACTIVE_ENV = "OPENSTRATEGY_STUDIO_PAGE_THEME"

# [theme] keys every file must set: the stylesheet, the logo, the matching chart
# theme and the pipeline diagram are drawn from them.
REQUIRED = ["base", "primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor", "borderColor"]
# Colours Python reads (the logo, the diagram's tints, the chart theme), so they must be
# hex: the CSS-only tokens (gradient, glow, shadows) may be any CSS value.
HEX_KEYS = {"theme": ["primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor", "borderColor"],
            "studio": ["accent", "surface", "muted", "on_primary", "positive", "negative"]}
HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")

# Streamlit's keywords for baseRadius, as CSS lengths for the stylesheet's own cards.
# ⚠️ VERIFY: close to Streamlit 1.67's values, not read from it; a theme that gives a
# length ("0.6rem") gets exactly that everywhere.
RADIUS_KEYWORDS = {"none": "0", "small": "0.25rem", "medium": "0.5rem", "large": "1rem", "full": "1.5rem"}


# ---------------------------------------------------------------------------
# Finding and checking theme files
# ---------------------------------------------------------------------------

def user_themes_dir():
    """The user's own page themes: beside settings.toml, so they survive a reinstall."""
    return config.config_dir() / "themes"


def theme_files():
    """{name: path} for every theme file: the packaged ones, then the user's, which win on a clash."""
    found = {}
    for folder in (PACKAGED, user_themes_dir()):
        if folder.is_dir():
            for path in sorted(folder.glob("*.toml")):
                found[path.stem] = path
    return found


def theme_problems(path):
    """What is wrong with a theme file, as sentences: empty when it is usable.

    The one check, made wherever a theme is chosen (cli.py at start, the Settings page,
    ui.boot), so a broken file of the user's falls back to the default instead of
    stopping the Studio.
    """
    path = Path(path)
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return [f"{path.name} cannot be read: {error}"]
    theme = parsed.get("theme")
    studio = parsed.get("studio", {})
    if not isinstance(theme, dict) or not isinstance(studio, dict):
        return [f"{path.name} needs a [theme] table, and [studio] must be a table too."]
    problems = [f"{path.name}: [theme] needs {key}." for key in REQUIRED if key not in theme]
    if theme.get("base", "light") not in ("light", "dark"):
        problems.append(f'{path.name}: base must be "light" or "dark" (a theme file cannot build on another file).')
    for table, keys in HEX_KEYS.items():
        values = theme if table == "theme" else studio
        for key in keys:
            if key in values and not (isinstance(values[key], str) and HEX.match(values[key])):
                problems.append(f"{path.name}: [{table}] {key} must be a hex colour like #4F46E5.")
    chart = studio.get("chart_theme", "page")
    if chart != "page":
        import plotly.io as pio
        import plotly_themes  # noqa: F401  (registers its themes)
        if chart not in pio.templates:
            problems.append(f'{path.name}: there is no chart theme called "{chart}". Use "page" or one of the '
                            "Settings page's chart themes.")
    return problems


def default_file():
    # Always the packaged file: a user's file of the same name may be the broken one.
    return PACKAGED / f"{DEFAULT_THEME}.toml"


def choose_theme_file(settings):
    """The theme file a new server should start with, and the problems that made it fall back.

    The Settings page's choice when it exists and is usable, else the default.
    """
    files = theme_files()
    name = settings.get("page_theme") or DEFAULT_THEME
    if name not in files:
        return default_file(), [f'There is no page theme called "{name}", so the Studio uses {DEFAULT_THEME}.']
    problems = theme_problems(files[name])
    if problems:
        return default_file(), problems + [f"The Studio uses {DEFAULT_THEME} until that is fixed."]
    return files[name], []


def active_theme(settings):
    """(theme, problems) for the page theme this process shows.

    In the running app, the file the server started with, which Streamlit has already
    applied. Elsewhere (tests, selfcheck) the file a new server would start with.
    The problems explain a fallback, and ui.boot shows them.
    """
    started = os.environ.get(ACTIVE_ENV)
    if started:
        problems = theme_problems(started)
        if not problems:
            return load_theme(started), []
        # Edited while the Studio runs and no longer parses: Streamlit keeps the colours it
        # started with, and the stylesheet falls back until the file is fixed.
        return load_theme(default_file()), problems + [f"The page uses {DEFAULT_THEME}'s styling until that is fixed."]
    path, problems = choose_theme_file(settings)
    return load_theme(path), problems


def catalogue():
    """For the Settings page: ({name: theme} for the usable files, [problems with the others])."""
    usable, problems = {}, []
    for name, path in theme_files().items():
        found = theme_problems(path)
        if found:
            problems += found
        else:
            usable[name] = load_theme(path)
    return usable, problems


def copy_theme(name):
    """Copy a theme into the user's folder under a new name, to edit. Returns the new file's path."""
    source = theme_files()[name]
    folder = user_themes_dir()
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    target = folder / f"{name}_copy.toml"
    while target.exists():
        number += 1
        target = folder / f"{name}_copy{number}.toml"
    text = source.read_text(encoding="utf-8")
    # Tell the copy apart in the picker. A file with no label line gets one from its name.
    suffix = "copy" if number == 1 else f"copy {number}"
    text = re.sub(r'^label = "(.*)"$', lambda match: f'label = "{match.group(1)} ({suffix})"', text,
                  count=1, flags=re.MULTILINE)
    target.write_text(text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# One theme's tokens
# ---------------------------------------------------------------------------

def load_theme(path):
    """A theme file as {"name", "path", "theme": Streamlit's table, "studio": the tokens}.

    Every [studio] token the file leaves out follows from [theme], so a theme can be
    as short as [theme]'s required colours. Check the file with theme_problems first.
    """
    path = Path(path)
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    theme = parsed["theme"]
    dark = theme.get("base") == "dark"
    studio = dict(parsed.get("studio") or {})
    studio.setdefault("label", path.stem.replace("_", " ").capitalize())
    studio.setdefault("description", "")
    studio.setdefault("accent", theme["primaryColor"])
    studio.setdefault("gradient", f"linear-gradient(120deg, {theme['primaryColor']}, {studio['accent']})")
    studio.setdefault("title_fill", studio["gradient"])          # the page title: the gradient, or one colour
    studio.setdefault("surface", theme["secondaryBackgroundColor"])   # cards: metrics, expanders, charts
    studio.setdefault("muted", theme["textColor"])
    studio.setdefault("on_primary", "#FFFFFF")                   # text on the gradient and the logo's mark
    studio.setdefault("shadow", "0 1px 2px rgba(0, 0, 0, 0.35)" if dark else "0 1px 3px rgba(16, 24, 40, 0.08)")
    studio.setdefault("shadow_hover", studio["shadow"])
    studio.setdefault("glow", "none")                            # the page background's decoration
    studio.setdefault("positive", "#4ADE80" if dark else "#16A34A")
    studio.setdefault("negative", "#F87171" if dark else "#DC2626")
    studio.setdefault("chart_theme", "page")                     # "page": drawn from these colours (analysis)
    studio.setdefault("css", "")                                 # extra rules, after the stylesheet
    return {"name": path.stem, "path": path, "theme": theme, "studio": studio}


def css_variables(theme):
    """The tokens as CSS custom properties. themes/studio.css uses these names and no colour of its own."""
    table, studio = theme["theme"], theme["studio"]
    sidebar = table.get("sidebar") or {}
    radius = str(table.get("baseRadius", "medium"))
    button_radius = str(table.get("buttonRadius", radius))
    tokens = {
        "--os-primary": table["primaryColor"],
        "--os-accent": studio["accent"],
        "--os-gradient": studio["gradient"],
        "--os-title-fill": studio["title_fill"],
        "--os-on-primary": studio["on_primary"],
        "--os-bg": table["backgroundColor"],
        "--os-surface": studio["surface"],
        "--os-text": table["textColor"],
        "--os-muted": studio["muted"],
        "--os-border": table["borderColor"],
        "--os-shadow": studio["shadow"],
        "--os-shadow-hover": studio["shadow_hover"],
        "--os-glow": studio["glow"],
        "--os-positive": studio["positive"],
        "--os-negative": studio["negative"],
        # Fonts for the Studio's own HTML (the At a glance panel); Streamlit sets its own.
        "--os-heading-font": table.get("headingFont", table.get("font", "sans-serif")),
        "--os-code-font": table.get("codeFont", "monospace"),
        "--os-radius": RADIUS_KEYWORDS.get(radius, radius),
        "--os-button-radius": RADIUS_KEYWORDS.get(button_radius, button_radius),
        # The sidebar has its own colours when [theme.sidebar] sets them.
        "--os-sidebar-primary": sidebar.get("primaryColor", table["primaryColor"]),
        "--os-sidebar-text": sidebar.get("textColor", table["textColor"]),
    }
    lines = [f"  {name}: {value};" for name, value in tokens.items()]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def stylesheet(theme):
    """The page's CSS: the tokens, the packaged stylesheet, then the theme's own extra rules.

    Read on every run (it is small), so an edit to studio.css or to a theme's [studio]
    table shows on the next rerun. Streamlit's [theme] table needs a restart.
    """
    rules = (PACKAGED / "studio.css").read_text(encoding="utf-8")
    css = css_variables(theme) + rules + "\n" + theme["studio"]["css"]
    # ⚠️ st.html sanitises with DOMPurify, which silently drops a <style> element whose text
    # holds "<" followed by a letter or "/", even inside a CSS comment (checked on Streamlit
    # 1.67): the page then loses its whole stylesheet. "\3c " is CSS's own escape for "<",
    # so a string like content: "<b>" still shows "<b>", and a comment just reads oddly.
    return css.replace("<", "\\3c ")


# ---------------------------------------------------------------------------
# Drawn from the tokens: the logo, the Settings preview, the pipeline's colours
# ---------------------------------------------------------------------------

def _attribute(value):
    # Theme values go inside HTML and SVG attributes; a font list may hold quotes.
    return html.escape(str(value), quote=True)


def _flat_radius(theme):
    radius = str(theme["theme"].get("baseRadius", "medium"))
    return RADIUS_KEYWORDS.get(radius, radius) in ("0", "0px", "0rem")


def _monospace(font):
    """Whether a CSS font list starts with a fixed-width font. Text in one is about a fifth wider."""
    first = str(font).split(",")[0].strip().strip("'\"").lower()
    return first == "monospace" or any(word in first for word in ("mono", "consolas", "courier", "menlo"))


def logo_svg(theme, mark_only=False):
    """The sidebar's logo: a rising line on the theme's gradient, then the name.

    In the sidebar's text colour. ⚠️ An <img> cannot use the page's web fonts, so the
    name is drawn in the theme's heading font only when it is installed on the machine
    (the packaged themes use system fonts), else in the browser's sans-serif.
    """
    table, studio = theme["theme"], theme["studio"]
    text = (table.get("sidebar") or {}).get("textColor", table["textColor"])
    heading = table.get("headingFont", table.get("font", "sans-serif"))
    font = _attribute(heading)
    # Room for the name: an <img> clips what does not fit, and a fixed-width font is wider.
    width = 280 if _monospace(heading) else 250
    corner = 3 if _flat_radius(theme) else 10
    mark = (f'<defs><linearGradient id="os-mark" x1="0" y1="1" x2="1" y2="0">'
            f'<stop offset="0" stop-color="{table["primaryColor"]}"/><stop offset="1" stop-color="{studio["accent"]}"/>'
            f'</linearGradient></defs>'
            f'<rect x="1" y="2" width="36" height="36" rx="{corner}" fill="url(#os-mark)"/>'
            f'<polyline points="9,28 16,20 22,24 29,13" fill="none" stroke="{studio["on_primary"]}" stroke-width="3.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="29" cy="13" r="2.7" fill="{studio["on_primary"]}"/>')
    if mark_only:
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">{mark}</svg>'
    name = (f'<text x="48" y="27" font-family="{font}" font-size="19" font-weight="700" fill="{text}">OpenStrategy'
            f'<tspan font-weight="400" fill-opacity="0.68"> Studio</tspan></text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 40">{mark}{name}</svg>'


def preview_html(theme):
    """A small mock page in the theme's own colours, for the Settings page's picker."""
    table, studio = theme["theme"], theme["studio"]
    sidebar = table.get("sidebar") or {}
    value = {
        "bg": table["backgroundColor"], "glow": studio["glow"], "border": table["borderColor"],
        "text": table["textColor"], "muted": studio["muted"], "surface": studio["surface"],
        "gradient": studio["gradient"], "title": studio["title_fill"], "on_primary": studio["on_primary"],
        "sidebar": sidebar.get("backgroundColor", table["backgroundColor"]),
        "sidebar_text": sidebar.get("textColor", table["textColor"]),
        "font": table.get("font", "sans-serif"), "heading": table.get("headingFont", table.get("font", "sans-serif")),
        "radius": RADIUS_KEYWORDS.get(str(table.get("baseRadius", "medium")), str(table.get("baseRadius", "medium"))),
    }
    value = {key: _attribute(item) for key, item in value.items()}
    dots = ""
    for colour in (table.get("chartCategoricalColors") or [])[:8]:
        dots += (f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                 f'background:{_attribute(colour)}"></span>')
    lines = ""
    for width in (70, 45, 58):
        lines += (f'<div style="height:6px;width:{width}%;margin:7px 0;border-radius:3px;'
                  f'background:{value["sidebar_text"]};opacity:.35"></div>')
    return f"""
<div style="display:flex;border:1px solid {value['border']};border-radius:{value['radius']};overflow:hidden;
            font-family:{value['font']};background-color:{value['bg']};background-image:{value['glow']}">
  <div style="width:22%;padding:14px 12px;background:{value['sidebar']}">{lines}</div>
  <div style="flex:1;padding:16px 18px">
    <div style="font-family:{value['heading']};font-weight:700;font-size:24px;display:inline-block;
                background:{value['title']};-webkit-background-clip:text;background-clip:text;color:transparent">
      Backtest</div>
    <div style="color:{value['muted']};font-size:13px;margin:2px 0 12px">{_attribute(studio['description'])}</div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span style="background:{value['gradient']};color:{value['on_primary']};padding:6px 14px;font-size:13px;
                   font-weight:600;border-radius:{value['radius']}">Run backtest</span>
      <span style="background:{value['surface']};color:{value['text']};border:1px solid {value['border']};
                   padding:5px 13px;font-size:13px;border-radius:{value['radius']}">Save run</span>
    </div>
    <div style="display:flex;gap:6px;margin-top:14px">{dots}</div>
  </div>
</div>"""


def _mix(colour, base, share):
    """`share` of `colour` over `base`, both hex: a tint Graphviz can draw (it has no color-mix)."""
    def channels(value):
        value = value.lstrip("#")
        if len(value) == 3:
            value = "".join(letter * 2 for letter in value)
        return [int(value[index:index + 2], 16) for index in (0, 2, 4)]
    mixed = [round(top * share + bottom * (1 - share)) for top, bottom in zip(channels(colour), channels(base))]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def pipeline_palette(theme):
    """The pipeline diagram's colours (analysis.pipeline_dot) from the page theme.

    Nodes in the page's background on the chart card's surface, tinted by role: rules in
    the primary colour, risk in the negative colour, the account in the text colour.
    ⚠️ The browser draws the labels in the page's font whatever Graphviz is told, but
    Graphviz sizes the boxes from the font it is told, and it only knows a few: Courier
    for a fixed-width page (else its labels overflow), Helvetica for the rest.
    """
    table, studio = theme["theme"], theme["studio"]
    background = table["backgroundColor"]
    return {"font": "Courier" if _monospace(table.get("font", "sans-serif")) else "Helvetica",
            "background": "transparent", "node": background, "border": table["borderColor"],
            "text": table["textColor"], "edge": studio["muted"],
            "rule": _mix(table["primaryColor"], background, 0.16),
            "risk": _mix(studio["negative"], background, 0.14),
            "account": _mix(table["textColor"], background, 0.07)}
