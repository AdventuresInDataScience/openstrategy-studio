"""Settings: settings.toml in platformdirs.user_config_dir("openstrategy-studio").

Read once at startup by apply_settings(). Background workers are separate
processes, so each worker is handed the settings dict and applies it before
any run. All user data lives under one configurable folder (data_dir).
"""

import os
import tomllib
from pathlib import Path

import platformdirs

APP_NAME = "openstrategy-studio"

# Keys written to vbt.settings["vectorbtx"]. None means "leave vectorbtX's default".
VECTORBTX_KEYS = ["rounding", "financing_markup", "carry_markup", "fx_markup", "slippage_max", "slippage_coef", "vol_window",
                  "triple_day", "end_of_day", "day_count", "fill_policy", "static_rate",
                  "triangulation_tolerance"]

DEFAULTS = {
    "data_dir": "",                 # blank: platformdirs.user_data_dir("openstrategy-studio")
    "default_theme": "plotly_white",
    "workers": max(1, (os.cpu_count() or 2) - 1),
    "offline": False,
    "spike_multiple": 10.0,         # import check: bars this many median ranges wide are flagged
    "fred_api_key": "",             # prefer the FRED_API_KEY environment variable or a .env file
    "vectorbtx": {key: None for key in VECTORBTX_KEYS},
}


def home_override():
    """OPENSTRATEGY_STUDIO_HOME puts settings and default data in one folder (tests, portable installs)."""
    value = os.environ.get("OPENSTRATEGY_STUDIO_HOME")
    return Path(value).expanduser() if value else None


def config_dir():
    home = home_override()
    return home / "config" if home else Path(platformdirs.user_config_dir(APP_NAME))


def settings_path():
    return config_dir() / "settings.toml"


def load_settings():
    settings = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    path = settings_path()
    if path.exists():
        stored = tomllib.loads(path.read_text(encoding="utf-8"))
        for key, value in stored.items():
            if key == "vectorbtx":
                settings["vectorbtx"].update(value)
            else:
                settings[key] = value
    return settings


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_settings(settings):
    """Write settings.toml. A tiny writer, because tomllib only reads.

    TOML has no null, so unset (None) values are left out and fall back to DEFAULTS.
    """
    lines = []
    for key, value in settings.items():
        if key != "vectorbtx" and value is not None:
            lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    lines.append("[vectorbtx]")
    for key, value in settings["vectorbtx"].items():
        if value is not None and value != "":
            lines.append(f"{key} = {_toml_value(value)}")
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_dir(settings):
    """The one folder holding every piece of user data, created on first use."""
    home = home_override()
    default = home / "data" if home else Path(platformdirs.user_data_dir(APP_NAME))
    folder = Path(settings.get("data_dir") or default).expanduser()
    for sub in ("prices", "strategies", "runs", "jobs", "drafts", "universes"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    return folder


def _load_dotenv(path):
    """KEY=VALUE lines from a .env file into os.environ, without overriding what is set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def apply_environment(settings):
    """Environment variables that must be set BEFORE vectorbt (numba) is imported."""
    folder = data_dir(settings)
    _load_dotenv(config_dir() / ".env")
    _load_dotenv(folder / ".env")
    if settings.get("fred_api_key"):
        os.environ.setdefault("FRED_API_KEY", settings["fred_api_key"])
    # Keep the numba cache in a folder we know is writable (it matters for a
    # bundled app, where the package folder may be read-only).
    os.environ.setdefault("NUMBA_CACHE_DIR", str(folder / "numba_cache"))
    # ⚠️ On Windows, the standalone CPython that uv installs aborts the whole process
    # ("OPENSSL_Uplink ... no OPENSSL_Applink") the moment any HTTPS context is created
    # while SSLKEYLOGFILE is set, because OpenSSL then writes the key log through a C
    # FILE* it cannot reach. Some IDEs, proxies and debugging tools set it. The Studio
    # never needs it, and the crash would take down the Streamlit server on the first
    # FRED or Yahoo download, so it is dropped for this process and its workers.
    if os.name == "nt":
        os.environ.pop("SSLKEYLOGFILE", None)


def apply_settings(settings):
    """Apply settings to this process: environment first, then vectorbtX's own settings."""
    from . import engine

    apply_environment(settings)
    engine.apply_vectorbtx_settings({key: value for key, value in settings["vectorbtx"].items()
                                     if value is not None and value != ""})
