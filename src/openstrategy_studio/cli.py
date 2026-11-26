"""The `openstrategy-studio` command, and the single entry point of a bundled app.

    openstrategy-studio                 launch the app in your browser (same as `run`)
    openstrategy-studio run [--port N] [--headless] [--address A]
    openstrategy-studio desktop         the app in its own window (needs the `desktop` extra)
    openstrategy-studio selfcheck       the phase 4 gate: data, build, backtest, optimise, pages
    openstrategy-studio worker <dir>    a background job (the app starts these itself)

Streamlit runs in this process (no shell, no reliance on the working
directory), with the app's files read from the installed package. In a frozen
(PyInstaller) build there is no `python -m`, so background jobs and the desktop
window re-run this same executable with a subcommand instead.
"""

import argparse
import os
import sys
from importlib import resources

from . import config


def frozen():
    return bool(getattr(sys, "frozen", False))


def self_command(*arguments):
    """The command line that re-runs this program with a subcommand."""
    if frozen():
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "openstrategy_studio.cli", *arguments]


def run_app(port=None, headless=False, address=None, extra=()):
    # Environment first (numba cache folder, FRED key), before anything imports vectorbt.
    settings = config.load_settings()
    config.apply_environment(settings)
    from streamlit.web import cli as streamlit_cli

    from . import style

    # The page theme. Streamlit reads the file's [theme] table itself (theme.base may name
    # a TOML file since 1.67), and the pages read the same file, through ACTIVE_ENV, for
    # the stylesheet, the logo and the chart theme that matches the page. A missing or
    # broken choice falls back to the default here, so the Studio always starts.
    theme_file, problems = style.choose_theme_file(settings)
    for problem in problems:
        print(f"Page theme: {problem}")
    os.environ[style.ACTIVE_ENV] = str(theme_file)

    with resources.as_file(resources.files("openstrategy_studio") / "app" / "Home.py") as home:
        command = ["streamlit", "run", str(home),
                   "--browser.gatherUsageStats", "false",
                   # Large CSV imports: the default upload limit is 200 MB.
                   "--server.maxUploadSize", "1000",
                   # A frozen build looks like a source checkout to Streamlit; this keeps it in normal mode.
                   "--global.developmentMode", "false",
                   "--theme.base", str(theme_file),
                   # An app, not a development server: no Deploy button, and no rerun or clear-cache
                   # items in the menu. `openstrategy-studio run --client.toolbarMode developer`
                   # brings them back (flags after ours win).
                   "--client.toolbarMode", "viewer"]
        if port:
            command += ["--server.port", str(port)]
        if headless:
            command += ["--server.headless", "true"]
        if address:
            command += ["--server.address", address]
        sys.argv = command + list(extra)
        return streamlit_cli.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="openstrategy-studio", description="OpenStrategy Studio.")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="Launch the app in your browser (the default).")
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--headless", action="store_true", help="Do not open a browser window.")
    run.add_argument("--address", default=None, help="Address to listen on (0.0.0.0 inside Docker).")
    desktop = commands.add_parser("desktop", help="Open the app in its own window (needs pywebview).")
    desktop.add_argument("--port", type=int, default=None)
    check = commands.add_parser("selfcheck", help="Check this machine can build, backtest and optimise.")
    check.add_argument("--quick", action="store_true", help="Skip the page checks and the background optimisation.")
    check.add_argument("--keep", action="store_true", help="Keep the temporary data folder for inspection.")
    worker = commands.add_parser("worker", help="Run a background job (the app starts these itself).")
    worker.add_argument("job_dir")

    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]                      # bare `openstrategy-studio [--port N]` means run
    args, extra = parser.parse_known_args(argv)
    if args.command == "run":
        return run_app(args.port, args.headless, args.address, extra)
    if args.command == "desktop":
        from . import desktop as desktop_module
        return desktop_module.main(port=args.port)
    if args.command == "selfcheck":
        from . import selfcheck
        return selfcheck.main(quick=args.quick, keep=args.keep)
    if args.command == "worker":
        from . import worker as worker_module
        return 0 if worker_module.main(args.job_dir) in ("done", "cancelled") else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
