"""Entry point of the bundled (PyInstaller) app: every openstrategy-studio subcommand, frozen.

Double-clicking the app opens the desktop window. The bundle re-runs itself
with a subcommand for the Streamlit server (`run`) and for background jobs
(`worker <dir>`), because a frozen app has no `python -m`.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    # Process pools in a frozen app re-launch this executable; this hands those
    # children to multiprocessing instead of starting another app.
    multiprocessing.freeze_support()
    from openstrategy_studio import cli

    sys.exit(cli.main(sys.argv[1:] or ["desktop"]))
