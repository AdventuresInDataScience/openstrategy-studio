"""The app in its own window: Streamlit in a background process, shown in a native webview.

Needs the `desktop` extra (pywebview): `uv sync --extra desktop`. On Linux,
pywebview also needs a GUI backend (GTK or Qt) from the system packages.
"""

import socket
import subprocess
import time
import urllib.request

from .cli import self_command


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_until_up(url, process, timeout=180):
    """Poll Streamlit's health endpoint. The first start can take a while (numba compiles)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The Studio's server stopped while starting; run `openstrategy-studio run` to see why.")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"The Studio's server did not start within {timeout} seconds.")


def main(port=None):
    import webview      # pywebview: the optional `desktop` extra

    port = port or free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(self_command("run", "--headless", "--port", str(port), "--address", "127.0.0.1"))
    try:
        wait_until_up(url, server)
        webview.create_window("OpenStrategy Studio", url, width=1440, height=920, min_size=(900, 600))
        webview.start()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0
