"""Phase 4: the command line, frozen-app plumbing, the desktop wrapper, installers and the self-check."""

import http.server
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from openstrategy_studio import cli, data as D, desktop

ROOT = Path(__file__).parent.parent


def test_self_command(monkeypatch):
    assert cli.self_command("worker", "x") == [sys.executable, "-m", "openstrategy_studio.cli", "worker", "x"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert cli.self_command("worker", "x") == [sys.executable, "worker", "x"]


def test_cli_routing(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_app", lambda port, headless, address, extra: calls.append(("run", port, headless, address)) or 0)
    assert cli.main([]) == 0
    assert cli.main(["--port", "9000", "--headless"]) == 0
    assert cli.main(["run", "--address", "0.0.0.0"]) == 0
    assert calls == [("run", None, False, None), ("run", 9000, True, None), ("run", None, False, "0.0.0.0")]
    from openstrategy_studio import worker
    monkeypatch.setattr(worker, "main", lambda job_dir: calls.append(("worker", job_dir)) or "done")
    assert cli.main(["worker", "/tmp/job"]) == 0 and calls[-1] == ("worker", "/tmp/job")


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/_stcore/health" else 404)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def test_desktop_waits_for_the_server():
    server = http.server.HTTPServer(("127.0.0.1", desktop.free_port()), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        desktop.wait_until_up(f"http://127.0.0.1:{server.server_port}", alive, timeout=10)
    finally:
        alive.kill()
        server.shutdown()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    with pytest.raises(RuntimeError, match="stopped while starting"):
        desktop.wait_until_up(f"http://127.0.0.1:{desktop.free_port()}", dead, timeout=5)


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs bash")
def test_installer_script_parses():
    subprocess.run(["bash", "-n", str(ROOT / "packaging" / "installer" / "install.sh")], check=True)


def test_packaging_files_reference_real_paths():
    dockerfile = (ROOT / "packaging" / "docker" / "Dockerfile").read_text()
    for name in ("pyproject.toml", ".python-version", "README.md"):
        assert name in dockerfile and (ROOT / name).exists()
    spec = (ROOT / "packaging" / "pyinstaller" / "studio.spec").read_text()
    assert '"entry.py"' in spec and (ROOT / "packaging" / "pyinstaller" / "entry.py").exists()


def test_synthetic_data_is_clean():
    frames = D.synthetic_ohlcv(["EURUSD=X", "SPY"], n=300, seed=1)
    assert frames["close"]["EURUSD=X"].iloc[0] < 2 and frames["close"]["SPY"].iloc[0] > 50
    for symbol in ("EURUSD=X", "SPY"):
        top = np.maximum(frames["open"][symbol], frames["close"][symbol])
        bottom = np.minimum(frames["open"][symbol], frames["close"][symbol])
        assert (frames["high"][symbol] >= top).all() and (frames["low"][symbol] <= bottom).all()


def test_selfcheck_passes(monkeypatch):
    """The phase 4 gate. Skipped unless everything is installed."""
    for name in ("vectorbtx", "ta_patterns", "optuna", "butchc", "streamlit", "pyarrow"):
        pytest.importorskip(name)
    from openstrategy_studio import selfcheck
    assert selfcheck.main(quick=True) == 0
