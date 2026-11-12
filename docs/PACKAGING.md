# Phase 4: shipping a standalone app

The plan asked for three spikes, a decision, and one shipped option. All three spikes are in `packaging/`. **The decision is to ship the first-run installer**, keep Docker as a supported second route, and leave the PyInstaller bundle as an experiment until it passes the gate on both operating systems.

None of the three was run end to end when this was written, because the build environment had no network. What was checked: the Linux installer ran against a stand-in `uv` and wrote a correct launcher and desktop entry; the desktop wrapper's start-up wait was tested against a real local server; and the frozen-app command lines are unit-tested. The gate below is what proves each route on a clean machine.

## The three spikes

| | First-run installer | Docker | PyInstaller bundle with a webview |
|---|---|---|---|
| Files | `packaging/installer/install.sh`, `install.ps1` | `packaging/docker/Dockerfile`, `compose.yaml` | `packaging/pyinstaller/studio.spec`, `entry.py`, `src/openstrategy_studio/desktop.py` |
| What the user gets | A desktop shortcut that opens the app in the browser | A container on `127.0.0.1:8501` | A folder with a double-clickable app in its own window |
| First run needs | The network once: uv, Python 3.12 and the locked dependencies (about 1 GB) | Docker and the network once, to build | Nothing: everything is in the folder |
| After that | Fully offline (`uv run --frozen --offline`) | Fully offline | Fully offline |
| numba | Normal: cache in the data folder | Normal: cache in the `/data` volume | Fragile (see below) |
| Updates | `git pull`, then `uv sync --locked` | Rebuild the image | Rebuild and redistribute the folder |
| Size | Shared uv cache, about 1 GB | About 1.5 GB image | About 1 GB per OS, built on each OS |
| Main risk | Needs git while vectorbtX comes from GitHub | Docker Desktop on Windows is heavy for one user | Hidden imports and numba cache loading |

### Why the installer

It runs exactly the environment the tests run (`uv.lock`), so a bug report from a user reproduces on a developer machine. numba and Streamlit behave as they do in development. It is the only option where an update is a `git pull`. Its one real cost is the first-run download, which the plan accepts ("works offline once data is stored locally").

### Shipping it

1. Commit `uv.lock` (`uv lock` once; CI uses `uv sync --locked`).
2. Users get the repository (a zip of a tagged release is enough), then run:
   - Linux: `bash packaging/installer/install.sh`
   - Windows: `powershell -ExecutionPolicy Bypass -File packaging\installer\install.ps1`
3. The installer installs uv if it is missing, syncs from the lock file, runs `openstrategy-studio selfcheck`, and only then writes the launcher (`openstrategy-studio.sh` or `OpenStrategy Studio.cmd`) and a desktop entry or shortcut.
4. Behind a TLS-inspecting proxy or antivirus, use `NATIVE_TLS=1` (Linux) or `-NativeTls` (Windows), which sets `UV_NATIVE_TLS`. The Yahoo and FRED downloaders need their own certificate settings; see the vectorbtX README.

### Without git: vectorbtX from a wheel

While vectorbtX is installed from GitHub, uv needs git, and Windows machines often lack it. To avoid that, build a wheel once (`uv build` in a vectorbtX checkout), put it in `packaging/wheels/`, and point the source at it in `pyproject.toml`:

```toml
[tool.uv.sources]
vectorbtx = { path = "packaging/wheels/vectorbtx-<version>-py3-none-any.whl" }
```

Then `uv lock` again and commit both files. Once vectorbtX is on PyPI, delete the source entry instead.

## Docker

```bash
docker compose -f packaging/docker/compose.yaml up --build
```

The Studio runs as a non-root user. All data, settings and the numba cache live in the `studio-data` volume (`OPENSTRATEGY_STUDIO_HOME=/data`). The port is bound to `127.0.0.1` only. Match `UV_VERSION` in the Dockerfile to the uv that wrote `uv.lock`.

## The PyInstaller bundle (experimental)

```bash
uv sync --extra desktop --group bundle
uv run pyinstaller packaging/pyinstaller/studio.spec --noconfirm
"dist/OpenStrategy Studio/OpenStrategy Studio"        # Windows: ...\OpenStrategy Studio.exe
```

How it works: double-clicking runs `desktop`, which starts the Streamlit server by re-running the same executable with `run --headless` and shows it in a pywebview window. Background jobs also re-run the executable (`worker <dir>`), because a frozen app has no `python -m`. `cli.self_command` handles both cases, and `multiprocessing.freeze_support()` in `entry.py` handles the optimiser's process pools.

What to expect when you try it:

* **numba's cache.** vectorbt's kernels use `cache=True`, and numba locates cached functions by source file. The spec keeps `.py` sources on disk (`include_py_files=True`), and the Studio points `NUMBA_CACHE_DIR` at the data folder. If start-up still fails with "cannot cache function ... no locator available", check that vectorbt's `.py` files really are in the bundle's `_internal` folder. This is the most likely failure, and the reason the bundle stays experimental.
* **Hidden imports.** `collect_all` covers the known plugin packages. Anything else shows up as a `ModuleNotFoundError` in the console window. Add it to `hiddenimports`.
* **Streamlit.** `--global.developmentMode false` is always passed, because a frozen app otherwise looks like a source checkout to Streamlit. Its front-end files come in through `collect_all("streamlit")`.
* **Linux webview.** pywebview needs GTK or Qt from the system packages (for example `python3-gi gir1.2-webkit2-4.1`).
* **Build on each OS.** PyInstaller does not cross-compile.

## The gate: "a clean machine runs build, backtest and optimise"

On a fresh Linux VM and a fresh Windows VM, with nothing installed but the OS and git:

1. Run the installer. It ends by running `openstrategy-studio selfcheck`, which must pass. The self-check needs no network and uses a temporary data folder. It checks:
   - the environment: every package, and a writable numba cache;
   - storing and loading synthetic data;
   - validating, compiling and exporting every template;
   - backtesting every template, then saving and redrawing each run;
   - background jobs across two processes;
   - a real background optimisation with a held-back test;
   - that every app page loads.
2. Disconnect the network, start the Studio from the shortcut, and by hand:
   - import a CSV;
   - build a strategy from a template;
   - run a backtest and save it;
   - start an optimisation, close the browser, reopen it, and see the job finish;
   - apply the best trial.
3. Repeat step 1 for Docker. Repeat it for the bundle too, before calling the bundle supported.

CI runs `openstrategy-studio selfcheck` on Linux and Windows after the test suite. A green CI run is the automated half of this gate.

## Licence

vectorbt carries the Commons Clause. Every route here is for personal or internal use: do not offer the Studio, or an image or bundle of it, as a paid service.
