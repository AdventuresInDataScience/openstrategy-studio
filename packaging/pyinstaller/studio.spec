# PyInstaller spec for the bundled desktop app (phase 4 spike; see docs/PACKAGING.md).
#
#   uv sync --extra desktop --group bundle
#   uv run pyinstaller packaging/pyinstaller/studio.spec --noconfirm
#
# Produces dist/OpenStrategy Studio/ (one folder, not one file: numba and the
# Streamlit front end start much faster unpacked). Build on each target OS.
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [], [], []
# Packages with data files, plugins or lazy imports. include_py_files keeps the
# .py sources on disk: numba's cache finds compiled functions by source file,
# and vectorbt's cache=True kernels fail to load without them.
for package in ("openstrategy_studio", "streamlit", "plotly", "plotly_themes", "vectorbt", "vectorbtx",
                "ta_patterns", "butchc", "optuna", "numba", "llvmlite", "pyarrow", "webview"):
    package_datas, package_binaries, package_hidden = collect_all(package, include_py_files=True)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
# importlib.metadata.version() is used by Streamlit and by engine.versions().
for distribution in ("streamlit", "vectorbt", "vectorbtx", "plotly", "pandas", "numpy", "ta_patterns", "butchc",
                     "optuna", "pyarrow", "openstrategy-studio"):
    datas += copy_metadata(distribution)

a = Analysis(["entry.py"], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["tkinter", "pytest", "IPython"], noarchive=False)
pyz = PYZ(a.pure)
# console=True while this is a spike, so start-up errors are visible.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="OpenStrategy Studio", console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="OpenStrategy Studio")
