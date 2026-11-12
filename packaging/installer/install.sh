#!/usr/bin/env bash
# First-run installer for OpenStrategy Studio on Linux (macOS should work but is untested).
#
#   bash packaging/installer/install.sh                 # from the repository folder
#   NATIVE_TLS=1 bash packaging/installer/install.sh    # behind a TLS-inspecting proxy or antivirus
#
# Needs the network once: to fetch uv (if missing), Python 3.12 and the locked
# dependencies. Afterwards the launcher runs fully offline.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
say() { printf '\n==> %s\n' "$*"; }

if [ "${NATIVE_TLS:-0}" = "1" ]; then
  export UV_NATIVE_TLS=1            # uv otherwise fails every fetch with UnknownIssuer behind such a proxy
fi

say "Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (a one-off download from astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
UV="$(command -v uv)"

# vectorbtX comes from GitHub until it is on PyPI, and uv needs git for that.
if grep -q 'vectorbtx = { git' "$REPO/pyproject.toml" && ! command -v git >/dev/null 2>&1; then
  echo "vectorbtX is installed from GitHub, which needs git (for example: sudo apt install git)." >&2
  echo "Or install it from a wheel instead: see docs/PACKAGING.md." >&2
  exit 1
fi

cd "$REPO"
if [ ! -f uv.lock ]; then
  say "No uv.lock yet: resolving the dependencies once (commit the file this writes)"
  "$UV" lock
fi
say "Installing Python and the Studio's dependencies (first run only)"
"$UV" sync --locked

say "Checking that this machine can build, backtest and optimise"
if ! "$UV" run --frozen openstrategy-studio selfcheck; then
  echo "The self-check failed; the messages above say which part. Nothing else was changed." >&2
  exit 1
fi

LAUNCHER="$REPO/openstrategy-studio.sh"
cat > "$LAUNCHER" <<LAUNCH
#!/usr/bin/env bash
# Starts OpenStrategy Studio. --frozen and --offline stop uv from touching the network.
exec "$UV" run --frozen --offline --project "$REPO" openstrategy-studio "\$@"
LAUNCH
chmod +x "$LAUNCHER"

if [ "$(uname -s)" = "Linux" ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/openstrategy-studio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=OpenStrategy Studio
Comment=Build, backtest and optimise trading strategies
Exec="$LAUNCHER"
Terminal=true
Categories=Office;Finance;
DESKTOP
fi
say "Done. Start the Studio from your applications menu, or run: $LAUNCHER"
