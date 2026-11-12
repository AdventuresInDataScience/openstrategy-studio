<#
First-run installer for OpenStrategy Studio on Windows.

    powershell -ExecutionPolicy Bypass -File packaging\installer\install.ps1
    powershell -ExecutionPolicy Bypass -File packaging\installer\install.ps1 -NativeTls   # behind a TLS-inspecting proxy

Needs the network once: to fetch uv (if missing), Python 3.12 and the locked
dependencies. Afterwards the desktop shortcut runs fully offline.
#>
param([switch]$NativeTls)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
function Say($Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }

if ($NativeTls) { $env:UV_NATIVE_TLS = "true" }   # uv otherwise fails every fetch with UnknownIssuer

Say "Checking for uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv (a one-off download from astral.sh)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
$Uv = (Get-Command uv).Source

# vectorbtX comes from GitHub until it is on PyPI, and uv needs git for that.
$FromGit = Select-String -Path (Join-Path $Repo "pyproject.toml") -Pattern 'vectorbtx = \{ git' -Quiet
if ($FromGit -and -not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "vectorbtX is installed from GitHub, which needs Git for Windows (https://git-scm.com). Or install it from a wheel instead: see docs\PACKAGING.md."
}

Push-Location $Repo
try {
    if (-not (Test-Path "uv.lock")) {
        Say "No uv.lock yet: resolving the dependencies once (commit the file this writes)"
        & $Uv lock; if ($LASTEXITCODE) { throw "uv lock failed." }
    }
    Say "Installing Python and the Studio's dependencies (first run only)"
    & $Uv sync --locked; if ($LASTEXITCODE) { throw "uv sync failed." }
    Say "Checking that this machine can build, backtest and optimise"
    & $Uv run --frozen openstrategy-studio selfcheck
    if ($LASTEXITCODE) { throw "The self-check failed; the messages above say which part. Nothing else was changed." }
}
finally { Pop-Location }

# A launcher next to the code, and a desktop shortcut to it. "Default" is the
# ANSI code page, which keeps cmd.exe happy with accented user names.
$Launcher = Join-Path $Repo "OpenStrategy Studio.cmd"
@"
@echo off
rem Starts OpenStrategy Studio. --frozen and --offline stop uv from touching the network.
"$Uv" run --frozen --offline --project "$Repo" openstrategy-studio %*
"@ | Set-Content -Encoding Default $Launcher
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "OpenStrategy Studio.lnk"))
$Shortcut.TargetPath = $Launcher
$Shortcut.WorkingDirectory = $Repo
$Shortcut.Save()
Say "Done. Double-click 'OpenStrategy Studio' on your desktop."
