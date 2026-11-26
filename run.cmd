@echo off
rem Starts OpenStrategy Studio from this checkout: double-click it, or type `run` here.
rem Anything after it is passed on: `run desktop`, `run selfcheck --quick`, `run --port 8600`.
rem `uv run` installs whatever uv.lock says is missing first; the desktop extra is named
rem every time, because a plain sync would remove it.
cd /d "%~dp0"
uv run --extra desktop openstrategy-studio %*
rem Keep the window open on an error, so a double-click does not hide the message.
if errorlevel 1 pause
