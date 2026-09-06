@echo off
rem ---------------------------------------------------------------------------
rem  Quick start / restart for the local llama.cpp chat app.
rem
rem  All the real logic (environment checks, killing the previous instance,
rem  waiting for ports to be released, opening the browser) lives in
rem  scripts\launcher.ps1. This file is a deliberately plain-ASCII shell:
rem  cmd.exe re-reads a .bat by byte offset, so non-ASCII text plus a codepage
rem  switch desyncs it and comment fragments get executed as commands, and any
rem  inline PowerShell gets re-parsed by start/quote handling. Neither can
rem  happen if this file stays ASCII and delegates.
rem
rem  Do not add non-ASCII text here. Put it in scripts\launcher.ps1, which
rem  PowerShell decodes by BOM regardless of the console codepage.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\launcher.ps1" -Root "%ROOT%"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\python.exe" run.py
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo App exited with code %RC%. The window is kept open so you can read the log.
    pause
)

endlocal & exit /b %RC%
