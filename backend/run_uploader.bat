@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to backend directory
cd /d "%~dp0"

REM Prefer project virtualenv Python if available
set "VENV_PY=%~dp0..\venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    echo [Eden] Using venv Python at %VENV_PY%
    start "EdenStatusUploader" /MIN "%VENV_PY%" "%~dp0status_uploader.py"
) else (
    echo [Eden] Using system Python
    start "EdenStatusUploader" /MIN python "%~dp0status_uploader.py"
)

endlocal
