@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
:: Add MT5 to PATH if it exists
if exist "C:\Program Files\MetaTrader 5 Terminal" set PATH=C:\Program Files\MetaTrader 5 Terminal;%PATH%

:: Check for venv
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe watchdog.py
) else (
    python watchdog.py
)
