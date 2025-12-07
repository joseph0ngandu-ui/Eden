@echo off
:: Eden One-Click Launcher
:: Starts API, Trading Bot, and Tailscale Funnel

echo Starting Eden System...
cd /d "%~dp0"

:: 1. Start API (Port 8000, HTTP)
:: Tailscale Funnel handles the SSL/HTTPS termination
echo Launching API...
start "Eden API" cmd /k "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

:: 2. Start Trading Bot
echo Launching Bot...
start "Eden Bot" cmd /k "set EDEN_SHADOW=0 && python bot_runner.py"

:: 3. Start Tailscale Funnel
echo Launching Network...
start "Eden Network" cmd /k "infrastructure\start_tailscale.bat"

echo.
echo All systems launched!
echo 1. API running on port 8000
echo 2. Bot process active
echo 3. Tailscale Funnel exposing port 8000
echo.
pause
