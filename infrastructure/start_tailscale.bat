@echo off
:: Start Tailscale Funnel for Eden Bot
:: Requires Tailscale to be installed and authenticated

echo Starting Tailscale Funnel...
echo Pointing to localhost:8000

:: Check if tailscale is installed
where tailscale >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: tailscale is not installed or not in PATH.
    echo Please install Tailscale from: https://tailscale.com/download
    pause
    exit /b 1
)

:: Enable Funnel (exposes localhost:8000 to the public internet)
:: You may need to run 'tailscale funnel status' to see your URL
echo Enabling Funnel for port 8000...
tailscale funnel 8000

pause
