@echo off
echo ================================================================================
echo Restarting Eden Backend API (HTTPS)
echo ================================================================================
echo.
echo Step 1: Stopping any running uvicorn processes...
taskkill /F /FI "WINDOWTITLE eq *uvicorn*" 2>nul
taskkill /F /FI "IMAGENAME eq python.exe" /FI "MEMUSAGE gt 50000" 2>nul
timeout /t 2 /nobreak >nul

echo Step 2: Starting backend with new code...
cd /d "%~dp0backend"
start "Eden Backend API" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem"

echo.
echo Step 3: Waiting for server to start...
timeout /t 5 /nobreak

echo.
echo Step 4: Running verification script...
cd /d "%~dp0"
python verify_api_contract.py

echo.
echo ================================================================================
echo Restart Complete
echo ================================================================================
pause
