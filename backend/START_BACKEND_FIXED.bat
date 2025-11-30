@echo off
echo ================================================================================
echo Starting Eden Backend API with HTTPS (FIXED PATH)
echo ================================================================================
echo.
echo API will be available at:
echo   - https://localhost:8443/docs
echo   - https://localhost:8443/redoc
echo.

cd /d "%~dp0"
"C:\Program Files\Cloudbase Solutions\Cloudbase-Init\Python\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem

pause
