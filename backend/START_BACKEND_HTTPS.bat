@echo off
echo ================================================================================
echo Starting Eden Backend API with HTTPS
echo ================================================================================
echo.
echo API will be available at:
echo   - https://localhost:8443/docs
echo   - https://localhost:8443/redoc
echo.
echo Default Login:
echo   Email: admin@eden.com
echo   Password: admin123
echo.
echo Note: You may see certificate warnings - this is normal for self-signed certs
echo ================================================================================
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem

pause
