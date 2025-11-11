@echo off
echo ================================================================================
echo Starting Eden Backend API
echo ================================================================================
echo.
echo API will be available at:
echo   - http://localhost:8000/docs
echo   - http://localhost:8000/redoc
echo.
echo Default Login:
echo   Email: admin@eden.com
echo   Password: admin123
echo.
echo ================================================================================
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000

pause
