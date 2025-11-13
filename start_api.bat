@echo off
cd /d C:\Users\Administrator\Eden\backend
set PYTHONUNBUFFERED=1
python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-certfile C:/Users/Administrator/Eden/backend/ssl/cert.pem --ssl-keyfile C:/Users/Administrator/Eden/backend/ssl/key.pem
