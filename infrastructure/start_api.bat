@echo off
cd /d C:\Users\Administrator\Eden\backend
set PYTHONUNBUFFERED=1
python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-certfile C:/EdenCerts/edenbot.duckdns.org-chain.pem --ssl-keyfile C:/EdenCerts/edenbot.duckdns.org-key.pem
