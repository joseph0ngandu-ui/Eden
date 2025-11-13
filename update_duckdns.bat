@echo off
REM DuckDNS Auto-Updater for EdenBot
REM Run this script every 5 minutes via Scheduled Task

set DUCKDNS_TOKEN=d8789fe0-6dc6-409c-9332-e7b8a1e0813e
set DOMAIN=edenbot
set LOG_FILE=C:\Users\Administrator\Eden\logs\duckdns.log

echo %date% %time% - Updating DuckDNS... >> "%LOG_FILE%"
curl.exe -s "https://www.duckdns.org/update?domains=%DOMAIN%&token=%DUCKDNS_TOKEN%&ip=" >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"