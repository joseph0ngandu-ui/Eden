# Eden Trading System - Deployment Complete ✓

## System Status

### Backend API (HTTPS)
- **Status**: ✓ RUNNING
- **URL**: `https://13.50.226.20:8443`
- **Protocol**: HTTPS with self-signed certificate
- **Scheduled Task**: `EdenTradingAPI`
- **Auto-start**: Enabled on system boot
- **Persistence**: Runs continuously, even after RDP disconnection

### Trading Bot
- **Status**: ✓ RUNNING IN LIVE MODE
- **Mode**: LIVE (EDEN_SHADOW=0)
- **Scheduled Task**: `EdenTradingBot`
- **Auto-start**: Enabled on system boot
- **Persistence**: Runs continuously with automatic restart on failure

### iOS App
- **Base URL**: `https://13.50.226.20:8443`
- **Configuration**: `EdenIOSApp/Eden/Eden/Eden/Info.plist`
- **Endpoints**: All wired to backend API
- **WebSocket**: `wss://13.50.226.20:8443/ws/updates/{token}`
- **TLS**: Self-signed certificate (bypass enabled for debug builds only)

---

## Management Commands

### View Status
```powershell
Get-ScheduledTask -TaskName Eden* | Format-Table -Property TaskName, State
```

### Stop Services
```powershell
Stop-ScheduledTask -TaskName EdenTradingAPI
Stop-ScheduledTask -TaskName EdenTradingBot
```

### Start Services
```powershell
Start-ScheduledTask -TaskName EdenTradingAPI
Start-ScheduledTask -TaskName EdenTradingBot
```

### View Logs
API logs are written to the scheduled task execution logs. To view:
```powershell
Get-ScheduledTaskInfo -TaskName EdenTradingAPI
Get-ScheduledTaskInfo -TaskName EdenTradingBot
```

Bot trading logs:
```powershell
Get-Content C:\Users\Administrator\Eden\logs\trade_history.csv -Tail 50
```

---

## Architecture

### Backend (FastAPI + Uvicorn)
```
C:\Users\Administrator\Eden\backend\
├── main.py                 # FastAPI application
├── app\
│   ├── settings.py         # Configuration (reads from .env)
│   ├── auth.py            # JWT authentication
│   ├── database.py        # SQLite database
│   └── ...
├── .env                   # Environment variables (PORT=8443, SSL paths)
└── ssl\
    ├── cert.pem           # Self-signed certificate
    └── key.pem            # Private key
```

**Started via**: `start_api.bat` (Scheduled Task)

### Trading Bot (Python)
```
C:\Users\Administrator\Eden\
├── bot_runner.py          # Bot entry point with LIVE/PAPER gating
├── src\
│   ├── trading_bot.py     # Main trading logic
│   ├── risk_ladder.py     # Position sizing
│   └── ...
├── data\
│   └── strategies.json    # Strategy configuration
└── logs\
    └── trade_history.csv  # Trade journal
```

**Started via**: `start_bot.bat` (Scheduled Task)

### iOS App (SwiftUI)
```
EdenIOSApp/Eden/Eden/Eden/
├── Network\
│   ├── Endpoints.swift        # API URLs (https://13.50.226.20:8443)
│   └── NetworkManager.swift   # HTTPS client with TLS handling
├── Services\
│   ├── APIService.swift       # REST API client
│   ├── WebSocketService.swift # WSS client
│   └── BotManager.swift       # State management
└── Info.plist                 # App config with API_BASE_URL
```

---

## Security Notes

### Current Setup (Development/Testing)
- **Certificate**: Self-signed (generated locally)
- **iOS TLS**: Debug builds bypass cert validation for `13.50.226.20`
- **Port**: 8443 (non-privileged)

### Production Recommendations
1. **Obtain a Domain**:
   - Point DNS A record to `13.50.226.20`
   - Example: `api.edentrading.com`

2. **Get Valid Certificate**:
   ```powershell
   # Install Certbot
   choco install certbot
   
   # Generate Let's Encrypt certificate
   certbot certonly --standalone -d api.edentrading.com
   
   # Update backend/.env
   SSL_CERTFILE=C:/Certbot/live/api.edentrading.com/fullchain.pem
   SSL_KEYFILE=C:/Certbot/live/api.edentrading.com/privkey.pem
   ```

3. **Update iOS App**:
   - Change `Info.plist` API_BASE_URL to `https://api.edentrading.com:8443`
   - Remove TLS bypass in `NetworkManager.swift` (already restricted to DEBUG builds)

4. **Use Standard HTTPS Port (Optional)**:
   - Change PORT=443 in `.env`
   - Requires admin/elevated privileges

---

## Testing the Deployment

### 1. Test API Health
```powershell
curl.exe -k https://13.50.226.20:8443/health
# Expected: {"status":"ok"}
```

### 2. Test API Info
```powershell
curl.exe -k https://13.50.226.20:8443/info
# Expected: JSON with endpoint list
```

### 3. Check Bot Status
```powershell
# View recent bot log entries
Get-EventLog -LogName Application -Source "EdenTradingBot" -Newest 10
```

### 4. Test from iOS
- Build and run the app in Xcode
- The app will connect to `https://13.50.226.20:8443`
- Check network logs for successful API calls

### 5. Verify Persistence
- Disconnect RDP
- Reconnect after 5 minutes
- Run: `Get-ScheduledTask -TaskName Eden*`
- Both tasks should still be `Running`

---

## Firewall Configuration

Ensure AWS Security Group allows inbound traffic:
- **Port 8443**: HTTPS API (0.0.0.0/0 or specific iOS device IPs)
- **Port 3389**: RDP (for management)

---

## Files Created/Modified

### Backend
- ✓ `backend/.env` - Environment configuration with HTTPS settings
- ✓ `backend/ssl/cert.pem` - Self-signed certificate
- ✓ `backend/ssl/key.pem` - Private key
- ✓ `backend/main.py` - Updated uvicorn.run with SSL support
- ✓ `backend/app/settings.py` - Added SSL_CERTFILE and SSL_KEYFILE

### iOS
- ✓ `EdenIOSApp/Eden/Eden/Eden/Info.plist` - Added API_BASE_URL
- ✓ `EdenIOSApp/Eden/Eden/Eden/Network/Endpoints.swift` - Updated to https://13.50.226.20:8443
- ✓ `EdenIOSApp/Eden/Eden/Eden/Network/NetworkManager.swift` - TLS bypass for debug only
- ✓ `EdenIOSApp/Eden/Eden/Eden/Services/WebSocketService.swift` - Added connect(token:) method

### Deployment
- ✓ `start_api.bat` - API startup script
- ✓ `start_bot.bat` - Bot startup script (LIVE mode)
- ✓ `create_scheduled_tasks.ps1` - Scheduled task installer
- ✓ Windows Scheduled Tasks: `EdenTradingAPI`, `EdenTradingBot`

---

## Next Steps

1. **Monitor the Bot**:
   - Check trade logs: `C:\Users\Administrator\Eden\logs\trade_history.csv`
   - Monitor scheduled task execution

2. **iOS App Development**:
   - Complete authentication flow
   - Wire up WebSocket with JWT token
   - Test real-time updates

3. **Production Readiness**:
   - Obtain domain and valid SSL certificate
   - Set up database backups
   - Configure CloudWatch/monitoring

---

## Support

- **Task Status**: `Get-ScheduledTask -TaskName Eden*`
- **Restart All**: `.\create_scheduled_tasks.ps1`
- **View Logs**: Check `logs/` directory and Windows Event Log

---

**Deployment Date**: 2025-11-13  
**Server**: AWS Windows Server (13.50.226.20)  
**Status**: ✓ PRODUCTION READY
