# Eden API - Endpoint Test Summary

## 🚀 Backend Status: **ONLINE**

**Base URL**: `https://localhost:8443`  
**Tailscale URL**: `https://desktop-p1p7892.taildbc5d3.ts.net`

---

## ✅ Verified Working Endpoints

### Health & System
- ✅ `GET /health` - Returns `{"status":"ok"}`
- ✅ `GET /info` - API information
- ✅ `GET /system/status` - System diagnostics (requires auth)

### Authentication
- ✅ `POST /auth/register-local` - Register new user
- ✅ `POST /auth/login-local` - Login and get JWT token

### Bot Control
- ✅ `GET /bot/status` - Get bot status (public, no auth)
- ✅ `POST /bot/start` - Start trading bot (requires auth)
- ✅ `POST /bot/stop` - Stop trading bot (requires auth)
- ✅ `POST /bot/pause` - Pause trading bot (requires auth)

### Trading
- ✅ `GET /trades/open` - Get open positions (public)
- ✅ `GET /trades/history?limit=100` - Trade history (requires auth)
- ✅ `GET /trades/recent?days=7` - Recent trades (requires auth)
- ✅ `GET /trades/logs?limit=100` - Trade logs (public)
- ✅ `POST /trades/close` - Close position (requires auth)

### Orders
- ✅ `POST /orders/test` - Place test order (requires auth)

### Performance
- ✅ `GET /performance/stats` - Performance statistics (public)
- ✅ `GET /performance/equity-curve` - Equity curve data (requires auth)
- ✅ `GET /performance/daily-summary` - Daily PnL summary (requires auth)

### Strategy Configuration
- ✅ `GET /strategy/config` - Get current strategy config (public)
- ✅ `POST /strategy/config` - Update strategy config (requires auth)
- ✅ `GET /strategy/symbols` - Get trading symbols (requires auth)

### Strategy Management
- ✅ `GET /strategies` - List all strategies
- ✅ `POST /strategies` - Upload new strategy
- ✅ `GET /strategies/validated` - List validated strategies
- ✅ `GET /strategies/active` - List active strategies
- ✅ `PUT /strategies/{id}/activate` - Activate strategy (requires auth)
- ✅ `PUT /strategies/{id}/deactivate` - Deactivate strategy (requires auth)
- ✅ `PUT /strategies/{id}/promote` - Promote to LIVE (requires auth)
- ✅ `PATCH /strategies/{id}/policy` - Update policy (requires auth)

### MT5 Accounts
- ✅ `GET /account/mt5` - List MT5 accounts (requires auth)
- ✅ `GET /account/mt5/primary` - Get primary account (requires auth)
- ✅ `POST /account/mt5` - Create MT5 account (requires auth)
- ✅ `PUT /account/mt5/{id}` - Update MT5 account (requires auth)
- ✅ `DELETE /account/mt5/{id}` - Delete MT5 account (requires auth)

### Device Registration
- ✅ `POST /device/register` - Register device for push notifications (requires auth)

### WebSocket
- ✅ `WSS /ws/notifications` - WebSocket for real-time updates
- ✅ `WSS /ws/updates/{token}` - Authenticated WebSocket

---

## 📱 iOS App Integration

Use the TypeScript endpoint definitions in `ios/ApiEndpoints.ts`:

```typescript
import { API_BASE_URL, AuthEndpoints, BotEndpoints, createAuthHeaders } from './ApiEndpoints';

// Example: Login
const response = await fetch(AuthEndpoints.login, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email: 'user@example.com', password: 'password123' })
});
const { access_token } = await response.json();

// Example: Get Bot Status
const botStatus = await fetch(BotEndpoints.status);
const data = await botStatus.json();
console.log('Bot is running:', data.is_running);

// Example: Authenticated Request
const headers = createAuthHeaders(access_token);
const trades = await fetch(TradeEndpoints.history + '?limit=10', { headers });
```

---

## 🔧 Backend Startup

The backend is now configured to start with the correct Python path:

```powershell
# Use the fixed startup script
cd backend
.\START_BACKEND_FIXED.bat
```

Or manually:
```powershell
cd backend
& "C:\Program Files\Cloudbase Solutions\Cloudbase-Init\Python\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem
```

---

## 📊 Total API Surface

- **40+ REST Endpoints**
- **2 WebSocket Endpoints**
- **Authentication**: JWT Bearer Token
- **Protocol**: HTTPS with self-signed certificates
- **CORS**: Enabled for all origins

---

## 🎯 Next Steps for iOS App

1. ✅ Import `ios/ApiEndpoints.ts` into your iOS TypeScript project
2. ✅ Update `API_BASE_URL` based on your network:
   - Local: `https://localhost:8443`
   - Tailscale: `https://desktop-p1p7892.taildbc5d3.ts.net`
3. ✅ Implement authentication flow with `/auth/login-local`
4. ✅ Store JWT token securely (KeyChain/SecureStore)
5. ✅ Use `createAuthHeaders()` helper for authenticated requests
6. ✅ Connect WebSocket for real-time updates

---

**Last Tested**: 2025-11-30 04:42 UTC  
**Backend Version**: 1.0.0  
**Status**: ✅ All endpoints operational
