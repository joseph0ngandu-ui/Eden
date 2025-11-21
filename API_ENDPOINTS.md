# Eden Trading Bot API Endpoints

**Base URL**: `https://desktop-p1p7892.taildbc5d3.ts.net`

## Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/register-local` | Register new user | No |
| POST | `/auth/login-local` | Login user | No |

**Register Request**:
```json
{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe"
}
```

**Login Request**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Login Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

## Bot Control Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/bot/status` | Get bot status | Yes |
| POST | `/bot/start` | Start the bot | Yes |
| POST | `/bot/stop` | Stop the bot | Yes |
| POST | `/bot/pause` | Pause the bot | Yes |

---

## Trading Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/trades/open` | Get open positions | Yes |
| GET | `/trades/history` | Get trade history | Yes |
| GET | `/trades/recent` | Get recent trades | Yes |
| GET | `/trades/logs` | Get trade logs | Yes |
| POST | `/trades/close` | Close a position | Yes |

---

## Performance Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/performance/stats` | Get performance statistics | Yes |
| GET | `/performance/equity-curve` | Get equity curve data | Yes |
| GET | `/performance/daily-summary` | Get daily summary | Yes |

---

## Account Management Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/account/mt5` | List all MT5 accounts | Yes |
| GET | `/account/mt5/primary` | Get primary MT5 account | Yes |
| POST | `/account/mt5` | Create MT5 account | Yes |
| PUT | `/account/mt5/{id}` | Update MT5 account | Yes |
| DELETE | `/account/mt5/{id}` | Delete MT5 account | Yes |

**Create MT5 Account Request**:
```json
{
  "account_number": "12345678",
  "account_name": "My Trading Account",
  "broker": "Deriv",
  "server": "Deriv-Demo",
  "password": "mt5password",
  "is_primary": true
}
```

---

## Strategy Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/strategy/config` | Get strategy config | Yes |
| POST | `/strategy/config` | Update strategy config | Yes |
| GET | `/strategy/symbols` | Get available symbols | Yes |

---

## System Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Health check | No |
| GET | `/system/status` | System status | Yes |
| GET | `/info` | API info | No |

---

## Device & Notifications

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/device/register` | Register device for push notifications | Yes |

**Register Device Request**:
```json
{
  "token": "device_push_token_here"
}
```

---

## Test Orders

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/orders/test` | Place test order | Yes |

**Test Order Request**:
```json
{
  "symbol": "Volatility 75 Index",
  "side": "BUY",
  "volume": 0.01
}
```

---

## Authentication Header

For all authenticated endpoints, include the JWT token:

```
Authorization: Bearer {access_token}
```

---

## WebSocket Endpoint

```
wss://desktop-p1p7892.taildbc5d3.ts.net/ws/notifications
```

---

## Important Notes

1. **NO `/api` prefix** - Endpoints do NOT use `/api/` prefix
2. **HTTPS only** - All requests must use HTTPS
3. **JWT required** - Most endpoints require authentication
4. **Content-Type** - Use `application/json` for request bodies
5. **CORS enabled** - API accepts requests from any origin

---

## Quick Test

Test connectivity:
```bash
curl https://desktop-p1p7892.taildbc5d3.ts.net/health
```

Expected response:
```json
{"status":"ok"}
```
