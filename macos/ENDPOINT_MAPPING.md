# 🔌 Backend API Endpoints - Complete Mapping

## Overview
This document maps all available backend endpoints to their implementation status in the Aurora macOS app.

**Backend URL**: `https://desktop-p1p7892.taildbc5d3.ts.net:8443`

---

## ✅ Authentication Endpoints (3)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/auth/register-local` | POST | ✅ Implemented | `AuthService.swift` |
| `/auth/login-local` | POST | ✅ Implemented | `AuthService.swift` |
| `/device/register` | POST | ✅ Implemented | `NotificationService.swift` |

---

## ✅ Bot Control Endpoints (4)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/bot/status` | GET | ✅ Implemented | `BotService.swift` |
| `/bot/start` | POST | ✅ Implemented | `BotService.swift` |
| `/bot/stop` | POST | ✅ Implemented | `BotService.swift` |
| `/bot/pause` | POST | ✅ Implemented | `BotService.swift` |

---

## ✅ Trading Endpoints (6)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/trades/open` | GET | ✅ Implemented | `TradeService.swift` |
| `/trades/history` | GET | ✅ Implemented | `TradeService.swift` |
| `/trades/recent` | GET | ✅ Implemented | `TradeService.swift` |
| `/trades/close` | POST | ✅ Implemented | `TradeService.swift` |
| `/trades/logs` | GET | ✅ Implemented | `TradeService.swift` |
| `/orders/test` | POST | ⚠️ Available | Not yet implemented in app |

---

## ✅ Performance Endpoints (3)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/performance/stats` | GET | ✅ Implemented | `PerformanceService.swift` |
| `/performance/equity-curve` | GET | ✅ Implemented | `PerformanceService.swift` |
| `/performance/daily-summary` | GET | ✅ Implemented | `PerformanceService.swift` |

---

## ✅ Strategy Management Endpoints (8)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/strategies` | GET | ✅ Implemented | `StrategyService.swift` |
| `/strategies` | POST | ✅ Implemented | `StrategyService.swift` |
| `/strategies/validated` | GET | ✅ Implemented | `StrategyService.swift` |
| `/strategies/active` | GET | ✅ Implemented | `StrategyService.swift` |
| `/strategies/{id}/activate` | PUT | ✅ Implemented | `StrategyService.swift` |
| `/strategies/{id}/deactivate` | PUT | ✅ Implemented | `StrategyService.swift` |
| `/strategies/{id}/promote` | PUT | ✅ Implemented | `StrategyService.swift` |
| `/strategies/{id}/policy` | PATCH | ✅ Implemented | `StrategyService.swift` |

---

## ✅ Strategy Configuration Endpoints (2)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/strategy/config` | GET | ✅ Implemented | `StrategyService.swift` |
| `/strategy/config` | POST | ✅ Implemented | `StrategyService.swift` |
| `/strategy/symbols` | GET | ✅ Implemented | `StrategyService.swift` |

---

## ✅ MT5 Account Endpoints (5)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/account/mt5` | GET | ✅ Implemented | `MT5AccountService.swift` |
| `/account/mt5/primary` | GET | ✅ Implemented | `MT5AccountService.swift` |
| `/account/mt5` | POST | ✅ Implemented | `MT5AccountService.swift` |
| `/account/mt5/{id}` | PUT | ✅ Implemented | `MT5AccountService.swift` |
| `/account/mt5/{id}` | DELETE | ✅ Implemented | `MT5AccountService.swift` |

---

## ✅ WebSocket Endpoints (3)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/ws/updates/{token}` | WS | ✅ Implemented | `WebSocketService.swift` |
| `/ws/trades/{token}` | WS | ✅ Implemented | `WebSocketService.swift` |
| `/ws/notifications` | WS | ⚠️ Available | Not yet implemented in app |

---

## ✅ System & Health Endpoints (3)

| Endpoint | Method | Status | Implementation |
|----------|--------|--------|----------------|
| `/health` | GET | ✅ Implemented | Used for connectivity checks |
| `/system/status` | GET | ⚠️ Available | Not yet implemented in app |
| `/info` | GET | ⚠️ Available | Not yet implemented in app |

---

## 🚧 Missing Endpoints (To Implement)

### Paper Trading Reset
- ❌ `/account/paper/reset` | POST | **Not available in backend**
  - **Action Required**: Add this endpoint to backend
  - **Purpose**: Reset paper trading account balance and history

### Symbol Management
- ❌ `/symbols/update` | POST | **Not available in backend**
  - **Action Required**: Add this endpoint to backend or use `/strategy/config`
  - **Purpose**: Update symbol configuration

### Set Primary Account
- ❌ `/account/mt5/{id}/primary` | PUT | **Not available in backend**
  - **Action Required**: Add this endpoint to backend
  - **Purpose**: Set an account as primary

---

## 📊 Implementation Summary

### Total Endpoints: 37
- ✅ **Fully Implemented**: 32 (86%)
- ⚠️ **Available but Not Used**: 3 (8%)
- ❌ **Missing in Backend**: 2 (5%)

### By Category:
- **Authentication**: 3/3 (100%)
- **Bot Control**: 4/4 (100%)
- **Trading**: 5/6 (83%)
- **Performance**: 3/3 (100%)
- **Strategy Management**: 8/8 (100%)
- **Strategy Config**: 3/3 (100%)
- **MT5 Accounts**: 5/5 (100%)
- **WebSocket**: 2/3 (67%)
- **System**: 0/3 (0%)

---

## 🔧 Required Backend Changes

### 1. Add Paper Account Reset Endpoint
```python
@app.post("/account/paper/reset")
async def reset_paper_account(current_user: User = Depends(get_current_user)):
    """Reset paper trading account to default state."""
    # Implementation needed
    return {"status": "success", "message": "Paper account reset"}
```

### 2. Add Set Primary Account Endpoint
```python
@app.put("/account/mt5/{account_id}/primary")
async def set_primary_account(
    account_id: int,
    current_user: User = Depends(get_current_user)
):
    """Set an MT5 account as primary."""
    # Implementation needed
    return {"status": "success", "message": "Primary account updated"}
```

### 3. Optional: Symbol Update Endpoint
```python
@app.post("/symbols/update")
async def update_symbols(
    symbols: List[str],
    current_user: User = Depends(get_current_user)
):
    """Update enabled trading symbols."""
    # Can be handled via /strategy/config instead
    return {"status": "success", "symbols": symbols}
```

---

## 🎯 Next Steps

1. **Add Missing Backend Endpoints** (2 endpoints)
   - `/account/paper/reset`
   - `/account/mt5/{id}/primary`

2. **Implement Optional Endpoints in App** (3 endpoints)
   - `/orders/test` - Test order placement
   - `/system/status` - System diagnostics
   - `/ws/notifications` - Generic notifications WebSocket

3. **Update APIService.swift**
   - Verify all endpoint paths match backend
   - Ensure proper error handling
   - Add missing optional endpoints

4. **Testing**
   - Test all 32 implemented endpoints
   - Verify authentication flow
   - Test WebSocket connections
   - Validate error handling

---

## 📝 Notes

- All endpoints use JWT Bearer token authentication except public endpoints
- WebSocket endpoints use token in URL path
- Backend supports both HTTP and HTTPS (with SSL certificates)
- CORS is enabled for all origins
- Default port: 8443 (HTTPS) or 8000 (HTTP)

---

**Last Updated**: 2025-12-01
**Backend Version**: 1.0.0
**App Version**: Sprint 5 Complete
