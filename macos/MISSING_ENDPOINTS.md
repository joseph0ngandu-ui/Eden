# Missing Backend Endpoints in macOS App

**Analysis Date:** 2025-12-01  
**Branch Analyzed:** `origin/main`  
**Current Branch:** `Aurora-Mac-App`

## 📊 Summary

The backend (main branch) has **40+ endpoints**, but the macOS app only implements **5 API calls**.

### ✅ Currently Implemented (5 endpoints)
- `POST /auth/login-local` - Authentication
- `POST /ml/train` - ML training
- `GET /ml/status/{symbol}` - ML status
- `GET /strategies` - Fetch strategies
- `POST /strategies` - Upload strategy

### ❌ Missing Implementation (35+ endpoints)

---

## 🔐 Authentication Endpoints

### Implemented
- ✅ `POST /auth/login-local` - Login with email/password

### Missing
- ❌ `POST /auth/register-local` - Register new user account

---

## 📱 Device Management

### Missing
- ❌ `POST /device/register` - Register device for push notifications

---

## 📊 Trading Endpoints

### Missing - Trades/Positions
- ❌ `GET /trades/logs` - Get detailed trade logs
- ❌ `GET /trades/open` - Get currently open positions
- ❌ `GET /trades/history` - Get complete trade history with filters
- ❌ `GET /trades/recent` - Get recent trades (last N)
- ❌ `POST /trades/close` - Manually close a position

### Missing - Performance Metrics
- ❌ `GET /performance/stats` - Get performance statistics (win rate, profit, etc.)
- ❌ `GET /performance/equity-curve` - Get equity curve data for charts
- ❌ `GET /performance/daily-summary` - Get daily P&L summaries

---

## 🤖 Bot Control Endpoints

### Missing
- ❌ `GET /bot/status` - Get current bot status (running/stopped/paused)
- ❌ `POST /bot/start` - Start the trading bot
- ❌ `POST /bot/stop` - Stop the trading bot
- ❌ `POST /bot/pause` - Pause the trading bot

---

## ⚙️ Strategy Management Endpoints

### Implemented
- ✅ `GET /strategies` - List all strategies
- ✅ `POST /strategies` - Upload new strategy

### Missing
- ❌ `GET /strategies/validated` - List validated strategies only
- ❌ `GET /strategies/active` - List active strategies only
- ❌ `PUT /strategies/{strategy_id}/activate` - Activate a strategy
- ❌ `PUT /strategies/{strategy_id}/deactivate` - Deactivate a strategy
- ❌ `PUT /strategies/{strategy_id}/promote` - Promote strategy from PAPER to LIVE
- ❌ `PATCH /strategies/{strategy_id}/policy` - Update strategy policy/settings
- ❌ `GET /strategy/config` - Get current strategy configuration
- ❌ `POST /strategy/config` - Update strategy configuration
- ❌ `GET /strategy/symbols` - Get tradeable symbols list

---

## 💳 MT5 Account Management

### Missing
- ❌ `GET /account/mt5` - List all MT5 accounts
- ❌ `GET /account/mt5/primary` - Get primary MT5 account
- ❌ `POST /account/mt5` - Add new MT5 account
- ❌ `PUT /account/mt5/{account_id}` - Update MT5 account
- ❌ `DELETE /account/mt5/{account_id}` - Delete MT5 account (soft delete)

---

## 🧪 Testing Endpoints

### Missing
- ❌ `POST /orders/test` - Test order placement without real execution

---

## 🏥 System Health Endpoints

### Missing
- ❌ `GET /health` - Health check endpoint
- ❌ `GET /system/status` - Detailed system status
- ❌ `GET /info` - API information and available endpoints
- ❌ `GET /` - Root endpoint with API metadata

---

## 🔌 WebSocket Endpoints

### Missing
- ❌ `WS /ws/updates/{token}` - Real-time bot status updates
- ❌ `WS /ws/trades/{token}` - Real-time trade updates
- ❌ `WS /ws/notifications` - Real-time notification stream

**Note:** WebSocketService exists but is not connected to these endpoints

---

## 📋 Implementation Priority

### 🔴 **Critical (Core Functionality)**
1. Bot Control
   - `GET /bot/status`
   - `POST /bot/start`
   - `POST /bot/stop`
   
2. Real-time Data
   - `WS /ws/updates/{token}`
   - `WS /ws/trades/{token}`
   
3. Trading Data
   - `GET /trades/open`
   - `GET /trades/history`
   - `GET /performance/stats`
   - `GET /performance/equity-curve`

### 🟡 **High Priority (Enhanced Features)**
4. Strategy Management
   - `GET /strategies/active`
   - `PUT /strategies/{id}/activate`
   - `PUT /strategies/{id}/deactivate`
   - `PUT /strategies/{id}/promote`
   
5. MT5 Account Management
   - `GET /account/mt5`
   - `GET /account/mt5/primary`
   - `POST /account/mt5`

### 🟢 **Medium Priority (Nice to Have)**
6. Additional Features
   - `POST /trades/close`
   - `GET /performance/daily-summary`
   - `POST /device/register`
   - `GET /strategy/symbols`

### ⚪ **Low Priority (Optional)**
7. Admin/Testing
   - `POST /orders/test`
   - `GET /health`
   - `GET /system/status`
   - `POST /auth/register-local`

---

## 🛠️ Recommended Implementation Plan

### Phase 1: Essential Trading Features (Week 1)
```swift
// New service: BotService.swift
- getBotStatus()
- startBot()
- stopBot()
- pauseBot()

// Extend APIService.swift
- getOpenPositions()
- getTradeHistory()
- getPerformanceStats()
- getEquityCurve()
```

### Phase 2: Real-time Updates (Week 1-2)
```swift
// Update WebSocketService.swift
- connectToUpdates(token: String)
- connectToTrades(token: String)
- Handle bot status updates
- Handle trade notifications
```

### Phase 3: Advanced Strategy Management (Week 2)
```swift
// New service: StrategyService.swift
- getActiveStrategies()
- getValidatedStrategies()
- activateStrategy(id: String)
- deactivateStrategy(id: String)
- promoteStrategy(id: String)
- updateStrategyPolicy(id: String, policy: Dict)
```

### Phase 4: MT5 Account Management (Week 3)
```swift
// New service: MT5AccountService.swift
- listAccounts()
- getPrimaryAccount()
- addAccount(accountData)
- updateAccount(id, data)
- deleteAccount(id)
```

---

## 📝 Models to Add

The following data models are missing and need to be created:

```swift
// Models/Position.swift
struct Position: Codable, Identifiable {
    let id: String
    let ticket: Int
    let symbol: String
    let type: String  // "BUY" or "SELL"
    let volume: Double
    let openPrice: Double
    let currentPrice: Double
    let profit: Double
    let openTime: Date
}

// Models/Trade.swift
struct Trade: Codable, Identifiable {
    let id: String
    let ticket: Int
    let symbol: String
    let type: String
    let volume: Double
    let openPrice: Double
    let closePrice: Double?
    let profit: Double
    let openTime: Date
    let closeTime: Date?
}

// Models/BotStatus.swift
struct BotStatus: Codable {
    let status: String  // "RUNNING", "STOPPED", "PAUSED"
    let uptime: Int
    let activeTrades: Int
    let todayProfit: Double
    let totalProfit: Double
}

// Models/PerformanceStats.swift
struct PerformanceStats: Codable {
    let totalTrades: Int
    let winningTrades: Int
    let losingTrades: Int
    let winRate: Double
    let totalProfit: Double
    let averageWin: Double
    let averageLoss: Double
    let profitFactor: Double
    let sharpeRatio: Double?
}

// Models/MT5Account.swift
struct MT5Account: Codable, Identifiable {
    let id: Int
    let accountNumber: String
    let server: String
    let broker: String
    let isPrimary: Bool
    let isActive: Bool
    let balance: Double?
    let equity: Double?
    let createdAt: Date
    let updatedAt: Date
}
```

---

## 🎯 Next Steps

1. **Review this document** with the team
2. **Prioritize endpoints** based on user needs
3. **Create implementation tickets** for each phase
4. **Update APIService.swift** to include new endpoints
5. **Create new service classes** (BotService, StrategyService, MT5AccountService)
6. **Add missing models** as listed above
7. **Update UI views** to use new data
8. **Test each endpoint** thoroughly
9. **Update documentation** as you implement

---

## 📚 Backend Documentation

For full API documentation, see:
- Backend main.py (main branch)
- Backend routers/strategies.py (main branch)
- FastAPI auto-generated docs at: `/docs` (when backend is running)
