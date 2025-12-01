# macOS App Implementation Roadmap

## 🎯 Goal
Transform the Aurora For Mac app from a basic **strategy uploader** into a **full-featured trading management application** with real-time monitoring, bot control, and comprehensive analytics.

---

## 📊 Current vs Target State

### Current Implementation (12% Complete)
```
✅ Authentication (Login only)
✅ Basic Strategy Upload
✅ ML Training Trigger
⬜ Bot Control
⬜ Real-time Monitoring
⬜ Trade Management
⬜ Performance Analytics
⬜ MT5 Account Management
⬜ Live Updates (WebSocket)
```

### Target Implementation (100% Complete)
```
✅ Full Authentication (Login + Register)
✅ Strategy Management (CRUD + Lifecycle)
✅ ML Training & Status
✅ Bot Control (Start/Stop/Pause)
✅ Real-time Monitoring (WebSocket)
✅ Trade Management (View/Close trades)
✅ Performance Analytics (Stats + Charts)
✅ MT5 Account Management
✅ Live Updates (WebSocket)
✅ Push Notifications
```

---

## 🗓️ Sprint-Based Implementation Plan

### 🔴 Sprint 1: Core Trading Features (Priority: CRITICAL)
**Duration:** 1 week  
**Goal:** Enable basic bot monitoring and control

#### Services to Create/Update
1. **Create `BotService.swift`**
   ```swift
   - getBotStatus() -> BotStatus
   - startBot() async throws
   - stopBot() async throws
   - pauseBot() async throws
   ```

2. **Create `TradeService.swift`**
   ```swift
   - getOpenPositions() -> [Position]
   - getTradeHistory(limit: Int?) -> [Trade]
   - getRecentTrades(limit: Int) -> [Trade]
   - closeTrade(ticket: Int) async throws
   ```

3. **Create `PerformanceService.swift`**
   ```swift
   - getStats() -> PerformanceStats
   - getEquityCurve() -> [EquityPoint]
   - getDailySummary() -> DailySummary
   ```

#### Models to Create
```swift
// Models/BotStatus.swift
// Models/Position.swift
// Models/Trade.swift
// Models/PerformanceStats.swift
// Models/EquityPoint.swift
```

#### Views to Create
```swift
// Views/BotControlView.swift - Start/stop/pause bot
// Views/PositionsView.swift - List open positions
// Views/TradeHistoryView.swift - Show trade history
// Views/PerformanceView.swift - Display stats & charts
```

#### Deliverables
- ✅ Bot can be started/stopped from Mac app
- ✅ Real-time bot status display
- ✅ Open positions visible in UI
- ✅ Trade history accessible
- ✅ Performance metrics displayed

**Estimated Effort:** 25-30 hours

---

### 🟡 Sprint 2: Real-time Updates (Priority: HIGH)
**Duration:** 1 week  
**Goal:** Add live data streaming via WebSocket

#### Services to Update
1. **Update `WebSocketService.swift`**
   ```swift
   - connectToUpdates(token: String)
   - connectToTrades(token: String)
   - connectToNotifications()
   - handleMessage(_ data: Data)
   - subscribeToUpdates(handler: (BotStatus) -> Void)
   - subscribeToTrades(handler: (Trade) -> Void)
   ```

#### Views to Update
```swift
// Update ContentView.swift - Add live status indicator
// Update PositionsView.swift - Auto-refresh on updates
// Update TradeHistoryView.swift - Show new trades in real-time
```

#### Deliverables
- ✅ WebSocket connects on app launch
- ✅ Bot status updates automatically
- ✅ New trades appear instantly
- ✅ Position changes reflect immediately
- ✅ Notification badges for events

**Estimated Effort:** 20-25 hours

---

### 🟡 Sprint 3: Advanced Strategy Management (Priority: HIGH)
**Duration:** 1 week  
**Goal:** Full strategy lifecycle management

#### Services to Create/Update
1. **Create `StrategyService.swift`**
   ```swift
   - getAllStrategies() -> [StrategyItem]
   - getActiveStrategies() -> [StrategyItem]
   - getValidatedStrategies() -> [StrategyItem]
   - activateStrategy(id: String) async throws
   - deactivateStrategy(id: String) async throws
   - promoteStrategy(id: String) async throws // PAPER -> LIVE
   - updatePolicy(id: String, policy: [String: Any]) async throws
   ```

2. **Update `APIService.swift`**
   ```swift
   - getStrategyConfig() -> StrategyConfig
   - updateStrategyConfig(_ config: StrategyConfig)
   - getTradableSymbols() -> [String]
   ```

#### Models to Create
```swift
// Models/StrategyItem.swift (update existing)
// Models/StrategyConfig.swift
// Models/StrategyPolicy.swift
```

#### Views to Create/Update
```swift
// Update StrategyListView.swift
  - Show strategy status (PAPER/LIVE/INACTIVE)
  - Add activate/deactivate buttons
  - Add promote to LIVE button with confirmation
  
// Create StrategyDetailView.swift
  - View full strategy details
  - Edit strategy policy
  - View strategy performance
  - Manage strategy lifecycle
```

#### Deliverables
- ✅ View all strategies with status
- ✅ Activate/deactivate strategies
- ✅ Promote strategies from PAPER to LIVE
- ✅ Edit strategy policies
- ✅ Filter strategies by status

**Estimated Effort:** 20-25 hours

---

### 🟢 Sprint 4: MT5 Account Management (Priority: MEDIUM)
**Duration:** 1 week  
**Goal:** Manage multiple MT5 trading accounts

#### Services to Create
1. **Create `MT5AccountService.swift`**
   ```swift
   - getAllAccounts() -> [MT5Account]
   - getPrimaryAccount() -> MT5Account?
   - addAccount(_ account: MT5AccountCreate) async throws
   - updateAccount(id: Int, data: MT5AccountUpdate) async throws
   - deleteAccount(id: Int) async throws
   - setPrimaryAccount(id: Int) async throws
   ```

#### Models to Create
```swift
// Models/MT5Account.swift
// Models/MT5AccountCreate.swift
// Models/MT5AccountUpdate.swift
```

#### Views to Create
```swift
// Views/MT5AccountsView.swift - List all accounts
// Views/AddMT5AccountView.swift - Add new account form
// Views/EditMT5AccountView.swift - Edit account details
```

#### Deliverables
- ✅ List all MT5 accounts
- ✅ Add new MT5 accounts with credentials
- ✅ Edit existing accounts
- ✅ Set primary account
- ✅ Delete (deactivate) accounts

**Estimated Effort:** 15-20 hours

---

### 🟢 Sprint 5: Enhanced Features (Priority: MEDIUM)
**Duration:** 1 week  
**Goal:** Polish and additional features

#### Features to Add
1. **Device Registration**
   ```swift
   - registerDevice(token: String, deviceInfo: DeviceInfo)
   ```

2. **Manual Trade Closing**
   ```swift
   - Swipe to close position
   - Confirmation dialog
   - Success/error feedback
   ```

3. **Symbol Management**
   ```swift
   - View available symbols
   - Enable/disable symbols for trading
   ```

4. **Testing Features**
   ```swift
   - Test order placement (paper trading)
   - Validate strategies before activation
   ```

#### Views to Create/Update
```swift
// Update PositionsView.swift - Add swipe to close
// Create SymbolsView.swift - Manage tradable symbols
// Create TestOrderView.swift - Test order placement
```

#### Deliverables
- ✅ Close positions from app
- ✅ Manage tradable symbols
- ✅ Test orders without risk
- ✅ Device push notifications

**Estimated Effort:** 15-20 hours

---

### ⚪ Sprint 6: Polish & Deployment (Priority: LOW)
**Duration:** 1 week  
**Goal:** Prepare for production deployment

#### Tasks
1. **Code Signing & Notarization**
   - Set up Apple Developer account
   - Configure code signing
   - Notarize the app

2. **App Icon & Branding**
   - Design app icon
   - Add to all required sizes
   - Update branding colors

3. **Error Handling**
   - Comprehensive error messages
   - Retry logic
   - Offline mode handling

4. **Settings & Configuration**
   - Environment selection (dev/staging/prod)
   - Custom backend URL
   - Notification preferences
   - Dark mode support

5. **Documentation**
   - User guide
   - Developer documentation
   - API integration guide

#### Views to Create
```swift
// Update SettingsView.swift
  - Backend URL configuration
  - Environment selection
  - Notification settings
  - About/version info
  - Debug logging toggle
```

#### Deliverables
- ✅ Production-ready build
- ✅ Code signed and notarized
- ✅ Professional app icon
- ✅ Comprehensive settings
- ✅ User documentation

**Estimated Effort:** 20-25 hours

---

## 📈 Progress Tracking

### Completion Metrics

| Sprint | Focus Area | Endpoints | Est. Hours | Status |
|--------|-----------|-----------|------------|--------|
| 1 | Core Trading | 9 endpoints | 25-30 | ⬜ Not Started |
| 2 | Real-time | 3 WebSockets | 20-25 | ⬜ Not Started |
| 3 | Strategies | 8 endpoints | 20-25 | ⬜ Not Started |
| 4 | MT5 Accounts | 5 endpoints | 15-20 | ⬜ Not Started |
| 5 | Enhanced | 4 endpoints | 15-20 | ⬜ Not Started |
| 6 | Polish | 0 endpoints | 20-25 | ⬜ Not Started |
| **Total** | **All Features** | **29+ endpoints** | **115-145 hrs** | **12% Complete** |

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- ✅ Sprint 1 complete (Core Trading)
- ✅ Sprint 2 complete (Real-time Updates)
- ⚠️ Basic code signing

### Full Feature Release (v1.0)
- ✅ All sprints complete
- ✅ Production deployment
- ✅ User documentation
- ✅ App Store ready (if desired)

---

## 🚀 Getting Started

### Immediate Next Steps

1. **Review Priority**
   - Confirm Sprint 1 is the right starting point
   - Adjust priorities based on user needs

2. **Set Up Development Environment**
   - Ensure Xcode is up to date
   - Configure backend URL
   - Test current endpoints

3. **Create Branch**
   ```bash
   git checkout -b feature/sprint-1-core-trading
   ```

4. **Start Implementation**
   - Begin with BotService.swift
   - Follow the sprint 1 plan
   - Test each endpoint as you implement

5. **Track Progress**
   - Update this document as you complete tasks
   - Mark deliverables as done
   - Note any blockers or changes

---

## 📞 Questions or Issues?

If you encounter issues during implementation:
1. Check the backend API docs at `/docs`
2. Verify endpoint paths in `backend/main.py`
3. Test endpoints with curl or Postman first
4. Review existing Swift implementations for patterns

---

## 📚 Additional Resources

- **Backend Endpoints:** See `MISSING_ENDPOINTS.md`
- **Current Implementation:** See `README.md`
- **Architecture:** See project structure in repo
- **iOS App Reference:** Check `ios/` directory for similar implementations
