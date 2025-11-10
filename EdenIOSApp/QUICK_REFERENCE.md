# 🚀 Eden iOS App - Quick Reference Card

## 📍 Location
```
/Users/josephngandu/Desktop/Eden/EdenIOSApp/
```

---

## 📂 What's Inside

### 📱 Source Code (23 files)
```
Eden/
├── EdenApp.swift              # Entry point
├── Views/ (6 files)           # All screens
├── Components/ (8 files)      # UI components
├── Models/ (1 file)           # Data models
└── Services/ (4 files)        # API & state
```

### 📖 Documentation (5 files)
```
├── PACKAGE_SUMMARY.md         # Overview
├── README.md                  # Full docs
├── VISUAL_GUIDE.md            # Step-by-step
├── SETUP_CHECKLIST.md         # Printable checklist
└── SETUP.sh                   # Quick setup script
```

---

## ⚡️ Quick Start (5 steps)

1. **Open Xcode**
   ```bash
   open /Applications/Xcode.app
   ```

2. **Create Project**
   - File → New → Project
   - App → SwiftUI → Name: "Eden"

3. **Add Files**
   - Drag `Eden` folder into Xcode
   - Copy items ✓

4. **Configure**
   - Update Info.plist (see VISUAL_GUIDE)
   - Update APIService.swift URLs

5. **Run**
   - Select iPhone 15 Pro simulator
   - Press ⌘R

---

## 🎯 Key Files to Know

### Must Configure:
```swift
Services/APIService.swift
  Line 14: baseURL = "your-n8n-url"
  Line 15: apiKey = "your-api-key"
```

### Main Entry:
```swift
EdenApp.swift
  → ContentView → All screens
```

### State Management:
```swift
Services/BotManager.swift
  → All app state & data
```

---

## 🔑 Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run | ⌘R |
| Stop | ⌘. |
| Clean Build | ⇧⌘K |
| Build | ⌘B |
| Console | ⌘⇧Y |
| Navigator | ⌘0 |
| Preview | ⌥⌘↩ |

---

## 📱 Test Devices

Recommended simulators:
- iPhone 15 Pro (best)
- iPhone 15 Pro Max
- iPhone 14 Pro

Minimum: iOS 17.0

---

## 🎨 Design System

### Colors
- Background: Black (#000)
- Primary: Purple → Blue gradient
- Success: Green (#10B981)
- Error: Red (#EF4444)

### Spacing
- Small: 8-12pt
- Medium: 16-20pt
- Large: 24-32pt

### Corner Radius
- Cards: 20-24pt
- Buttons: 12-16pt
- Small elements: 8-12pt

---

## 🔌 API Endpoints Needed

```
POST /webhook/eden-webhook     # Receive events
GET  /webhook/eden-status       # Get bot status
POST /webhook/eden-control      # Control bot
GET  /webhook/eden-positions    # Get positions
GET  /webhook/eden-trades       # Get trades
```

---

## 📋 Files Checklist

Core (4):
- [x] EdenApp.swift
- [x] ContentView.swift
- [x] BotManager.swift
- [x] Models.swift

Views (6):
- [x] HeaderView.swift
- [x] OverviewView.swift
- [x] PositionsView.swift
- [x] AnalyticsView.swift
- [x] SettingsView.swift

Components (8):
- [x] StatCard.swift
- [x] EquityCurveView.swift
- [x] RecentTradesView.swift
- [x] TradeRow.swift
- [x] PositionCard.swift
- [x] MetricRow.swift
- [x] SettingField.swift
- [x] CustomTabBar.swift

Services (4):
- [x] BotManager.swift
- [x] APIService.swift
- [x] WebSocketService.swift
- [x] NotificationManager.swift

---

## 🐛 Quick Fixes

**Build fails:**
```bash
⇧⌘K # Clean build folder
```

**Simulator won't start:**
```
Window → Devices → Reset Simulator
```

**Code not updating:**
```bash
⌘. # Stop
⌘B # Build
⌘R # Run
```

---

## 📖 Documentation Order

1. **PACKAGE_SUMMARY.md** - Start here
2. **VISUAL_GUIDE.md** - Follow step-by-step
3. **SETUP_CHECKLIST.md** - Check off items
4. **README.md** - Full reference

---

## ✨ Features

- ✅ 4 main screens
- ✅ Real-time updates (3s)
- ✅ Animated charts
- ✅ Bot controls
- ✅ 23 Swift files
- ✅ Full docs
- ✅ Mock data ready
- ✅ API integration ready

---

## 🎯 Success = ✅

- App opens in simulator
- Black background + gradients
- All tabs work
- Data updates
- No errors in console

---

## 📞 Help

1. Check console: ⌘⇧Y
2. Read error message
3. Check VISUAL_GUIDE.md
4. Google error + "SwiftUI"
5. Clean & rebuild

---

## 💡 Pro Tip

**Start simple:**
1. Get it running with mock data
2. Test all UI features
3. Then connect real APIs

---

**Version:** 1.0
**iOS:** 17.0+
**Xcode:** 15+
**Files:** 28 total

---

🚀 **Ready to build? Open Xcode now!**

```bash
cd ~/Desktop/Eden/EdenIOSApp
./SETUP.sh
```
