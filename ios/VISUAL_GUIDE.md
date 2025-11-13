# Eden iOS App - Visual Setup Guide

## 🎯 Step-by-Step Setup

### Step 1: Open Xcode

```
Applications → Xcode.app
```

Or use Terminal:
```bash
open /Applications/Xcode.app
```

---

### Step 2: Create New Project

**File → New → Project (⇧⌘N)**

1. Choose template: **iOS → App**
2. Click **Next**

---

### Step 3: Configure Project

Fill in these details:

```
Product Name:        Eden
Team:                Your Apple ID
Organization ID:     com.yourname
Bundle Identifier:   com.yourname.eden (auto-generated)
Interface:           SwiftUI ✓
Language:            Swift ✓
Storage:             None
Include Tests:       ☐ (optional)
```

Click **Next**

---

### Step 4: Choose Save Location

**IMPORTANT:** Navigate to:
```
Desktop/Eden/EdenIOSApp/
```

Click **Create**

---

### Step 5: Add Source Files

**Method 1 - Drag & Drop:**

1. Open Finder
2. Navigate to: `Desktop/Eden/EdenIOSApp/Eden/`
3. Drag the **entire Eden folder** into Xcode's left sidebar
4. In the dialog:
   - ✓ Copy items if needed
   - ✓ Create groups
   - ✓ Add to target: Eden
5. Click **Finish**

**Method 2 - File Menu:**

1. File → Add Files to "Eden"
2. Select the `Eden` folder
3. Same options as above

---

### Step 6: Verify File Structure

Your Xcode sidebar should show:

```
Eden (Blue folder icon)
├── 📱 EdenApp.swift
├── 📁 Views/
│   ├── ContentView.swift
│   ├── HeaderView.swift
│   ├── OverviewView.swift
│   ├── PositionsView.swift
│   ├── AnalyticsView.swift
│   └── SettingsView.swift
├── 📁 Components/
│   ├── StatCard.swift
│   ├── EquityCurveView.swift
│   ├── RecentTradesView.swift
│   ├── TradeRow.swift
│   ├── PositionCard.swift
│   ├── MetricRow.swift
│   ├── SettingField.swift
│   └── CustomTabBar.swift
├── 📁 Models/
│   └── Models.swift
├── 📁 Services/
│   ├── BotManager.swift
│   ├── APIService.swift
│   ├── WebSocketService.swift
│   └── NotificationManager.swift
└── 📁 Assets.xcassets
```

---

### Step 7: Configure Info.plist

1. Click on **Info.plist** in the sidebar
2. Right-click → **Open As → Source Code**
3. Add this inside the `<dict>` tag:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>

<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>remote-notification</string>
</array>

<key>Privacy - Notifications Usage Description</key>
<string>Eden needs notifications to alert you about trades</string>
```

---

### Step 8: Update API Configuration

1. Open **Services/APIService.swift**
2. Find line 14:
   ```swift
   private let baseURL = "https://your-n8n-instance.com/webhook"
   ```
3. Replace with your actual n8n webhook URL
4. Find line 15:
   ```swift
   private let apiKey = "YOUR_API_KEY_HERE"
   ```
5. Replace with your actual API key

---

### Step 9: Select Target Device

Top toolbar → Select target:
- **iPhone 15 Pro** (simulator) - recommended for testing
- Or your physical iPhone (requires Apple Developer account)

---

### Step 10: Run the App

**Press ⌘R** or click the **▶ Play** button

The app will:
1. Build (15-30 seconds first time)
2. Launch simulator
3. Open Eden app with mock data
4. Show live updates every 3 seconds

---

## ✅ Success Checklist

After running, you should see:

- ✓ Black background with purple/blue gradients
- ✓ "Eden" logo with bolt icon
- ✓ Balance showing ~$347.82
- ✓ Green "Active" button
- ✓ Four stat cards (Win Rate, Risk Tier, etc.)
- ✓ Animated equity curve chart
- ✓ Recent trades list
- ✓ Bottom tab bar with 4 tabs

---

## 🎨 Testing the UI

### Try These:

1. **Toggle Bot Status**
   - Tap "Active" button → Should turn red "Paused"
   - Tap again → Back to green "Active"

2. **Hide Balance**
   - Tap eye icon next to balance
   - Should show "••••••"

3. **Switch Tabs**
   - Tap "Positions" → See active trades
   - Tap "Analytics" → See performance metrics
   - Tap "Settings" → See configuration

4. **Watch Real-time Updates**
   - Balance changes every 3 seconds
   - Equity curve updates
   - Position values change

---

## 🔧 Common Issues

### Build Fails

**Error: "No such module 'SwiftUI'"**
- Solution: Select iOS 17+ as deployment target

**Error: "Ambiguous use of..."**
- Solution: Clean build folder (⇧⌘K)
- Restart Xcode

### Simulator Issues

**Simulator won't launch**
- Solution: Xcode → Window → Devices and Simulators
- Delete old simulators
- Create new iPhone 15 Pro simulator

**App crashes on launch**
- Check console for error messages (⌘0)
- Look for red error logs

### UI Not Showing

**Black screen only**
- Check if EdenApp.swift is in target
- Verify ContentView.swift exists
- Check console for SwiftUI errors

---

## 🚀 Next: Connect Real Data

Once the app works with mock data:

1. **Test n8n webhook** (Postman/curl)
2. **Add webhook to Eden bot** (webhook_notifier.py)
3. **Update bot to send events** (trading_bot.py integration)
4. **Enable real API calls** (uncomment in BotManager)
5. **Test end-to-end** (Eden bot → n8n → iOS app)

---

## 📱 Preview on Device

### Requirements:
- Apple Developer account ($99/year)
- Physical iPhone
- Lightning/USB-C cable

### Steps:
1. Connect iPhone via cable
2. Select your iPhone as target
3. Xcode → Signing & Capabilities
4. Select your team
5. Trust certificate on iPhone (Settings → General → VPN & Device Management)
6. Run (⌘R)

---

## 🎉 You're Done!

Your Eden iOS app is now ready for:
- ✅ Monitoring your trading bot
- ✅ Real-time position tracking
- ✅ Performance analytics
- ✅ Bot control (start/stop)

**Enjoy your bleeding-edge trading dashboard!** 📈⚡️
