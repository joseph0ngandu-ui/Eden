# Eden iOS Widgets Setup Guide

Complete guide for setting up and using Eden Trading Bot widgets on iOS.

## Overview

Eden now supports **Lock Screen** and **Home Screen** widgets that display real-time trading bot status, balance, profit/loss, and performance metrics directly on your iPhone.

### Widget Types

#### Lock Screen Widgets (iOS 16+)
1. **Circular Widget** - Shows bot active status and profit direction (↑/↓)
2. **Rectangular Widget** - Displays bot status, balance, and daily P&L
3. **Inline Widget** - Compact view with balance and profit in status bar area

#### Home Screen Widgets
1. **Small Widget** - Bot status, balance, and daily P&L
2. **Medium Widget** - Balance, daily P&L, win rate, positions, profit factor, and risk tier
3. **Large Widget** - Full dashboard with comprehensive metrics and performance data

## Features

- ✅ Real-time bot status (Active/Stopped)
- ✅ Current balance display
- ✅ Daily profit/loss with color indicators
- ✅ Win rate and profit factor
- ✅ Active positions count
- ✅ Total trades counter
- ✅ Risk tier display
- ✅ Last update timestamp
- ✅ Automatic updates every 5 minutes
- ✅ Dark mode optimized design

## Setup Instructions

### 1. Add Widget Extension to Xcode Project

1. Open `Eden.xcodeproj` in Xcode
2. Go to **File → New → Target**
3. Select **Widget Extension**
4. Name it: `EdenWidget`
5. Choose **Include Configuration Intent**: No
6. Click **Finish**
7. When prompted to activate scheme, click **Activate**

### 2. Configure App Groups (Required)

App Groups enable data sharing between the main app and widgets.

#### For Main App Target:
1. Select the **Eden** target
2. Go to **Signing & Capabilities** tab
3. Click **+ Capability**
4. Add **App Groups**
5. Click **+** and add: `group.com.eden.trading`
6. Check the box to enable it

#### For Widget Target:
1. Select the **EdenWidget** target
2. Go to **Signing & Capabilities** tab
3. Click **+ Capability**
4. Add **App Groups**
5. Click **+** and add: `group.com.eden.trading` (same group)
6. Check the box to enable it

⚠️ **Important**: Both targets must use the **exact same** App Group identifier.

### 3. Add Files to Widget Target

1. In Xcode, select `EdenWidget.swift` (already created in `ios/EdenWidget/`)
2. In the **Target Membership** panel (right side), ensure **EdenWidget** is checked
3. Add `SharedDataService.swift` to both **Eden** and **EdenWidget** targets
4. Add `Models.swift` to both targets (for shared data structures)

### 4. Update Info.plist for Widget

The widget extension needs proper configuration:

1. Select **EdenWidget** target
2. Go to **Info** tab
3. Add the following keys if not present:
   - `NSExtension` → `NSExtensionPointIdentifier` = `com.apple.widgetkit-extension`
   - `CFBundleDisplayName` = `Eden Widget`

### 5. Build and Run

1. Select the **Eden** scheme (not EdenWidget)
2. Build and run on a physical device or simulator (iOS 16+)
3. Once the app runs, the widgets will be available

### 6. Add Widgets to iPhone

#### Home Screen Widgets:
1. Long-press on home screen
2. Tap **+** button in top-left corner
3. Search for **Eden**
4. Choose widget size (Small, Medium, or Large)
5. Tap **Add Widget**
6. Position and tap **Done**

#### Lock Screen Widgets:
1. Lock your iPhone
2. Long-press on lock screen
3. Tap **Customize**
4. Tap the widget area (below the time)
5. Scroll to find **Eden Bot Lock Screen**
6. Choose widget style (Circular, Rectangular, or Inline)
7. Tap outside to save
8. Tap **Done**

## Widget Update Frequency

- Widgets update automatically every **5 minutes**
- Widget data is refreshed when the main app updates bot status
- Manual refresh: Force-touch widget → select "Reload"
- iOS system may throttle updates to preserve battery

## Customization

### Changing Update Interval

In `EdenWidget.swift`, modify the timeline policy:

```swift
// Update every 5 minutes (default)
let nextUpdate = Calendar.current.date(byAdding: .minute, value: 5, to: currentDate)!

// Update every 1 minute (more frequent)
let nextUpdate = Calendar.current.date(byAdding: .minute, value: 1, to: currentDate)!

// Update every 15 minutes (less frequent, better battery)
let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: currentDate)!
```

### Changing App Group Identifier

If you need to use a different App Group:

1. Update the identifier in both targets' capabilities
2. Update `SharedDataService.swift`:
   ```swift
   private let appGroupIdentifier = "group.your.new.identifier"
   ```

## Troubleshooting

### Widget Shows "No Data"

**Causes:**
- App Groups not configured correctly
- App hasn't run yet to populate data
- App Group identifier mismatch

**Solutions:**
1. Verify both targets use the same App Group identifier
2. Run the main app at least once
3. Check that `SharedDataService.swift` is included in both targets
4. Rebuild the project

### Widget Not Appearing in Widget Gallery

**Solutions:**
1. Clean build folder: **Product → Clean Build Folder**
2. Rebuild the project
3. Restart Xcode
4. Delete app from device and reinstall

### Widget Not Updating

**Solutions:**
1. Check that the main app is running and updating data
2. Force refresh the widget (long-press → reload)
3. Verify timeline policy in `EdenWidget.swift`
4. Check for iOS background app refresh settings

### Build Errors

**Common Issues:**
- Missing imports: Add `import WidgetKit` and `import SwiftUI`
- Missing targets: Ensure all required files are in both targets
- App Group errors: Verify App Groups capability is added

## Technical Details

### Data Flow

```
Main App (BotManager)
    ↓ (updates every 3s)
    ↓
SharedDataService
    ↓ (saves to App Group UserDefaults)
    ↓
Widget Extension
    ↓ (reads from UserDefaults)
    ↓
Widget UI (updates every 5m)
```

### File Structure

```
ios/
├── Eden/
│   └── Eden/
│       └── Eden/
│           ├── Services/
│           │   ├── BotManager.swift (updated)
│           │   └── SharedDataService.swift (new)
│           └── Models.swift
└── EdenWidget/
    └── EdenWidget.swift (new)
```

### Shared Data Model

```swift
struct WidgetData: Codable {
    let isRunning: Bool
    let balance: Double
    let dailyPnL: Double
    let activePositions: Int
    let winRate: Double
    let totalTrades: Int
    let profitFactor: Double
    let riskTier: String
    let lastUpdate: Date
}
```

## Widget Preview Screenshots

### Home Screen Widgets

**Small Widget:**
- Bot status indicator (green/red circle)
- Current balance
- Daily P&L with arrow indicator
- Last update time

**Medium Widget:**
- Everything from small widget
- Win rate percentage
- Active positions count
- Profit factor
- Risk tier

**Large Widget:**
- Full bot dashboard
- Balance with daily P&L
- Win rate, profit factor
- Active positions, total trades
- Risk tier and profit status
- Last update timestamp

### Lock Screen Widgets

**Circular:**
- Bot status icon (checkmark/X)
- Profit direction (↑/↓)

**Rectangular:**
- Bot status with text
- Balance and daily P&L

**Inline:**
- Compact: Status icon + Balance + P&L

## Performance Considerations

- Widgets use minimal battery (updates every 5 minutes)
- Data is cached in App Group UserDefaults
- No network requests in widget (data from main app only)
- Widgets automatically suspended when not visible
- Background app refresh should be enabled for best experience

## Security Notes

- Widget data is stored locally in App Group container
- No sensitive data displayed (only metrics)
- Widgets follow iOS sandbox security model
- App Group data is isolated from other apps

## Future Enhancements

Planned features for future versions:
- Interactive widgets (iOS 17+) for bot control
- Multiple account support in widgets
- Equity curve chart in large widget
- Customizable refresh intervals per widget
- Widget configuration options
- Live Activities support (iOS 16.1+)

## Support

If you encounter issues:
1. Check this troubleshooting guide first
2. Verify all setup steps are completed
3. Check Xcode console for error messages
4. Rebuild the project with a clean build

## Additional Resources

- [Apple WidgetKit Documentation](https://developer.apple.com/documentation/widgetkit)
- [App Groups Guide](https://developer.apple.com/documentation/xcode/configuring-app-groups)
- [Lock Screen Widgets](https://developer.apple.com/documentation/widgetkit/creating-lock-screen-widgets-and-watch-complications)

---

**Version:** 1.0  
**Last Updated:** 2025  
**iOS Requirements:** iOS 16.0 or later  
**Xcode Requirements:** Xcode 14.0 or later
