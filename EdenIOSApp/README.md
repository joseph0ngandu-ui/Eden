# Eden iOS App

A bleeding-edge iOS app for monitoring and controlling your Eden AI trading bot.

## 🚀 Quick Start

### Prerequisites
- macOS with Xcode 15+ installed
- iOS 17+ device or simulator
- Apple Developer account (for device deployment)

### Setup Instructions

1. **Open Xcode**
   ```bash
   open /Applications/Xcode.app
   ```

2. **Create New Project**
   - File → New → Project
   - Choose **App** template
   - Product Name: `Eden`
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Organization Identifier: `com.yourname.eden`
   - Save in: `Desktop/Eden/EdenIOSApp`

3. **Add All Files**
   - Drag the `Eden` folder from Finder into your Xcode project
   - Make sure "Copy items if needed" is checked
   - Select "Create groups"

4. **Configure Info.plist**
   Add these keys (Right-click Info.plist → Open As → Source Code):
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
   ```

5. **Update API Endpoints**
   - Open `Services/APIService.swift`
   - Replace `baseURL` with your n8n webhook URL
   - Replace `apiKey` with your API key

6. **Run the App**
   - Select a simulator or connected device
   - Press ⌘R or click the Play button
   - App will launch with mock data

## 📁 Project Structure

```
Eden/
├── EdenApp.swift                 # Main app entry point
├── Views/
│   ├── ContentView.swift         # Main container
│   ├── HeaderView.swift          # Header with balance
│   ├── OverviewView.swift        # Dashboard screen
│   ├── PositionsView.swift       # Active positions
│   ├── AnalyticsView.swift       # Performance metrics
│   └── SettingsView.swift        # Configuration
├── Components/
│   ├── StatCard.swift            # Stat display card
│   ├── EquityCurveView.swift     # Chart component
│   ├── RecentTradesView.swift    # Trades list
│   ├── TradeRow.swift            # Trade row item
│   ├── PositionCard.swift        # Position display
│   ├── MetricRow.swift           # Metric display
│   ├── SettingField.swift        # Settings input
│   └── CustomTabBar.swift        # Bottom navigation
├── Models/
│   └── Models.swift              # Data models
└── Services/
    ├── BotManager.swift          # State management
    ├── APIService.swift          # REST API calls
    ├── WebSocketService.swift    # Real-time updates
    └── NotificationManager.swift # Push notifications
```

## 🔌 Eden Bot Integration

### 1. Add Webhook to Eden

Create `src/webhook_notifier.py` in your Eden bot:

```python
import requests
from datetime import datetime

class WebhookNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_trade_opened(self, symbol, entry_price, direction, confidence):
        payload = {
            "event": "trade_opened",
            "symbol": symbol,
            "entry_price": entry_price,
            "direction": direction,
            "confidence_level": confidence,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        requests.post(self.webhook_url, json=payload, timeout=5)
```

### 2. Integrate with trading_bot.py

```python
from webhook_notifier import WebhookNotifier

# In __init__
self.webhook = WebhookNotifier("https://your-n8n.com/webhook/eden-webhook")

# In place_order
if order_placed:
    self.webhook.send_trade_opened(symbol, entry_price, direction, confidence)
```

## 🎨 Features

- ✅ Real-time balance monitoring
- ✅ Active position tracking
- ✅ Recent trades feed
- ✅ Performance analytics
- ✅ Bot start/stop controls
- ✅ Balance privacy toggle
- ✅ Animated equity curve
- ✅ Bleeding-edge UI design
- ✅ Dark mode only
- ✅ Smooth animations

## 🔐 Security Setup

1. **Store API Key Securely** (Recommended)
   ```swift
   // Use Keychain for production
   import Security
   
   // Save to Keychain
   func saveAPIKey(_ key: String) {
       let data = key.data(using: .utf8)!
       let query = [
           kSecClass: kSecClassGenericPassword,
           kSecAttrAccount: "eden_api_key",
           kSecValueData: data
       ] as CFDictionary
       SecItemAdd(query, nil)
   }
   ```

2. **Enable HTTPS Only**
   - Remove `NSAllowsArbitraryLoads` in production
   - Use SSL certificates for all endpoints

## 🧪 Testing

### Run with Mock Data
- App runs with simulated data by default
- Data updates every 3 seconds
- Perfect for UI testing

### Connect to Real Bot
1. Update `APIService.swift` with your endpoints
2. Uncomment `fetchBotStatus()` in `BotManager.init()`
3. Run Eden bot with webhook integration
4. Test API responses

## 📲 Push Notifications Setup

1. **Apple Developer Portal**
   - Enable Push Notifications for App ID
   - Create APNs key
   - Download .p8 key file

2. **Xcode Configuration**
   - Select project → Signing & Capabilities
   - Add "Push Notifications" capability
   - Add "Background Modes" → Check "Remote notifications"

3. **Backend Setup**
   - Configure n8n to send push notifications
   - Use APNs HTTP/2 API
   - Send device token from app to server

## 🚀 Deployment

### TestFlight (Beta Testing)
1. Select "Any iOS Device (arm64)" as target
2. Product → Archive
3. Distribute App → TestFlight
4. Upload to App Store Connect
5. Add internal testers

### App Store Release
1. Complete App Store listing
2. Add screenshots (use simulator)
3. Submit for review
4. Publish when approved

## 🎯 Next Steps

- [ ] Replace mock data with real API calls
- [ ] Add Face ID authentication
- [ ] Create home screen widget
- [ ] Build Apple Watch companion app
- [ ] Add Siri shortcuts
- [ ] Implement chart library (Charts framework)
- [ ] Add more trade analytics
- [ ] Create onboarding flow

## 🐛 Troubleshooting

**App won't build:**
- Clean build folder: Shift+⌘K
- Delete derived data
- Restart Xcode

**API not connecting:**
- Check network permissions
- Verify webhook URL is correct
- Test endpoint with Postman first

**Simulator issues:**
- Reset simulator: Device → Erase All Content and Settings
- Try different simulator version

## 📝 Notes

- App uses mock data by default for testing
- Real-time updates simulate every 3 seconds
- All colors and gradients are customizable
- Dark mode only (no light mode)
- Minimum iOS version: 17.0

## 💡 Tips

- Use `@AppStorage` for user preferences
- Enable "Debug View Hierarchy" to inspect UI
- Use Instruments for performance profiling
- Test on physical device for accurate performance
- Keep Xcode and iOS updated

## 📞 Support

For issues or questions:
1. Check Xcode console for errors
2. Verify all files are included in target
3. Test with simulator first
4. Check API service logs

---

**Built with SwiftUI** | **Minimum iOS 17** | **Dark Mode Only**
