# ✅ Eden iOS App - Setup Checklist

Print this and check off as you go!

---

## 📋 Pre-Setup

- [ ] macOS computer with Xcode installed
- [ ] Xcode version 15 or newer
- [ ] iOS 17+ simulator or device
- [ ] Eden iOS App files at: `Desktop/Eden/EdenIOSApp/`
- [ ] Read PACKAGE_SUMMARY.md
- [ ] Have n8n webhook URL ready (optional for now)
- [ ] Have API key ready (optional for now)

---

## 🔧 Xcode Setup (15 minutes)

### Create Project
- [ ] Open Xcode
- [ ] File → New → Project
- [ ] Select "App" template
- [ ] Click Next
- [ ] Product Name: **Eden**
- [ ] Interface: **SwiftUI** ✓
- [ ] Language: **Swift** ✓
- [ ] Click Next
- [ ] Save location: `Desktop/Eden/EdenIOSApp/`
- [ ] Click Create

### Add Source Files
- [ ] Open Finder
- [ ] Navigate to `Desktop/Eden/EdenIOSApp/Eden/`
- [ ] Drag **Eden folder** into Xcode sidebar
- [ ] Check "Copy items if needed" ✓
- [ ] Check "Create groups" ✓
- [ ] Click Finish
- [ ] Verify all 23 Swift files are visible in Xcode

### Configure Info.plist
- [ ] Click Info.plist in sidebar
- [ ] Right-click → Open As → Source Code
- [ ] Add network permissions (see VISUAL_GUIDE.md)
- [ ] Save (⌘S)

### Update API Settings
- [ ] Open `Services/APIService.swift`
- [ ] Line 14: Update `baseURL` (or leave as is for mock data)
- [ ] Line 15: Update `apiKey` (or leave as is for mock data)
- [ ] Save (⌘S)

---

## ▶️ First Run (5 minutes)

### Select Target
- [ ] Top toolbar: Select "iPhone 15 Pro" simulator
- [ ] Wait for simulator to download (if needed)

### Build & Run
- [ ] Press ⌘R or click Play button
- [ ] Wait for build (15-30 seconds first time)
- [ ] Simulator launches
- [ ] App opens automatically

### Verify UI
- [ ] Black background visible ✓
- [ ] Purple/blue gradients visible ✓
- [ ] "Eden" logo with bolt icon ✓
- [ ] Balance shows ~$347.82 ✓
- [ ] Green "Active" button visible ✓
- [ ] Four stat cards displayed ✓
- [ ] Equity curve chart visible ✓
- [ ] Recent trades list shown ✓
- [ ] Bottom tab bar with 4 tabs ✓

---

## 🧪 Test Features (5 minutes)

### Basic Interactions
- [ ] Tap "Active" button → turns red "Paused"
- [ ] Tap again → back to green "Active"
- [ ] Tap eye icon → balance hides (••••••)
- [ ] Tap eye icon again → balance shows

### Navigation
- [ ] Tap "Positions" tab → see active trades
- [ ] Tap "Analytics" tab → see metrics
- [ ] Tap "Settings" tab → see config
- [ ] Tap "Overview" tab → back to dashboard

### Real-time Updates
- [ ] Watch balance change (every 3 seconds)
- [ ] Watch equity curve update
- [ ] Watch position values change
- [ ] All updates smooth and animated ✓

---

## 🔌 Eden Bot Integration (Later)

### n8n Setup
- [ ] Create webhook: `/webhook/eden-webhook`
- [ ] Create endpoint: `/webhook/eden-status`
- [ ] Create endpoint: `/webhook/eden-control`
- [ ] Create endpoint: `/webhook/eden-positions`
- [ ] Create endpoint: `/webhook/eden-trades`
- [ ] Test with Postman/curl

### Eden Bot Updates
- [ ] Create `src/webhook_notifier.py`
- [ ] Update `src/trading_bot.py` with webhook calls
- [ ] Add WEBHOOK_URL to `.env.eden`
- [ ] Add WEBHOOK_ENABLED=true to `.env.eden`
- [ ] Test webhook sending

### iOS App Connection
- [ ] Update APIService.swift with real URLs
- [ ] Update API key in APIService.swift
- [ ] Uncomment `fetchBotStatus()` in BotManager
- [ ] Test API connection
- [ ] Verify real data displays

---

## 📲 Push Notifications (Optional)

### Apple Developer Setup
- [ ] Apple Developer account ($99/year)
- [ ] Create App ID with Push enabled
- [ ] Create APNs key (.p8 file)
- [ ] Download key file

### Xcode Configuration
- [ ] Select project → Signing & Capabilities
- [ ] Add "Push Notifications" capability
- [ ] Add "Background Modes" capability
- [ ] Check "Remote notifications"

### Backend Setup
- [ ] Configure n8n to send push notifications
- [ ] Use APNs HTTP/2 API
- [ ] Test notification delivery

---

## 🚀 Deployment (Optional)

### TestFlight
- [ ] Archive app (Product → Archive)
- [ ] Distribute to TestFlight
- [ ] Upload to App Store Connect
- [ ] Add internal testers
- [ ] Install TestFlight on iPhone
- [ ] Install Eden beta app

### App Store
- [ ] Create App Store listing
- [ ] Add app screenshots (6 required)
- [ ] Write app description
- [ ] Submit for review
- [ ] Wait for approval (1-7 days)
- [ ] Publish app

---

## 🐛 Troubleshooting

### Build Errors
- [ ] Clean Build Folder (⇧⌘K)
- [ ] Delete Derived Data
- [ ] Restart Xcode
- [ ] Check all files in target

### Simulator Issues
- [ ] Reset simulator content
- [ ] Try different simulator model
- [ ] Restart Mac

### UI Issues
- [ ] Check console for errors (⌘⇧Y)
- [ ] Verify all files imported
- [ ] Check preview canvas (⌥⌘↩)

---

## 📝 Notes

**Current Status:**
- Working: ___________________
- Mock data: ✅ Yes / ⬜ No
- Real API: ⬜ Yes / ⬜ No
- Push notifications: ⬜ Yes / ⬜ No

**Issues Found:**
```
_________________________________________________
_________________________________________________
_________________________________________________
```

**Next Steps:**
```
_________________________________________________
_________________________________________________
_________________________________________________
```

---

## ✨ Success Criteria

You're done when:
- ✅ App opens without errors
- ✅ All 4 tabs work
- ✅ Data updates in real-time
- ✅ Animations are smooth
- ✅ UI looks like preview
- ✅ No console errors

---

**Setup Date:** _______________
**Completed By:** _______________
**Time Taken:** _______________

---

🎉 **Congratulations!** Your Eden iOS app is ready!
