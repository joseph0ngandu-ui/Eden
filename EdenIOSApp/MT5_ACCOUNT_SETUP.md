# MetaTrader 5 Account Configuration

## Overview

The Eden iOS app now includes **MetaTrader 5 account configuration** so users know which trading account the bot is using.

## Features

### ✅ Settings View
- **MT5 Account Section** with dedicated fields
- **Persistent storage** using UserDefaults
- **Visual confirmation** when account is configured

### ✅ Overview Dashboard
- **Account Info Card** displayed at the top
- Shows: Account Number, Account Name, and Broker
- **Green checkmark** indicator when configured

## Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| **Account Number** | MT5 account number | `12345678` |
| **Account Name** | Display name for the account | `My Trading Account` |
| **Broker** | Broker name | `Exness`, `IC Markets`, `XM` |
| **Server** | MT5 server name | `Exness-MT5Real`, `ICMarkets-Live` |
| **Password** | Account password (securely stored) | `********` |

## How to Use

### Step 1: Open Settings
1. Launch Eden app
2. Tap **Settings** in the tab bar

### Step 2: Configure MT5 Account
1. Scroll to **MetaTrader 5 Account** section
2. Fill in your account details:
   - **Account Number**: Your MT5 account number
   - **Account Name**: A friendly name (e.g., "Live Trading")
   - **Broker**: Your broker name
   - **Server**: MT5 server (find in MT5 app)
   - **Password**: Your MT5 account password

### Step 3: Save Settings
1. Tap **Save All Settings** button
2. Check for success message in console

### Step 4: Verify
1. Go back to **Overview** tab
2. You should see your account info card at the top
3. Green checkmark indicates successful configuration

## Account Info Display

### In Settings View
When account is configured, you'll see:
```
✓ Account 12345678 - My Trading Account
  Exness - Exness-MT5Real
```

### In Overview Dashboard
Top of dashboard shows:
```
🏛️ Trading Account
   12345678 - My Trading Account
   Exness
   ✓
```

## Data Storage

All settings are stored locally using **UserDefaults**:
- `mt5AccountNumber`
- `mt5AccountName`
- `mt5Broker`
- `mt5Server`
- `mt5Password` (stored securely)

## Security Notes

⚠️ **Password Storage**
- Password is stored in UserDefaults (plain text)
- For production, consider using **Keychain** for password storage
- iOS Keychain provides encrypted storage for sensitive data

### Recommended Enhancement
```swift
// Future: Use KeychainSwift or similar
import KeychainSwift

let keychain = KeychainSwift()
keychain.set(mt5Password, forKey: "mt5Password")
let password = keychain.get("mt5Password")
```

## Backend Integration

### Current State
Settings are stored locally in the iOS app only.

### Future Enhancement
Consider syncing MT5 account details with the backend:
- Add `/api/account/mt5` endpoint
- Store account details in database
- Sync across multiple devices
- Validate account with broker API

## UI Components

### SettingField Component
Reusable text field with label:
```swift
SettingField(label: "Account Number", text: $mt5AccountNumber)
    .keyboardType(.numberPad)

SettingField(label: "Password", text: $mt5Password, isSecure: true)
```

### Account Info Card
Displays configured account:
- Green border and background
- Checkmark icon
- Account details formatted nicely

## Testing

### Test in Simulator
1. Open Eden app in Xcode simulator
2. Navigate to Settings
3. Fill in test account details:
   - Number: `12345678`
   - Name: `Test Account`
   - Broker: `Exness`
   - Server: `Exness-MT5Real`
   - Password: `test123`
4. Save settings
5. Check Overview tab for account card

### Verify Persistence
1. Close app completely (swipe up in app switcher)
2. Reopen app
3. Check Settings - fields should be filled
4. Check Overview - account card should display

## User Benefits

✅ **Know which account is trading**
- Always visible in Overview dashboard
- No confusion about which account Eden is using

✅ **Multiple account support** (future)
- Can switch between accounts
- Each account has its own settings

✅ **Transparency**
- Users see exactly which broker/server
- Account number clearly displayed

✅ **Easy configuration**
- Simple form in Settings
- All details in one place

## Screenshots

### Settings View - MT5 Section
```
╔════════════════════════════════════════╗
║  MetaTrader 5 Account                  ║
║  Configure the MT5 account Eden is     ║
║  trading on                            ║
║                                        ║
║  Account Number: [12345678]            ║
║  Account Name: [Live Trading]          ║
║  Broker: [Exness]                      ║
║  Server: [Exness-MT5Real]              ║
║  Password: [••••••••]                  ║
║                                        ║
║  ┌─────────────────────────────────┐  ║
║  │ ✓ Trading Account:              │  ║
║  │   12345678 - Live Trading       │  ║
║  │   Exness - Exness-MT5Real       │  ║
║  └─────────────────────────────────┘  ║
╚════════════════════════════════════════╝
```

### Overview Dashboard - Account Card
```
╔════════════════════════════════════════╗
║  🏛️  Trading Account                   ║
║       12345678 - Live Trading       ✓  ║
║       Exness                            ║
╚════════════════════════════════════════╝
```

## Code Changes

### Files Modified
- ✅ `SettingsView.swift` - Added MT5 account fields and save logic
- ✅ `OverviewView.swift` - Added account info display card

### New State Variables
```swift
// In SettingsView
@State private var mt5AccountNumber = ""
@State private var mt5AccountName = ""
@State private var mt5Broker = ""
@State private var mt5Server = ""
@State private var mt5Password = ""
```

### New Methods
```swift
// Load settings from UserDefaults
private func loadSettings()

// Save all settings including MT5 account
private func saveAllSettings()

// Load MT5 info for display in Overview
private func loadMT5AccountInfo()
```

## Future Enhancements

### 1. Keychain Integration
Store password securely in iOS Keychain instead of UserDefaults.

### 2. Account Validation
Validate account details with broker API before saving.

### 3. Multiple Accounts
Allow users to configure and switch between multiple MT5 accounts.

### 4. Backend Sync
Sync account configuration with Eden backend for cross-device support.

### 5. Account Balance Display
Fetch and display current account balance from MT5.

### 6. Trade History Filter
Filter trades by selected account in Positions/Trades view.

## Support

If you encounter issues:
1. Check that all fields are filled correctly
2. Verify account details in MT5 app
3. Check console logs for error messages
4. Clear UserDefaults if needed: 
   ```swift
   UserDefaults.standard.removeObject(forKey: "mt5AccountNumber")
   ```

## Summary

✅ Users can now input and view MT5 account details  
✅ Account info displayed prominently in Overview  
✅ Settings persist across app restarts  
✅ Clean, organized UI in Settings view  
✅ Visual confirmation when configured  

**Users now always know which MT5 account Eden is trading on!**
