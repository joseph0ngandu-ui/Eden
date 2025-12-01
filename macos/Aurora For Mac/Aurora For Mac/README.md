# Aurora Mac App

A native macOS application for ML strategy training and management for the Eden trading bot.

## Requirements

- macOS 13.0+
- Xcode 15.0+
- Swift 5.9+

## Features

### ✅ Implemented
- **Authentication** - Secure login with Keychain storage
- **Strategy Management** - Create, edit, and upload strategies
- **Backend Integration** - Full API connectivity with Eden backend

### 🚧 Coming Soon
- **ML Training** - Train Create ML models on historical data
- **Backtesting** - Validate strategies on historical market conditions
- **Live Monitoring** - Real-time strategy performance tracking

## Getting Started

### 1. Open in Xcode

Since this is a pure SwiftUI app without a .xcodeproj file, you have two options:

**Option A: Create Xcode Project (Recommended)**
1. Open Xcode
2. File → New → Project
3. Select "macOS" → "App"
4. Name: "AuroraMacApp", Bundle ID: "com.eden.auroramac"
5. Replace the default files with the files in this directory

**Option B: Use Swift Package**
1. Run `swift package init --type executable` in this directory
2. Delete the generated files
3. The existing Swift files will be used

### 2. Configure Backend URL

In `Services/APIService.swift`, update the `baseURL` to point to your Eden backend:

```swift
@Published var baseURL: String = "https://your-tailscale-url.ts.net"
```

### 3. Build and Run

Press `Cmd + R` in Xcode to build and run the app.

## Architecture

```
AuroraMacApp/
├── Models/              # Data structures
│   ├── Strategy.swift
│   └── BacktestResult.swift
├── Services/            # Backend communication
│   ├── APIService.swift
│   └── AuthService.swift
├── ViewModels/          # Business logic
│   └── StrategyViewModel.swift
└── Views/               # UI components
    ├── ContentView.swift
    ├── LoginView.swift
    ├── StrategyListView.swift
    ├── StrategyEditorView.swift
    └── ...
```

## Usage

### Creating a Strategy

1. Click "New Strategy" in the Strategies tab
2. Fill in the parameters (risk, stop loss, take profit, etc.)
3. Add indicators and entry/exit conditions
4. Click "Save & Upload" to send to the bot

### API Integration

The app connects to your Eden backend endpoints:
- `POST /auth/login-local` - Authentication
- `GET /strategies` - Fetch all strategies
- `POST /strategies` - Upload new strategy

## License

Same as Eden project
