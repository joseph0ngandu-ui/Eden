# Aurora Mac App

A premium native macOS application for managing and monitoring your Eden trading bot strategies with real-time updates and ML training capabilities.

![Aurora Logo](https://img.shields.io/badge/Platform-macOS_13.0+-blue.svg)
![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### ✅ Fully Implemented

- **🔐 Secure Authentication** 
  - Login with email/password
  - Keychain integration for secure credential storage
  - Beautiful glassmorphism login UI
  - Auto-login on app launch

- **📊 Strategy Management**
  - Create, edit, and delete trading strategies
  - Configure parameters (risk, stop loss, take profit, timeframe)
  - Define technical indicators and entry/exit conditions
  - Upload strategies directly to your trading bot
  - Real-time strategy status tracking

- **📈 Real-Time Monitoring**
  - Live WebSocket connection to trading bot
  - Active positions dashboard with P&L tracking
  - Recent trades history
  - Real-time statistics (Total P&L, Active Positions, Win Rate)
  - Position filtering (All, Long, Short, Profitable)
  - Beautiful card-based UI with live updates

- **🔔 Smart Notifications**
  - Trade execution alerts
  - Position change notifications
  - Connection status updates
  - Configurable in Settings

- **⚙️ Advanced Settings**
  - Custom API endpoint configuration
  - Connection testing
  - Notification preferences
  - Dark mode support
  - Auto-connect option

### 🚧 Coming Soon

- **🤖 ML Training** - Train Create ML models on historical data to generate strategies
- **⏮️ Backtesting** - Validate strategies against historical market conditions
- **📉 Analytics Dashboard** - Advanced performance metrics and charts
- **💾 Strategy Templates** - Pre-built strategy templates for quick deployment

## 🚀 Quick Start

### Prerequisites

- macOS 13.0 (Ventura) or later
- Xcode 15.0 or later
- Swift 5.9 or later
- Eden trading bot backend running

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eden.git
   cd eden/macos
   ```

2. **Open in Xcode**
   ```bash
   open "Aurora For Mac.xcodeproj"
   ```

3. **Configure Backend URL**
   
   The app is pre-configured to connect to `https://edenbot.duckdns.org:8443`. You can change this in Settings after logging in.

4. **Build and Run**
   
   Press `⌘R` in Xcode to build and run the app.

5. **Login**
   
   Use your Eden trading bot credentials to log in.

## 📱 App Structure

```
Aurora For Mac/
├── Models/                 # Data structures
│   ├── Strategy.swift      # Trading strategy model
│   ├── Position.swift      # Active position model
│   ├── Trade.swift         # Trade history model
│   └── BacktestResult.swift
│
├── Services/               # Business logic & networking
│   ├── APIService.swift    # REST API client
│   ├── AuthService.swift   # Authentication & Keychain
│   ├── WebSocketService.swift # Real-time updates
│   └── NotificationManager.swift # Push notifications
│
├── ViewModels/             # State management
│   └── StrategyViewModel.swift
│
├── Views/                  # UI components
│   ├── ContentView.swift   # Main navigation
│   ├── LoginView.swift     # Authentication UI
│   ├── MonitorView.swift   # Real-time monitoring dashboard
│   ├── StrategyListView.swift # Strategy management
│   ├── StrategyEditorView.swift # Strategy editor
│   ├── MLTrainingView.swift # ML training (coming soon)
│   ├── BacktestView.swift  # Backtesting (coming soon)
│   ├── SettingsView.swift  # App settings
│   └── Components/         # Reusable UI components
│       ├── PositionCard.swift
│       ├── TradeRow.swift
│       └── StatCard.swift
│
└── Utilities/              # Helper utilities
    └── ErrorPresenter.swift # Error handling & display
```

## 🎨 UI/UX Highlights

- **Glassmorphism Design** - Modern, premium UI with frosted glass effects
- **Animated Gradients** - Beautiful gradient backgrounds throughout the app
- **Real-Time Updates** - Live data streaming via WebSocket
- **Dark Mode** - Full dark mode support
- **Responsive Layouts** - Adaptive UI that works on all Mac screen sizes
- **SF Symbols** - Native macOS iconography
- **Native Controls** - Uses standard macOS UI components

## 🔌 API Integration

The app communicates with your Eden backend through two channels:

### REST API
- `POST /auth/login-local` - User authentication
- `GET /strategies` - Fetch all strategies
- `POST /strategies` - Upload new strategy
- More endpoints as needed

### WebSocket
- `ws://your-backend/ws/monitor` - Real-time position and trade updates

## 🔒 Security

- **Keychain Integration** - Credentials stored securely in macOS Keychain
- **App Sandbox** - Runs in sandboxed environment
- **Network Security** - HTTPS/WSS only with certificate validation
- **No Plain Text Storage** - Passwords never stored in plain text

## ⚙️ Configuration

### Changing Backend URL

1. Open Settings (`⌘,` or menu bar → Aurora → Settings)
2. Update the "API Base URL" field
3. Test the connection
4. Restart the app if needed

### Enabling Notifications

1. Open Settings
2. Toggle "Enable Notifications"
3. Grant permission when prompted
4. Configure notification preferences

## 🐛 Troubleshooting

### Cannot connect to backend
- Verify the backend URL is correct in Settings
- Ensure your backend is running and accessible
- Check firewall settings
- Try the "Test Connection" button in Settings

### Login fails
- Verify your credentials
- Check backend logs for authentication errors
- Ensure your account exists in the backend

### WebSocket not connecting
- Check backend WebSocket endpoint is running
- Verify URL scheme (wss:// for HTTPS backends)
- Check network proxy settings

### Notifications not working
- Grant notification permissions in Settings
- Check macOS System Settings → Notifications → Aurora
- Restart the app after granting permissions

## 📝 Development

### Adding New Features

1. Create models in `Models/`
2. Add service methods in `Services/`
3. Create ViewModels if needed in `ViewModels/`
4. Build UI in `Views/`
5. Update this README

### Building for Release

```bash
xcodebuild -project "Aurora For Mac.xcodeproj" \
  -scheme "Aurora For Mac" \
  -configuration Release \
  -archivePath "build/Aurora.xcarchive" \
  archive
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of the Eden trading bot ecosystem. See the main repository for license information.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Contact the development team
- Check the main Eden documentation

## 🙏 Acknowledgments

- Built with SwiftUI and modern macOS APIs
- Uses SF Symbols for iconography
- Inspired by modern fintech applications

---

**Made with ❤️ for algorithmic traders**
