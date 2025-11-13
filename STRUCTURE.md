# Eden Project Structure

## Directory Organization

```
Eden/
├── ios/                    # iOS Application
│   ├── Eden/              # Xcode project
│   └── *.md               # iOS-specific documentation
│
├── backend/               # Backend API Service
│   ├── app/              # API application code
│   ├── ssl/              # SSL certificates
│   └── main.py           # Entry point
│
├── trading/              # Trading Bot Core Logic
│   ├── backtest_engine.py
│   ├── trading_bot.py
│   ├── volatility_burst.py
│   └── ...               # Trading strategies & utilities
│
├── infrastructure/       # Deployment & Management
│   ├── scripts/         # Build & deployment scripts
│   ├── lambda/          # AWS Lambda functions
│   ├── *.py             # Management scripts (bot_runner, optimizer, etc.)
│   ├── *.bat            # Windows batch scripts
│   └── *.ps1            # PowerShell scripts
│
├── config/              # Configuration Files
│   └── *.yaml, *.yml    # Strategy and system configs
│
├── docs/                # Documentation
│   ├── ARCHITECTURE.md
│   ├── BACKTESTING.md
│   ├── DEPLOYMENT.md
│   └── ...
│
├── data/                # Data Storage
│   └── strategies.json
│
├── logs/                # Application Logs
│
├── reports/             # Backtest & Performance Reports
│
├── tests/               # Test Suite
│
└── venv/                # Python Virtual Environment
```

## Root Files
- `README.md` - Main project documentation
- `QUICKSTART.md` - Quick start guide
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker configuration
- `.env.example` - Environment template
- `.gitignore` - Git ignore rules

## Quick Access

### iOS App Main Files
- **App Entry**: `ios/Eden/Eden/Eden/EdenApp.swift`
- **Views**: `ios/Eden/Eden/Eden/Views/`
- **Components**: `ios/Eden/Eden/Eden/Components/`
- **Models**: `ios/Eden/Eden/Eden/Models.swift`
- **Network**: `ios/Eden/Eden/Eden/Network/`

### Backend API
- **Entry Point**: `backend/main.py`
- **API Service**: `backend/api_service.py`
- **Routes**: `backend/app/`

### Trading Bot
- **Main Bot**: `trading/trading_bot.py`
- **Strategies**: `trading/volatility_burst.py`, `trading/strategy_backtester.py`
- **Backtest**: `trading/backtest_engine.py`

### Management Scripts
- **Run Bot**: `infrastructure/bot_runner.py`
- **Optimizer**: `infrastructure/autonomous_optimizer.py`
- **Deployment**: `infrastructure/deployment_manager.py`
- **Start Scripts**: `infrastructure/*.bat`, `infrastructure/*.ps1`
