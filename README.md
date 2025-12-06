# Eden Trading Bot

A professional algorithmic trading system built with Python and MetaTrader 5, featuring advanced ICT (Inner Circle Trader) strategies, real-time market analysis, and autonomous optimization capabilities.

## 🚀 Features

- **Advanced Trading Strategies**
  - Volatility Burst v1.3 with enhanced entry/exit logic
  - ICT 2023 Silver Bullet Strategy
  - ICT 2024 Unicorn Model
  - ICT 2025 Venom Strategy
  
- **Backend API (FastAPI)**
  - RESTful API with comprehensive endpoints
  - WebSocket support for real-time updates
  - JWT authentication
  - Systematic status monitoring
  
- **Autonomous Optimization**
  - Real-time parameter tuning
  - Performance-based adjustments
  - Risk management optimization
  
- **Professional Infrastructure**
  - Health monitoring and watchdog systems
  - Comprehensive logging and error tracking
  - Trade journaling and performance analytics
  - SSL/HTTPS support

## 📋 Prerequisites

- Python 3.10+
- MetaTrader 5 terminal
- Windows OS (for MT5 integration)
- Active MT5 account (demo or live)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Eden.git
   cd Eden
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env`
   - Update with your MT5 credentials and settings
   ```env
   MT5_LOGIN=your_account
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server
   ```

4. **Configure strategies**
   - Edit `data/strategies.json` to enable/configure strategies
   - Adjust risk parameters in `config/risk_config.json`

## 🎯 Quick Start

### Start the Backend API
```bash
cd backend
python main.py
```
The API will be available at `https://localhost:8443`

### Run the Trading Bot
```bash
python infrastructure/bot_runner.py
```

### Complete System Restart
```powershell
.\restart_all.ps1
```

## 📁 Project Structure

```
Eden/
├── backend/              # FastAPI backend server
│   ├── app/             # API application
│   │   ├── routers/     # API endpoints
│   │   ├── models.py    # Pydantic models
│   │   └── ...
│   └── main.py          # API entry point
├── trading/             # Trading bot core
│   ├── trading_bot.py   # Main bot logic
│   ├── ict_strategies.py # ICT strategy implementations
│   ├── volatility_burst_enhanced.py
│   ├── models.py        # Shared trading models
│   └── ...
├── infrastructure/      # System infrastructure
│   ├── bot_runner.py    # Bot execution wrapper
│   └── ...
├── config/              # Configuration files
├── data/                # Strategy configs and data
├── scripts/             # Utility scripts
│   └── debug/          # Debug and verification tools
├── tests/               # Unit and integration tests
└── docs/                # Additional documentation
```

## 🔌 API Documentation

See [API_ENDPOINTS.md](API_ENDPOINTS.md) for complete API documentation.

Key endpoints:
- `GET /health` - System health status
- `GET /strategies` - List all strategies
- `GET /trades` - Trade history
- `GET /performance` - Performance metrics
- `WebSocket /ws` - Real-time updates

## 📊 Trading Strategies

### Volatility Squeeze (H1)
Trend following strategy running on 28 Pairs. Identifies volatility contractions followed by explosive breakouts.
- **Edge:** +15.5R (High Win Rate)
- **Timeframe:** H1
- **Role:** Portfolio Anchor (Wide & Slow)

### Quiet Before Storm (H1)
Sniper strategy for GBP and XAUUSD. Capitalizes on rare, extreme volatility contractions.
- **Edge:** +0.6R
- **Role:** High Impact / Low Frequency

### VWAP Reversion (M5)
High-frequency mean reversion strategy fading 3-ATR deviations from Intraday VWAP.
- **Edge:** +1200R Gross (Spread Sensitive)
- **Volume:** ~15,000 trades/month
- **Role:** Cash Flow (User Enabled)

## 🔧 Configuration

### Risk Management
Edit `config/risk_config.json`:
```json
{
  "max_risk_per_trade": 0.02,
  "max_daily_drawdown": 0.05,
  "position_sizing": "dynamic"
}
```

### Strategy Selection
Edit `data/strategies.json` to enable/disable strategies:
```json
{
  "strategies": [
    {
      "name": "Volatility Burst v1.3",
      "enabled": true,
      "version": "1.3.0"
    }
  ]
}
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run specific strategy tests:
```bash
pytest tests/test_ict_strategies.py -v
```

## 📈 Monitoring

### Health Checks
```bash
python check_bot.ps1
```

### View Logs
- Backend: `backend/api.log`
- Trading: `logs/trading_*.log`
- System: `watchdog.log`

### Performance Reports
Generated in `reports/` directory with CSV and JSON formats.

## 🔐 Security

- JWT-based authentication for API access
- Environment variables for sensitive data
- SSL/HTTPS encryption for all communications
- No hardcoded credentials

## 📝 License

This project is proprietary software. All rights reserved.

## 🤝 Contributing

This is a private trading system. Contact the repository owner for collaboration inquiries.

## ⚠️ Disclaimer

Trading forex and CFDs involves significant risk. This software is provided for educational and research purposes. Always test strategies in a demo environment before live trading. Past performance does not guarantee future results.

## 📞 Support

For questions or issues:
- Check [QUICKSTART.md](QUICKSTART.md) for setup help
- Review [API_ENDPOINTS.md](API_ENDPOINTS.md) for API details
- See [STRUCTURE.md](STRUCTURE.md) for architecture overview

---

**Built with ❤️ for algorithmic trading excellence**
