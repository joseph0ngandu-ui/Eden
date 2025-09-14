# Eden - Professional Trading System

**The Origin of Order**

## Project Structure

Eden has been professionally organized with Apple-inspired design and automatic theme detection. The project is now clean, intuitive, and robust.

### Core Files

```
├── Eden.py                 # Unified entry point - launches GUI or CLI
├── run_ui.py              # Main GUI application with Apple styling
├── installer.py           # Professional Windows installer
├── splash_screen.py       # Apple-style splash screen with motivational quotes
├── theme_manager.py       # Automatic light/dark theme detection
├── requirements.txt       # Python dependencies
└── QUICK_START.md        # User guide
```

### Launcher Options

```
├── launch.py              # Cross-platform Python launcher
├── launch.bat             # Windows batch file launcher
├── Launch_Eden.ps1        # Windows PowerShell launcher
└── build_setup.py         # Professional installer builder
```

### Core Package

```
eden/                      # Main package
├── __init__.py
├── cli.py                 # Command-line interface
├── config.py              # Configuration management
├── logging_conf.py        # Logging setup
├── mt5_integration.py     # MetaTrader 5 integration
├── backtest/              # Backtesting engine
├── data/                  # Data loading and processing
├── execution/             # Trading execution
├── features/              # Feature engineering
├── ml/                    # Machine learning components
├── risk/                  # Risk management
├── strategies/            # Trading strategies
├── tests/                 # Test suite
├── ui_app.py             # UI application core
├── ui_components.py       # UI components
├── ui_modern.py          # Modern UI elements
├── ui_theme.py           # UI theming
└── utils/                # Utility functions
```

### Distribution

```
dist/                      # Built distributions
├── Eden Setup.exe         # Professional Windows installer (46.6 MB)
├── eden.exe              # Standalone executable (110.1 MB)
└── Eden_Portable.zip     # Portable package (0.1 MB)
```

## Key Features Implemented

### 1. Apple-Style Design System
- **Automatic theme detection** based on Windows system preferences
- **Clean typography** using Apple system fonts (-apple-system, SF Pro)
- **Minimalist interface** with proper spacing and subtle shadows
- **Real-time theme switching** when user changes system theme
- **Professional color palettes** for light and dark modes

### 2. Unified Entry Point
- **Single `Eden.py` file** for all application access
- **Automatic GUI/CLI detection** based on command-line arguments
- **Graceful fallbacks** when GUI dependencies unavailable
- **Professional splash screen** with motivational quotes

### 3. Professional Installation System
- **`Eden Setup.exe`** - One-click Windows installer
- **Automatic dependency management** 
- **Desktop and Start Menu shortcuts**
- **Windows Add/Remove Programs integration**
- **Theme-aware installer UI**

### 4. Multiple Launch Options
- **Double-click `Eden Setup.exe`** - Install and run automatically
- **Double-click `launch.bat`** - Quick Windows launcher
- **Run `python Eden.py`** - Direct Python execution
- **PowerShell `.\Launch_Eden.ps1`** - Advanced Windows launcher

## Installation Instructions

### For End Users (Recommended)
1. **Download `Eden Setup.exe`** from the `dist/` folder
2. **Double-click** to install automatically
3. **Eden will be installed** with all dependencies
4. **Launch from desktop shortcut** or Start Menu

### For Developers
1. **Clone the repository**
2. **Run `python Eden.py`** for development mode
3. **Use launcher scripts** for convenience

### Portable Version
1. **Extract `Eden_Portable.zip`**
2. **Double-click `Start_Eden.bat`**
3. **No installation required**

## Technical Excellence

### Apple Design Principles Applied
- **Typography**: Proper font hierarchy with Apple system fonts
- **Color System**: Semantic colors that adapt to system theme
- **Layout**: Clean spacing with consistent 8px grid
- **Interactions**: Subtle hover states and smooth transitions
- **Accessibility**: High contrast ratios in both themes

### Professional Code Structure
- **Unified entry point** eliminates confusion
- **Proper error handling** with graceful fallbacks  
- **Automatic dependency management**
- **Clean imports** and organized modules
- **Professional logging** and monitoring

### Windows Integration
- **System theme detection** via Windows Registry
- **Proper installer** with Windows standards compliance
- **Desktop and Start Menu shortcuts**
- **Add/Remove Programs registration**
- **Professional executable metadata**

## Usage Examples

### Launch GUI Application
```bash
# Any of these methods work:
python Eden.py
python launch.py
.\launch.bat
.\Launch_Eden.ps1
```

### Launch CLI Application
```bash
python Eden.py --help
python Eden.py backtest --config config.yml
```

### Build Installer
```bash
python build_setup.py
```

## Apple-Style Quality Standards

Eden now meets professional Apple-quality standards:

✅ **Clean, minimalist design** with no visual clutter  
✅ **Automatic theme adaptation** to user preferences  
✅ **Consistent typography** using system fonts  
✅ **Professional color system** with semantic naming  
✅ **Smooth animations** with proper easing curves  
✅ **Intuitive user interface** with clear hierarchy  
✅ **Reliable installation** with automatic setup  
✅ **Professional documentation** and user guides  
✅ **Robust error handling** with user-friendly messages  
✅ **Clean codebase** with no duplicate or redundant files  

## Summary

Eden is now a **professional, production-ready trading system** with:

- 🎨 **Apple-inspired design** that automatically adapts to light/dark themes
- 📦 **Professional Windows installer** named "Eden Setup.exe"
- 🚀 **Multiple launch options** for different user preferences  
- 🧹 **Clean project structure** with no redundant files
- 💎 **Premium user experience** matching modern app standards
- 🛠️ **Robust technical foundation** ready for production deployment

The system is **intuitive, robust, and professionally structured** - ready for distribution to end users.

---

**Eden - The Origin of Order**  
*Professional Trading System v2.0*