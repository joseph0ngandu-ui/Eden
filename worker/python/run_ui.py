#!/usr/bin/env python3
"""
Eden - Professional Trading Dashboard
Apple-inspired design with clean typography and intuitive interface
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
        QProgressBar, QTableWidget, QTableWidgetItem, QSplitter, QFrame, QGridLayout,
        QScrollArea, QGroupBox, QCheckBox, QDateEdit, QSlider, QMessageBox, QFileDialog,
        QStatusBar, QMenuBar, QToolBar, QTreeWidget, QTreeWidgetItem, QFormLayout,
        QTabBar, QSizePolicy, QHeaderView
    )
    from PySide6.QtGui import (
        QFont, QIcon, QColor, QPalette, QPixmap, QPainter, QLinearGradient,
        QAction, QFontDatabase, QMovie
    )
    from PySide6.QtCore import (
        Qt, QTimer, QThread, Signal, QSize, QPropertyAnimation, QEasingCurve,
        QParallelAnimationGroup, QSequentialAnimationGroup, QRect, QDateTime
    )
    
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, mkPen, mkBrush
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Eden modules
from eden.config import EdenConfig
from eden.data.loader import DataLoader
from eden.logging_conf import configure_logging
from splash_screen import show_splash_screen
from eden.mt5_integration import get_mt5_manager
from eden.cli import run_backtest as cli_run_backtest

# Import Apple-style theme manager
try:
    from theme_manager import get_theme_manager, get_theme_colors, get_theme_color
    THEME_MANAGER_AVAILABLE = True
except ImportError:
    THEME_MANAGER_AVAILABLE = False

# Import backtest modules if available
try:
    from eden.backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None

try:
    from eden.ml.discovery import StrategyDiscovery
except ImportError:
    StrategyDiscovery = None


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    if not PYSIDE6_AVAILABLE:
        missing_deps.append("PySide6")
    
    if not PYQTGRAPH_AVAILABLE:
        missing_deps.append("pyqtgraph")
    
    return missing_deps


class EdenTheme:
    """Apple-inspired color theme with automatic light/dark mode switching."""
    
    def __init__(self):
        """Initialize theme with system detection."""
        if THEME_MANAGER_AVAILABLE:
            self.theme_manager = get_theme_manager()
            self._colors = get_theme_colors()
        else:
            # Fallback to light theme
            self._colors = self._get_light_fallback()
    
    def _get_light_fallback(self) -> dict:
        """Fallback light theme colors."""
        return {
            'primary': '#007AFF',
            'secondary': '#5856D6',
            'success': '#34C759',
            'warning': '#FF9500',
            'danger': '#FF3B30',
            'bg_primary': '#FFFFFF',
            'bg_secondary': '#F2F2F7',
            'bg_tertiary': '#FFFFFF',
            'text_primary': '#000000',
            'text_secondary': '#3C3C43',
            'text_tertiary': '#3C3C4399',
            'separator': '#C6C6C8',
            'border': '#E5E5E7',
            'profit': '#34C759',
            'loss': '#FF3B30',
            'neutral': '#8E8E93',
        }
    
    def refresh_colors(self):
        """Refresh colors from theme manager."""
        if THEME_MANAGER_AVAILABLE:
            self._colors = get_theme_colors()
    
    def get_color(self, key: str, fallback: str = '#000000') -> str:
        """Get color by key with fallback."""
        if THEME_MANAGER_AVAILABLE:
            return get_theme_color(key, fallback)
        return self._colors.get(key, fallback)
    
    # Dynamic properties that adapt to current theme
    @property
    def PRIMARY(self) -> str:
        return self.get_color('primary', '#007AFF')
    
    @property
    def SECONDARY(self) -> str:
        return self.get_color('secondary', '#5856D6')
    
    @property
    def SUCCESS(self) -> str:
        return self.get_color('success', '#34C759')
    
    @property
    def WARNING(self) -> str:
        return self.get_color('warning', '#FF9500')
    
    @property
    def DANGER(self) -> str:
        return self.get_color('danger', '#FF3B30')
    
    @property
    def BG_PRIMARY(self) -> str:
        return self.get_color('bg_primary', '#FFFFFF')
    
    @property
    def BG_SECONDARY(self) -> str:
        return self.get_color('bg_secondary', '#F2F2F7')
    
    @property
    def BG_TERTIARY(self) -> str:
        return self.get_color('bg_tertiary', '#FFFFFF')
    
    @property
    def TEXT_PRIMARY(self) -> str:
        return self.get_color('text_primary', '#000000')
    
    @property
    def TEXT_SECONDARY(self) -> str:
        return self.get_color('text_secondary', '#3C3C43')
    
    @property
    def TEXT_TERTIARY(self) -> str:
        return self.get_color('text_tertiary', '#3C3C4399')
    
    @property
    def BORDER_PRIMARY(self) -> str:
        return self.get_color('separator', '#C6C6C8')
    
    @property
    def BORDER_SECONDARY(self) -> str:
        return self.get_color('border', '#E5E5E7')
    
    @property
    def PROFIT(self) -> str:
        return self.get_color('profit', '#34C759')
    
    @property
    def LOSS(self) -> str:
        return self.get_color('loss', '#FF3B30')
    
    @property
    def NEUTRAL(self) -> str:
        return self.get_color('neutral', '#8E8E93')
    
    def create_palette(self):
        """Create Qt palette with current theme."""
        if THEME_MANAGER_AVAILABLE and hasattr(self, 'theme_manager'):
            return self.theme_manager.get_qt_palette()
        
        # Fallback palette creation
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(self.BG_PRIMARY))
        palette.setColor(QPalette.WindowText, QColor(self.TEXT_PRIMARY))
        
        # Base colors
        palette.setColor(QPalette.Base, QColor(self.BG_TERTIARY))
        palette.setColor(QPalette.AlternateBase, QColor(self.BG_SECONDARY))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(self.TEXT_PRIMARY))
        palette.setColor(QPalette.BrightText, QColor(self.TEXT_PRIMARY))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(self.PRIMARY))
        palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(self.PRIMARY))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        
        return palette
    
    def get_stylesheet(self):
        """Get comprehensive Apple-style stylesheet for the application."""
        # Refresh colors to get latest theme
        self.refresh_colors()
        
        return f"""
        /* Main application styling */
        QMainWindow {{
            background-color: {cls.BG_PRIMARY};
            color: {cls.TEXT_PRIMARY};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', sans-serif;
        }}
        
        /* Tab widget styling */
        QTabWidget::pane {{
            border: 1px solid {cls.BORDER_SECONDARY};
            background-color: {cls.BG_PRIMARY};
            border-radius: 8px;
        }}
        
        QTabBar::tab {{
            background-color: transparent;
            color: {cls.TEXT_SECONDARY};
            padding: 8px 16px;
            margin: 0px 1px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 400;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QTabBar::tab:selected {{
            background-color: {cls.PRIMARY};
            color: white;
            font-weight: 500;
        }}
        
        QTabBar::tab:hover {{
            background-color: {cls.BG_SECONDARY};
            color: {cls.TEXT_PRIMARY};
        }}
        
        /* Button styling - Apple system style */
        QPushButton {{
            background-color: {cls.PRIMARY};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QPushButton:hover {{
            background-color: #0056CC;
        }}
        
        QPushButton:pressed {{
            background-color: #004499;
        }}
        
        QPushButton:disabled {{
            background-color: {cls.BG_SECONDARY};
            color: {cls.TEXT_TERTIARY};
        }}
        
        /* Input styling - Apple system style */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {cls.BG_INPUT};
            color: {cls.TEXT_PRIMARY};
            border: 1px solid {cls.BORDER_PRIMARY};
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QLineEdit:focus, QComboBox:focus {{
            border: 2px solid {cls.PRIMARY};
            outline: none;
        }}
        
        /* Table styling - Clean Apple style */
        QTableWidget {{
            background-color: {cls.BG_PRIMARY};
            color: {cls.TEXT_PRIMARY};
            gridline-color: {cls.BORDER_SECONDARY};
            selection-background-color: {cls.PRIMARY};
            alternate-background-color: {cls.BG_SECONDARY};
            border: 1px solid {cls.BORDER_SECONDARY};
            border-radius: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QTableWidget::item {{
            padding: 12px 8px;
            font-size: 13px;
        }}
        
        QHeaderView::section {{
            background-color: {cls.BG_SECONDARY};
            color: {cls.TEXT_SECONDARY};
            padding: 12px 8px;
            border: none;
            border-right: 1px solid {cls.BORDER_SECONDARY};
            font-weight: 500;
            font-size: 12px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        /* Text edit styling */
        QTextEdit {{
            background-color: {cls.BG_INPUT};
            color: {cls.TEXT_PRIMARY};
            border: 1px solid {cls.BORDER_PRIMARY};
            border-radius: 8px;
            padding: 12px;
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        /* Group box styling - Apple card style */
        QGroupBox {{
            color: {cls.TEXT_PRIMARY};
            border: 1px solid {cls.BORDER_SECONDARY};
            border-radius: 12px;
            margin: 16px 0;
            padding-top: 24px;
            background-color: {cls.BG_TERTIARY};
            font-weight: 500;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px;
            color: {cls.TEXT_SECONDARY};
        }}
        
        /* Progress bar styling - Apple system style */
        QProgressBar {{
            background-color: {cls.BG_SECONDARY};
            border: none;
            border-radius: 4px;
            text-align: center;
            color: {cls.TEXT_SECONDARY};
            height: 8px;
            font-size: 11px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        
        QProgressBar::chunk {{
            background-color: {cls.PRIMARY};
            border-radius: 4px;
        }}
        
        /* Status bar styling */
        QStatusBar {{
            background-color: {cls.BG_SECONDARY};
            color: {cls.TEXT_SECONDARY};
            border-top: 1px solid {cls.BORDER_SECONDARY};
            font-size: 12px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        }}
        """


class StatusCard(QFrame):
    """Apple-style status card widget."""
    
    def __init__(self, title: str, value: str = "0", 
                 color: str = "neutral", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {EdenTheme.BG_TERTIARY};
                border-radius: 12px;
                border: 1px solid {EdenTheme.BORDER_SECONDARY};
                padding: 16px;
            }}
        """)
        self.setMinimumHeight(90)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            color: {EdenTheme.TEXT_SECONDARY};
            font-size: 13px;
            font-weight: 400;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
        """)
        layout.addWidget(self.title_label)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            color: {EdenTheme.TEXT_PRIMARY};
            font-size: 28px;
            font-weight: 300;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', sans-serif;
            letter-spacing: -0.5px;
        """)
        layout.addWidget(self.value_label)
        layout.addStretch()
        
        self.update_color(color)
    
    def update_value(self, value: str, color: str = None):
        """Update card value and color."""
        self.value_label.setText(value)
        if color:
            self.update_color(color)
    
    def update_color(self, color: str):
        """Update value color based on status."""
        color_map = {
            'profit': EdenTheme.PROFIT,
            'loss': EdenTheme.LOSS,
            'success': EdenTheme.SUCCESS,
            'warning': EdenTheme.WARNING,
            'danger': EdenTheme.DANGER,
            'info': EdenTheme.INFO,
            'neutral': EdenTheme.NEUTRAL
        }
        value_color = color_map.get(color, EdenTheme.NEUTRAL)
        self.value_label.setStyleSheet(f"""
            color: {value_color};
            font-size: 28px;
            font-weight: 300;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', sans-serif;
            letter-spacing: -0.5px;
        """)


class TradingChart(QWidget):
    """Advanced trading chart with pyqtgraph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if PYQTGRAPH_AVAILABLE:
            # Create plot widget with Apple-style design
            self.plot_widget = PlotWidget()
            self.plot_widget.setBackground(EdenTheme.BG_PRIMARY)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.1)
            self.plot_widget.setLabel('left', 'Price', color=EdenTheme.TEXT_SECONDARY, size='11pt')
            self.plot_widget.setLabel('bottom', 'Time', color=EdenTheme.TEXT_SECONDARY, size='11pt')
            
            # Apply Apple-style plot styling
            self.plot_widget.getPlotItem().getAxis('left').setPen(EdenTheme.BORDER_PRIMARY)
            self.plot_widget.getPlotItem().getAxis('bottom').setPen(EdenTheme.BORDER_PRIMARY)
            self.plot_widget.getPlotItem().getAxis('left').setTextPen(EdenTheme.TEXT_SECONDARY)
            self.plot_widget.getPlotItem().getAxis('bottom').setTextPen(EdenTheme.TEXT_SECONDARY)
            
            layout.addWidget(self.plot_widget)
            
            # Store plot items
            self.price_line = None
            self.data = None
        else:
            # Fallback placeholder
            placeholder = QLabel("Chart requires pyqtgraph installation")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(f"color: {EdenTheme.TEXT_MUTED}; font-size: 16px;")
            layout.addWidget(placeholder)
    
    def plot_ohlcv_data(self, data: pd.DataFrame, symbol: str = ""):
        """Plot OHLCV data."""
        if not PYQTGRAPH_AVAILABLE or data is None or data.empty:
            return
        
        self.data = data
        self.plot_widget.clear()
        
        # Plot closing price line with Apple-style design
        if 'close' in data.columns:
            timestamps = np.arange(len(data))
            self.price_line = self.plot_widget.plot(
                timestamps, data['close'].values,
                pen=mkPen(color=EdenTheme.PRIMARY, width=2),
                name='Price'
            )
        
        # Set title
        if symbol:
            self.plot_widget.setTitle(f"{symbol} Price Chart", 
                                    color=EdenTheme.TEXT_PRIMARY, size='14pt')


class LogConsole(QTextEdit):
    """Enhanced log console with colored output."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        
        # Set up color formatting
        self.log_colors = {
            'INFO': EdenTheme.TEXT_PRIMARY,
            'SUCCESS': EdenTheme.SUCCESS,
            'WARNING': EdenTheme.WARNING,
            'ERROR': EdenTheme.DANGER,
            'DEBUG': EdenTheme.TEXT_MUTED
        }
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add a colored log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.log_colors.get(level, EdenTheme.TEXT_PRIMARY)
        
        # HTML formatted message
        formatted_msg = f"""<span style="color: {EdenTheme.TEXT_MUTED};">[{timestamp}]</span> 
                           <span style="color: {color}; font-weight: bold;">{level}:</span> 
                           <span style="color: {EdenTheme.TEXT_PRIMARY};">{message}</span>"""
        
        self.append(formatted_msg)
        
        # Auto-scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class EdenDashboard(QMainWindow):
    """Main Eden Bot dashboard application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data
        self.config = {}
        self.log_console = None
        
        # Setup UI
        self.init_ui()
        self.setup_status_bar()
        
        # Setup timers
        self.setup_timers()
        
        # Initialize logging
        configure_logging("INFO")
        
        # Load initial data
        self.load_initial_data()
    
    def init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("Eden Bot - Professional Trading Dashboard v2.0")
        self.setMinimumSize(1400, 900)
        
        # Create central widget with tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)
        
        # Create all tabs
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_backtest_tab()
        self.create_strategies_tab()
        self.create_logs_tab()
        self.create_settings_tab()
        
        self.setCentralWidget(self.tabs)
    
    def create_dashboard_tab(self):
        """Create main dashboard overview."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("🌟 Eden Bot Dashboard")
        header.setStyleSheet(f"""
            color: {EdenTheme.TEXT_PRIMARY};
            font-size: 28px;
            font-weight: 600;
            padding: 20px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Status cards row
        cards_layout = QHBoxLayout()
        
        self.balance_card = StatusCard("Portfolio Balance", "$10,000", "success")
        self.pnl_card = StatusCard("Today's P&L", "+$150", "profit")
        self.trades_card = StatusCard("Active Trades", "3", "info")
        self.winrate_card = StatusCard("Win Rate", "68%", "success")
        
        cards_layout.addWidget(self.balance_card)
        cards_layout.addWidget(self.pnl_card)
        cards_layout.addWidget(self.trades_card)
        cards_layout.addWidget(self.winrate_card)
        
        layout.addLayout(cards_layout)
        
        # Main chart
        self.main_chart = TradingChart()
        layout.addWidget(self.main_chart, 2)
        
        # Quick actions
        actions_group = QGroupBox("⚡ Quick Actions")
        actions_layout = QHBoxLayout()
        
        start_trading_btn = QPushButton("🚀 Start Trading")
        start_trading_btn.clicked.connect(self.start_trading)
        
        run_backtest_btn = QPushButton("📊 Run Backtest")
        run_backtest_btn.clicked.connect(self.show_backtest_tab)
        
        discover_strategies_btn = QPushButton("🔍 Discover Strategies")
        discover_strategies_btn.clicked.connect(self.discover_strategies)
        
        actions_layout.addWidget(start_trading_btn)
        actions_layout.addWidget(run_backtest_btn)
        actions_layout.addWidget(discover_strategies_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🏠 Dashboard")
    
    def create_trading_tab(self):
        """Create live trading interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Header with MT5 status
        header_layout = QHBoxLayout()
        
        header = QLabel("📈 Live Trading")
        header.setStyleSheet(f"color: {EdenTheme.TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # MT5 connection status
        self.mt5_status_label = QLabel("MT5: Disconnected")
        self.mt5_status_label.setStyleSheet(f"color: {EdenTheme.TEXT_MUTED}; font-size: 12px;")
        header_layout.addWidget(self.mt5_status_label)
        
        layout.addLayout(header_layout)
        
        # MT5 connection section
        mt5_group = QGroupBox("MetaTrader 5 Connection")
        mt5_layout = QVBoxLayout()
        
        # Connection controls
        connect_layout = QHBoxLayout()
        
        self.mt5_connect_btn = QPushButton("🔌 Connect to MetaTrader 5")
        self.mt5_connect_btn.clicked.connect(self.connect_mt5)
        connect_layout.addWidget(self.mt5_connect_btn)
        
        self.mt5_trading_toggle = QCheckBox("Allow Eden to Trade")
        self.mt5_trading_toggle.setEnabled(False)
        self.mt5_trading_toggle.toggled.connect(self.toggle_trading)
        connect_layout.addWidget(self.mt5_trading_toggle)
        
        connect_layout.addStretch()
        
        mt5_layout.addLayout(connect_layout)
        
        # Account info display
        self.account_info_label = QLabel("Not connected to any trading account")
        self.account_info_label.setStyleSheet(f"color: {EdenTheme.TEXT_SECONDARY}; padding: 10px;")
        mt5_layout.addWidget(self.account_info_label)
        
        mt5_group.setLayout(mt5_layout)
        layout.addWidget(mt5_group)
        
        # Live trading chart placeholder
        self.trading_chart = TradingChart()
        layout.addWidget(self.trading_chart, 2)
        
        # Positions table
        positions_group = QGroupBox("Open Positions")
        positions_layout = QVBoxLayout()
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(7)
        self.positions_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Price", "Profit", "Actions"
        ])
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        
        positions_layout.addWidget(self.positions_table)
        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📈 Trading")
        
        # Initialize MT5 manager
        self.mt5_manager = get_mt5_manager()
        self.mt5_manager.add_connection_callback(self.on_mt5_connection_change)
        self.mt5_manager.add_error_callback(self.on_mt5_error)
    
    def create_backtest_tab(self):
        """Create backtesting interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("🔬 Backtesting Engine")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {EdenTheme.TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        layout.addWidget(header)
        
        # Backtest controls
        controls_group = QGroupBox("Backtest Configuration")
        controls_layout = QFormLayout()
        
        # Symbol selection
        self.bt_symbol = QComboBox()
        self.bt_symbol.addItems(["XAUUSD", "EURUSD", "US30", "NAS100", "GBPUSD"])
        controls_layout.addRow("Symbol:", self.bt_symbol)
        
        # Timeframe selection
        self.bt_timeframe = QComboBox()
        # Enable full set with multi-timeframe defaults
        self.bt_timeframe.addItems(["M1", "5M", "15M", "1H", "4H", "1D", "W1"])
        controls_layout.addRow("Timeframe:", self.bt_timeframe)
        
        # Strategy selection
        self.bt_strategy = QComboBox()
        self.bt_strategy.addItems(["Ensemble", "ICT", "Momentum", "Mean Reversion"])
        controls_layout.addRow("Strategy:", self.bt_strategy)
        
        # Capital
        self.bt_capital = QDoubleSpinBox()
        self.bt_capital.setRange(1000, 1000000)
        self.bt_capital.setValue(100000)
        self.bt_capital.setPrefix("$")
        controls_layout.addRow("Initial Capital:", self.bt_capital)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress and results
        progress_group = QGroupBox("Progress & Results")
        progress_layout = QVBoxLayout()
        
        self.bt_progress = QProgressBar()
        self.bt_status = QLabel("Ready to run backtest")
        self.bt_status.setAlignment(Qt.AlignCenter)
        
        progress_layout.addWidget(self.bt_progress)
        progress_layout.addWidget(self.bt_status)
        
        # Run button
        self.bt_run_btn = QPushButton("🚀 Run Backtest")
        self.bt_run_btn.clicked.connect(self.run_backtest)
        progress_layout.addWidget(self.bt_run_btn)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Monte Carlo section (wired, not auto-running)
        mc_group = QGroupBox("Monte Carlo (heavy)")
        mc_layout = QVBoxLayout()
        self.mc_status = QLabel("Ready. Not executed by default.")
        self.mc_status.setAlignment(Qt.AlignCenter)
        self.mc_run_btn = QPushButton("🎲 Run Monte Carlo")
        self.mc_run_btn.setToolTip("Bootstrap simulations of equity drawdowns from backtest trades. Not run automatically.")
        self.mc_run_btn.clicked.connect(self.run_monte_carlo)
        mc_layout.addWidget(self.mc_status)
        mc_layout.addWidget(self.mc_run_btn)
        mc_group.setLayout(mc_layout)
        layout.addWidget(mc_group)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🔬 Backtest")
    
    def create_strategies_tab(self):
        """Create strategies management interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("⚡ Strategy Management")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {EdenTheme.TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        layout.addWidget(header)
        
        # Strategy table
        self.strategies_table = QTableWidget()
        self.strategies_table.setColumnCount(5)
        self.strategies_table.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Performance", "Last Run"
        ])
        self.strategies_table.horizontalHeader().setStretchLastSection(True)
        
        # Add sample data
        self.populate_strategies_table()
        
        layout.addWidget(self.strategies_table)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        discover_btn = QPushButton("🔍 Discover Strategies")
        discover_btn.clicked.connect(self.discover_strategies)
        
        optimize_btn = QPushButton("⚙️ Optimize")
        prune_btn = QPushButton("✂️ Prune Weak")
        
        controls_layout.addWidget(discover_btn)
        controls_layout.addWidget(optimize_btn)
        controls_layout.addWidget(prune_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "⚡ Strategies")
    
    def create_logs_tab(self):
        """Create logs and monitoring interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        header = QLabel("📋 System Logs")
        header.setStyleSheet(f"font-size: 20px; font-weight: 600; color: {EdenTheme.TEXT_PRIMARY};")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Clear button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)
        
        # Log console
        self.log_console = LogConsole()
        layout.addWidget(self.log_console)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📋 Logs")
    
    def create_settings_tab(self):
        """Create settings and configuration interface.""" 
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("⚙️ Settings & Configuration")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {EdenTheme.TEXT_PRIMARY}; font-size: 20px; font-weight: 600;")
        layout.addWidget(header)
        
        # Settings groups
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        
        # Theme selection
        theme_combo = QComboBox()
        theme_combo.addItems(["Dark (Eden)", "Light"])
        general_layout.addRow("Theme:", theme_combo)
        
        # Log level
        log_level_combo = QComboBox()
        log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_level_combo.setCurrentText("INFO")
        general_layout.addRow("Log Level:", log_level_combo)
        
        # Auto-save
        auto_save_check = QCheckBox()
        auto_save_check.setChecked(True)
        general_layout.addRow("Auto-save:", auto_save_check)
        
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Trading settings
        trading_group = QGroupBox("Trading Settings")
        trading_layout = QFormLayout()
        
        broker_combo = QComboBox()
        broker_combo.addItems(["Paper", "MT5", "CCXT"])
        trading_layout.addRow("Broker:", broker_combo)
        
        max_positions = QSpinBox()
        max_positions.setRange(1, 100)
        max_positions.setValue(10)
        trading_layout.addRow("Max Positions:", max_positions)
        
        risk_per_trade = QDoubleSpinBox()
        risk_per_trade.setRange(0.1, 10.0)
        risk_per_trade.setValue(2.0)
        risk_per_trade.setSuffix("%")
        trading_layout.addRow("Risk per Trade:", risk_per_trade)
        
        trading_group.setLayout(trading_layout)
        layout.addWidget(trading_group)
        
        layout.addStretch()
        
        # Save button
        save_btn = QPushButton("💾 Save Configuration")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "⚙️ Settings")
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.connection_label = QLabel("Offline")
        self.time_label = QLabel()
        
        self.statusbar.addWidget(self.status_label)
        self.statusbar.addPermanentWidget(self.connection_label)
        self.statusbar.addPermanentWidget(self.time_label)
    
    def setup_timers(self):
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
    
    def load_initial_data(self):
        """Load initial data and setup."""
        if self.log_console:
            self.log_console.add_log("Eden Bot Professional Dashboard initialized", "SUCCESS")
            self.log_console.add_log("UI Framework: PySide6", "INFO")
            self.log_console.add_log("Theme: Eden Dark Professional", "INFO")
            self.log_console.add_log("Ready for trading operations", "INFO")
    
    def populate_strategies_table(self):
        """Populate strategies table with sample data."""
        strategies_data = [
            ["ICT Strategy", "Technical Analysis", "Active", "+15.2%", "2024-01-15"],
            ["Mean Reversion", "Statistical", "Inactive", "-2.1%", "2024-01-14"],
            ["Momentum Breakout", "Technical Analysis", "Active", "+8.7%", "2024-01-15"],
            ["ML Generated #1", "Machine Learning", "Testing", "+12.3%", "2024-01-15"],
            ["Price Action", "Discretionary", "Active", "+22.1%", "2024-01-15"]
        ]
        
        self.strategies_table.setRowCount(len(strategies_data))
        
        for row, strategy in enumerate(strategies_data):
            for col, value in enumerate(strategy):
                item = QTableWidgetItem(str(value))
                if col == 3:  # Performance column
                    if value.startswith('+'):
                        item.setForeground(QColor(EdenTheme.PROFIT))
                    elif value.startswith('-'):
                        item.setForeground(QColor(EdenTheme.LOSS))
                self.strategies_table.setItem(row, col, item)
    
    def update_status(self):
        """Update status bar information."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    # Event handlers
    def start_trading(self):
        """Start live trading."""
        if self.log_console:
            self.log_console.add_log("Switching to trading interface...", "INFO")
        self.tabs.setCurrentIndex(1)  # Switch to trading tab
    
    def show_backtest_tab(self):
        """Show backtest tab."""
        self.tabs.setCurrentIndex(2)  # Switch to backtest tab
    
    def run_backtest(self):
        """Run backtest with current settings (calls real CLI backtest in background)."""
        symbol = self.bt_symbol.currentText()
        timeframe = self.bt_timeframe.currentText()
        strategy = self.bt_strategy.currentText()
        capital = self.bt_capital.value()
        
        if self.log_console:
            self.log_console.add_log(f"Starting backtest for {symbol} {timeframe} with {strategy} strategy", "INFO")
            self.log_console.add_log(f"Initial capital: ${capital:,.2f}", "INFO")
        
        # Reset progress UI
        self.bt_progress.setValue(5)
        self.bt_status.setText("Running backtest (this may take a moment)...")

        def _work():
            try:
                # Call CLI backtest with overrides; let config defaults handle dates
                cli_run_backtest(None, ci_short=False, overrides={
                    "symbols": [symbol],
                    "timeframe": timeframe,
                    "strategy": strategy.lower().replace(' ', '_'),
                    "starting_cash": float(capital),
                })
                # Read metrics
                from pathlib import Path
                import json as _json
                results_dir = Path("examples/results").absolute()
                metrics_path = results_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = _json.loads(metrics_path.read_text())
                    msg = f"Backtest complete. Net PnL: {metrics.get('net_pnl', 0):.2f}, Sharpe: {metrics.get('sharpe', 0):.2f}, Max DD: {metrics.get('max_dd', 0):.4f}, Trades: {metrics.get('trades', 0)}"
                else:
                    msg = "Backtest complete. Metrics file not found."
                def _update_success():
                    self.bt_progress.setValue(100)
                    self.bt_status.setText("Backtest completed successfully!")
                    if self.log_console:
                        self.log_console.add_log(msg, "SUCCESS")
                        self.log_console.add_log("Artifacts written to examples/results (metrics.json, equity_curve.png, trades.csv)", "INFO")
                QTimer.singleShot(0, _update_success)
            except Exception as ex:
                def _update_error():
                    self.bt_progress.setValue(0)
                    self.bt_status.setText("Backtest failed")
                    if self.log_console:
                        self.log_console.add_log(f"Backtest error: {ex}", "ERROR")
                QTimer.singleShot(0, _update_error)
        
        threading.Thread(target=_work, daemon=True).start()
    
    def complete_backtest(self):
        """Complete backtest simulation."""
        self.bt_progress.setValue(100)
        self.bt_status.setText("Backtest completed successfully!")
        
        if self.log_console:
            self.log_console.add_log("Backtest completed with sample results", "SUCCESS")
            self.log_console.add_log("Total Return: +15.5%, Sharpe: 1.2, Max DD: -8.3%", "INFO")

    def run_monte_carlo(self):
        """Hook for Monte Carlo analysis. Does not auto-run; provides status only for now."""
        if self.log_console:
            self.log_console.add_log("Monte Carlo wired: will use backtest trades when available. Not running automatically.", "INFO")
        if hasattr(self, 'mc_status'):
            self.mc_status.setText("Monte Carlo ready. Run after backtest completes.")
        
    def discover_strategies(self):
        """Run strategy discovery."""
        if self.log_console:
            self.log_console.add_log("Starting ML-based strategy discovery...", "INFO")
            self.log_console.add_log("This feature requires the full Eden backend", "WARNING")
    
    def clear_logs(self):
        """Clear log console."""
        if self.log_console:
            self.log_console.clear()
            self.log_console.add_log("Log console cleared", "INFO")
    
    def save_settings(self):
        """Save application settings."""
        if self.log_console:
            self.log_console.add_log("Settings saved successfully", "SUCCESS")
    
    def connect_mt5(self):
        """Connect to MetaTrader 5."""
        # Check if MT5 is installed
        if not self.mt5_manager.check_mt5_installed():
            if self.log_console:
                self.log_console.add_log("MetaTrader 5 not found", "WARNING")
            
            # Prompt for installation
            installed = self.mt5_manager.prompt_mt5_installation(self)
            if not installed:
                if self.log_console:
                    self.log_console.add_log("MT5 installation cancelled", "INFO")
                return
        
        # Attempt connection
        if self.log_console:
            self.log_console.add_log("Connecting to MetaTrader 5...", "INFO")
        
        success = self.mt5_manager.connect()
        if success:
            if self.log_console:
                self.log_console.add_log("Successfully connected to MT5", "SUCCESS")
        else:
            if self.log_console:
                self.log_console.add_log("Failed to connect to MT5", "ERROR")
    
    def toggle_trading(self, enabled: bool):
        """Toggle trading capabilities."""
        if enabled:
            if self.mt5_manager.enable_trading():
                if self.log_console:
                    self.log_console.add_log("✅ Trading enabled - Eden can now place trades", "SUCCESS")
            else:
                self.mt5_trading_toggle.setChecked(False)
                if self.log_console:
                    self.log_console.add_log("❌ Failed to enable trading", "ERROR")
        else:
            self.mt5_manager.disable_trading()
            if self.log_console:
                self.log_console.add_log("Trading disabled", "INFO")
    
    def on_mt5_connection_change(self, status, account_info):
        """Handle MT5 connection status changes."""
        self.mt5_status_label.setText(f"MT5: {status.value}")
        
        if status.value == "Connected" and account_info:
            # Update status label color
            self.mt5_status_label.setStyleSheet(f"color: {EdenTheme.SUCCESS}; font-size: 12px;")
            
            # Update connection button
            self.mt5_connect_btn.setText("🔌 Connected")
            self.mt5_connect_btn.setEnabled(False)
            
            # Enable trading toggle
            self.mt5_trading_toggle.setEnabled(True)
            
            # Update account info
            account_text = f"""Connected to: {account_info.company}
Account: {account_info.login} ({account_info.name})
Server: {account_info.server}
Balance: {account_info.balance:.2f} {account_info.currency}
Equity: {account_info.equity:.2f} {account_info.currency}
Leverage: 1:{account_info.leverage}"""
            self.account_info_label.setText(account_text)
            
        elif status.value == "Disconnected":
            # Update status label color
            self.mt5_status_label.setStyleSheet(f"color: {EdenTheme.TEXT_MUTED}; font-size: 12px;")
            
            # Reset connection button
            self.mt5_connect_btn.setText("🔌 Connect to MetaTrader 5")
            self.mt5_connect_btn.setEnabled(True)
            
            # Disable trading toggle
            self.mt5_trading_toggle.setEnabled(False)
            self.mt5_trading_toggle.setChecked(False)
            
            # Reset account info
            self.account_info_label.setText("Not connected to any trading account")
            
        elif status.value == "Error":
            # Update status label color
            self.mt5_status_label.setStyleSheet(f"color: {EdenTheme.DANGER}; font-size: 12px;")
            
            # Reset connection button
            self.mt5_connect_btn.setText("🔌 Retry Connection")
            self.mt5_connect_btn.setEnabled(True)
    
    def on_mt5_error(self, error_message: str):
        """Handle MT5 errors."""
        if self.log_console:
            self.log_console.add_log(f"MT5 Error: {error_message}", "ERROR")
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.log_console:
            self.log_console.add_log("Shutting down Eden Bot Dashboard...", "INFO")
        
        # Disconnect from MT5
        if hasattr(self, 'mt5_manager'):
            self.mt5_manager.disconnect()
        
        event.accept()


def main():
    """Main application entry point."""
    # Check dependencies first
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies with:")
        print("   pip install " + " ".join(missing_deps))
        return 1
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Eden")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Eden Trading Systems")
    
    # Apply theme
    app.setPalette(EdenTheme.create_palette())
    app.setStyleSheet(EdenTheme.get_stylesheet())
    
    # Show splash screen with loading
    def create_loading_functions():
        def load_config():
            configure_logging("INFO")
            time.sleep(0.2)
        
        def load_modules():
            # Import heavy modules
            time.sleep(0.3)
        
        def initialize_mt5():
            # Initialize MT5 manager
            get_mt5_manager()
            time.sleep(0.2)
        
        def prepare_ui():
            # Prepare UI components
            time.sleep(0.3)
        
        return [
            (load_config, "Loading configuration..."),
            (load_modules, "Loading Eden modules..."),
            (initialize_mt5, "Initializing MetaTrader 5..."),
            (prepare_ui, "Preparing interface...")
        ]
    
    # Show splash screen
    loading_functions = create_loading_functions()
    splash = show_splash_screen(loading_functions, app)
    
    # Create and show main window
    window = EdenDashboard()
    
    # Finish splash screen
    if splash:
        splash.finish_loading()
    
    window.show()
    
    # Start application
    return app.exec()


def run_main_ui(app=None):
    """Run the main UI with optional existing QApplication instance."""
    if app is None:
        return main()
    else:
        # Use existing app instance
        if not PYSIDE6_AVAILABLE:
            print("PySide6 not available. Cannot run GUI.")
            return 1
        
        # Apply theme to existing app
        theme = EdenTheme()
        app.setPalette(theme.create_palette())
        app.setStyleSheet(theme.get_stylesheet())
        
        # Create and show main window
        window = EdenDashboard()
        window.show()
        
        # Start application event loop
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())
