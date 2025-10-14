#!/usr/bin/env python3
"""
Eden VIX 100 Trading Bot
A professional automated trading bot for VIX 100 via MetaTrader 5

Features:
- Automated VIX 100 trading strategies
- MetaTrader 5 integration
- Risk management
- Real-time market analysis
- Position management
"""

import sys
import os
import time
import logging
import threading
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Add the worker directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "worker" / "python"))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✅ MetaTrader5 module loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    print("❌ MetaTrader5 module not found. Install with: pip install MetaTrader5")

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install pandas numpy")

class VIX100TradingBot:
    """Eden VIX 100 Automated Trading Bot"""
    
    def __init__(self):
        self.symbol = "Volatility 100 Index"  # VIX 100 symbol in MT5
        self.lot_size = 0.01  # Start with small positions
        self.max_positions = 3  # Maximum open positions
        self.stop_loss_pips = 50  # Stop loss in pips
        self.take_profit_pips = 100  # Take profit in pips
        
        self.running = False
        self.positions = []
        self.account_info = None
        self.logger = self._setup_logging()
        
        # Trading strategy parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.moving_avg_period = 20
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('eden_vix100_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('EdenVIX100Bot')
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 module not available")
            return False
            
        if not mt5.initialize():
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return False
            
        self.account_info = account_info._asdict()
        self.logger.info(f"Connected to MT5 - Account: {self.account_info['login']}")
        self.logger.info(f"Balance: {self.account_info['balance']}")
        self.logger.info(f"Equity: {self.account_info['equity']}")
        
        # Verify VIX 100 symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {self.symbol} not found")
            # Try alternative symbol names for VIX 100
            alt_symbols = ["VIX100", "Volatility100", "Vol100", "VIX 100"]
            for alt in alt_symbols:
                if mt5.symbol_info(alt):
                    self.symbol = alt
                    self.logger.info(f"Using alternative symbol: {alt}")
                    break
            else:
                return False
        
        return True
    
    def disconnect_mt5(self):
        """Disconnect from MetaTrader 5"""
        if MT5_AVAILABLE:
            mt5.shutdown()
            self.logger.info("Disconnected from MT5")
    
    def get_market_data(self, timeframe=mt5.TIMEFRAME_M1, count=100) -> Optional[pd.DataFrame]:
        """Get market data for VIX 100"""
        if not MT5_AVAILABLE:
            return None
            
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None:
            self.logger.error(f"Failed to get market data: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_market(self) -> Dict[str, Any]:
        """Analyze market conditions and generate trading signals"""
        df = self.get_market_data()
        if df is None or len(df) < self.rsi_period:
            return {"signal": "WAIT", "reason": "Insufficient data"}
        
        # Calculate technical indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['sma'] = df['close'].rolling(window=self.moving_avg_period).mean()
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        
        analysis = {
            "price": current_price,
            "rsi": current_rsi,
            "sma": current_sma,
            "signal": "WAIT",
            "reason": "No clear signal"
        }
        
        # Generate trading signals based on RSI and moving average
        if current_rsi < self.rsi_oversold and current_price < current_sma:
            analysis["signal"] = "BUY"
            analysis["reason"] = f"RSI oversold ({current_rsi:.2f}) and price below SMA"
        elif current_rsi > self.rsi_overbought and current_price > current_sma:
            analysis["signal"] = "SELL"
            analysis["reason"] = f"RSI overbought ({current_rsi:.2f}) and price above SMA"
        
        return analysis
    
    def get_current_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not MT5_AVAILABLE:
            return []
            
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
            
        return [pos._asdict() for pos in positions]
    
    def place_order(self, order_type: str, volume: float = None) -> bool:
        """Place a trading order"""
        if not MT5_AVAILABLE:
            return False
            
        if volume is None:
            volume = self.lot_size
            
        # Get current positions count
        current_positions = len(self.get_current_positions())
        if current_positions >= self.max_positions:
            self.logger.info(f"Max positions ({self.max_positions}) reached")
            return False
        
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.logger.error("Failed to get current price")
            return False
        
        if order_type == "BUY":
            price = tick.ask
            sl = price - (self.stop_loss_pips * mt5.symbol_info(self.symbol).point)
            tp = price + (self.take_profit_pips * mt5.symbol_info(self.symbol).point)
            order_type_mt5 = mt5.ORDER_TYPE_BUY
        else:  # SELL
            price = tick.bid
            sl = price + (self.stop_loss_pips * mt5.symbol_info(self.symbol).point)
            tp = price - (self.take_profit_pips * mt5.symbol_info(self.symbol).point)
            order_type_mt5 = mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Eden VIX100 Bot {order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None:
            self.logger.error(f"Order failed: {mt5.last_error()}")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.retcode}")
            return False
        
        self.logger.info(f"Order placed: {order_type} {volume} lots at {price}")
        return True
    
    def manage_positions(self):
        """Monitor and manage existing positions"""
        positions = self.get_current_positions()
        
        for pos in positions:
            # Log position info
            pnl = pos['profit']
            self.logger.info(f"Position {pos['ticket']}: P&L = {pnl:.2f}")
            
            # Here you could add additional position management logic
            # such as trailing stops, partial closes, etc.
    
    def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Analyze market
            analysis = self.analyze_market()
            self.logger.info(f"Market Analysis - {analysis['reason']}")
            
            # Manage existing positions
            self.manage_positions()
            
            # Execute trading signal
            if analysis["signal"] in ["BUY", "SELL"]:
                self.logger.info(f"Signal: {analysis['signal']} - {analysis['reason']}")
                success = self.place_order(analysis["signal"])
                if success:
                    self.logger.info(f"Order executed successfully")
                else:
                    self.logger.error(f"Order execution failed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def start(self):
        """Start the trading bot"""
        if not DEPENDENCIES_OK:
            self.logger.error("Missing dependencies. Cannot start bot.")
            return
            
        self.logger.info("🚀 Starting Eden VIX 100 Trading Bot")
        
        if not self.connect_mt5():
            self.logger.error("Failed to connect to MT5. Exiting.")
            return
        
        self.running = True
        
        try:
            while self.running:
                self.run_trading_cycle()
                
                # Wait before next cycle (60 seconds)
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("🛑 Stopping Eden VIX 100 Trading Bot")
        self.running = False
        self.disconnect_mt5()
        self.logger.info("Bot stopped successfully")


def main():
    """Main entry point"""
    print("=" * 60)
    print("🌟 EDEN VIX 100 TRADING BOT")
    print("=" * 60)
    print("Professional automated trading for VIX 100")
    print("Press Ctrl+C to stop the bot")
    print("=" * 60)
    
    bot = VIX100TradingBot()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal...")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the bot
    bot.start()


if __name__ == "__main__":
    main()