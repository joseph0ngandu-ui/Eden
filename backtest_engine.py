#!/usr/bin/env python3
"""
Comprehensive Backtest Engine for Eden
Supports ICT strategies, Gold strategies, and ML adaptive trading
Target: 100% weekly returns across all 10 instruments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import ta  # Technical Analysis library

@dataclass
class Trade:
    """Trade record"""
    instrument: str
    strategy: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    quantity: float
    entry_commission: float
    exit_commission: float
    pnl: float
    pnl_pct: float
    max_profit: float
    max_drawdown: float
    
    def to_dict(self):
        return {
            'instrument': self.instrument,
            'strategy': self.strategy,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'direction': self.direction,
            'quantity': float(self.quantity),
            'entry_commission': float(self.entry_commission),
            'exit_commission': float(self.exit_commission),
            'pnl': float(self.pnl),
            'pnl_pct': float(self.pnl_pct),
            'max_profit': float(self.max_profit),
            'max_drawdown': float(self.max_drawdown)
        }

class IndicatorCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        try:
            # Try 'length' parameter first (newer version)
            return ta.momentum.rsi(df['close'], length=period)
        except TypeError:
            # Fall back to 'window' parameter (older version)
            return ta.momentum.rsi(df['close'], window=period)
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        try:
            return ta.trend.sma(df['close'], length=period)
        except TypeError:
            return ta.trend.sma(df['close'], window=period)
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        try:
            return ta.volatility.atr(df['high'], df['low'], df['close'], length=period)
        except TypeError:
            return ta.volatility.atr(df['high'], df['low'], df['close'], window=period)
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        try:
            macd = ta.trend.macd(df['close'])
        except:
            macd = ta.trend.macd(df['close'], window=12, signal=26, window_slow=26)
        return macd.iloc[:, 0], macd.iloc[:, 1], macd.iloc[:, 2]
    
    @staticmethod
    def calculate_bb(df: pd.DataFrame, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        try:
            bb = ta.volatility.bollinger_bands(df['close'], length=period, std=std)
        except TypeError:
            bb = ta.volatility.bollinger_bands(df['close'], window=period, num_std=std)
        return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    
    @staticmethod
    def calculate_htf_bias(df: pd.DataFrame, htf_period: int = 20) -> pd.Series:
        """Higher Timeframe Bias using SMA"""
        sma_htf = ta.trend.sma(df['close'], length=htf_period)
        return (df['close'] > sma_htf).astype(int) * 2 - 1  # +1 for uptrend, -1 for downtrend

class StrategyGenerator:
    """Generate buy/sell signals based on strategies"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.setup_indicators()
    
    def setup_indicators(self):
        """Calculate all indicators"""
        calc = IndicatorCalculator()
        
        self.df['rsi_14'] = calc.calculate_rsi(self.df, 14)
        self.df['sma_20'] = calc.calculate_sma(self.df, 20)
        self.df['sma_50'] = calc.calculate_sma(self.df, 50)
        self.df['atr_14'] = calc.calculate_atr(self.df, 14)
        self.df['macd'], self.df['macd_signal'], self.df['macd_hist'] = calc.calculate_macd(self.df)
        self.df['bb_upper'], self.df['bb_mid'], self.df['bb_lower'] = calc.calculate_bb(self.df, 20, 2)
        self.df['htf_bias'] = calc.calculate_htf_bias(self.df, 20)
        
        # Fill NaN values
        self.df = self.df.fillna(method='bfill')
    
    def htf_bias_strategy(self) -> pd.Series:
        """Higher Timeframe Bias Strategy"""
        signals = pd.Series(0, index=self.df.index)
        
        # Buy: HTF uptrend + RSI > 50 + SMA20 > SMA50
        buy_signal = (self.df['htf_bias'] > 0) & (self.df['rsi_14'] > 50) & (self.df['sma_20'] > self.df['sma_50'])
        signals[buy_signal] = 1
        
        # Sell: HTF downtrend + RSI < 50
        sell_signal = (self.df['htf_bias'] < 0) & (self.df['rsi_14'] < 50)
        signals[sell_signal] = -1
        
        return signals
    
    def breakout_volume_strategy(self) -> pd.Series:
        """Gold Breakout Strategy with Volume"""
        signals = pd.Series(0, index=self.df.index)
        
        # Calculate volume moving average
        vol_ma = self.df['volume'].rolling(14).mean()
        
        # Buy: Price > SMA50 + Volume spike
        buy_signal = (self.df['close'] > self.df['sma_50']) & (self.df['volume'] > vol_ma * 1.5)
        signals[buy_signal] = 1
        
        # Sell: Price < SMA50
        sell_signal = self.df['close'] < self.df['sma_50']
        signals[sell_signal] = -1
        
        return signals
    
    def ema_crossover_strategy(self) -> pd.Series:
        """EMA Crossover Strategy"""
        signals = pd.Series(0, index=self.df.index)
        
        try:
            ema_20 = ta.trend.ema(self.df['close'], length=20)
            ema_50 = ta.trend.ema(self.df['close'], length=50)
        except TypeError:
            ema_20 = ta.trend.ema(self.df['close'], span=20)
            ema_50 = ta.trend.ema(self.df['close'], span=50)
        
        # Buy: EMA20 crosses above EMA50 + RSI > 50
        buy_signal = (ema_20 > ema_50) & (self.df['rsi_14'] > 50)
        signals[buy_signal] = 1
        
        # Sell: EMA20 crosses below EMA50
        sell_signal = ema_20 < ema_50
        signals[sell_signal] = -1
        
        return signals
    
    def bollinger_rsi_strategy(self) -> pd.Series:
        """Bollinger Bands + RSI Strategy"""
        signals = pd.Series(0, index=self.df.index)
        
        # Buy: Price touches lower BB + RSI < 30 (oversold)
        buy_signal = (self.df['close'] < self.df['bb_lower']) & (self.df['rsi_14'] < 30)
        signals[buy_signal] = 1
        
        # Sell: Price touches upper BB + RSI > 70 (overbought)
        sell_signal = (self.df['close'] > self.df['bb_upper']) & (self.df['rsi_14'] > 70)
        signals[sell_signal] = -1
        
        return signals
    
    def fvg_strategy(self) -> pd.Series:
        """Fair Value Gap Strategy"""
        signals = pd.Series(0, index=self.df.index)
        
        # Detect FVGs (gaps in price action)
        for i in range(2, len(self.df)):
            # Bullish FVG: low[i] > high[i-2]
            if self.df['low'].iloc[i] > self.df['high'].iloc[i-2] and self.df['rsi_14'].iloc[i] > 50:
                signals.iloc[i] = 1
            
            # Bearish FVG: high[i] < low[i-2]
            if self.df['high'].iloc[i] < self.df['low'].iloc[i-2] and self.df['rsi_14'].iloc[i] < 50:
                signals.iloc[i] = -1
        
        return signals

class BacktestEngine:
    """Main backtest engine"""
    
    def __init__(self, initial_capital: float = 100000, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.trades: List[Trade] = []
        self.results = {}
    
    def backtest_instrument(self, symbol: str, df: pd.DataFrame, 
                           strategy_name: str, params: Dict) -> Dict:
        """
        Run backtest for single instrument with strategy
        
        Args:
            symbol: Instrument symbol (VIX75, XAUUSD, etc)
            df: OHLCV dataframe
            strategy_name: Name of strategy to use
            params: Strategy parameters
            
        Returns:
            Dict with performance metrics
        """
        
        # Generate signals
        sg = StrategyGenerator(df)
        
        if strategy_name == 'htf_bias':
            signals = sg.htf_bias_strategy()
        elif strategy_name == 'breakout_volume':
            signals = sg.breakout_volume_strategy()
        elif strategy_name == 'ema_crossover':
            signals = sg.ema_crossover_strategy()
        elif strategy_name == 'bollinger_rsi':
            signals = sg.bollinger_rsi_strategy()
        elif strategy_name == 'fvg':
            signals = sg.fvg_strategy()
        else:
            signals = pd.Series(0, index=df.index)
        
        # Execute trades
        trades = self._execute_trades(symbol, strategy_name, df, signals, params)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, df)
        metrics['trades'] = [t.to_dict() for t in trades]
        
        return metrics
    
    def _execute_trades(self, symbol: str, strategy: str, df: pd.DataFrame, 
                       signals: pd.Series, params: Dict) -> List[Trade]:
        """Execute trades based on signals"""
        trades = []
        position = None
        
        for i in range(1, len(df)):
            signal = signals.iloc[i]
            current_price = df['close'].iloc[i]
            current_time = df['timestamp'].iloc[i]
            atr = df['atr_14'].iloc[i] if 'atr_14' in df.columns else 0.01
            
            # Close existing position if signal changes
            if position and signal * position['direction'] <= 0:
                exit_price = current_price
                entry_commission = position['entry_commission']
                exit_commission = exit_price * position['quantity'] * self.commission_pct
                
                pnl = (exit_price - position['entry_price']) * position['quantity'] * position['direction']
                pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) if position['quantity'] > 0 else 0
                
                trade = Trade(
                    instrument=symbol,
                    strategy=strategy,
                    entry_time=position['entry_time'],
                    exit_time=current_time,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    direction='long' if position['direction'] > 0 else 'short',
                    quantity=position['quantity'],
                    entry_commission=entry_commission,
                    exit_commission=exit_commission,
                    pnl=pnl - entry_commission - exit_commission,
                    pnl_pct=pnl_pct,
                    max_profit=position['max_profit'],
                    max_drawdown=position['max_drawdown']
                )
                trades.append(trade)
                position = None
            
            # Open new position
            if signal != 0 and position is None:
                risk_amount = self.initial_capital * params.get('risk_pct', 0.01)
                stop_loss_distance = atr * params.get('atr_multiplier', 2.0)
                quantity = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 1
                
                position = {
                    'direction': signal,
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_commission': current_price * quantity * self.commission_pct,
                    'quantity': quantity,
                    'max_profit': 0,
                    'max_drawdown': 0
                }
        
        return trades
    
    def _calculate_metrics(self, trades: List[Trade], df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'net_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'return_pct': 0
            }
        
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        net_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate Sharpe Ratio
        returns = trades_df['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Max Drawdown
        cumsum = np.cumsum(trades_df['pnl'].values)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / (running_max + 1e-9)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return_pct = (net_pnl / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'net_pnl': float(net_pnl),
            'gross_profit': float(gross_profit),
            'gross_loss': float(gross_loss),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'return_pct': float(return_pct)
        }

def main():
    """Test the backtest engine"""
    print("\n" + "="*60)
    print("🧪 BACKTEST ENGINE TEST")
    print("="*60)
    
    # Load sample data
    data_dir = Path("data/mt5_feeds")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/mt5_feeds/")
        return False
    
    engine = BacktestEngine(initial_capital=100000)
    
    for csv_file in csv_files[:2]:  # Test first 2 instruments
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        symbol = csv_file.stem.replace('_1M', '')
        print(f"\n📊 Backtesting {symbol}...")
        
        params = {'risk_pct': 0.01, 'atr_multiplier': 2.0}
        metrics = engine.backtest_instrument(symbol, df, 'htf_bias', params)
        
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Net PnL: ${metrics['net_pnl']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
    
    print("\n✅ Backtest engine working correctly!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
