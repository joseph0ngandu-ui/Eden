#!/usr/bin/env python3
"""
Trade Journal Exporter

Export each trade (timestamp, symbol, entry/exit price, PnL) to CSV for later analytics.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TradeJournal:
    """Manages trade history export to CSV."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize trade journal.
        
        Args:
            log_dir: Directory to store trade history CSV
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "trade_history.csv"
        self.trades: List[Dict[str, Any]] = []
        self.load_existing()
    
    def load_existing(self) -> None:
        """Load existing trade history from CSV."""
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                self.trades = df.to_dict('records')
                logger.info(f"Loaded {len(self.trades)} existing trades from {self.csv_path}")
            except Exception as e:
                logger.warning(f"Could not load existing trades: {e}")
    
    def add_trade(
        self,
        symbol: str,
        trade_type: str,
        entry_price: float,
        entry_time: datetime,
        exit_price: float = None,
        exit_time: datetime = None,
        volume: float = 1.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        notes: str = ""
    ) -> None:
        """
        Add a trade to the journal.
        
        Args:
            symbol: Trading symbol
            trade_type: "BUY" or "SELL"
            entry_price: Entry price
            entry_time: Entry datetime
            exit_price: Exit price (None if still open)
            exit_time: Exit datetime (None if still open)
            volume: Trade volume in lots
            commission: Commission paid
            slippage: Slippage in pips
            notes: Additional notes about the trade
        """
        pnl = None
        pnl_percent = None
        
        if exit_price is not None:
            if trade_type == "BUY":
                pnl = (exit_price - entry_price) * volume * 100000  # Assuming 100k unit
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl = (entry_price - exit_price) * volume * 100000
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            
            pnl -= commission
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': trade_type,
            'volume': volume,
            'entry_price': entry_price,
            'entry_time': entry_time.isoformat() if entry_time else None,
            'exit_price': exit_price,
            'exit_time': exit_time.isoformat() if exit_time else None,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'commission': commission,
            'slippage': slippage,
            'status': 'CLOSED' if exit_price is not None else 'OPEN',
            'notes': notes,
        }
        
        self.trades.append(trade)
        logger.debug(f"Trade added: {symbol} {trade_type} @ {entry_price}")
    
    def close_trade(
        self,
        entry_index: int,
        exit_price: float,
        exit_time: datetime = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        notes: str = ""
    ) -> None:
        """
        Close an open trade.
        
        Args:
            entry_index: Index of the trade to close
            exit_price: Exit price
            exit_time: Exit time
            commission: Commission paid
            slippage: Slippage in pips
            notes: Additional notes
        """
        if entry_index < 0 or entry_index >= len(self.trades):
            logger.error(f"Invalid trade index: {entry_index}")
            return
        
        trade = self.trades[entry_index]
        
        if exit_time is None:
            exit_time = datetime.now()
        
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time.isoformat()
        trade['commission'] = commission
        trade['slippage'] = slippage
        trade['status'] = 'CLOSED'
        
        # Calculate PnL
        entry_price = trade['entry_price']
        volume = trade['volume']
        trade_type = trade['type']
        
        if trade_type == "BUY":
            pnl = (exit_price - entry_price) * volume * 100000
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL
            pnl = (entry_price - exit_price) * volume * 100000
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
        
        trade['pnl'] = pnl - commission
        trade['pnl_percent'] = pnl_percent
        
        if notes:
            trade['notes'] = notes
        
        logger.debug(f"Trade closed: {trade['symbol']} {trade_type} @ {exit_price}, PnL: {trade['pnl']:.2f}")
    
    def export_csv(self, filepath: str = None) -> str:
        """
        Export trades to CSV.
        
        Args:
            filepath: Custom filepath (default: logs/trade_history.csv)
            
        Returns:
            Path to exported CSV file
        """
        if filepath is None:
            filepath = str(self.csv_path)
        
        try:
            df = pd.DataFrame(self.trades)
            
            # Sort by entry time
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                df = df.sort_values('entry_time', ascending=False)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Trade history exported: {filepath} ({len(self.trades)} trades)")
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting trades to CSV: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {}
        
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        stats = {
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(df[df['status'] == 'OPEN']),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0,
            'total_pnl': closed_trades['pnl'].sum(),
            'avg_pnl': closed_trades['pnl'].mean(),
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'max_profit': closed_trades['pnl'].max(),
            'min_profit': closed_trades['pnl'].min(),
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0,
        }
        
        return stats
    
    def print_summary(self) -> None:
        """Print trade summary to logger."""
        stats = self.get_statistics()
        if not stats:
            logger.info("No trade statistics available")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRADE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trades: {stats.get('total_trades', 0)}")
        logger.info(f"Closed Trades: {stats.get('closed_trades', 0)}")
        logger.info(f"Open Trades: {stats.get('open_trades', 0)}")
        logger.info(f"Winning Trades: {stats.get('winning_trades', 0)}")
        logger.info(f"Losing Trades: {stats.get('losing_trades', 0)}")
        logger.info(f"Win Rate: {stats.get('win_rate', 0):.2f}%")
        logger.info(f"Total PnL: {stats.get('total_pnl', 0):.2f}")
        logger.info(f"Avg PnL: {stats.get('avg_pnl', 0):.2f}")
        logger.info(f"Max Profit: {stats.get('max_profit', 0):.2f}")
        logger.info(f"Min Profit: {stats.get('min_profit', 0):.2f}")
        logger.info(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
        logger.info(f"{'='*60}\n")