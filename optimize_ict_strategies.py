#!/usr/bin/env python3
"""
ICT Strategy Optimization Framework

Comprehensive quant analysis tool for optimizing ICT strategies:
- Extended backtesting (6+ months)
- Parameter grid search
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Portfolio optimization
- Monte Carlo simulation
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.ict_strategies import ICTStrategyBot, Bar
from trading.models import Trade

@dataclass
class RiskMetrics:
    """Comprehensive risk and performance metrics."""
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_duration: float
    expectancy: float
    
def fetch_extended_data(symbol: str, months: int = 6) -> Optional[pd.DataFrame]:
    """Fetch extended historical data."""
    if not mt5.initialize():
        print(f"MT5 initialize() failed")
        return None
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        mt5.shutdown()
        return None
    
    # Use H1 instead of M1 for better historical data availability
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol} from {start_date} to {end_date}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Fetched {len(df)} H1 bars")
    return df

def calculate_risk_metrics(trades: List[Dict], initial_capital: float = 100000) -> RiskMetrics:
    """Calculate comprehensive risk metrics."""
    if not trades:
        return None
    
    # Basic metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    
    sum_wins = sum(t['pnl'] for t in wins)
    sum_losses = abs(sum(t['pnl'] for t in losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0
    
    # Calculate equity curve
    equity = initial_capital
    equity_curve = [equity]
    returns = []
    
    for trade in trades:
        equity += trade['pnl'] * initial_capital  # Convert pips to dollars (simplified)
        equity_curve.append(equity)
        ret = trade['pnl'] / equity_curve[-2] if equity_curve[-2] > 0 else 0
        returns.append(ret)
    
    # Drawdown calculation
    peak = equity_curve[0]
    max_dd = 0
    max_dd_pct = 0
    
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Sortino Ratio (only downside deviation)
    downside_returns = [r for r in returns if r < 0]
    if downside_returns:
        downside_std = np.std(downside_returns)
        sortino = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        sortino = sharpe
    
    # Calmar Ratio (return / max drawdown)
    total_return =  (equity_curve[-1] - initial_capital) / initial_capital
    calmar = (total_return / (max_dd_pct / 100)) if max_dd_pct > 0 else 0
    
    # Trade duration
    durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in trades]
    avg_duration = np.mean(durations) if durations else 0
    
    # Expectancy
    prob_win = len(wins) / len(trades) if trades else 0
    prob_loss = len(losses) / len(trades) if trades else 0
    expectancy = (prob_win * avg_win) - (prob_loss * abs(avg_loss))
    
    return RiskMetrics(
        total_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_trade_duration=avg_duration,
        expectancy=expectancy
    )

def optimize_strategy_parameters(
    symbol: str,
    strategy_name: str,
    df: pd.DataFrame,
    param_grid: Dict
) -> List[Tuple[Dict, RiskMetrics]]:
    """Grid search optimization over parameter space."""
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nOptimizing {strategy_name} on {symbol}")
    print(f"Testing {len(combinations)} parameter combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # Run backtest with these parameters
        trades = run_backtest_with_params(symbol, strategy_name, df, params)
        
        if trades and len(trades) > 10:  # Minimum trade threshold
            metrics = calculate_risk_metrics(trades)
            results.append((params, metrics))
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(combinations)}")
    
    # Sort by Sharpe ratio (risk-adjusted returns)
    results.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
    
    return results

def run_backtest_with_params(
    symbol: str,
    strategy_name: str,
    df: pd.DataFrame,
    params: Dict
) -> List[Dict]:
    """Run backtest with specific parameters."""
    bot = ICTStrategyBot()
    trades = []
    open_trade = None
    
    for idx, row in df.iterrows():
        bar = Bar(
            time=row['time'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['tick_volume']
        )
        
        bot.bars.append(bar)
        bot.engine.update_structure(bot.bars)
        
        if open_trade is None:
            # Apply parameters to strategy
            signal = get_signal_with_params(bot, bar, symbol, strategy_name, params)
            
            if signal:
                open_trade = signal
                open_trade.entry_time = bar.time
        else:
            # Check exit
            hit_tp = False
            hit_sl = False
            
            if open_trade.direction == "LONG":
                hit_tp = bar.high >= open_trade.tp
                hit_sl = bar.low <= open_trade.sl
            else:
                hit_tp = bar.low <= open_trade.tp
                hit_sl = bar.high >= open_trade.sl
            
            if hit_tp or hit_sl:
                exit_price = open_trade.tp if hit_tp else open_trade.sl
                pnl = (exit_price - open_trade.entry_price) if open_trade.direction == "LONG" else (open_trade.entry_price - exit_price)
                
                trades.append({
                    'entry_time': open_trade.entry_time,
                    'exit_time': bar.time,
                    'direction': open_trade.direction,
                    'entry_price': open_trade.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': 'TP' if hit_tp else 'SL'
                })
                open_trade = None
    
    return trades

def get_signal_with_params(bot, bar, symbol, strategy_name, params):
    """Get signal with modified parameters."""
    if strategy_name == "Silver Bullet":
        signal = bot.run_2023_silver_bullet(bar, symbol)
        if signal:
            # Modify R:R based on params
            rr = params.get('rr_ratio', 2.0)
            risk = abs(signal.entry_price - signal.sl)
            if signal.direction == "LONG":
                signal.tp = signal.entry_price + (risk * rr)
            else:
                signal.tp = signal.entry_price - (risk * rr)
        return signal
    
    elif strategy_name == "Unicorn":
        signal = bot.run_2024_unicorn(bar, symbol)
        if signal:
            rr = params.get('rr_ratio', 3.0)
            risk = abs(signal.entry_price - signal.sl)
            if signal.direction == "LONG":
                signal.tp = signal.entry_price + (risk * rr)
            else:
                signal.tp = signal.entry_price - (risk * rr)
        return signal
    
    return None

def print_optimization_results(results: List[Tuple[Dict, RiskMetrics]], top_n: int = 5):
    """Print top optimization results."""
    print(f"\n{'='*120}")
    print(f"TOP {top_n} PARAMETER COMBINATIONS (by Sharpe Ratio)")
    print(f"{'='*120}")
    
    for i, (params, metrics) in enumerate(results[:top_n]):
        print(f"\n#{i+1}: {params}")
        print(f"  Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1f}% | PF: {metrics.profit_factor:.2f}")
        print(f"  Total P&L: {metrics.total_pnl:.5f} | Expectancy: {metrics.expectancy:.5f}")
        print(f"  Sharpe: {metrics.sharpe_ratio:.2f} | Sortino: {metrics.sortino_ratio:.2f} | Calmar: {metrics.calmar_ratio:.2f}")
        print(f"  Max DD: {metrics.max_drawdown_pct:.2f}% | Avg Trade: {metrics.avg_trade_duration:.0f} min")

if __name__ == "__main__":
    print("ICT Strategy Optimization Framework")
    print("=====================================\n")
    
    # Configuration
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    MONTHS = 6
    
    # Parameter grids for optimization
    silver_bullet_params = {
        'rr_ratio': [2.0, 2.5, 3.0, 3.5]
    }
    
    unicorn_params = {
        'rr_ratio': [2.5, 3.0, 3.5, 4.0, 4.5]
    }
    
    all_optimization_results = {}
    
    # Run optimization for each symbol and strategy
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"FETCHING DATA FOR {symbol}")
        print(f"{'='*80}")
        
        df = fetch_extended_data(symbol, MONTHS)
        if df is None:
            print(f"Failed to fetch data for {symbol}")
            continue
        
        print(f"Fetched {len(df)} bars ({MONTHS} months)")
        
        # Optimize Silver Bullet
        sb_results = optimize_strategy_parameters(symbol, "Silver Bullet", df, silver_bullet_params)
        all_optimization_results[f"SilverBullet_{symbol}"] = sb_results
        print_optimization_results(sb_results, top_n=3)
        
        # Optimize Unicorn
        un_results = optimize_strategy_parameters(symbol, "Unicorn", df, unicorn_params)
        all_optimization_results[f"Unicorn_{symbol}"] = un_results
        print_optimization_results(un_results, top_n=3)
    
    # Save results to JSON
    results_file = "ict_optimization_results.json"
    summary = {}
    for key, results in all_optimization_results.items():
        if results:
            best = results[0]
            summary[key] = {
                'params': best[0],
                'sharpe': best[1].sharpe_ratio,
                'sortino': best[1].sortino_ratio,
                'calmar': best[1].calmar_ratio,
                'max_dd_pct': best[1].max_drawdown_pct,
                'win_rate': best[1].win_rate,
                'profit_factor': best[1].profit_factor,
                'total_trades': best[1].total_trades,
                'total_pnl': best[1].total_pnl
            }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
