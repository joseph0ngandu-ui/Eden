#!/usr/bin/env python3
"""
Eden Optimal Micro Account Runner
Executes the optimal configuration for $15 micro accounts based on backtesting results.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Ensure 'eden' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker' / 'python'))

from run_backtests import run_all_backtests, BacktestConfig
from postprocess_metrics import main as postprocess_main
from datetime import datetime, timedelta

def load_optimal_config():
    """Load the optimal micro account configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'optimal_micro_account.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def run_optimal_backtest(
    days_back: int = 7,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str = 'results_optimal'
):
    """
    Run backtest with optimal micro account settings
    """
    config = load_optimal_config()
    
    # Create BacktestConfig from optimal settings
    cfg = BacktestConfig(
        days_back=days_back,
        starting_cash=config['account_settings']['starting_cash'],
        per_order_risk_fraction=config['risk_management']['per_order_risk_fraction'],
        min_trade_value=config['risk_management']['min_trade_value'],
        growth_factor=config['risk_management']['growth_factor'],
        commission_bps=config['execution_settings']['commission_bps'],
        slippage_bps=config['execution_settings']['slippage_bps']
    )
    
    # Use optimal strategies only
    strategies = [config['primary_strategy']['name'], config['secondary_strategy']['name']]
    exec_tfs = config['data_settings']['execution_timeframes']
    htf_tfs = config['data_settings']['htf_timeframes']
    instrument = config['data_settings']['instrument']
    
    print(f"🎯 Running Eden Optimal Configuration")
    print(f"📊 Account: ${config['account_settings']['starting_cash']} micro account")
    print(f"📈 Strategies: {', '.join(strategies)}")
    print(f"⏰ Timeframes: {'/'.join(exec_tfs)} execution, {'/'.join(htf_tfs)} HTF bias")
    print(f"🎲 Risk: {config['risk_management']['per_order_risk_fraction']*100}% per trade, ${config['risk_management']['min_trade_value']} minimum")
    print()
    
    # Run backtests
    metrics = run_all_backtests(
        cfg=cfg,
        execution_tfs=exec_tfs,
        htf_tfs=htf_tfs,
        strategies=strategies,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )
    
    # Post-process results
    results_dir = Path(output_dir)
    consolidated = postprocess_main(results_dir)
    
    # Generate optimal summary
    print("\n🏆 === OPTIMAL CONFIGURATION RESULTS ===")
    
    primary_key = f"{config['primary_strategy']['name']}_{config['primary_strategy']['timeframe']}"
    secondary_key = f"{config['secondary_strategy']['name']}_{config['secondary_strategy']['timeframe']}"
    
    for key in [primary_key, secondary_key]:
        if key in consolidated:
            v = consolidated[key]
            strategy_type = "PRIMARY" if "ml_generated" in key else "SECONDARY"
            print(f"\n{strategy_type} - {key}:")
            print(f"  💰 Net PnL: ${v.get('net_pnl', 0):>8.2f}")
            print(f"  📊 Sharpe:  {v.get('sharpe', 0):>8.2f}")
            print(f"  📉 Max DD:  {v.get('max_drawdown_pct', 0):>7.1f}%")
            print(f"  🎯 Win Rate: {v.get('win_rate', 0):>6.1f}%")
            print(f"  🔢 Trades:  {v.get('trades', 0):>8d}")
            print(f"  📈 Growth:  {v.get('equity_growth_pct', 0):>7.1f}%")
    
    # Calculate combined performance estimate
    if primary_key in consolidated and secondary_key in consolidated:
        p_pnl = consolidated[primary_key]['net_pnl'] * 0.7  # 70% allocation
        s_pnl = consolidated[secondary_key]['net_pnl'] * 0.3  # 30% allocation
        combined_pnl = p_pnl + s_pnl
        combined_growth = (combined_pnl / config['account_settings']['starting_cash']) * 100
        
        print(f"\n🎯 COMBINED PORTFOLIO ESTIMATE (70/30 allocation):")
        print(f"  💰 Combined PnL: ${combined_pnl:>8.2f}")
        print(f"  📈 Combined Growth: {combined_growth:>6.1f}%")
    
    # Risk warnings
    print(f"\n⚠️  RISK MONITORING:")
    print(f"  🛑 Max Drawdown Alert: {config['monitoring']['drawdown_alert_pct']}%")
    print(f"  🚨 Emergency Stop: {config['account_settings']['emergency_stop_loss']}%")
    print(f"  📅 Weekly Review Required: {config['monitoring']['weekly_review_required']}")
    
    # Save configuration used
    (results_dir / 'optimal_config_used.json').write_text(json.dumps(config, indent=2))
    print(f"\n✅ Results saved to: {output_dir}/")
    print(f"✅ Optimal config saved to: {output_dir}/optimal_config_used.json")
    
    return consolidated

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Eden Optimal Micro Account Configuration')
    parser.add_argument('--days-back', type=int, default=7, help='Days of historical data')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')  
    parser.add_argument('--output-dir', type=str, default='results_optimal', help='Output directory')
    
    args = parser.parse_args()
    
    run_optimal_backtest(
        days_back=args.days_back,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir
    )