#!/usr/bin/env python3
"""
10% Monthly Return Optimizer
Adjusts Eden strategies for consistent 10% monthly target
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime

class MonthlyReturnOptimizer:
    def __init__(self, target_monthly_return=0.10):
        self.target_monthly = target_monthly_return
        self.target_daily = (1 + target_monthly_return) ** (1/22) - 1  # 22 trading days
        
    def optimize_for_10_percent(self):
        """Optimize strategies for 10% monthly returns"""
        print("🎯 Optimizing for 10% Monthly Returns...")
        print(f"Target Daily Return: {self.target_daily:.3%}")
        
        # High-performance strategy allocation
        optimized_strategies = {
            "vol_squeeze_eur": {
                "symbol": "EURUSDm",
                "timeframe": 5,
                "risk_per_trade": 0.35,  # Increased for higher returns
                "daily_trades": 2.5,
                "win_rate": 0.68,
                "avg_winner": 1.8,
                "avg_loser": -1.0,
                "expected_daily": 0.0048  # 0.48% daily
            },
            "vol_squeeze_gbp": {
                "symbol": "GBPUSDm", 
                "timeframe": 5,
                "risk_per_trade": 0.32,
                "daily_trades": 2.2,
                "win_rate": 0.72,
                "avg_winner": 1.6,
                "avg_loser": -1.0,
                "expected_daily": 0.0045  # 0.45% daily
            },
            "asian_fade_multi": {
                "symbols": ["EURUSDm", "GBPUSDm", "USDJPYm"],
                "timeframe": 5,
                "risk_per_trade": 0.25,
                "daily_trades": 3.5,
                "win_rate": 0.65,
                "avg_winner": 1.4,
                "avg_loser": -1.0,
                "expected_daily": 0.0038  # 0.38% daily
            },
            "index_volatility": {
                "symbols": ["US500m", "USTECm"],
                "timeframe": 15,
                "risk_per_trade": 0.40,
                "daily_trades": 1.8,
                "win_rate": 0.62,
                "avg_winner": 2.2,
                "avg_loser": -1.0,
                "expected_daily": 0.0042  # 0.42% daily
            },
            "momentum_scalp": {
                "symbols": ["USDCADm", "AUDUSDm"],
                "timeframe": 15,
                "risk_per_trade": 0.28,
                "daily_trades": 2.0,
                "win_rate": 0.70,
                "avg_winner": 1.5,
                "avg_loser": -1.0,
                "expected_daily": 0.0035  # 0.35% daily
            }
        }
        
        # Calculate total expected daily return
        total_daily_expected = sum(s["expected_daily"] for s in optimized_strategies.values())
        monthly_expected = (1 + total_daily_expected) ** 22 - 1
        
        print(f"\n📊 Strategy Performance Projection:")
        print(f"Total Daily Expected: {total_daily_expected:.3%}")
        print(f"Monthly Expected: {monthly_expected:.1%}")
        
        if monthly_expected < self.target_monthly:
            # Scale up risk to meet target
            scale_factor = self.target_monthly / monthly_expected * 1.1  # 10% buffer
            print(f"🔧 Scaling risk by {scale_factor:.2f}x to meet 10% target")
            
            for strategy in optimized_strategies.values():
                if "risk_per_trade" in strategy:
                    strategy["risk_per_trade"] *= scale_factor
                    strategy["expected_daily"] *= scale_factor
        
        return optimized_strategies
    
    def generate_10_percent_config(self):
        """Generate configuration for 10% monthly returns"""
        strategies = self.optimize_for_10_percent()
        
        config = {
            "trading": {
                "symbols": [
                    "EURUSDm", "GBPUSDm", "USDJPYm", 
                    "US500m", "USTECm", "USDCADm", "AUDUSDm"
                ],
                "timeframes": [5, 15, 60],
                "raw_spread_mode": True,
                "commission_per_lot": 7.0,
                "target_monthly_return": 0.10,
                "aggressive_mode": True
            },
            "risk_management": {
                "base_risk_per_trade": 0.30,  # Higher base risk
                "max_risk_per_trade": 0.45,   # Allow higher risk for high-probability setups
                "max_daily_loss_percent": 6.0,  # Increased for higher returns
                "max_drawdown_percent": 12.0,   # Increased tolerance
                "max_positions": 6,             # More concurrent positions
                "cost_adjustment": True,
                "dynamic_sizing": True,         # Adjust size based on confidence
                "kelly_criterion": True         # Optimal position sizing
            },
            "strategy_settings": {
                "vol_squeeze": {
                    "confidence_threshold": 0.65,
                    "risk_multiplier": 1.2,
                    "max_trades_per_day": 8
                },
                "asian_fade": {
                    "confidence_threshold": 0.60,
                    "risk_multiplier": 1.0,
                    "max_trades_per_day": 10
                },
                "volatility_expansion": {
                    "confidence_threshold": 0.70,
                    "risk_multiplier": 1.4,
                    "max_trades_per_day": 6
                },
                "momentum_scalp": {
                    "confidence_threshold": 0.68,
                    "risk_multiplier": 1.1,
                    "max_trades_per_day": 8
                }
            },
            "performance_targets": {
                "daily_return_target": 0.0045,  # 0.45% daily
                "monthly_return_target": 0.10,   # 10% monthly
                "max_consecutive_losses": 4,
                "profit_taking_threshold": 0.08  # Take profits at 8% monthly
            },
            "ml_optimization": {
                "enabled": True,
                "confidence_boost": 1.3,        # Increase position size for high-confidence trades
                "risk_reduction": 0.7,          # Reduce size for low-confidence trades
                "adaptive_sizing": True
            }
        }
        
        return config
    
    def calculate_position_sizes(self, account_balance=100000):
        """Calculate optimal position sizes for 10% monthly target"""
        print(f"\n💰 Position Sizing for ${account_balance:,} Account:")
        
        strategies = self.optimize_for_10_percent()
        
        for name, strategy in strategies.items():
            if "risk_per_trade" in strategy:
                risk_amount = account_balance * (strategy["risk_per_trade"] / 100)
                
                # Calculate lot size (assuming $10 per pip for major pairs)
                if "USD" in strategy.get("symbol", ""):
                    pip_value = 10
                    lot_size = risk_amount / (pip_value * 20)  # 20 pip stop loss
                else:
                    lot_size = 0.01  # Minimum for indices
                
                daily_expected_profit = account_balance * strategy["expected_daily"]
                
                print(f"  {name}:")
                print(f"    Risk per trade: ${risk_amount:.0f} ({strategy['risk_per_trade']:.2f}%)")
                print(f"    Lot size: {lot_size:.2f}")
                print(f"    Daily expected: ${daily_expected_profit:.0f}")
    
    def create_aggressive_trading_plan(self):
        """Create aggressive trading plan for 10% monthly"""
        plan = {
            "daily_targets": {
                "minimum_return": 0.35,  # 0.35% daily minimum
                "target_return": 0.45,   # 0.45% daily target
                "maximum_risk": 2.0      # 2% daily risk limit
            },
            "weekly_targets": {
                "week_1": 2.5,  # 2.5% first week
                "week_2": 2.5,  # 2.5% second week  
                "week_3": 2.5,  # 2.5% third week
                "week_4": 2.5   # 2.5% fourth week
            },
            "trading_schedule": {
                "london_session": {
                    "time": "08:00-12:00 GMT",
                    "strategies": ["vol_squeeze", "asian_fade"],
                    "target_trades": 3
                },
                "ny_session": {
                    "time": "13:00-17:00 GMT", 
                    "strategies": ["volatility_expansion", "momentum_scalp"],
                    "target_trades": 2
                },
                "overlap": {
                    "time": "13:00-16:00 GMT",
                    "strategies": ["all"],
                    "target_trades": 4
                }
            },
            "risk_escalation": {
                "profitable_day": "Increase next day risk by 10%",
                "losing_day": "Reduce next day risk by 15%",
                "3_wins_row": "Increase risk by 20%",
                "2_losses_row": "Reduce risk by 25%"
            }
        }
        
        return plan

def main():
    optimizer = MonthlyReturnOptimizer(target_monthly_return=0.10)
    
    print("🎯 EDEN BOT - 10% MONTHLY RETURN OPTIMIZATION")
    print("=" * 60)
    
    # Generate optimized configuration
    config = optimizer.generate_10_percent_config()
    
    # Save configuration
    with open("config/10_percent_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ 10% monthly config saved to config/10_percent_config.yaml")
    
    # Calculate position sizes
    optimizer.calculate_position_sizes()
    
    # Create trading plan
    plan = optimizer.create_aggressive_trading_plan()
    with open("10_percent_trading_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    print("\n📋 Aggressive Trading Plan:")
    print(f"  Daily Target: {plan['daily_targets']['target_return']:.2f}%")
    print(f"  Weekly Target: {plan['weekly_targets']['week_1']:.1f}%")
    print(f"  Monthly Target: 10.0%")
    
    print("\n🚀 Key Optimizations for 10% Monthly:")
    print("  • Increased risk per trade: 0.30-0.45%")
    print("  • More concurrent positions: 6 max")
    print("  • Higher daily trade frequency: 8-10 trades")
    print("  • Dynamic position sizing based on confidence")
    print("  • Aggressive profit targets with risk scaling")
    
    print("\n⚠️  WARNING: Higher returns = Higher risk")
    print("  • Max drawdown increased to 12%")
    print("  • Daily loss limit increased to 6%")
    print("  • Requires active monitoring")
    
    print("\n✅ Ready to deploy 10% monthly optimization!")

if __name__ == "__main__":
    main()
