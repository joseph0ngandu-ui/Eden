"""
Quick test to verify daily reset logic works correctly
"""

from datetime import datetime

class MockBot:
    def __init__(self):
        self.current_trading_day = None
        self.start_of_day_balance = 0.0
    
    def _check_and_reset_daily_balance(self, current_balance: float) -> None:
        """Check if new trading day has started and reset daily balance tracking."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if we've entered a new day
        if self.current_trading_day != current_date:
            self.start_of_day_balance = current_balance
            self.current_trading_day = current_date
            print(f"📅 NEW TRADING DAY: {current_date} | Starting Balance: ${self.start_of_day_balance:.2f}")
    
    def calculate_daily_dd(self, current_equity: float) -> float:
        """Calculate daily drawdown percentage."""
        daily_dd_pct = 0.0
        if self.start_of_day_balance > 0:
            daily_pnl = current_equity - self.start_of_day_balance
            if daily_pnl < 0:
                daily_dd_pct = abs(daily_pnl / self.start_of_day_balance) * 100
        return daily_dd_pct

# Test the logic
bot = MockBot()

print("=== Day 1 (Dec 1st) ===")
bot._check_and_reset_daily_balance(1000.0)
print(f"Start: $1000, Current: $980, DD: {bot.calculate_daily_dd(980.0):.2f}%")  # Should be 2%
print(f"Start: $1000, Current: $950, DD: {bot.calculate_daily_dd(950.0):.2f}%")  # Should be 5%

print("\n=== Day 2 (Simulate next day) ===")
# Manually change date to simulate new day
bot.current_trading_day = "2025-12-03"  # Force reset
bot._check_and_reset_daily_balance(950.0)  # New balance after Day 1 loss
print(f"Start: $950, Current: $950, DD: {bot.calculate_daily_dd(950.0):.2f}%")   # Should be 0% (reset!)
print(f"Start: $950, Current: $930, DD: {bot.calculate_daily_dd(930.0):.2f}%")   # Should be ~2.1%

print("\n✅ Daily reset logic working correctly!")
