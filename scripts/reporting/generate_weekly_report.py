
import sys
import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Add project root to path
sys.path.append("c:\\Users\\opc\\Desktop\\Eden")

def generate_report():
    print("="*60)
    print("📊 WEEKLY PERFORMANCE REPORT (MT5 SOURCE)")
    print("="*60)
    
    if not mt5.initialize():
        print(f"❌ MT5 CONNECTION FAILED: {mt5.last_error()}")
        return

    # Define Period (Last 7 Days)
    end = datetime.now()
    start = end - timedelta(days=7)
    
    print(f"Fetching history from {start} to {end}...")
    
    # Get History
    deals = mt5.history_deals_get(start, end)
    
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Filter for Entries/Exits
    # Entry=0, Exit=1 (usually)
    # We want Profit != 0 (Closed trades)
    closed_trades = df[df['entry'] == 1] # Out deals usually have profit
    
    # Actually, simpler to look at all deals with profit != 0
    # Or 'commission' + 'swap' + 'profit'
    
    # Let's filter useful columns
    cols = ['time', 'symbol', 'type', 'volume', 'price', 'profit', 'commission', 'swap', 'comment']
    report_df = pd.DataFrame(deals, columns=deals[0]._asdict().keys())
    
    # Filter for realized PnL (Deals that are exits)
    # Entry Type: 0=IN, 1=OUT, 2=IN/OUT
    exits = report_df[report_df['entry'] == 1].copy()
    
    if len(exits) == 0:
        print("⚠️ No CLOSED trades found (only entries?).")
        # Check open positions
        positions = mt5.positions_get()
        if positions:
            print(f"\n🔓 OPEN POSITIONS: {len(positions)}")
            for pos in positions:
                print(f"   {pos.symbol} {pos.type} {pos.volume} | Profit: {pos.profit:.2f}")
    else:
        exits['total_pnl'] = exits['profit'] + exits['commission'] + exits['swap']
        
        total_pnl = exits['total_pnl'].sum()
        win_count = len(exits[exits['total_pnl'] > 0])
        loss_count = len(exits[exits['total_pnl'] <= 0])
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        print("\n📜 TRADE LIST:")
        for _, row in exits.iterrows():
            print(f"   {row['time']} | {row['symbol']:<7} | {row['type']} | {row['volume']} | PnL: ${row['total_pnl']:6.2f} | {row['comment']}")

        print(f"\n📈 WEEKLY SUMMARY:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Net PnL:      ${total_pnl:.2f}")
        print(f"   Win Rate:     {win_rate:.1f}%")
        print("="*60)
            
    mt5.shutdown()

if __name__ == "__main__":
    generate_report()
