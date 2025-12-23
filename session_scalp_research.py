import pandas as pd
import numpy as np

def session_scalp(symbol, tf):
    """Session-based scalping - trade during high volatility sessions"""
    try:
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
    except:
        return None
    
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
    df["hour"] = df["time"].dt.hour
    
    # Indicators
    df["ema5"] = df["close"].ewm(span=5).mean()
    df["ema13"] = df["close"].ewm(span=13).mean()
    
    tr = np.maximum(df["high"] - df["low"], 
                   np.maximum(abs(df["high"] - df["close"].shift(1)),
                             abs(df["low"] - df["close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    
    # Volume filter
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["high_vol"] = df["volume"] > df["vol_ma"] * 1.2
    
    pip = 10000 if "JPY" not in symbol else 100
    commission = 7
    
    trades = []
    pos = None
    
    for i in range(20, len(df)):
        c = df.iloc[i]
        p = df.iloc[i-1]
        
        # Only trade during London/NY sessions (7-17 UTC)
        if not (7 <= c["hour"] <= 17):
            continue
        
        if pos is None:
            # Long: EMA5 crosses above EMA13 with volume
            if p["ema5"] <= p["ema13"] and c["ema5"] > c["ema13"] and c["high_vol"]:
                pos = {"type": "long", "entry": c["close"], "idx": i,
                       "sl": c["close"] - c["atr"]*1.0, "tp": c["close"] + c["atr"]*1.5}
            # Short: EMA5 crosses below EMA13 with volume
            elif p["ema5"] >= p["ema13"] and c["ema5"] < c["ema13"] and c["high_vol"]:
                pos = {"type": "short", "entry": c["close"], "idx": i,
                       "sl": c["close"] + c["atr"]*1.0, "tp": c["close"] - c["atr"]*1.5}
        else:
            exit_trade = False
            exit_price = c["close"]
            
            if pos["type"] == "long":
                if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            else:
                if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            
            if i - pos["idx"] >= 12: exit_trade = True
            
            if exit_trade:
                pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                trades.append(pips - commission)
                pos = None
    
    if trades:
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t <= 0]
        return {
            "trades": len(trades),
            "win_rate": len(winners)/len(trades),
            "pips": sum(trades),
            "pf": abs(sum(winners)/sum(losers)) if losers and sum(losers) != 0 else 0
        }
    return None

def range_breakout_scalp(symbol, tf):
    """Asian range breakout scalping"""
    try:
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
    except:
        return None
    
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.date
    
    tr = np.maximum(df["high"] - df["low"], 
                   np.maximum(abs(df["high"] - df["close"].shift(1)),
                             abs(df["low"] - df["close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    
    pip = 10000 if "JPY" not in symbol else 100
    commission = 7
    
    trades = []
    pos = None
    daily_range = {}
    
    for i in range(20, len(df)):
        c = df.iloc[i]
        date = c["date"]
        hour = c["hour"]
        
        # Calculate Asian range (0-7 UTC)
        if hour < 7:
            if date not in daily_range:
                daily_range[date] = {"high": c["high"], "low": c["low"]}
            else:
                daily_range[date]["high"] = max(daily_range[date]["high"], c["high"])
                daily_range[date]["low"] = min(daily_range[date]["low"], c["low"])
            continue
        
        if date not in daily_range:
            continue
        
        asian_high = daily_range[date]["high"]
        asian_low = daily_range[date]["low"]
        
        if pos is None and 7 <= hour <= 12:  # Trade during London open
            # Breakout above Asian high
            if c["close"] > asian_high and c["close"] > c["open"]:
                pos = {"type": "long", "entry": c["close"], "idx": i,
                       "sl": asian_low, "tp": c["close"] + (asian_high - asian_low)}
            # Breakout below Asian low
            elif c["close"] < asian_low and c["close"] < c["open"]:
                pos = {"type": "short", "entry": c["close"], "idx": i,
                       "sl": asian_high, "tp": c["close"] - (asian_high - asian_low)}
        
        elif pos is not None:
            exit_trade = False
            exit_price = c["close"]
            
            if pos["type"] == "long":
                if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            else:
                if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            
            if hour >= 16: exit_trade = True  # Close by end of NY session
            
            if exit_trade:
                pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                trades.append(pips - commission)
                pos = None
    
    if trades:
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t <= 0]
        return {
            "trades": len(trades),
            "win_rate": len(winners)/len(trades),
            "pips": sum(trades),
            "pf": abs(sum(winners)/sum(losers)) if losers and sum(losers) != 0 else 0
        }
    return None

print("SESSION-BASED SCALPING RESEARCH")
print("=" * 50)

results = []

for sym in ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]:
    for tf in ["M5", "M15"]:
        # Session scalp
        r = session_scalp(sym, tf)
        if r and r["pips"] > 0:
            results.append({"symbol": sym, "tf": tf, "strategy": "Session", **r})
            print(f"+ {sym} {tf} Session: {r['trades']} trades, {r['win_rate']*100:.0f}% win, +{r['pips']:.0f} pips")
        
        # Range breakout
        r = range_breakout_scalp(sym, tf)
        if r and r["pips"] > 0:
            results.append({"symbol": sym, "tf": tf, "strategy": "Range Breakout", **r})
            print(f"+ {sym} {tf} Range Breakout: {r['trades']} trades, {r['win_rate']*100:.0f}% win, +{r['pips']:.0f} pips")

print(f"\nTotal profitable: {len(results)}")

if results:
    print("\nBEST SCALPING STRATEGIES:")
    results.sort(key=lambda x: x["pips"], reverse=True)
    for r in results[:5]:
        print(f"  {r['symbol']} {r['tf']} {r['strategy']}: +{r['pips']:.0f} pips, {r['win_rate']*100:.0f}% win, PF: {r['pf']:.2f}")
