import pandas as pd
import numpy as np

def test_strategy(symbol, tf, sl_mult, tp_mult):
    try:
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
    except:
        return None
    
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain/loss))
    
    tr = np.maximum(df["high"] - df["low"], 
                   np.maximum(abs(df["high"] - df["close"].shift(1)),
                             abs(df["low"] - df["close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    
    pip = 10000 if "JPY" not in symbol else 100
    commission = 7
    
    trades = []
    pos = None
    
    for i in range(55, len(df)):
        c = df.iloc[i]
        p = df.iloc[i-1]
        
        if pos is None:
            if p["ema9"] <= p["ema21"] and c["ema9"] > c["ema21"] and c["close"] > c["sma50"] and c["rsi"] > 50:
                pos = {"type": "long", "entry": c["close"], "idx": i, 
                       "sl": c["close"] - c["atr"]*sl_mult, "tp": c["close"] + c["atr"]*tp_mult}
            elif p["ema9"] >= p["ema21"] and c["ema9"] < c["ema21"] and c["close"] < c["sma50"] and c["rsi"] < 50:
                pos = {"type": "short", "entry": c["close"], "idx": i,
                       "sl": c["close"] + c["atr"]*sl_mult, "tp": c["close"] - c["atr"]*tp_mult}
        else:
            exit_trade = False
            exit_price = c["close"]
            
            if pos["type"] == "long":
                if c["high"] >= pos["tp"]:
                    exit_price, exit_trade = pos["tp"], True
                elif c["low"] <= pos["sl"]:
                    exit_price, exit_trade = pos["sl"], True
            else:
                if c["low"] <= pos["tp"]:
                    exit_price, exit_trade = pos["tp"], True
                elif c["high"] >= pos["sl"]:
                    exit_price, exit_trade = pos["sl"], True
            
            if i - pos["idx"] >= 100:
                exit_trade = True
            
            if exit_trade:
                pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                trades.append(pips - commission)
                pos = None
    
    if trades:
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t <= 0]
        win_rate = len(winners)/len(trades)
        total = sum(trades)
        pf = abs(sum(winners)/sum(losers)) if losers and sum(losers) != 0 else 0
        return {"symbol": symbol, "tf": tf, "trades": len(trades), "win_rate": win_rate, "pips": total, "pf": pf, "sl": sl_mult, "tp": tp_mult}
    return None

print("PARAMETER OPTIMIZATION WITH REAL DATA")
print("=" * 60)

results = []
symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
timeframes = ["M15", "H1", "H4"]

for sym in symbols:
    for tf in timeframes:
        for sl in [1.5, 2.0, 2.5]:
            for tp in [2.0, 2.5, 3.0, 4.0]:
                r = test_strategy(sym, tf, sl, tp)
                if r and r["pips"] > 0:
                    results.append(r)

results.sort(key=lambda x: x["pips"], reverse=True)

print("\nPROFITABLE COMBINATIONS:")
for r in results[:15]:
    print(f"{r['symbol']} {r['tf']}: {r['trades']} trades, {r['win_rate']*100:.0f}% win, {r['pips']:.0f} pips, PF: {r['pf']:.2f}")

print(f"\nFound {len(results)} profitable combinations")
