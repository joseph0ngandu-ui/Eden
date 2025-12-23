import pandas as pd
import numpy as np

def test_scalp(symbol, tf, sl, tp):
    try:
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
    except:
        return None
    
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    
    df["mom3"] = df["close"] - df["close"].shift(3)
    df["mom5"] = df["close"] - df["close"].shift(5)
    
    tr = np.maximum(df["high"] - df["low"], 
                   np.maximum(abs(df["high"] - df["close"].shift(1)),
                             abs(df["low"] - df["close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()
    
    pip = 10000 if "JPY" not in symbol else 100
    commission = 7
    
    trades = []
    pos = None
    
    for i in range(20, len(df)):
        c = df.iloc[i]
        p = df.iloc[i-1]
        
        if pos is None:
            if c["mom3"] > 0 and c["mom5"] > 0 and p["mom3"] <= 0:
                pos = {"type": "long", "entry": c["close"], "idx": i,
                       "sl": c["close"] - c["atr"]*sl, "tp": c["close"] + c["atr"]*tp}
            elif c["mom3"] < 0 and c["mom5"] < 0 and p["mom3"] >= 0:
                pos = {"type": "short", "entry": c["close"], "idx": i,
                       "sl": c["close"] + c["atr"]*sl, "tp": c["close"] - c["atr"]*tp}
        else:
            exit_trade = False
            exit_price = c["close"]
            
            if pos["type"] == "long":
                if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            else:
                if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
            
            if i - pos["idx"] >= 8: exit_trade = True
            
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

print("MOMENTUM SCALPING PARAMETER SEARCH")
print("=" * 50)

results = []
for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
    for tf in ["M5", "M15"]:
        for sl in [0.5, 0.8, 1.0, 1.2]:
            for tp in [0.8, 1.0, 1.2, 1.5, 2.0]:
                r = test_scalp(sym, tf, sl, tp)
                if r and r["pips"] > 0:
                    results.append({"symbol": sym, "tf": tf, "sl": sl, "tp": tp, **r})

results.sort(key=lambda x: x["pips"], reverse=True)

print("\nPROFITABLE SCALPING CONFIGS:")
for r in results[:15]:
    print(f"{r['symbol']} {r['tf']}: {r['trades']} trades, {r['win_rate']*100:.0f}% win, +{r['pips']:.0f} pips (SL:{r['sl']} TP:{r['tp']})")

print(f"\nFound {len(results)} profitable configs")
