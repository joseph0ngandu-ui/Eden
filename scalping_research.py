"""
Scalping Strategy Research - Finding profitable M5/M15 strategies
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ScalpingResearch:
    def __init__(self):
        self.commission = 7
        
    def load_data(self, symbol, tf):
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        return df
    
    def add_indicators(self, df):
        # Fast EMAs for scalping
        df["ema3"] = df["close"].ewm(span=3).mean()
        df["ema8"] = df["close"].ewm(span=8).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain/loss))
        
        # Stochastic
        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        # ATR
        tr = np.maximum(df["high"] - df["low"], 
                       np.maximum(abs(df["high"] - df["close"].shift(1)),
                                 abs(df["low"] - df["close"].shift(1))))
        df["atr"] = tr.rolling(14).mean()
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        
        # MACD
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Volume
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"]
        
        # Momentum
        df["mom"] = df["close"] - df["close"].shift(5)
        
        # Candle patterns
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["body_ratio"] = df["body"] / df["range"]
        
        return df
    
    def stoch_rsi_scalp(self, df, symbol):
        """Stochastic + RSI oversold/overbought scalping"""
        pip = 10000 if "JPY" not in symbol else 100
        trades = []
        pos = None
        
        for i in range(25, len(df)):
            c = df.iloc[i]
            p = df.iloc[i-1]
            
            if pos is None:
                # Long: Stoch oversold crossing up + RSI < 40
                if (p["stoch_k"] < 20 and c["stoch_k"] > p["stoch_k"] and 
                    c["stoch_k"] > c["stoch_d"] and c["rsi"] < 40):
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["close"] - c["atr"]*1.0, "tp": c["close"] + c["atr"]*1.5}
                # Short: Stoch overbought crossing down + RSI > 60
                elif (p["stoch_k"] > 80 and c["stoch_k"] < p["stoch_k"] and 
                      c["stoch_k"] < c["stoch_d"] and c["rsi"] > 60):
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
                    trades.append(pips - self.commission)
                    pos = None
        
        return trades
    
    def bb_squeeze_scalp(self, df, symbol):
        """Bollinger Band squeeze breakout scalping"""
        pip = 10000 if "JPY" not in symbol else 100
        trades = []
        pos = None
        
        # Calculate squeeze (low volatility)
        df["squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)
        
        for i in range(55, len(df)):
            c = df.iloc[i]
            p = df.iloc[i-1]
            
            if pos is None:
                # Long: Squeeze release + breakout above BB mid + positive momentum
                if (p["squeeze"] and not c["squeeze"] and 
                    c["close"] > c["bb_mid"] and c["mom"] > 0 and c["vol_ratio"] > 1.2):
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["bb_mid"], "tp": c["close"] + c["atr"]*2.0}
                # Short: Squeeze release + breakout below BB mid + negative momentum
                elif (p["squeeze"] and not c["squeeze"] and 
                      c["close"] < c["bb_mid"] and c["mom"] < 0 and c["vol_ratio"] > 1.2):
                    pos = {"type": "short", "entry": c["close"], "idx": i,
                           "sl": c["bb_mid"], "tp": c["close"] - c["atr"]*2.0}
            else:
                exit_trade = False
                exit_price = c["close"]
                
                if pos["type"] == "long":
                    if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                else:
                    if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                
                if i - pos["idx"] >= 20: exit_trade = True
                
                if exit_trade:
                    pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                    trades.append(pips - self.commission)
                    pos = None
        
        return trades
    
    def macd_divergence_scalp(self, df, symbol):
        """MACD histogram divergence scalping"""
        pip = 10000 if "JPY" not in symbol else 100
        trades = []
        pos = None
        
        for i in range(30, len(df)):
            c = df.iloc[i]
            p = df.iloc[i-1]
            pp = df.iloc[i-2]
            
            if pos is None:
                # Bullish: MACD hist turning positive + price above EMA8
                if (pp["macd_hist"] < p["macd_hist"] < 0 and c["macd_hist"] > p["macd_hist"] and
                    c["close"] > c["ema8"]):
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["close"] - c["atr"]*1.2, "tp": c["close"] + c["atr"]*1.8}
                # Bearish: MACD hist turning negative + price below EMA8
                elif (pp["macd_hist"] > p["macd_hist"] > 0 and c["macd_hist"] < p["macd_hist"] and
                      c["close"] < c["ema8"]):
                    pos = {"type": "short", "entry": c["close"], "idx": i,
                           "sl": c["close"] + c["atr"]*1.2, "tp": c["close"] - c["atr"]*1.8}
            else:
                exit_trade = False
                exit_price = c["close"]
                
                if pos["type"] == "long":
                    if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                else:
                    if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                
                if i - pos["idx"] >= 15: exit_trade = True
                
                if exit_trade:
                    pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                    trades.append(pips - self.commission)
                    pos = None
        
        return trades
    
    def ema_pullback_scalp(self, df, symbol):
        """EMA pullback scalping - trade pullbacks to fast EMA in trend"""
        pip = 10000 if "JPY" not in symbol else 100
        trades = []
        pos = None
        
        for i in range(30, len(df)):
            c = df.iloc[i]
            p = df.iloc[i-1]
            
            if pos is None:
                # Uptrend pullback: EMA3 > EMA21, price touches EMA8 from above
                if (c["ema3"] > c["ema21"] and p["low"] > p["ema8"] and c["low"] <= c["ema8"] and
                    c["close"] > c["ema8"] and c["rsi"] > 40):
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["ema21"], "tp": c["close"] + c["atr"]*1.5}
                # Downtrend pullback: EMA3 < EMA21, price touches EMA8 from below
                elif (c["ema3"] < c["ema21"] and p["high"] < p["ema8"] and c["high"] >= c["ema8"] and
                      c["close"] < c["ema8"] and c["rsi"] < 60):
                    pos = {"type": "short", "entry": c["close"], "idx": i,
                           "sl": c["ema21"], "tp": c["close"] - c["atr"]*1.5}
            else:
                exit_trade = False
                exit_price = c["close"]
                
                if pos["type"] == "long":
                    if c["high"] >= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["low"] <= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                else:
                    if c["low"] <= pos["tp"]: exit_price, exit_trade = pos["tp"], True
                    elif c["high"] >= pos["sl"]: exit_price, exit_trade = pos["sl"], True
                
                if i - pos["idx"] >= 10: exit_trade = True
                
                if exit_trade:
                    pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                    trades.append(pips - self.commission)
                    pos = None
        
        return trades
    
    def analyze(self, trades):
        if not trades:
            return None
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t <= 0]
        return {
            "trades": len(trades),
            "win_rate": len(winners)/len(trades) if trades else 0,
            "pips": sum(trades),
            "pf": abs(sum(winners)/sum(losers)) if losers and sum(losers) != 0 else 0
        }
    
    def run_research(self):
        print("=" * 60)
        print("SCALPING STRATEGY RESEARCH - M5 & M15")
        print("=" * 60)
        
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
        timeframes = ["M5", "M15"]
        
        strategies = {
            "Stoch RSI": self.stoch_rsi_scalp,
            "BB Squeeze": self.bb_squeeze_scalp,
            "MACD Divergence": self.macd_divergence_scalp,
            "EMA Pullback": self.ema_pullback_scalp
        }
        
        results = []
        
        for tf in timeframes:
            print(f"\n{tf} TIMEFRAME:")
            print("-" * 40)
            
            for symbol in symbols:
                try:
                    df = self.load_data(symbol, tf)
                    df = self.add_indicators(df)
                    
                    for name, func in strategies.items():
                        trades = func(df, symbol)
                        analysis = self.analyze(trades)
                        
                        if analysis and analysis["pips"] > 0:
                            results.append({
                                "symbol": symbol, "tf": tf, "strategy": name,
                                **analysis
                            })
                            print(f"  + {symbol} {name}: {analysis['trades']} trades, {analysis['win_rate']:.0%} win, +{analysis['pips']:.0f} pips, PF: {analysis['pf']:.2f}")
                except Exception as e:
                    pass
        
        print("\n" + "=" * 60)
        print("PROFITABLE SCALPING STRATEGIES FOUND:")
        print("=" * 60)
        
        results.sort(key=lambda x: x["pips"], reverse=True)
        for r in results[:10]:
            print(f"{r['symbol']} {r['tf']} {r['strategy']}: {r['trades']} trades, {r['win_rate']:.0%} win, +{r['pips']:.0f} pips, PF: {r['pf']:.2f}")
        
        print(f"\nTotal profitable combinations: {len(results)}")
        
        return results

if __name__ == "__main__":
    research = ScalpingResearch()
    research.run_research()
