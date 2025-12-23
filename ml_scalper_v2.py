import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLScalperV2:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.commission = 7
        
    def load_data(self, symbol, tf):
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        return df
    
    def create_features(self, df):
        # Returns
        df["ret1"] = df["close"].pct_change()
        df["ret5"] = df["close"].pct_change(5)
        df["ret10"] = df["close"].pct_change(10)
        
        # EMAs
        df["ema5"] = df["close"].ewm(span=5).mean()
        df["ema10"] = df["close"].ewm(span=10).mean()
        df["ema20"] = df["close"].ewm(span=20).mean()
        
        # EMA crossovers
        df["ema5_10"] = df["ema5"] / df["ema10"] - 1
        df["ema10_20"] = df["ema10"] / df["ema20"] - 1
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain/loss))
        df["rsi_norm"] = (df["rsi"] - 50) / 50
        
        # ATR
        tr = np.maximum(df["high"] - df["low"], 
                       np.maximum(abs(df["high"] - df["close"].shift(1)),
                                 abs(df["low"] - df["close"].shift(1))))
        df["atr"] = tr.rolling(14).mean()
        
        # Volatility regime
        df["vol"] = df["ret1"].rolling(20).std()
        df["vol_ratio"] = df["vol"] / df["vol"].rolling(50).mean()
        
        # Price position
        df["high20"] = df["high"].rolling(20).max()
        df["low20"] = df["low"].rolling(20).min()
        df["price_pos"] = (df["close"] - df["low20"]) / (df["high20"] - df["low20"])
        
        # Candle patterns
        df["body"] = (df["close"] - df["open"]) / df["atr"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["atr"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["atr"]
        
        # Volume
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] / df["vol_ma"]
        
        # Trend strength
        df["adx_proxy"] = abs(df["ema5_10"]) + abs(df["ema10_20"])
        
        return df
    
    def create_target(self, df, lookahead=5, min_profit=10):
        """Target: direction of next N bars with minimum profit threshold"""
        pip = 10000
        
        targets = []
        for i in range(len(df) - lookahead):
            future = df.iloc[i+1:i+lookahead+1]
            current = df["close"].iloc[i]
            
            max_up = (future["high"].max() - current) * pip
            max_down = (current - future["low"].min()) * pip
            
            # Only label if clear direction with profit potential
            if max_up > min_profit and max_up > max_down * 1.5:
                targets.append(1)  # Long
            elif max_down > min_profit and max_down > max_up * 1.5:
                targets.append(0)  # Short
            else:
                targets.append(-1)  # No trade
        
        targets.extend([-1] * lookahead)
        df["target"] = targets
        return df
    
    def train(self, symbols, tf="M5"):
        print("Loading data...")
        all_dfs = []
        
        for symbol in symbols:
            try:
                df = self.load_data(symbol, tf)
                df = self.create_features(df)
                df = self.create_target(df)
                df["symbol"] = symbol
                all_dfs.append(df)
                print(f"  {symbol}: {len(df)} bars")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.dropna()
        
        # Only use tradeable signals
        tradeable = combined[combined["target"] != -1].copy()
        
        features = ["ret1", "ret5", "ret10", "ema5_10", "ema10_20", "rsi_norm",
                   "vol_ratio", "price_pos", "body", "upper_wick", "lower_wick",
                   "vol_spike", "adx_proxy"]
        
        X = tradeable[features]
        y = tradeable["target"]
        
        print(f"\nTraining samples: {len(X)}")
        print(f"Long: {(y==1).sum()}, Short: {(y==0).sum()}")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=42
            )
            model.fit(X_train_s, y_train)
            
            score = accuracy_score(y_test, model.predict(X_test_s))
            scores.append(score)
        
        print(f"\nCV Accuracy: {np.mean(scores):.1%} (+/- {np.std(scores):.1%})")
        
        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Feature importance
        imp = pd.DataFrame({"feature": features, "importance": self.model.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)
        print("\nTop Features:")
        for _, row in imp.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return np.mean(scores)
    
    def backtest(self, symbol, tf="M5"):
        df = self.load_data(symbol, tf)
        df = self.create_features(df)
        df = df.dropna()
        
        features = ["ret1", "ret5", "ret10", "ema5_10", "ema10_20", "rsi_norm",
                   "vol_ratio", "price_pos", "body", "upper_wick", "lower_wick",
                   "vol_spike", "adx_proxy"]
        
        X = df[features]
        X_scaled = self.scaler.transform(X)
        
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        pip = 10000 if "JPY" not in symbol else 100
        trades = []
        pos = None
        
        for i in range(len(df) - 10):
            pred = preds[i]
            conf = max(probs[i])
            c = df.iloc[i]
            
            if pos is None and conf > 0.55:  # Confidence threshold
                atr = c["atr"]
                if pred == 1:  # Long
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["close"] - atr*1.5, "tp": c["close"] + atr*2.5}
                else:  # Short
                    pos = {"type": "short", "entry": c["close"], "idx": i,
                           "sl": c["close"] + atr*1.5, "tp": c["close"] - atr*2.5}
            
            elif pos is not None:
                c = df.iloc[i]
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
                
                if i - pos["idx"] >= 15:
                    exit_trade = True
                
                if exit_trade:
                    pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                    trades.append(pips - self.commission)
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
    
    def save(self, path="ml_scalper_v2.pkl"):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"Saved to {path}")

def main():
    print("=" * 60)
    print("ML SCALPER V2 - IMPROVED MODEL")
    print("=" * 60)
    
    scalper = MLScalperV2()
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
    
    print("\n1. TRAINING...")
    accuracy = scalper.train(symbols, tf="M5")
    
    print("\n2. BACKTESTING...")
    print("-" * 40)
    
    total = 0
    for sym in symbols:
        result = scalper.backtest(sym, "M5")
        if result:
            sign = "+" if result["pips"] > 0 else ""
            print(f"{sym}: {result['trades']} trades, {result['win_rate']:.0%} win, {sign}{result['pips']:.0f} pips, PF: {result['pf']:.2f}")
            total += result["pips"]
    
    print("-" * 40)
    print(f"TOTAL: {total:.0f} pips")
    
    scalper.save()
    
    # Also test on H1 timeframe
    print("\n3. TESTING ON H1 TIMEFRAME...")
    print("-" * 40)
    
    scalper2 = MLScalperV2()
    scalper2.train(symbols, tf="H1")
    
    total_h1 = 0
    for sym in symbols:
        result = scalper2.backtest(sym, "H1")
        if result:
            sign = "+" if result["pips"] > 0 else ""
            print(f"{sym} H1: {result['trades']} trades, {result['win_rate']:.0%} win, {sign}{result['pips']:.0f} pips, PF: {result['pf']:.2f}")
            total_h1 += result["pips"]
    
    print("-" * 40)
    print(f"TOTAL H1: {total_h1:.0f} pips")

if __name__ == "__main__":
    main()
