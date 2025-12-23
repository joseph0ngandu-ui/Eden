import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLScalper:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.commission = 7  # pips
        
    def load_data(self, symbol, tf):
        df = pd.read_csv(f"{symbol}_{tf}_data.csv", sep="\t", encoding="utf-8-sig")
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        return df
    
    def create_features(self, df):
        # Price features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"]/df["close"].shift(1))
        
        # Moving averages
        for p in [5, 10, 20, 50]:
            df[f"sma{p}"] = df["close"].rolling(p).mean()
            df[f"ema{p}"] = df["close"].ewm(span=p).mean()
            df[f"close_sma{p}"] = df["close"] / df[f"sma{p}"] - 1
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain/loss))
        
        # MACD
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR
        tr = np.maximum(df["high"] - df["low"], 
                       np.maximum(abs(df["high"] - df["close"].shift(1)),
                                 abs(df["low"] - df["close"].shift(1))))
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        
        # Momentum
        df["mom5"] = df["close"] / df["close"].shift(5) - 1
        df["mom10"] = df["close"] / df["close"].shift(10) - 1
        df["mom20"] = df["close"] / df["close"].shift(20) - 1
        
        # Volatility
        df["vol5"] = df["returns"].rolling(5).std()
        df["vol20"] = df["returns"].rolling(20).std()
        
        # Price patterns
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        df["body"] = abs(df["close"] - df["open"]) / df["atr"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["atr"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["atr"]
        
        # Volume features
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        return df
    
    def create_labels(self, df, pip_mult, lookahead=10):
        """Create labels: 1=profitable long, -1=profitable short, 0=no trade"""
        pip = 10000 if "JPY" not in df.columns else 100
        
        labels = []
        for i in range(len(df) - lookahead):
            future_high = df["high"].iloc[i+1:i+lookahead+1].max()
            future_low = df["low"].iloc[i+1:i+lookahead+1].min()
            current = df["close"].iloc[i]
            
            long_profit = (future_high - current) * pip_mult
            short_profit = (current - future_low) * pip_mult
            
            # Need at least 15 pips profit to cover commission + profit
            if long_profit > 15 and long_profit > short_profit:
                labels.append(1)
            elif short_profit > 15 and short_profit > long_profit:
                labels.append(-1)
            else:
                labels.append(0)
        
        # Pad with zeros for last rows
        labels.extend([0] * lookahead)
        df["label"] = labels
        return df
    
    def prepare_data(self, symbols, tf="M5"):
        all_data = []
        
        for symbol in symbols:
            try:
                df = self.load_data(symbol, tf)
                df = self.create_features(df)
                pip_mult = 10000 if "JPY" not in symbol else 100
                df = self.create_labels(df, pip_mult)
                df["symbol"] = symbol
                all_data.append(df)
                print(f"Loaded {symbol}: {len(df)} bars")
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def train(self, df):
        # Feature columns
        feature_cols = ["returns", "log_returns", "rsi", "macd", "macd_hist", "bb_pos",
                       "atr_pct", "mom5", "mom10", "mom20", "vol5", "vol20",
                       "close_sma5", "close_sma10", "close_sma20", "close_sma50",
                       "higher_high", "lower_low", "body", "upper_wick", "lower_wick", "vol_ratio"]
        
        # Clean data
        df = df.dropna()
        
        # Only train on tradeable signals (not 0)
        df_trades = df[df["label"] != 0].copy()
        
        X = df_trades[feature_cols]
        y = df_trades["label"]
        
        print(f"\nTraining data: {len(X)} samples")
        print(f"Long signals: {(y == 1).sum()}, Short signals: {(y == -1).sum()}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.1%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Short", "Long"]))
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        print("\nTop 10 Features:")
        print(importance.head(10).to_string(index=False))
        
        return accuracy
    
    def backtest(self, df, symbol):
        feature_cols = ["returns", "log_returns", "rsi", "macd", "macd_hist", "bb_pos",
                       "atr_pct", "mom5", "mom10", "mom20", "vol5", "vol20",
                       "close_sma5", "close_sma10", "close_sma20", "close_sma50",
                       "higher_high", "lower_low", "body", "upper_wick", "lower_wick", "vol_ratio"]
        
        df = df.dropna()
        pip = 10000 if "JPY" not in symbol else 100
        
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions with probability
        predictions = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)
        
        trades = []
        pos = None
        
        for i in range(len(df) - 10):
            pred = predictions[i]
            conf = max(probas[i])
            
            if pos is None and conf > 0.6:  # Only trade high confidence
                c = df.iloc[i]
                atr = c["atr"]
                
                if pred == 1:  # Long
                    pos = {"type": "long", "entry": c["close"], "idx": i,
                           "sl": c["close"] - atr*1.5, "tp": c["close"] + atr*2}
                elif pred == -1:  # Short
                    pos = {"type": "short", "entry": c["close"], "idx": i,
                           "sl": c["close"] + atr*1.5, "tp": c["close"] - atr*2}
            
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
                
                if i - pos["idx"] >= 20:  # Max 20 bars
                    exit_trade = True
                
                if exit_trade:
                    pips = (exit_price - pos["entry"]) * pip if pos["type"] == "long" else (pos["entry"] - exit_price) * pip
                    trades.append(pips - self.commission)
                    pos = None
        
        if trades:
            winners = [t for t in trades if t > 0]
            losers = [t for t in trades if t <= 0]
            win_rate = len(winners)/len(trades)
            total = sum(trades)
            pf = abs(sum(winners)/sum(losers)) if losers and sum(losers) != 0 else 0
            
            return {
                "symbol": symbol,
                "trades": len(trades),
                "win_rate": win_rate,
                "total_pips": total,
                "profit_factor": pf,
                "avg_trade": np.mean(trades)
            }
        return None
    
    def save(self, path="ml_scalper_model.pkl"):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"Model saved to {path}")
    
    def load(self, path="ml_scalper_model.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        print(f"Model loaded from {path}")

def main():
    print("=" * 60)
    print("ML SCALPER - TRAINING ON REAL MT5 DATA")
    print("=" * 60)
    
    scalper = MLScalper()
    
    # Prepare data from all symbols
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
    
    print("\n1. LOADING AND PREPARING DATA...")
    df = scalper.prepare_data(symbols, tf="M5")
    
    print("\n2. TRAINING ML MODEL...")
    accuracy = scalper.train(df)
    
    print("\n3. BACKTESTING ON EACH SYMBOL...")
    print("-" * 60)
    
    total_pips = 0
    for symbol in symbols:
        try:
            test_df = scalper.load_data(symbol, "M5")
            test_df = scalper.create_features(test_df)
            result = scalper.backtest(test_df, symbol)
            
            if result:
                status = "+" if result["total_pips"] > 0 else ""
                print(f"{symbol}: {result['trades']} trades, {result['win_rate']:.0%} win, {status}{result['total_pips']:.0f} pips, PF: {result['profit_factor']:.2f}")
                total_pips += result["total_pips"]
        except Exception as e:
            print(f"{symbol}: Error - {e}")
    
    print("-" * 60)
    print(f"TOTAL: {total_pips:.0f} pips")
    
    # Save model
    scalper.save()
    
    print("\n" + "=" * 60)
    print("ML SCALPER TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
