#!/usr/bin/env python3
"""
Multi-Timeframe Backtest Engine for Eden
Analyzes trading signals across multiple timeframes with HTF bias confirmation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis and signal generation"""
    
    def __init__(self, instruments_data: Dict):
        """
        Args:
            instruments_data: Dict with structure {symbol: {timeframe: df}}
        """
        self.data = instruments_data
        self.timeframe_order = ['M1', 'M5', 'M15', 'H1', 'H4']
        
    def get_htf_bias(self, df: pd.DataFrame, timeframe: str = 'H1') -> pd.Series:
        """Get Higher Timeframe Bias"""
        if len(df) < 20:
            return pd.Series(0, index=df.index)
        
        sma_20 = df['close'].rolling(20).mean()
        bias = (df['close'] > sma_20).astype(int) * 2 - 1
        return bias
    
    def align_signals_to_base(self, base_df: pd.DataFrame, 
                             htf_df: pd.DataFrame) -> pd.Series:
        """Align higher timeframe signal to base timeframe"""
        aligned_signal = pd.Series(0, index=base_df.index)
        
        if len(htf_df) < 2:
            return aligned_signal
        
        # Forward fill HTF signal to lower TF
        htf_bias = self.get_htf_bias(htf_df)
        
        for i, ts in enumerate(base_df['timestamp']):
            # Find nearest HTF timestamp
            mask = htf_df['timestamp'] <= ts
            if mask.any():
                idx = mask.sum() - 1
                if idx < len(htf_bias):
                    aligned_signal.iloc[i] = htf_bias.iloc[idx]
        
        return aligned_signal
    
    def analyze_instrument(self, symbol: str) -> Dict:
        """Analyze single instrument across timeframes"""
        
        if symbol not in self.data:
            return None
        
        timeframes_data = self.data[symbol]
        result = {
            'symbol': symbol,
            'timeframes': {},
            'multi_tf_signals': []
        }
        
        # Get base timeframe (M1) data
        if 'M1' not in timeframes_data or timeframes_data['M1'] is None:
            return result
        
        base_df = timeframes_data['M1'].copy()
        
        # Analyze each timeframe
        for tf in self.timeframe_order:
            if tf not in timeframes_data or timeframes_data[tf] is None:
                continue
            
            tf_data = timeframes_data[tf].copy()
            if len(tf_data) < 2:
                continue
            
            # Calculate HTF bias
            htf_bias = self.get_htf_bias(tf_data)
            
            result['timeframes'][tf] = {
                'candles': len(tf_data),
                'bias_up_pct': float((htf_bias > 0).sum() / len(htf_bias) * 100),
                'bias_down_pct': float((htf_bias < 0).sum() / len(htf_bias) * 100),
                'current_bias': int(htf_bias.iloc[-1]) if len(htf_bias) > 0 else 0
            }
        
        # Multi-timeframe confluence analysis
        if 'H4' in timeframes_data and timeframes_data['H4'] is not None:
            h4_bias = self.get_htf_bias(timeframes_data['H4'])
            h1_bias = self.get_htf_bias(timeframes_data['H1']) if 'H1' in timeframes_data else None
            
            if len(h4_bias) > 0 and h1_bias is not None and len(h1_bias) > 0:
                # Count confluences
                confluence_up = ((h4_bias.iloc[-1] > 0) and (h1_bias.iloc[-1] > 0))
                confluence_down = ((h4_bias.iloc[-1] < 0) and (h1_bias.iloc[-1] < 0))
                
                result['confluence'] = {
                    'h4_h1_aligned': confluence_up or confluence_down,
                    'h4_direction': 'UP' if h4_bias.iloc[-1] > 0 else 'DOWN',
                    'h1_direction': 'UP' if h1_bias.iloc[-1] > 0 else 'DOWN'
                }
        
        return result

class MultiTimeframeBacktester:
    """Backtest engine using multi-timeframe signals"""
    
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.analyzer = None
        
    def backtest_all(self, instruments_data: Dict) -> Dict:
        """Run backtest on all instruments"""
        
        self.analyzer = MultiTimeframeAnalyzer(instruments_data)
        
        print("\n" + "="*70)
        print("MULTI-TIMEFRAME BACKTEST ANALYSIS")
        print("="*70)
        
        all_results = {}
        
        for symbol in self.analyzer.data.keys():
            print(f"\nAnalyzing {symbol}...")
            result = self.analyzer.analyze_instrument(symbol)
            
            if result:
                all_results[symbol] = result
                
                # Print summary
                if 'timeframes' in result and 'H4' in result['timeframes']:
                    h4_stats = result['timeframes']['H4']
                    print(f"  H4 Trend: {'UP' if h4_stats['current_bias'] > 0 else 'DOWN'}")
                    print(f"  H4 Data: {h4_stats['candles']} candles")
                
                if 'confluence' in result:
                    conf = result['confluence']
                    aligned = "✓" if conf['h4_h1_aligned'] else "✗"
                    print(f"  H4/H1 Confluence: {aligned} (H4: {conf['h4_direction']}, H1: {conf['h1_direction']})")
        
        return all_results
    
    def generate_report(self, results: Dict, output_dir: Path = Path("results/backtest")):
        """Generate backtest report"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / "multi_timeframe_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_instruments': len(results),
            'instruments_analyzed': list(results.keys()),
            'timeframes_used': ['M1', 'M5', 'M15', 'H1', 'H4']
        }
        
        summary_file = output_dir / "multi_timeframe_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("BACKTEST REPORT GENERATED")
        print("="*70)
        print(f"Results saved to: {output_dir}/")
        print(f"Files:")
        print(f"  - multi_timeframe_analysis.json")
        print(f"  - multi_timeframe_summary.json")
        print("="*70)

def load_all_instrument_data() -> Dict:
    """Load all instruments across all timeframes"""
    
    data_dir = Path("data/mt5_feeds")
    instruments_data = {}
    
    # Get unique instrument names
    csv_files = list(data_dir.glob("*.csv"))
    instrument_names = set()
    
    for f in csv_files:
        name = f.stem.rsplit('_', 1)[0]
        instrument_names.add(name)
    
    print(f"\nLoading {len(instrument_names)} instruments with 5 timeframes each...")
    
    for instrument in sorted(instrument_names):
        instruments_data[instrument] = {}
        
        for tf in ['M1', 'M5', 'M15', 'H1', 'H4']:
            csv_file = data_dir / f"{instrument}_{tf}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                instruments_data[instrument][tf] = df
                print(f"  ✓ {instrument} {tf}: {len(df)} candles")
            else:
                instruments_data[instrument][tf] = None
    
    return instruments_data

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("EDEN MULTI-TIMEFRAME BACKTEST ENGINE")
    print("="*70)
    
    # Load data
    instruments_data = load_all_instrument_data()
    
    # Run analysis
    backtester = MultiTimeframeBacktester()
    results = backtester.backtest_all(instruments_data)
    
    # Generate report
    backtester.generate_report(results)
    
    print("\n✅ Multi-timeframe analysis complete!")
    return results

if __name__ == "__main__":
    results = main()
