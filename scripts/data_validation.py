#!/usr/bin/env python3
"""
Data Validation and Fetching Module
Validates data integrity, checks for missing bars, and exports clean parquet files
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    def __init__(self, config_dir: str = "configs/instruments", data_dir: str = "data"):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_instrument_pool(self) -> List[str]:
        """Load instrument pool from configuration"""
        pool_path = self.config_dir / "instruments_pool.json"
        with open(pool_path, 'r') as f:
            pool_config = json.load(f)
        return pool_config['instruments']
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   resolution: str = '1m') -> pd.DataFrame:
        """
        Fetch historical data for a symbol
        In production, this would connect to your data provider
        For now, generates sample data structure
        """
        print(f"  Fetching {symbol} data from {start_date} to {end_date} ({resolution})")
        
        # TODO: Replace with actual data provider API call
        # Example: MetaTrader5, broker API, or data vendor
        
        # Generate sample data structure (replace with real implementation)
        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': np.random.randn(len(date_range)).cumsum() + 100,
            'high': np.random.randn(len(date_range)).cumsum() + 102,
            'low': np.random.randn(len(date_range)).cumsum() + 98,
            'close': np.random.randn(len(date_range)).cumsum() + 100,
            'volume': np.random.randint(100, 10000, len(date_range))
        })
        
        return df
    
    def validate_data(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict]:
        """Validate data quality and check for missing bars"""
        validation_results = {
            'symbol': symbol,
            'total_rows': len(df),
            'missing_timestamps': 0,
            'null_values': {},
            'duplicates': 0,
            'price_anomalies': 0,
            'status': 'PASS'
        }
        
        # Check for null values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_results['null_values'][col] = int(null_count)
        
        # Check for duplicates
        validation_results['duplicates'] = int(df.duplicated(subset=['timestamp']).sum())
        
        # Check for price anomalies (high < low, negative prices)
        if 'high' in df.columns and 'low' in df.columns:
            anomalies = (df['high'] < df['low']).sum()
            validation_results['price_anomalies'] = int(anomalies)
        
        # Check for missing timestamps (gaps)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            expected_diff = pd.Timedelta('1min')  # Adjust based on resolution
            gaps = (time_diffs > expected_diff * 2).sum()
            validation_results['missing_timestamps'] = int(gaps)
        
        # Determine overall status
        if (validation_results['null_values'] or 
            validation_results['duplicates'] > 0 or 
            validation_results['price_anomalies'] > 0):
            validation_results['status'] = 'WARNING'
        
        if validation_results['missing_timestamps'] > len(df) * 0.1:  # >10% gaps
            validation_results['status'] = 'FAIL'
        
        return validation_results['status'] != 'FAIL', validation_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for backtesting"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Forward fill missing values (conservative approach)
        df = df.fillna(method='ffill')
        
        # Remove any remaining nulls
        df = df.dropna()
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str) -> str:
        """Save cleaned data to parquet format"""
        output_path = self.data_dir / f"{symbol}_clean.parquet"
        df.to_parquet(output_path, compression='snappy', index=False)
        return str(output_path)
    
    def run_validation_pipeline(self, start_date: str = None, end_date: str = None):
        """Run complete validation pipeline for all instruments"""
        # Default to last 6 months
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"\n=== Data Validation Pipeline ===")
        print(f"Date range: {start_date} to {end_date}\n")
        
        instruments = self.load_instrument_pool()
        validation_summary = {
            'run_date': datetime.now().isoformat(),
            'date_range': {'start': start_date, 'end': end_date},
            'instruments': {}
        }
        
        for symbol in instruments:
            print(f"\n[{symbol}] Processing...")
            
            # Fetch data
            df = self.fetch_data(symbol, start_date, end_date)
            
            # Validate
            is_valid, results = self.validate_data(df, symbol)
            
            # Clean
            df_clean = self.clean_data(df)
            
            # Save to parquet
            output_path = self.save_to_parquet(df_clean, symbol)
            
            results['output_file'] = output_path
            results['cleaned_rows'] = len(df_clean)
            validation_summary['instruments'][symbol] = results
            
            status_emoji = "✓" if is_valid else "⚠"
            print(f"  {status_emoji} Status: {results['status']}")
            print(f"  Rows: {results['total_rows']} → {results['cleaned_rows']}")
            print(f"  Output: {output_path}")
        
        # Save validation summary
        summary_path = self.data_dir / "data_validation.json"
        with open(summary_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        print(f"\n=== Validation Complete ===")
        print(f"Summary saved to: {summary_path}")
        
        # Print summary statistics
        total = len(instruments)
        passed = sum(1 for v in validation_summary['instruments'].values() if v['status'] == 'PASS')
        warnings = sum(1 for v in validation_summary['instruments'].values() if v['status'] == 'WARNING')
        failed = sum(1 for v in validation_summary['instruments'].values() if v['status'] == 'FAIL')
        
        print(f"\nResults: {passed} PASS, {warnings} WARNING, {failed} FAIL (of {total} total)")
        
        return validation_summary


if __name__ == "__main__":
    validator = DataValidator()
    validator.run_validation_pipeline()
