#!/usr/bin/env python3
"""
Master Execution Script for Eden 100% Weekly Returns Optimization
Orchestrates: Data Fetching → Backtesting → Optimization → Tuning Loop
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

class OptimizationOrchestrator:
    """Orchestrate the entire optimization pipeline"""
    
    def __init__(self):
        self.log = []
        self.start_time = datetime.now()
        self.results_dir = Path("results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def log_msg(self, msg: str, level: str = "INFO"):
        """Log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {msg}"
        print(log_entry)
        self.log.append(log_entry)
    
    def run_command(self, cmd: str, description: str) -> bool:
        """Run command and return success status"""
        self.log_msg(f"Starting: {description}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                self.log_msg(f"✅ Completed: {description}")
                return True
            else:
                self.log_msg(f"❌ Failed: {description}", "ERROR")
                self.log_msg(f"   Error: {result.stderr}", "ERROR")
                return False
        except subprocess.TimeoutExpired:
            self.log_msg(f"❌ Timeout: {description}", "ERROR")
            return False
        except Exception as e:
            self.log_msg(f"❌ Exception: {str(e)}", "ERROR")
            return False
    
    def fetch_data(self) -> bool:
        """Step 1: Fetch MT5 data for all instruments"""
        self.log_msg("="*60)
        self.log_msg("STEP 1: FETCH MT5 DATA")
        self.log_msg("="*60)
        
        return self.run_command(
            "python fetch_mt5_data.py",
            "Fetching 1-week data for all 10 instruments"
        )
    
    def validate_data(self) -> bool:
        """Step 2: Validate data quality"""
        self.log_msg("="*60)
        self.log_msg("STEP 2: VALIDATE DATA")
        self.log_msg("="*60)
        
        data_dir = Path("data/mt5_feeds")
        csv_files = list(data_dir.glob("*.csv"))
        
        self.log_msg(f"Found {len(csv_files)} CSV files")
        
        if len(csv_files) >= 10:
            self.log_msg("✅ All instruments data present")
            return True
        else:
            self.log_msg("❌ Missing data for some instruments", "ERROR")
            return False
    
    def run_optimization(self, trials: int = 100, round_num: int = 1) -> Dict:
        """Run optimization with specified trials"""
        self.log_msg("="*60)
        self.log_msg(f"OPTIMIZATION ROUND {round_num}: {trials} trials per instrument")
        self.log_msg("="*60)
        
        try:
            # Import and run optimizer
            from optimizer import PortfolioOptimizer
            
            optimizer = PortfolioOptimizer()
            results = optimizer.optimize_all_instruments(trials=trials, max_workers=4)
            metrics = optimizer.calculate_portfolio_metrics()
            optimizer.save_results()
            
            self.log_msg(f"Round {round_num} Results:")
            self.log_msg(f"  Combined Return: {metrics['total_return_pct']:.2f}%")
            self.log_msg(f"  Winning Instruments: {metrics['winning_instruments']}/{metrics['total_instruments']}")
            self.log_msg(f"  Total Trades: {metrics['total_trades']}")
            
            return metrics
            
        except Exception as e:
            self.log_msg(f"❌ Optimization failed: {str(e)}", "ERROR")
            return None
    
    def iterative_optimization(self, max_rounds: int = 5) -> Dict:
        """Run iterative optimization with increasing trials"""
        
        self.log_msg("="*60)
        self.log_msg("🚀 ITERATIVE OPTIMIZATION TO 100% WEEKLY RETURNS")
        self.log_msg("="*60)
        
        trial_schedule = [50, 100, 200, 300, 500]
        best_metrics = None
        
        for round_num, trials in enumerate(trial_schedule[:max_rounds], 1):
            self.log_msg(f"\n{'='*60}")
            self.log_msg(f"ROUND {round_num}: {trials} trials")
            self.log_msg(f"{'='*60}")
            
            metrics = self.run_optimization(trials=trials, round_num=round_num)
            
            if metrics:
                best_metrics = metrics
                
                # Check if target achieved
                if metrics['target_achieved']:
                    self.log_msg("\n🎉 TARGET ACHIEVED: 100% Weekly Return!")
                    return best_metrics
                
                # Calculate shortfall and recommend next action
                shortfall = 100.0 - metrics['total_return_pct']
                self.log_msg(f"⚠️  Shortfall: {shortfall:.2f}%")
                
                # If we have good progress, continue
                if metrics['total_return_pct'] > 80.0:
                    self.log_msg(f"   Progress: {metrics['total_return_pct']:.2f}% - Close to target")
                elif metrics['total_return_pct'] > 50.0:
                    self.log_msg(f"   Progress: {metrics['total_return_pct']:.2f}% - Good progress")
                else:
                    self.log_msg(f"   Progress: {metrics['total_return_pct']:.2f}% - Consider strategy adjustments")
                
                # Wait before next round
                if round_num < max_rounds:
                    self.log_msg(f"Waiting before Round {round_num+1}...")
                    time.sleep(2)
            else:
                self.log_msg("Optimization round failed, aborting", "ERROR")
                break
        
        if best_metrics and best_metrics['target_achieved']:
            return best_metrics
        else:
            self.log_msg("\n⚠️  100% target not achieved in allocated rounds", "WARNING")
            return best_metrics
    
    def generate_report(self, final_metrics: Dict):
        """Generate final optimization report"""
        
        self.log_msg("="*60)
        self.log_msg("OPTIMIZATION REPORT")
        self.log_msg("="*60)
        
        report = {
            'execution_date': self.start_time.isoformat(),
            'total_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'target_achieved': final_metrics['target_achieved'] if final_metrics else False,
            'final_return_pct': final_metrics['total_return_pct'] if final_metrics else 0,
            'winning_instruments': final_metrics['winning_instruments'] if final_metrics else 0,
            'total_trades': final_metrics['total_trades'] if final_metrics else 0,
            'execution_log': self.log
        }
        
        # Save report
        report_file = self.results_dir / "execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_msg(f"\n💾 Report saved to {report_file}")
        
        return report
    
    def run_full_pipeline(self):
        """Execute full optimization pipeline"""
        
        print("\n" + "="*70)
        print("🎯 EDEN 100% WEEKLY RETURNS OPTIMIZATION PIPELINE")
        print("="*70)
        print(f"Start Time: {self.start_time}")
        print("="*70 + "\n")
        
        # Step 1: Fetch data
        if not self.fetch_data():
            self.log_msg("Cannot proceed without data", "ERROR")
            return False
        
        # Step 2: Validate data
        if not self.validate_data():
            self.log_msg("Cannot proceed with invalid data", "ERROR")
            return False
        
        # Step 3: Iterative optimization
        final_metrics = self.iterative_optimization(max_rounds=5)
        
        # Step 4: Generate report
        report = self.generate_report(final_metrics)
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Target Achieved: {'✅ YES' if report['target_achieved'] else '❌ NO'}")
        print(f"Final Return: {report['final_return_pct']:.2f}%")
        print(f"Winning Instruments: {report['winning_instruments']}")
        print(f"Total Duration: {report['total_duration_hours']:.2f} hours")
        print("="*70)
        
        return report['target_achieved']

def main():
    """Main entry point"""
    orchestrator = OptimizationOrchestrator()
    success = orchestrator.run_full_pipeline()
    
    if success:
        print("\n🎉 OPTIMIZATION SUCCESSFUL - 100% WEEKLY RETURNS ACHIEVED!")
        return 0
    else:
        print("\n⚠️  Optimization incomplete - continue tuning for higher returns")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
