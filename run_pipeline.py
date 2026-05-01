import time
import sys
from src.data_ingestion import run_data_ingestion
from src.feature_engineering import run_feature_engineering
from src.optimization import run_optimization
from src.model import run_model
from src.backtest import run_backtest


def main():
    pipeline_start = time.time()
    
    # Define pipeline stages
    steps = [
        ("Data Ingestion", run_data_ingestion),
        ("Feature Engineering", run_feature_engineering),
        ("Optimization", run_optimization),
        ("Model Training", run_model),
        ("Backtesting", run_backtest)
    ]

    print(f"--- Pipeline Started: {len(steps)} stages to execute ---")

    for i, (name, func) in enumerate(steps, 1):
        step_start = time.time()
        
        # Progress status
        sys.stdout.write(f"[{i}/{len(steps)}] {name}... ")
        sys.stdout.flush()
        
        try:
            func()
            elapsed = (time.time() - step_start) / 60
            sys.stdout.write(f"Done ({elapsed:.2f}m)\n")
        except Exception as e:
            sys.stdout.write("FAILED\n")
            print(f"\n[!] Critical error during {name}: {e}")
            sys.exit(1)

    # Final summary
    total_time = (time.time() - pipeline_start) / 60
    print(f"--- Pipeline Completed Successfully in {total_time:.2f}m ---\n")

if __name__ == "__main__":
    main()