import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_path = os.path.join(current_dir, '../../../hummingbot/')
sys.path.append(root_path)

from backtesting import backtesting_engine

from datetime import datetime

if __name__ == '__main__':
    engine = backtesting_engine.BacktestingEngine()
    engine.run_backtest(current_dir, 'backtesting_pmm_simple_1.yml', datetime(2025, 4, 29), datetime(2025, 4, 30))
    
