import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

from backtesting.legacy import backtesting_engine

from datetime import datetime

if __name__ == '__main__':
    engine = backtesting_engine.BacktestingEngine()
    engine.run_backtest(current_dir, 'backtesting_pmm_simple_2.yml', datetime(2025, 4, 29), datetime(2025, 5, 1), '1m')
