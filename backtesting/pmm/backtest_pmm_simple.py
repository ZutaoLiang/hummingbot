import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

from backtesting import backtest_engine
from backtesting import backtesting_engine

from datetime import datetime

if __name__ == '__main__':
    # engine1 = backtesting_engine.BacktestingEngine()
    # engine1.run_backtest(current_dir, 'backtesting_pmm_trending_adaptive_v2_1.yml', datetime(2025, 5, 9), datetime(2025, 5, 10), '1m')
    
    engine = backtest_engine.BacktestEngine()
    engine.run_backtest(current_dir, 'backtesting_pmm_trending_adaptive_v2_1.yml', datetime(2025, 5, 9), datetime(2025, 5, 10), '1m')
