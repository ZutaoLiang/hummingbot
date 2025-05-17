import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

from backtesting import backtest_engine

from datetime import datetime

if __name__ == '__main__':
    start_date = datetime(2025, 5, 15)
    end_date = datetime(2025, 5, 17)

    config_file = 'backtesting_pmm_trending_adaptive_v2_2.yml'

    # engine = backtest_engine.BacktestEngine(batch=1, base_dir=current_dir)
    # engine.run_backtest(current_dir, config_file, start_date, end_date, '1m')
    
    param_optimization = backtest_engine.ParamOptimization()
    param_optimization.run(current_dir, config_file, start_date, end_date, '1m')
