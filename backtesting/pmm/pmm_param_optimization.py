import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from backtesting import backtest_engine

from datetime import datetime

if __name__ == '__main__':
    start_date = datetime(2025, 5, 24)
    end_date = datetime(2025, 5, 31)

    config_file = 'pmm_param_optimization.yml'

    # engine = backtest_engine.BacktestEngine(batch=1, base_dir=current_dir)
    # engine.run_backtest(current_dir, config_file, start_date, end_date, '1m')
    
    space_level = 100
    param_optimization = backtest_engine.ParamOptimization()
    param_optimization.run(current_dir, config_file, start_date, end_date, space_level, '1m', 0.0005, 0.0005)
