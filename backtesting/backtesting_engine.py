import os
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_path = os.path.join(current_dir, '../../hummingbot/')
sys.path.append(root_path)


import logging as _logging
_logger = _logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")
from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import socket
import socks
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 10810)
socket.socket = socks.socksocket

import asyncio

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase


class BacktestingResult:
    def __init__(self, backtesting_result: Dict, controller_config: ControllerConfigBase, start_date: datetime, end_date: datetime):
        self.processed_data = backtesting_result["processed_data"]["features"]
        self.results = backtesting_result["results"]
        self.executors = backtesting_result["executors"]
        self.controller_config = controller_config
        self.start_date = start_date
        self.end_date = end_date

    def get_results_summary(self, results: Optional[Dict] = None):
        if results is None:
            results = self.results
        net_pnl_quote = results["net_pnl_quote"]
        net_pnl_pct = results["net_pnl"]
        max_drawdown = results["max_drawdown_usd"]
        max_drawdown_pct = results["max_drawdown_pct"]
        total_volume = results["total_volume"]
        sharpe_ratio = results["sharpe_ratio"]
        profit_factor = results["profit_factor"]
        total_executors = results["total_executors"]
        accuracy_long = results["accuracy_long"]
        accuracy_short = results["accuracy_short"]
        take_profit = results["close_types"].get("TAKE_PROFIT", 0)
        stop_loss = results["close_types"].get("STOP_LOSS", 0)
        time_limit = results["close_types"].get("TIME_LIMIT", 0)
        trailing_stop = results["close_types"].get("TRAILING_STOP", 0)
        early_stop = results["close_types"].get("EARLY_STOP", 0)
        return f"""
=====================================================================================================================================    
Backtest result From: {self.start_date} to: {self.end_date}
Net PNL: ${net_pnl_quote:.2f} ({net_pnl_pct*100:.2f}%) | Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct*100:.2f}%)
Total Volume ($): {total_volume:.2f} | Sharpe Ratio: {sharpe_ratio:.2f} | Profit Factor: {profit_factor:.2f}
Total Executors: {total_executors} | Accuracy Long: {accuracy_long:.2%} | Accuracy Short: {accuracy_short:.2%}
Close Types: Take Profit: {take_profit} | Trailing Stop: {trailing_stop} | Stop Loss: {stop_loss} | Time Limit: {time_limit} | Early Stop: {early_stop}
=====================================================================================================================================
"""

    @property
    def executors_df(self):
        executors_df = pd.DataFrame([e.dict() for e in self.executors])
        executors_df["side"] = executors_df["config"].apply(lambda x: x["side"].name)
        return executors_df

    def _get_bt_candlestick_trace(self):
        self.processed_data.index = pd.to_datetime(self.processed_data.timestamp, unit='s')
        return go.Scatter(x=self.processed_data.index,
                          y=self.processed_data['close'],
                          mode='lines',
                          line=dict(color="blue"),
                          )

    @staticmethod
    def _get_pnl_trace(executors, line_style: str = "dash"):
        pnl = [e.net_pnl_quote for e in executors]
        cum_pnl = np.cumsum(pnl)
        return go.Scatter(
            x=pd.to_datetime([e.close_timestamp for e in executors], unit="s"),
            y=cum_pnl,
            mode='lines',
            line=dict(color='gold', width=2, dash=line_style if line_style == "dash" else None),
            name='Cumulative PNL'
        )

    @staticmethod
    def _get_default_layout(title=None, height=800, width=1200):
        layout = {
            "template": "plotly_dark",
            "plot_bgcolor": 'rgba(0, 0, 0, 0)',  # Transparent background
            "paper_bgcolor": 'rgba(0, 0, 0, 0.1)',  # Lighter shade for the paper
            "font": {"color": 'white', "size": 12},  # Consistent font color and size
            "height": height,
            "width": width,
            "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
            "xaxis_rangeslider_visible": False,
            "hovermode": "x unified",
            "showlegend": False,
        }
        if title:
            layout["title"] = title
        return layout

    @staticmethod
    def _add_executors_trace(fig, executors, row=1, col=1, line_style="dash"):
        for executor in executors:
            entry_time = pd.to_datetime(executor.timestamp, unit='s')
            entry_price = executor.custom_info["current_position_average_price"]
            exit_time = pd.to_datetime(executor.close_timestamp, unit='s')
            exit_price = executor.custom_info["close_price"]
            name = "Buy Executor" if executor.config.side == TradeType.BUY else "Sell Executor"

            if executor.filled_amount_quote == 0:
                fig.add_trace(
                    go.Scatter(x=[entry_time, exit_time], y=[entry_price, entry_price], mode='lines', showlegend=False,
                               line=dict(color='grey', width=2, dash=line_style if line_style == "dash" else None),
                               name=name), row=row, col=col)
            else:
                if executor.net_pnl_quote > Decimal(0):
                    fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=False,
                                             line=dict(color='green', width=2,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row,
                                  col=col)
                else:
                    fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=False,
                                             line=dict(color='red', width=2,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row, col=col)

        return fig

    def get_backtesting_figure(self):
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, subplot_titles=('Candlestick', 'PNL Quote'),
                            row_heights=[0.7, 0.3])

        # Add candlestick trace
        fig.add_trace(self._get_bt_candlestick_trace(), row=1, col=1)

        # Add executors trace
        fig = self._add_executors_trace(fig, self.executors, row=1, col=1)

        # Add PNL trace
        fig.add_trace(self._get_pnl_trace(self.executors), row=2, col=1)

        # Apply the theme layout
        layout_settings = self._get_default_layout(f"Trading Pair: {self.controller_config.dict().get('trading_pair')}")
        layout_settings["showlegend"] = False
        fig.update_layout(**layout_settings)

        # Update axis properties
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="PNL", row=2, col=1)
        return fig


class BacktestingEngine(BacktestingEngineBase):
    
    def __init__(self):
        super().__init__()
        
    def run_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, backtest_resolution: str = '3m', trade_cost: float = 0.0005):
        asyncio.run(self.async_backtest(config_dir, config_path, start_date, end_date, backtest_resolution, trade_cost))
    
    async def async_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, backtest_resolution: str, trade_cost: float):
        controller_config = self.get_controller_config_instance_from_yml(controllers_conf_dir_path=config_dir, config_path=config_path)
        start = int(start_date.timestamp())
        end = int(end_date.timestamp())

        result = await self.run_backtesting(controller_config, start, end, backtest_resolution, trade_cost)
        
        backtesting_result = BacktestingResult(result, controller_config, start_date, end_date)
        
        _logger.info(backtesting_result.get_results_summary())
        print(backtesting_result.get_results_summary())