import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import warnings
warnings.filterwarnings("ignore")
import time
from decimal import Decimal
from typing import List, Dict, Optional
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
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation, ExecutorSimulatorBase
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase


class BacktestResult:
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
        trading_pair = self.controller_config.dict().get('trading_pair')
        return f"""
=====================================================================================================================================    
Backtest result for {trading_pair} From: {self.start_date} to: {self.end_date}
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
        self.processed_data.index = pd.to_datetime(self.processed_data.timestamp, unit='s') + pd.Timedelta(hours=8)
        
        return go.Candlestick(
            x=self.processed_data.index,
            open=self.processed_data['open'],
            high=self.processed_data['high'],
            low=self.processed_data['low'],
            close=self.processed_data['close'],
            increasing_line_color='#2ECC71',  # 上涨K线颜色（绿色）
            decreasing_line_color='#E74C3C',  # 下跌K线颜色（红色）
            name='K线图',
        )
        
        return go.Scatter(x=self.processed_data.index,
                          y=self.processed_data['close'],
                          mode='lines',
                          line=dict(color='#6A8AFF', width=2),
                          )

    @staticmethod
    def _get_pnl_trace(executors, line_style: str = "solid"):
        pnl = [e.net_pnl_quote for e in executors]
        cum_pnl = np.cumsum(pnl)
        return go.Scatter(
            x=pd.to_datetime([e.close_timestamp for e in executors], unit="s") + pd.Timedelta(hours=8),
            y=cum_pnl,
            mode='lines',
            line=dict(color='gold', width=2, dash=line_style if line_style == "dash" else None),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            name='Cumulative PNL'
        )

    @staticmethod
    def _get_default_layout(title=None, height=800, width=1200):
        layout = {
            "template": "plotly_dark",
            "plot_bgcolor": 'rgba(0, 0, 0, 0)',  # Transparent background
            "paper_bgcolor": 'rgba(0, 0, 0, 0.1)',  # Lighter shade for the paper
            "font": {"color": 'white', "size": 15},  # Consistent font color and size
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
    def _add_executors_trace(fig, executors, row=1, col=1, line_style="solid"):
        for executor in executors:
            entry_time = pd.to_datetime(executor.timestamp, unit='s') + pd.Timedelta(hours=8)
            entry_price = executor.custom_info["current_position_average_price"]
            exit_time = pd.to_datetime(executor.close_timestamp, unit='s') + pd.Timedelta(hours=8)
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
                                             line=dict(color='green', width=3,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row,
                                  col=col)
                else:
                    fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=False,
                                             line=dict(color='red', width=3,
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


class MyPositionExecutorSimulator(ExecutorSimulatorBase):
    
    def simulate(self, df: pd.DataFrame, config: PositionExecutorConfig, trade_cost: float) -> ExecutorSimulation:
        if config.triple_barrier_config.open_order_type.is_limit_type():
            entry_condition = (df['low'] <= config.entry_price) if config.side == TradeType.BUY else (df['high'] >= config.entry_price)
            start_timestamp = df[entry_condition]['timestamp'].min()
        else:
            start_timestamp = df['timestamp'].min()        
        last_timestamp = df['timestamp'].max()

        # Set up barriers
        take_profit = float(config.triple_barrier_config.take_profit) if config.triple_barrier_config.take_profit else None
        stop_loss = float(config.triple_barrier_config.stop_loss)
        trailing_sl_trigger_pct = None
        trailing_sl_delta_pct = None
        if config.triple_barrier_config.trailing_stop:
            trailing_sl_trigger_pct = float(config.triple_barrier_config.trailing_stop.activation_price)
            trailing_sl_delta_pct = float(config.triple_barrier_config.trailing_stop.trailing_delta)
        tl = config.triple_barrier_config.time_limit if config.triple_barrier_config.time_limit else None
        tl_timestamp = config.timestamp + tl if tl else last_timestamp

        # Filter dataframe based on the conditions
        executor_simulation = df[df['timestamp'] <= tl_timestamp].copy()
        executor_simulation['net_pnl_pct'] = 0.0
        executor_simulation['net_pnl_quote'] = 0.0
        executor_simulation['cum_fees_quote'] = 0.0
        executor_simulation['filled_amount_quote'] = 0.0
        executor_simulation["current_position_average_price"] = float(config.entry_price)

        if pd.isna(start_timestamp):
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.TIME_LIMIT)

        entry_time = datetime.fromtimestamp(start_timestamp).strftime("%Y%m%d:%H%M%S")
        
        # entry_price = df.loc[df['timestamp'] == start_timestamp, 'close'].values[0]
        entry_price = float(config.entry_price)
        
        simulation_filterd = executor_simulation[executor_simulation['timestamp'] >= start_timestamp]
        
        timestamps = simulation_filterd['timestamp'].values
        lows = simulation_filterd['low'].values
        highs = simulation_filterd['high'].values
        closes = simulation_filterd['close'].values
        
        close_type = CloseType.TIME_LIMIT
        close_timestamp = None
        max_profit_ratio = -1e6
        trailing_stop_activated = False
        net_pnl_pct = 0
        count = len(simulation_filterd)
        
        if config.side == TradeType.BUY:
            side_multiplier = 1
            take_profit_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                low = lows[i]
                
                if low <= stop_loss_price:
                    close_timestamp = timestamp
                    close_price = stop_loss_price
                    close_type = CloseType.STOP_LOSS
                    break

                high = highs[i]
                if high >= take_profit_price:
                    close_timestamp = timestamp
                    close_price = take_profit_price
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (high/entry_price - 1) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, high/entry_price - 1)
                    if (max_profit_ratio - (low/entry_price - 1)) > trailing_sl_delta_pct:
                        close_timestamp = timestamp
                        close_price = entry_price * (1 + max_profit_ratio - trailing_sl_delta_pct)
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (close_price - entry_price) / entry_price - trade_cost
        else:
            side_multiplier = -1
            take_profit_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                high = highs[i]
                
                if stop_loss_price <= high:
                    close_timestamp = timestamp
                    close_price = stop_loss_price
                    close_type = CloseType.STOP_LOSS
                    break

                low = lows[i]
                if low <= take_profit_price:
                    close_timestamp = timestamp
                    close_price = take_profit_price
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (1 - low/entry_price) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, 1 - low/entry_price)
                    if (max_profit_ratio - (1 - high/entry_price)) > trailing_sl_delta_pct:
                        close_timestamp = timestamp
                        close_price = entry_price * (1 - (max_profit_ratio - trailing_sl_delta_pct))
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (entry_price - close_price) / entry_price - trade_cost
            
        filled_amount_quote = float(config.amount) * entry_price
        executor_simulation['filled_amount_quote'] = filled_amount_quote
        cum_fees_quote = trade_cost * filled_amount_quote
        executor_simulation['cum_fees_quote'] = cum_fees_quote
        
        net_pnl_quote = filled_amount_quote * net_pnl_pct
        
        close_time = "End"
        if close_timestamp is not None:
            close_time = datetime.fromtimestamp(close_timestamp).strftime("%Y%m%d:%H%M%S")
            executor_simulation = executor_simulation[executor_simulation['timestamp'] <= close_timestamp]
        
        last_loc = executor_simulation.index[-1]
        executor_simulation.loc[last_loc, "net_pnl_pct"] = net_pnl_pct
        executor_simulation.loc[last_loc, "filled_amount_quote"] = filled_amount_quote * 2
        executor_simulation.loc[last_loc, "net_pnl_quote"] = net_pnl_quote
        executor_simulation.loc[last_loc, "cum_fees_quote"] = cum_fees_quote
        
        print(f'{config.level_id} {close_type}({entry_time}-{close_time}), entry:{entry_price:.7f}, close:{close_price:.7f}, amount:{config.amount:.2f}, quote:{filled_amount_quote:.2f}, net_pnl:{net_pnl_quote:.2f}')
        
        simulation = ExecutorSimulation(
            config=config,
            executor_simulation=executor_simulation,
            close_type=close_type
        )
        return simulation


class BacktestEngine(BacktestingEngineBase):
    
    def __init__(self):
        super().__init__()
        
    def run_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, backtest_resolution: str = '3m', trade_cost: float = 0.0005):
        try:
            __IPYTHON__
            self.async_backtest(config_dir, config_path, start_date, end_date, backtest_resolution, trade_cost)
        except:
            asyncio.run(self.async_backtest(config_dir, config_path, start_date, end_date, backtest_resolution, trade_cost))
    
    async def async_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, backtest_resolution: str = '3m', trade_cost: float = 0.0005):
        controller_config = self.get_controller_config_instance_from_yml(controllers_conf_dir_path=config_dir, config_path=config_path)
        start = int(start_date.timestamp())
        end = int(end_date.timestamp())

        result = await self.do_backtest(controller_config, start, end, backtest_resolution, trade_cost)
        
        backtest_result = BacktestResult(result, controller_config, start_date, end_date)
        print(backtest_result.get_results_summary())
        
        return backtest_result
    
    async def do_backtest(self,
                          controller_config: ControllerConfigBase,
                          start: int, end: int,
                          backtesting_resolution: str = "1m",
                          trade_cost=0.0006):
        self.active_executor_simulations: List[ExecutorSimulation] = []
        self.stopped_executors_info: List[ExecutorInfo] = []
        self.backtesting_resolution = backtesting_resolution
        
        t = time.time()
        self.backtesting_data_provider.update_backtesting_time(start, end)
        await self.backtesting_data_provider.initialize_trading_rules(controller_config.connector_name)
        
        # TODO: 新增的Executor需要注册
        controller_class = controller_config.get_controller_class()
        self.controller = controller_class(config=controller_config, market_data_provider=self.backtesting_data_provider, actions_queue=None)
        
        await self.initialize_backtesting_data_provider()
        
        market_data = self.prepare_market_data()
        
        print(f'Prepare market data:{int(time.time() - t)} seconds')
        t = time.time()
        
        for i, row in market_data.iterrows():
            if len(controller_config.candles_config) > 0:
                current_start = start - (450 * CandlesBase.interval_to_seconds[controller_config.candles_config[0].interval])
                current_end = int(row["timestamp_bt"])
                self.backtesting_data_provider.update_backtesting_time(current_start, current_end)
                
            await self.controller.update_processed_data()
            await self.update_state(row)
            for action in self.controller.determine_executor_actions():
                if isinstance(action, CreateExecutorAction):
                    market_data_from_start = market_data.loc[i:]
                    executor_simulation = MyPositionExecutorSimulator().simulate(market_data_from_start, action.executor_config, trade_cost)
                    if executor_simulation.close_type != CloseType.FAILED:
                        self.manage_active_executors(executor_simulation)
                elif isinstance(action, StopExecutorAction):
                    self.handle_stop_action(action, row["timestamp"])
        
        executors_info = self.controller.executors_info
        
        print(f'Simulation:{int(time.time() - t)} seconds')
        t = time.time()
        
        results = self.summarize_results(executors_info, controller_config.total_amount_quote)
        
        print(f'Summraize results:{int(time.time() - t)} seconds')
        
        return {
            "executors": executors_info,
            "results": results,
            "processed_data": self.controller.processed_data,
        }
 