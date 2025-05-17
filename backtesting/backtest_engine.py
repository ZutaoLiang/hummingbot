import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import warnings
warnings.filterwarnings("ignore")
import time
from decimal import Decimal
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import socket
# import socks
# socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 10810)
# socket.socket = socks.socksocket

import asyncio
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import aiomultiprocess as amp

from hummingbot.core.data_type.common import TradeType
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation, ExecutorSimulatorBase
from hummingbot.strategy_v2.backtesting.backtesting_data_provider import BacktestingDataProvider
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers.market_making_controller_base import MarketMakingControllerConfigBase

local_timezone_offset_hours = 8

class BacktestResult:
    
    close_type_info_map = {
            CloseType.TIME_LIMIT: 'TimeLmit', CloseType.STOP_LOSS: 'StopLoss', CloseType.TAKE_PROFIT: 'TakeProfit', 
            CloseType.EXPIRED: 'Expired', CloseType.EARLY_STOP: 'EarlyStop', CloseType.TRAILING_STOP: 'TrailingStop', 
            CloseType.INSUFFICIENT_BALANCE: 'InsufficientBalance', CloseType.FAILED: 'Failed', 
            CloseType.COMPLETED: 'Completed', CloseType.POSITION_HOLD: 'PositionHold'}
    
    def __init__(self, backtesting_result: Dict, controller_config: ControllerConfigBase, backtest_resolution, 
                 start_date: datetime, end_date: datetime, trade_cost: float, slippage: float):
        self.processed_data = backtesting_result["processed_data"]["features"]
        self.results = backtesting_result["results"]
        self.executors = backtesting_result["executors"]
        self.controller_config = controller_config
        self.backtest_resolution = backtest_resolution
        self.start_date = start_date
        self.end_date = end_date
        self.trade_cost = trade_cost
        self.slippage = slippage

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
Backtest result for {trading_pair}({self.backtest_resolution}) From: {self.start_date} to: {self.end_date} with trade cost: {self.trade_cost:.2%} and slippage:{self.slippage:.2%}
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
        self.processed_data.index = pd.to_datetime(self.processed_data.timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
        
        return go.Candlestick(
            x=self.processed_data.index,
            open=self.processed_data['open'],
            high=self.processed_data['high'],
            low=self.processed_data['low'],
            close=self.processed_data['close'],
            increasing_line_color='#2ECC71',  # 上涨K线颜色（绿色）
            decreasing_line_color='#E74C3C',  # 下跌K线颜色（红色）
            name='K-Lines',
        )
        

    @staticmethod
    def _get_pnl_trace(executors, line_style: str = "solid"):
        pnl = [e.net_pnl_quote for e in executors]
        cum_pnl = np.cumsum(pnl)
        return go.Scatter(
            x=pd.to_datetime([e.close_timestamp for e in executors], unit="s") + pd.Timedelta(hours=local_timezone_offset_hours),
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
    def _map_close_type(close_type: CloseType):
        return BacktestResult.close_type_info_map.get(close_type, 'None')

    @staticmethod
    def _add_executors_trace(fig, executors: List[ExecutorInfo], row=1, col=1, line_style="solid"):
        for executor in executors:
            start_time = pd.to_datetime(executor.timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
            
            entry_time = executor.custom_info.get("entry_timestamp", None)
            if entry_time is not None:
                entry_time = pd.to_datetime(entry_time, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
                
            entry_price = executor.custom_info["current_position_average_price"]
            end_time = pd.to_datetime(executor.close_timestamp, unit='s') + pd.Timedelta(hours=local_timezone_offset_hours)
            exit_price = executor.custom_info["close_price"]
            close_type = BacktestResult._map_close_type(executor.close_type)
            name = f"Buy-{close_type}" if executor.config.side == TradeType.BUY else f"Sell-{close_type}"

            if executor.filled_amount_quote == 0:
                fig.add_trace(
                    go.Scatter(x=[start_time, end_time], y=[entry_price, entry_price], mode='lines', showlegend=False,
                               line=dict(color='grey', width=2, dash=line_style if line_style == "dash" else None),
                               name=name), row=row, col=col)
            else:
                if executor.net_pnl_quote > Decimal(0):
                    if entry_time is not None:
                        fig.add_trace(go.Scatter(x=[start_time, entry_time], y=[entry_price, entry_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='blue', width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                      row=row, col=col)
                        start_time = entry_time
                    
                    fig.add_trace(go.Scatter(x=[start_time, end_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='green', width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                  row=row, col=col)
                else:
                    if entry_time is not None:
                        fig.add_trace(go.Scatter(x=[start_time, entry_time], y=[entry_price, entry_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='blue', width=4,
                                                       dash=line_style if line_style == "dash" else None), name=name), 
                                      row=row, col=col)
                        start_time = entry_time
                        
                    fig.add_trace(go.Scatter(x=[start_time, end_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=True,
                                             line=dict(color='red', width=4,
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


class CacheableBacktestingDataProvider(BacktestingDataProvider):
    
    def __init__(self, connectors: Dict[str, ConnectorBase], base_dir: str = '.'):
        super().__init__(connectors)
        self.data_dir = os.path.join(base_dir, 'data')
    
    async def init_data(self, start_time: int, end_time: int, candles_config: CandlesConfig):
        self.update_start_end_time(start_time, end_time)
        await self.initialize_trading_rules(candles_config.connector)
        await self.get_candles_feed(candles_config)
    
    async def initialize_trading_rules(self, connector: str):
        if len(self.trading_rules.get(connector, {})) == 0:
            file_path = os.path.join(self.data_dir, self._get_local_trading_rules_file(connector))
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    print(f'Loaded {connector} trading rules from {file_path}')
                    self.trading_rules[connector] = pickle.load(f)
                    return
        
        await super().initialize_trading_rules(connector)
        
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            
        with open(file_path, "wb") as f:
            pickle.dump(self.trading_rules[connector], f)
            print(f'Saved {connector} trading rules to {file_path}')
    
    def update_start_end_time(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time
    
    def _get_local_trading_rules_file(self, key: str):
        return f'{key}-trading-rules.pkl'
    
    def _get_local_market_data_file(self, key: str):
        return f'{key}-{self.start_time}-{self.end_time}.parquet'
    
    def _load_market_data_from_local(self, key: str):
        file_path = os.path.join(self.data_dir, self._get_local_market_data_file(key))
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        print(f'Loaded market data from {file_path}')
        return pd.read_parquet(file_path, engine="pyarrow")
    
    def _save_to_local(self, df: pd.DataFrame, key: str):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        file_path = os.path.join(self.data_dir, self._get_local_market_data_file(key))
        df.to_parquet(file_path, engine="pyarrow")
        print(f'Saved market data to {file_path}')
        
    def _filter_existing_feed(self, existing_feed: pd.DataFrame):
        if existing_feed.empty:
            return existing_feed
        
        existing_feed_start_time = existing_feed["timestamp"].min()
        existing_feed_end_time = existing_feed["timestamp"].max()
        if existing_feed_start_time <= self.start_time and existing_feed_end_time >= self.end_time:
            return existing_feed[existing_feed["timestamp"] <= self.end_time]
        else:
            return pd.DataFrame()

    async def get_candles_feed(self, config: CandlesConfig):
        key = self._generate_candle_feed_key(config)
        existing_feed = self._filter_existing_feed(self.candles_feeds.get(key, pd.DataFrame()))
        if not existing_feed.empty:
            return existing_feed
        
        candles_df = self._load_market_data_from_local(key)
        if candles_df.empty:
            candles_df = await super().get_candles_feed(config)
            self._save_to_local(candles_df, key)
        else:
            self.candles_feeds[key] = candles_df
        return self._filter_existing_feed(candles_df)


class MyPositionExecutorSimulator(ExecutorSimulatorBase):
    
    def simulate(self, df: pd.DataFrame, config: PositionExecutorConfig, executor_refresh_time: int, trade_cost: float, slippage: float) -> ExecutorSimulation:
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
        time_limit = config.triple_barrier_config.time_limit if config.triple_barrier_config.time_limit else None
        time_limit_timestamp = config.timestamp + time_limit if time_limit else last_timestamp
        early_stop_timestamp = min(last_timestamp, config.timestamp + executor_refresh_time) if executor_refresh_time > 0 else last_timestamp

        # Filter dataframe based on the conditions
        executor_simulation = df[df['timestamp'] <= time_limit_timestamp].copy()
        executor_simulation['net_pnl_pct'] = 0.0
        executor_simulation['net_pnl_quote'] = 0.0
        executor_simulation['cum_fees_quote'] = 0.0
        executor_simulation['filled_amount_quote'] = 0.0
        executor_simulation["current_position_average_price"] = float(config.entry_price)

        if pd.isna(start_timestamp):
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.TIME_LIMIT)

        if start_timestamp > early_stop_timestamp:
            executor_simulation = executor_simulation[executor_simulation['timestamp'] <= early_stop_timestamp].copy()
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.EARLY_STOP)

        simulation_filterd = executor_simulation[executor_simulation['timestamp'] >= start_timestamp]
        if simulation_filterd.empty:
            return ExecutorSimulation(config=config, executor_simulation=executor_simulation, close_type=CloseType.TIME_LIMIT)
        
        entry_time = datetime.fromtimestamp(start_timestamp).strftime("%m%d:%H%M")
        last_timestamp = executor_simulation['timestamp'].max()
        
        # entry_price = df.loc[df['timestamp'] == start_timestamp, 'close'].values[0]
        entry_price = float(config.entry_price)
        
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
            take_profit_trigger_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_trigger_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                next_timestamp = timestamps[i+1] if i < count-1 else timestamp
                low = lows[i]
                
                if low <= stop_loss_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = stop_loss_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.STOP_LOSS
                    break

                high = highs[i]
                if high >= take_profit_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = take_profit_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (high/entry_price - 1) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, high/entry_price - 1)
                    if (max_profit_ratio - (low/entry_price - 1)) > trailing_sl_delta_pct:
                        close_timestamp = next_timestamp
                        close_price = entry_price * (1 + max_profit_ratio - trailing_sl_delta_pct) * (1 - side_multiplier*slippage)
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (close_price - entry_price) / entry_price - trade_cost
        else:
            side_multiplier = -1
            take_profit_trigger_price = entry_price * (1 + side_multiplier * (take_profit + trade_cost))
            stop_loss_trigger_price = entry_price * (1 - side_multiplier * stop_loss)
            
            for i in range(count):
                timestamp = timestamps[i]
                next_timestamp = timestamps[i+1] if i < count-1 else timestamp
                high = highs[i]
                
                if stop_loss_trigger_price <= high:
                    close_timestamp = next_timestamp
                    close_price = stop_loss_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.STOP_LOSS
                    break

                low = lows[i]
                if low <= take_profit_trigger_price:
                    close_timestamp = next_timestamp
                    close_price = take_profit_trigger_price * (1 - side_multiplier*slippage)
                    close_type = CloseType.TAKE_PROFIT
                    break
                
                if not trailing_stop_activated:
                    if (1 - low/entry_price) >= trailing_sl_trigger_pct:
                        trailing_stop_activated = True
                        
                if trailing_stop_activated:
                    max_profit_ratio = max(max_profit_ratio, 1 - low/entry_price)
                    if (max_profit_ratio - (1 - high/entry_price)) > trailing_sl_delta_pct:
                        close_timestamp = next_timestamp
                        close_price = entry_price * (1 - (max_profit_ratio - trailing_sl_delta_pct)) * (1 - side_multiplier*slippage)
                        close_type = CloseType.TRAILING_STOP
                        break
                
                close_price = closes[i]
                
            net_pnl_pct = (entry_price - close_price) / entry_price - trade_cost
            
        close_time = "End"
        if close_timestamp is not None:
            close_time = datetime.fromtimestamp(close_timestamp).strftime("%m%d:%H%M")
        else:
            close_timestamp = last_timestamp
        
        executor_simulation = executor_simulation[executor_simulation['timestamp'] <= close_timestamp]
        
        filled_amount_quote = float(config.amount) * entry_price
        net_pnl_quote = filled_amount_quote * net_pnl_pct
        cum_fees_quote = filled_amount_quote * trade_cost
        
        executor_simulation.loc[executor_simulation['timestamp'] >= start_timestamp, ['filled_amount_quote', 'cum_fees_quote']] = [filled_amount_quote, cum_fees_quote]
        
        last_loc = executor_simulation.index[-1]
        executor_simulation.loc[last_loc, "net_pnl_pct"] = net_pnl_pct
        executor_simulation.loc[last_loc, "net_pnl_quote"] = net_pnl_quote
        executor_simulation.loc[last_loc, "filled_amount_quote"] = filled_amount_quote * 2
        executor_simulation.loc[last_loc, "cum_fees_quote"] = cum_fees_quote * 2
        
        # info = f'Pnl:{net_pnl_quote:+.2f}, entry:{entry_price:.7f}, close:{close_price:.7f}, amount:{config.amount:.2f}, quote:{filled_amount_quote:.2f}, ' \
        #         f'[{config.level_id}] {close_type}({entry_time}-{close_time})[{int(close_timestamp-start_timestamp)}s], id:{config.id}'
        # print(f'\033[92m{info}\033[0m' if net_pnl_quote > 0 else f'\033[91m{info}\033[0m')
        
        simulation = ExecutorSimulation(
            config=config,
            executor_simulation=executor_simulation,
            close_type=close_type
        )
        return simulation


class BacktestEngine(BacktestingEngineBase):
    
    def __init__(self, batch: int = 1, base_dir: str = '.'):
        super().__init__()
        self.base_dir = base_dir
        self.batch = batch
        self.backtesting_data_provider = CacheableBacktestingDataProvider(connectors={}, base_dir=base_dir)
        
    def run_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
                     backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001):
        return asyncio.run(self.async_backtest(config_dir, config_path, start_date, end_date, backtest_resolution, trade_cost, slippage))
    
    def get_controller_config(self, config_dir: str, config_path: str):
        return self.get_controller_config_instance_from_yml(controllers_conf_dir_path=config_dir, config_path=config_path)
    
    async def async_backtest(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
                             backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001):
        controller_config = self.get_controller_config(config_dir, config_path)
        return await self.async_backtest_with_config(controller_config, start_date, end_date, backtest_resolution, trade_cost, slippage)
    
    async def async_backtest_with_config(self, controller_config: ControllerConfigBase, start_date: datetime, end_date: datetime, 
                             backtest_resolution: str, trade_cost: float, slippage: float) -> BacktestResult:
        executor_refresh_time = 0
        if isinstance(controller_config, MarketMakingControllerConfigBase):
            executor_refresh_time = int(controller_config.executor_refresh_time)
        
        start = int(start_date.timestamp())
        end = int(end_date.timestamp())
        result = await self.do_backtest(controller_config, start, end, executor_refresh_time, backtest_resolution, trade_cost, slippage)
        
        backtest_result = BacktestResult(result, controller_config, backtest_resolution, start_date, end_date, trade_cost, slippage)
        print(f'[Batch-{self.batch}] {backtest_result.get_results_summary()}')
        
        return backtest_result
    
    async def do_backtest(self,
                          controller_config: ControllerConfigBase,
                          start: int, end: int,
                          executor_refresh_time: int,
                          backtesting_resolution: str = "1m",
                          trade_cost=0.0006, slippage: float=0.001):
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
        # processed_data will be recreated by MarketMakingControllerBase.update_processed_data() if not implemented by sub-class, so we keep it for backtest result
        market_data_features = self.controller.processed_data["features"]
        
        print(f'[Batch-{self.batch}] Prepare market data:{int(time.time() - t)} seconds')
        t = time.time()
        
        for i, row in market_data.iterrows():
            if len(controller_config.candles_config) > 0:
                current_start = start - (450 * CandlesBase.interval_to_seconds[controller_config.candles_config[0].interval])
                current_end = int(row["timestamp_bt"])
                self.backtesting_data_provider.update_start_end_time(current_start, current_end)
                
            await self.controller.update_processed_data()
            await self.update_state(row)
            active_executor_simulation_ids = set([e.config.id for e in self.active_executor_simulations])
            
            for action in self.controller.determine_executor_actions():
                if isinstance(action, CreateExecutorAction):
                    market_data_from_start = market_data.loc[i:]
                    simulation_result = MyPositionExecutorSimulator().simulate(market_data_from_start, action.executor_config, executor_refresh_time, trade_cost, slippage)
                    if simulation_result.executor_simulation.empty or simulation_result.config.id in active_executor_simulation_ids:
                        continue
                    
                    if simulation_result.close_type != CloseType.FAILED:
                        self.active_executor_simulations.append(simulation_result)
                elif isinstance(action, StopExecutorAction):
                    self.handle_stop_action(action, row["timestamp"])
        
        executors_info = self.controller.executors_info
        
        print(f'[Batch-{self.batch}] Simulation:{int(time.time() - t)} seconds')
        t = time.time()
        
        results = self.summarize_results(executors_info, controller_config.total_amount_quote)
        
        print(f'[Batch-{self.batch}] Summraize results:{int(time.time() - t)} seconds')
        
        self.controller.processed_data['features'] = market_data_features
        
        return {
            "executors": executors_info,
            "results": results,
            "processed_data": self.controller.processed_data,
        }


@dataclass
class BacktestParam:
    
    batch: int
    base_dir: str
    config_dict: Dict
    start_date: datetime
    end_date: datetime
    backtest_resolution: str
    trade_cost: float
    slippage: float
    

class ParamOptimization:
    
    async def run_one(self, backtest_param: BacktestParam):
        controller_config = BacktestEngine.get_controller_config_instance_from_dict(backtest_param.config_dict)
        return (
            backtest_param, 
            await BacktestEngine(backtest_param.batch, backtest_param.base_dir)
                        .async_backtest_with_config(controller_config, backtest_param.start_date, backtest_param.end_date, 
                                                    backtest_param.backtest_resolution, backtest_param.trade_cost, backtest_param.slippage)
        )
        
    async def async_run_all(self, backtest_params: List[BacktestParam]):
        backtest_param = backtest_params[0]
        result_dir = os.path.join(backtest_param.base_dir, 'result')
        start_time = datetime.fromtimestamp(backtest_param.start_date.timestamp()).strftime("%y%m%d%H%M%S")
        end_time = datetime.fromtimestamp(backtest_param.end_date.timestamp()).strftime("%y%m%d%H%M%S")
        result_file = f'{backtest_param.config_dict.get("trading_pair")}-{backtest_param.backtest_resolution}-{start_time}-{end_time}.csv'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        async with amp.Pool(processes = min(os.cpu_count()-1, 128)) as pool:
            results = await pool.map(self.run_one, backtest_params)
            
            rows = []
            for backtest_param, backtest_summary in results:
                row = backtest_param.config_dict
                row.update(backtest_summary.results)
                del row["close_types"]
                rows.append(row)
                
            result_df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
            result_df_cols = ['net_pnl'] + [col for col in result_df.columns if col != 'net_pnl']
            result_df = result_df[result_df_cols]
            result_df.to_csv(os.path.join(result_dir, result_file), index=False)
    
    def run(self, config_dir: str, config_path: str, start_date: datetime, end_date: datetime, 
            backtest_resolution: str = '3m', trade_cost: float = 0.0005, slippage: float = 0.0001):
        base_config_dict = BacktestEngine.load_controller_config(config_path=config_path, controllers_conf_dir_path=config_dir)
        trading_pair = base_config_dict.get('trading_pair')
        candles_config = CandlesConfig(
            connector=base_config_dict.get('connector_name'),
            trading_pair=trading_pair,
            interval=backtest_resolution
        )
        data_provider = CacheableBacktestingDataProvider(connectors={}, base_dir=config_dir)
        
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        asyncio.run(data_provider.init_data(int(start_timestamp), int(end_timestamp), candles_config))
        
        backtest_params = []
        batch = 0
        
        executor_refresh_time_space = range(60, 901, 60)
        stop_loss_space = np.arange(0.02, 0.036, 0.005)
        for executor_refresh_time in executor_refresh_time_space:
            for stop_loss in stop_loss_space:
                config_dict = base_config_dict.copy()
                
                config_dict['executor_refresh_time'] = executor_refresh_time
                config_dict['stop_loss'] = stop_loss
                
                batch += 1
                backtest_param = BacktestParam(batch, config_dir, config_dict, start_date, end_date, backtest_resolution, trade_cost, slippage)
                backtest_params.append(backtest_param)
        
        asyncio.run(self.async_run_all(backtest_params))
        
