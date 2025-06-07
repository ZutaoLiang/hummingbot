from decimal import Decimal
from typing import List, Tuple
from datetime import datetime

import pandas_ta as ta
from pydantic import Field

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TrailingStop, TripleBarrierConfig
from controllers.market_making import pmm_common

class PMMTrendingAdaptiveV6ControllerConfig(MarketMakingControllerConfigBase):
    controller_name: str = "pmm_trending_adaptive_v6"
    candles_config: List[CandlesConfig] = []
    buy_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of buy spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of sell spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sleep_interval: int = Field(
        default="3",
        json_schema_extra={
            "prompt": "Enter the sleep interval seconds(3): ",
            "prompt_on_new": True})
    update_interval: int = Field(
        default="180",
        json_schema_extra={
            "prompt": "Enter the update data interval seconds(180): ",
            "prompt_on_new": True})
    candle_interval: str = Field(
        default="15m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    sma_short_length: int = Field(
        default=10,
        json_schema_extra={"prompt": "Enter the SMA short length(10): ", "prompt_on_new": True})
    sma_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the SMA length(30): ", "prompt_on_new": True})
    cci_length: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the CCI length(30): ", "prompt_on_new": True})
    cci_threshold: int = Field(
        default=60,
        json_schema_extra={"prompt": "Enter the CCI threshold(60): ", "prompt_on_new": True})
    natr_length: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the NATR length(14): ", "prompt_on_new": True})
    widen_spread_multiplier: float = Field(
        default=3,
        json_schema_extra={"prompt": "Enter the widen spread multiplier(3): ", "prompt_on_new": True})
    narrow_spread_multiplier: float = Field(
        default=0.6,
        json_schema_extra={"prompt": "Enter the narrow spread multiplier(0.5): ", "prompt_on_new": True})
    early_stop_activation_price_min: float = Field(
        default=0.008,
        json_schema_extra={"prompt": "Enter the min trailing stop activation (0.008): ", "prompt_on_new": True})
    early_stop_decrease_interval: int = Field(
        default="0",
        json_schema_extra={
            "prompt": "Enter the decrease trailing stop interval seconds(0): ",
            "prompt_on_new": True})
    refresh_time_align: bool = Field(
        default=True, json_schema_extra={"is_updatable": True})


class PMMTrendingAdaptiveV6Controller(MarketMakingControllerBase):
    def __init__(self, config: PMMTrendingAdaptiveV6ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = int(max(config.sma_length, config.cci_length, config.natr_length) * 3)
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector_name,
                trading_pair=config.trading_pair,
                interval=config.candle_interval,
                max_records=self.max_records
            )]
        super().__init__(config, update_interval=self.config.sleep_interval, *args, **kwargs)
        self.update_data_interval = self.config.update_interval
        self.last_update_data_time = self.market_data_provider.time() - self.update_data_interval - 10
        self.last_candle_timestamp = 0
        self.time_align_refreshable = False

    def log_msg(self, msg, current_timestamp=None):
        if not current_timestamp:
            current_timestamp = self.market_data_provider.time()
        return self.logger().info(pmm_common.format_msg(msg, current_timestamp))

    async def update_processed_data(self):
        current_time = self.market_data_provider.time()
        
        if current_time - self.last_update_data_time < self.update_data_interval:
            return
        
        candles = self.market_data_provider.get_candles_df(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           interval=self.config.candle_interval,
                                                           max_records=self.max_records)
        last_candle_timestamp = candles["timestamp"].max()
        if last_candle_timestamp > self.last_candle_timestamp:
            self.last_candle_timestamp = last_candle_timestamp
            self.log_msg(f'Update candles {self.config.trading_pair}-{self.config.candle_interval} processed data up to {pmm_common.format_timestamp(self.last_candle_timestamp)} at {pmm_common.format_timestamp(current_time)}')
        else:
            return
        
        sma_short = ta.sma(candles["close"], length=self.config.sma_short_length, talib=False)
        sma = ta.sma(candles["close"], length=self.config.sma_length, talib=False)
        cci = ta.cci(candles["high"], candles["low"], candles["close"], length=self.config.cci_length, talib=False)
        natr = ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length, scalar=1, talib=False)
        # adx = ta.adx(candles["high"], candles["low"], candles["close"], length=self.config.adx_length)
        
        candle_close = candles["close"].iloc[-1]
        candle_sma_short = sma_short.iloc[-1]
        candle_sma = sma.iloc[-1]
        candle_cci = cci.iloc[-1]
        
        trend = ""
        if candle_close > candle_sma_short and candle_close > candle_sma and candle_cci > self.config.cci_threshold:
            trend = "up"
            if not pmm_common.BACKTESTING:
                self.log_msg(f"Up trend => candle_close:{candle_close:.5f} > candle_sma_short:{candle_sma_short:.5f} and candle_sma:{candle_sma:.5f}, " \
                    f"candle_cci:{candle_cci:.1f} > threshold:{self.config.cci_threshold:.1f}")
        elif candle_close < candle_sma_short and candle_close < candle_sma and candle_cci < -self.config.cci_threshold:
            trend = "down"
            if not pmm_common.BACKTESTING:
                self.log_msg(f"Down trend => candle_close:{candle_close:.5f} < candle_sma_short:{candle_sma_short:.5f} and candle_sma:{candle_sma:.5f}, " \
                    f"candle_cci:{candle_cci:.1f} < threshold:{-self.config.cci_threshold:.1f}")
        
        # should always use 1m candles
        last_high = candles["high"].iloc[-1]
        last_low = candles["low"].iloc[-1]
        reference_price = (last_high + last_low) / 2
        # reference_price = self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
        
        self.processed_data.update(
            {
                "reference_price": Decimal(reference_price),
                "natr": Decimal(natr.iloc[-1]),
                "trend": trend
            }
        )
        
        self.last_update_data_time = current_time
        self.time_align_refreshable = True

        if not pmm_common.BACKTESTING:
            order_book_snapshot = self.market_data_provider.get_order_book_snapshot(self.config.connector_name, self.config.trading_pair)
            self.log_msg(f"{order_book_snapshot.head()}")

    def get_price_and_amount(self, level_id: str) -> Tuple[Decimal, Decimal]:
        """
        Get the spread and amount in quote for a given level id.
        """
        level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)
        spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)
        reference_price = Decimal(self.processed_data["reference_price"])
        base_spread_multiplier = Decimal(self.processed_data["natr"])
        spread_in_pct = Decimal(spreads[int(level)]) * base_spread_multiplier
        
        trend = self.processed_data["trend"]
        
        if trend == "up":
            if trade_type == TradeType.BUY:
                spread_in_pct *= Decimal(self.config.narrow_spread_multiplier)
                if not pmm_common.BACKTESTING:
                    self.log_msg(f"Up trend [Narrow] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
            elif trade_type == TradeType.SELL:
                spread_in_pct *= Decimal(self.config.widen_spread_multiplier)
                if not pmm_common.BACKTESTING:
                    self.log_msg(f"Up trend [Widen] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
        elif trend == "down":
            if trade_type == TradeType.BUY:
                spread_in_pct *= Decimal(self.config.widen_spread_multiplier)
                if not pmm_common.BACKTESTING:
                    self.log_msg(f"Down trend [Widen] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
            elif trade_type == TradeType.SELL:
                spread_in_pct *= Decimal(self.config.narrow_spread_multiplier)
                if not pmm_common.BACKTESTING:
                    self.log_msg(f"Down trend [Narrow] {level_id} spread:{spread_in_pct:.2%}, base spread_multiplier:{base_spread_multiplier:.2%}, reference_price:{reference_price:.5f}")
        
        side_multiplier = Decimal("-1") if trade_type == TradeType.BUY else Decimal("1")
        order_price = reference_price * (1 + side_multiplier * spread_in_pct)
        if order_price <= 0:
            return 0, 0
        
        return order_price, round(Decimal(amounts_quote[int(level)]) / order_price, 2)

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        if price == 0:
            return None
        if amount == 0:
            return None

        # spreads, amounts_quote = self.config.get_spreads_and_amounts_in_quote(trade_type)
        # level = self.get_level_from_level_id(level_id)
        trade_type = self.get_trade_type_from_level_id(level_id)

        reference_price = Decimal(self.processed_data["reference_price"])
        natr = Decimal(self.processed_data["natr"])
        
        initial_config = self.config.triple_barrier_config
        
        # stop_loss = initial_config.stop_loss
        # take_profit = max(initial_config.take_profit, base_spread_multiplier * 4)
        # trailing_stop_activation_price = max(initial_config.trailing_stop.activation_price, base_spread_multiplier * 3)
        # trailing_stop_delta = max(initial_config.trailing_stop.trailing_delta, base_spread_multiplier)
        
        stop_loss  = abs(initial_config.stop_loss * natr)
        take_profit = abs(initial_config.take_profit * natr)
        trailing_stop_activation_price = abs(initial_config.trailing_stop.activation_price * natr)
        trailing_stop_delta = abs(natr * initial_config.trailing_stop.trailing_delta * natr)
        time_limit = initial_config.time_limit
        
        trailing_stop = TrailingStop(activation_price=trailing_stop_activation_price, trailing_delta=trailing_stop_delta)
        
        triple_barrier_config = TripleBarrierConfig(
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_limit=time_limit,
            trailing_stop=trailing_stop,
            open_order_type=OrderType.LIMIT,  # Defaulting to LIMIT as is a Maker Controller
            take_profit_order_type=self.config.triple_barrier_config.take_profit_order_type,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET  # Defaulting to MARKET as per requirement
        )
        
        if not pmm_common.BACKTESTING or pmm_common.LOG_DETAIL:
            self.log_msg(f"Creating executor {level_id} with price: {price:.7f}(reference:{reference_price:.7f}), "
                         f"quote: {amount * price:.7f}, amount: {amount:.1f}, trade_type: {trade_type}, "
                         f"stop_loss: {stop_loss:.3%}, take_profit: {take_profit:.3%}, "
                         f"trailing_stop_activation_price: {trailing_stop_activation_price:.3%}, "
                         f"trailing_stop_delta: {trailing_stop_delta:.3%}")
        
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )

    def get_levels_to_execute(self) -> List[str]:
        current_timestamp = self.market_data_provider.time()
        current_minute = datetime.fromtimestamp(current_timestamp).minute
        candle_seconds = CandlesBase.interval_to_seconds[self.config.candles_config[0].interval]
        prevent_reentry_close_types = [CloseType.TAKE_PROFIT, CloseType.STOP_LOSS, CloseType.TRAILING_STOP]
        working_levels = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: 
                x.is_active
                or (x.close_type == CloseType.STOP_LOSS and current_timestamp - x.close_timestamp < self.config.cooldown_time)
                or (x.close_type in prevent_reentry_close_types and current_timestamp - x.close_timestamp < candle_seconds and current_minute == datetime.fromtimestamp(x.close_timestamp).minute)
        )
        working_levels_ids = [executor.custom_info["level_id"] for executor in working_levels]
        return self.get_not_active_levels_ids(working_levels_ids)

    def determine_early_stop(self):
        executors_to_early_stop = []
        
        if not self.config.refresh_time_align and self.config.early_stop_decrease_interval <= 0:
            return executors_to_early_stop
            
        if self.config.refresh_time_align and not pmm_common.BACKTESTING and self.time_align_refreshable:
            current_timestamp = self.market_data_provider.time()
            current_second = datetime.fromtimestamp(current_timestamp).second
            if current_second < 5:
                for executor_info in self.executors_info:
                    if executor_info.is_active and not executor_info.is_trading:
                        execution_end_timestamp = executor_info.timestamp + self.config.executor_refresh_time
                        zero_second_timestamp = datetime.fromtimestamp(execution_end_timestamp).replace(second=0, microsecond=0).timestamp()
                        if current_timestamp >= zero_second_timestamp:
                            executors_to_early_stop.append(executor_info)
                            self.log_msg(f'Add {executor_info.config.level_id} executor to early stop')
                self.time_align_refreshable = False
                
        if self.config.early_stop_decrease_interval <= 0:
            return executors_to_early_stop
        
        for executor_info in self.executors_info:
            execution_seconds = self.market_data_provider.time() - executor_info.timestamp
            time_factor = round(execution_seconds / float(self.config.early_stop_decrease_interval), 1)
            if time_factor <= 1:
                continue
            
            if not executor_info.is_active or not executor_info.is_trading:
                continue
            early_take_profit = max(self.config.take_profit / Decimal(time_factor), self.config.early_stop_activation_price_min)
            if executor_info.net_pnl_pct > early_take_profit:
                executors_to_early_stop.append(executor_info)
                self.log_msg(f'Add {executor_info} to early take profit, pnl:{executor_info.net_pnl_pct:.2%}, threshold:{early_take_profit:.2f}, execution_time:{execution_seconds}s')
        
        if len(executors_to_early_stop) > 0:
            self.log_msg(f'Added {len(executors_to_early_stop)} executors to early take profit')
            
        return executors_to_early_stop

    def executors_to_early_stop(self) -> List[ExecutorAction]:
        executors = self.determine_early_stop()
        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in executors]
