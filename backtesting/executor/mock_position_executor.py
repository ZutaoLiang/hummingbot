from datetime import datetime
import os
import sys

from hummingbot.core.data_type.trade_fee import TradeFeeBase, TradeFeeSchema

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))

import uuid
from typing import Dict
from decimal import Decimal

from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate 
from hummingbot.core.event.events import BuyOrderCompletedEvent, BuyOrderCreatedEvent, OrderCancelledEvent, OrderFilledEvent, SellOrderCompletedEvent, SellOrderCreatedEvent
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.backtesting.backtesting_data_provider import BacktestingDataProvider
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor


class MockConnectorStrategy(ScriptStrategyBase):
    
    def __init__(self, market_data_provider: BacktestingDataProvider, connectors={}, config = None):
        super().__init__(connectors, config)
        self.ready_to_trade = True
        self.market_data_provider = market_data_provider
        
    def buy(self, connector_name, trading_pair, amount, order_type, price=Decimal("NaN"), position_action=PositionAction.OPEN):
        return uuid.uuid4()
    
    def sell(self, connector_name, trading_pair, amount, order_type, price=Decimal("NaN"), position_action=PositionAction.OPEN):
        return uuid.uuid4()
    
    def cancel(self, connector_name, trading_pair, order_id):
        return
    
    @property
    def current_timestamp(self) -> Decimal:
        return self.market_data_provider.time()
    

class MockPositionExecutor(PositionExecutor):
    
    def __init__(self, config: PositionExecutorConfig, market_data_provider: BacktestingDataProvider, trade_cost: float, strategy: MockConnectorStrategy = None, update_interval = 1, max_retries = 10):
        self.market_data_provider = market_data_provider
        self.trade_cost = trade_cost
        self.in_flight_orders: Dict[str, InFlightOrder] = {}
        self.entry_timestamp = None
        
        if strategy is None:
            strategy = MockConnectorStrategy(market_data_provider=market_data_provider, connectors={})
        
        super().__init__(strategy, config, update_interval, max_retries)
        
        self._status = RunnableStatus.RUNNING
    
    def current_time(self):
        return self.market_data_provider.time()
    
    def get_custom_info(self) -> Dict:
        custom_info = super().get_custom_info()
        custom_info['entry_timestamp'] = self.entry_timestamp
        return custom_info
    
    def get_trading_rules(self, connector_name, trading_pair):
        return self.market_data_provider.get_trading_rules(connector_name, trading_pair)
    
    def register_events(self):
        return
    
    def unregister_events(self):
        return

    def stop(self):
        self.place_close_order_and_cancel_open_orders(CloseType.EARLY_STOP)
        self._status = RunnableStatus.TERMINATED
    
    def quantize_order_amount(self, amount):
        trading_rule = self.market_data_provider.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        order_size_quantum = trading_rule.min_order_size
        return (amount // order_size_quantum) * order_size_quantum
    
    @property
    def open_filled_amount(self) -> Decimal:
        if self._open_order:
            if self._open_order.fee_asset == self.config.trading_pair.split("-")[0]:
                open_filled_amount = self._open_order.executed_amount_base - self._open_order.cum_fees_base
            else:
                open_filled_amount = self._open_order.executed_amount_base
            return self.quantize_order_amount(amount=open_filled_amount)
        else:
            return Decimal("0")
    
    def get_in_flight_order(self, connector_name, order_id: str) -> InFlightOrder:
        return self.in_flight_orders.get(order_id, None)
    
    def update_in_flight_order(self, order_id: str, order: InFlightOrder):
        self.in_flight_orders[order_id] = order
        
    def remove_in_flight_order(self, order_id: str):
        if order_id and order_id in self.in_flight_orders.keys():
            del self.in_flight_orders[order_id]
    
    def can_fill(self, order_type: OrderType, side: TradeType, price: Decimal):
        if order_type == OrderType.MARKET:
            return True
        
        market_price = self.get_market_price()
        is_buy = side == TradeType.BUY
        if (is_buy and market_price > price) or (not is_buy and market_price < price):
            return False
        return True            
    
    def update_order_state(self, order: InFlightOrder, order_state: OrderState):
        if not order:
            return
        
        order.update_with_order_update(OrderUpdate(
            trading_pair=order.trading_pair,
            update_timestamp=self.current_time(),
            new_state=order_state,
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id
        ))
    
    def calc_fee(self, order: InFlightOrder) -> TradeFeeBase:
        fee_schema = TradeFeeSchema()
        if self.is_perpetual:
            fee = TradeFeeBase.new_perpetual_fee(
                fee_schema=fee_schema,
                position_action=order.position,
                percent=Decimal(self.trade_cost),
                percent_token=None,
                flat_fees=[]
            )
        else:
            fee = TradeFeeBase.new_spot_fee(
                fee_schema=fee_schema,
                trade_type=order.trade_type,
                percent=Decimal(self.trade_cost),
                percent_token=None,
                flat_fees=[]
            )
        return fee
    
    def update_order_trade(self, order: InFlightOrder):
        if not order:
            return
            
        order.update_with_trade_update(TradeUpdate(
            trade_id=uuid.uuid4(),
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            trading_pair=order.trading_pair,
            fill_timestamp=self.current_time(),
            fill_price=order.price,
            fill_base_amount=order.amount,
            fill_quote_amount=order.amount*order.price,
            fee=self.calc_fee(order)
        ))
    
    def place_order(self, connector_name, trading_pair, order_type, side, amount, position_action = PositionAction.NIL, price=Decimal("NaN")):
        price = self.process_nan_price(price)
        order_id = super().place_order(connector_name, trading_pair, order_type, side, amount, position_action, price)
        
        order = InFlightOrder(
            client_order_id=order_id,
            trading_pair=trading_pair,
            order_type=order_type,
            trade_type=side,
            amount=amount,
            creation_timestamp=self.current_time(),
            price=price,
            exchange_order_id=uuid.uuid4(),
            leverage=self.config.leverage,
            position=position_action,
        )
        
        self.update_in_flight_order(order_id, order)
        return order_id
    
    def place_open_order(self):
        super().place_open_order()
        
        order_id = self._open_order.order_id
        order: InFlightOrder = self.get_in_flight_order(self.config.connector_name, order_id)
        
        event_class = BuyOrderCreatedEvent if self.is_buy() else SellOrderCreatedEvent
        created_event = event_class(
            timestamp=self.current_time(),
            type=order.order_type,
            trading_pair=order.trading_pair,
            amount=order.amount,
            price=order.price,
            order_id=order_id,
            creation_timestamp=order.creation_timestamp,
            exchange_order_id=order.exchange_order_id,
            leverage=order.leverage,
            position=order.position
        )
        
        self.update_order_state(order, OrderState.OPEN)
        self.update_in_flight_order(order_id, order)
        self.process_order_created_event(None, None, created_event)

    def process_nan_price(self, price: Decimal):
        if price.is_nan():
            return self.current_market_price
        return price
        
    def place_close_order_and_cancel_open_orders(self, close_type, price=Decimal("NaN")):
        price = self.process_nan_price(price)
        super().place_close_order_and_cancel_open_orders(close_type, price)
        
        if not self._close_order:
            return
        
        order_id = self._close_order.order_id
        order = self.get_in_flight_order(self.config.connector_name, order_id)
        if not order:
            return
        
        if self._open_order:
            order.creation_timestamp = self._open_order.creation_timestamp
            self.update_in_flight_order(order_id, order)
        
        if self.can_fill(order.order_type, order.trade_type, price):
            self.update_order_state(order, OrderState.FILLED)
            self.update_order_trade(order)
            self.build_and_process_filled_event(order, close_type)

    def format_timestamp(self, current_timestamp):
        return datetime.fromtimestamp(current_timestamp).strftime('%m%d/%H:%M:%S')
    
    def build_and_process_filled_event(self, order: InFlightOrder, close_type: CloseType = None):
        filled_event = OrderFilledEvent(
            timestamp=self.current_time(),
            order_id=order.client_order_id,
            trading_pair=order.trading_pair,
            trade_type=order.trade_type,
            order_type=order.order_type,
            price=order.price,
            amount=order.amount,
            trade_fee=self.calc_fee(order),
            exchange_order_id=order.exchange_order_id,
            leverage=order.leverage,
            position=order.position
        )
        self.process_order_filled_event(None, None, filled_event)
        
        if order.position == PositionAction.CLOSE:
            self.logger().warning(f"[{self.format_timestamp(order.creation_timestamp)}-{self.format_timestamp(filled_event.timestamp)}] Close {close_type.name}: pnl={self.net_pnl_quote:.5f}, price={filled_event.price:.5f}, amount={filled_event.amount*filled_event.price:.5f}, pct={self.net_pnl_pct:.2%}, {filled_event.trade_type.name}-{filled_event.order_type.name}-{filled_event.position.name}")
        else:
            self.logger().info(f"[{self.format_timestamp(order.creation_timestamp)}-{self.format_timestamp(filled_event.timestamp)}] Open filled: price={filled_event.price:.5f}, amount={filled_event.amount*filled_event.price:.5f}, {filled_event.trade_type.name}-{filled_event.order_type.name}-{filled_event.position.name}")

    def place_take_profit_limit_order(self):
        super().place_take_profit_limit_order()
        
        if not self._take_profit_limit_order:
            return
        
        order_id = self._take_profit_limit_order.order_id
        order = self.get_in_flight_order(self.config.connector_name, order_id)
        if not order:
            return
            
        if self._open_order:
            order.creation_timestamp = self._open_order.creation_timestamp
            self.update_in_flight_order(order_id, order)
 
        if self.can_fill(order.order_type, order.trade_type, order.price):
            self.update_order_state(order, OrderState.FILLED)
            self.update_order_trade(order)
            self.build_and_process_completed_event(order)

    def build_and_process_completed_event(self, order: InFlightOrder):
        event_class = BuyOrderCompletedEvent if self.is_buy() else SellOrderCompletedEvent
        if order.order_type == OrderType.MARKET:
            price = self.get_market_price()
        else:
            price = order.price
            
        event = event_class(
            timestamp=self.current_time(),
            order_id=order.client_order_id,
            base_asset=order.base_asset,
            quote_asset=order.quote_asset,
            base_asset_amount=order.amount,
            quote_asset_amount=order.amount*price,
            order_type=order.order_type
        )
        self.process_order_completed_event(None, None, event)
        
    def cancel_order(self, order: TrackedOrder):
        if not order:
            return
        
        order_id = order.order_id
        order = self.get_in_flight_order(self.config.connector_name, order_id)
        if not order:
            return
        
        self.update_order_state(order, OrderState.CANCELED)
        # self.remove_in_flight_order(order_id)
        
        event = OrderCancelledEvent(
            timestamp=self.current_time(),
            order_id=order_id,
            exchange_order_id=order.exchange_order_id
        )
        self.process_order_canceled_event(None, None, event)
    
    def cancel_open_order(self):
        super().cancel_open_order()
        self.cancel_order(self._open_order)
        
    def cancel_take_profit(self):
        super().cancel_take_profit()
        self.cancel_order(self._take_profit_limit_order)
    
    def get_side(self):
        return self.config.side
    
    def is_buy(self):
        return self.get_side() == TradeType.BUY
    
    def get_price(self, connector_name, trading_pair, price_type = PriceType.MidPrice):
        return self.get_market_price(price_type)
    
    def get_market_price(self, price_type = PriceType.MidPrice):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)
    
    def on_market_data(self, market_data) -> bool:
        prices = []
        is_buy = self.is_buy()
        if is_buy:
            prices.extend([market_data['low'], market_data['high']])
        else:
            prices.extend([market_data['high'], market_data['low']])
        
        prices.append(market_data['close'])

        key = f"{self.config.connector_name}_{self.config.trading_pair}"
        for price in prices:
            self.market_data_provider.prices = {key: Decimal(price)}
 
            self.control_open_order()
            
            if self.determine_filled(self._open_order, is_buy):
                self.entry_timestamp = self.current_time()
                
            self.determine_filled(self._take_profit_limit_order, is_buy)
            
            self.control_barriers()
            
            if self.status == RunnableStatus.SHUTTING_DOWN:
                return False
        
        return True
    
    def determine_filled(self, tracked_order: TrackedOrder, is_buy: bool) -> bool:
        if not tracked_order:
            return False
        
        order = tracked_order.order
        if not order or order.is_filled:
            return False
        
        if not self.can_fill(order.order_type, order.trade_type, order.price):
            return False
        
        self.update_order_state(order, OrderState.FILLED)
        self.update_order_trade(order)
        self.build_and_process_filled_event(order, None)
        return True
    