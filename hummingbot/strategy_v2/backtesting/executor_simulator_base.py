from decimal import Decimal
from typing import Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class ExecutorSimulation(BaseModel):
    config: Union[PositionExecutorConfig, DCAExecutorConfig]
    executor_simulation: pd.DataFrame
    close_type: CloseType
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('executor_simulation', mode="before")
    @classmethod
    def validate_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("executor_simulation must be a pandas DataFrame")
        return v

    def get_executor_info_at_timestamp(self, timestamp: float) -> ExecutorInfo:
        # Filter the DataFrame up to the specified timestamp
        df_up_to_timestamp = self.executor_simulation[self.executor_simulation['timestamp'] <= timestamp]
        if df_up_to_timestamp.empty:
            return ExecutorInfo(
                id=self.config.id,
                timestamp=self.config.timestamp,
                type=self.config.type,
                status=RunnableStatus.TERMINATED,
                config=self.config,
                net_pnl_pct=Decimal(0),
                net_pnl_quote=Decimal(0),
                cum_fees_quote=Decimal(0),
                filled_amount_quote=Decimal(0),
                is_active=False,
                is_trading=False,
                custom_info={}
            )

        entry_timestamp = self.executor_simulation[
            pd.to_numeric(self.executor_simulation['filled_amount_quote'], errors='coerce') > 0
        ]['timestamp'].min()
        last_entry = df_up_to_timestamp.iloc[-1]
        is_active = last_entry['timestamp'] < self.executor_simulation['timestamp'].max()
        return ExecutorInfo(
            id=self.config.id,
            timestamp=self.config.timestamp,
            type=self.config.type,
            close_timestamp=None if is_active else float(last_entry['timestamp']),
            close_type=None if is_active else self.close_type,
            status=RunnableStatus.RUNNING if is_active else RunnableStatus.TERMINATED,
            config=self.config,
            net_pnl_pct=Decimal(last_entry['net_pnl_pct']),
            net_pnl_quote=Decimal(last_entry['net_pnl_quote']),
            cum_fees_quote=Decimal(last_entry['cum_fees_quote']),
            filled_amount_quote=Decimal(last_entry['filled_amount_quote']),
            is_active=is_active,
            is_trading=last_entry['filled_amount_quote'] > 0 and is_active,
            custom_info=self.get_custom_info(last_entry, entry_timestamp)
        )

    def get_custom_info(self, last_entry: pd.Series, entry_timestamp) -> dict:
        current_position_average_price = last_entry['current_position_average_price'] if "current_position_average_price" in last_entry else None
        return {
            "entry_timestamp": entry_timestamp,
            "close_price": last_entry['close'],
            "level_id": self.config.level_id,
            "side": self.config.side,
            "current_position_average_price": current_position_average_price
        }


class ExecutorSimulatorBase:
    """Base class for trading simulators."""
    def simulate(self, df: pd.DataFrame, config, trade_cost: float) -> ExecutorSimulation:
        """Simulates trading based on provided configuration and market data."""
        # This method should be generic enough to handle various trading strategies.
        raise NotImplementedError
