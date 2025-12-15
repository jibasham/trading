"""Core type definitions for the trading system.

All data models use Pydantic BaseModel for automatic validation, JSON
serialization, and better error messages.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, NewType

from pydantic import BaseModel, ConfigDict, Field

# Type aliases for domain-specific identifiers
Symbol = NewType("Symbol", str)
DatasetId = NewType("DatasetId", str)
RunId = NewType("RunId", str)


# ---------------------------------------------------------------------------
# Base Configuration
# ---------------------------------------------------------------------------


class FrozenModel(BaseModel):
    """Base model with frozen (immutable) configuration."""

    model_config = ConfigDict(frozen=True)


class MutableModel(BaseModel):
    """Base model for mutable state objects."""

    model_config = ConfigDict(validate_assignment=True)


# ---------------------------------------------------------------------------
# Date/Time Types
# ---------------------------------------------------------------------------


class DateRange(FrozenModel):
    """Inclusive start, exclusive end range for time-bounded queries.

    :param start: Start of the range (inclusive).
    :param end: End of the range (exclusive).
    """

    start: datetime
    end: datetime


# ---------------------------------------------------------------------------
# Market Data Types
# ---------------------------------------------------------------------------


class Bar(FrozenModel):
    """Raw bar of market data for a symbol.

    This is the common structure produced by all data sources before any
    additional normalization or enrichment.

    :param symbol: Market symbol for this bar.
    :param timestamp: Timestamp for this bar (should be timezone-aware).
    :param open: Opening price.
    :param high: Highest price during the bar period.
    :param low: Lowest price during the bar period.
    :param close: Closing price.
    :param volume: Trading volume during the bar period.
    """

    symbol: Symbol
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class NormalizedBar(Bar):
    """Bar that has been normalized into the engine's canonical format.

    For now this is identical to ``Bar`` but is modeled separately so future
    transformations (currency conversion, corporate actions, etc.) can be
    reflected without changing call sites.
    """

    pass


# ---------------------------------------------------------------------------
# Account Types
# ---------------------------------------------------------------------------


class Position(MutableModel):
    """Represents an open position along with pending quantity.

    :param symbol: Market symbol for the position.
    :param quantity: Net quantity of shares held (long positive, short negative).
    :param cost_basis: Average cost per share for the position.
    :param pending_quantity: Quantity that is still pending settlement.
    """

    symbol: Symbol
    quantity: float
    cost_basis: float
    pending_quantity: float = 0.0


class PendingTransaction(FrozenModel):
    """Tracks a transaction whose results are pending clearing.

    :param transaction_id: Unique identifier for the transaction.
    :param symbol: Market symbol impacted by the transaction.
    :param quantity: Quantity change to apply once the transaction clears.
    :param amount: Cash impact once the transaction clears.
    :param timestamp: Execution timestamp for the transaction.
    :param side: Side of the trade that produced the transaction.
    """

    transaction_id: str
    symbol: Symbol
    quantity: float
    amount: float
    timestamp: datetime
    side: str


class Account(MutableModel):
    """Training account state including balances and pending activity.

    :param account_id: Identifier for the account.
    :param base_currency: Currency used for the account.
    :param cleared_balance: Funds currently available for trading.
    :param pending_balance: Funds that are awaiting clearing.
    :param reserved_balance: Funds reserved for pending orders.
    :param positions: Open positions keyed by symbol.
    :param clearing_delay_hours: Clearing delay, measured in hours.
    :param use_business_days: Whether clearing delays count only business days.
    :param pending_transactions: Transactions that are still pending settlement.
    """

    account_id: str
    base_currency: str
    cleared_balance: float
    pending_balance: float
    reserved_balance: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    clearing_delay_hours: int = 24
    use_business_days: bool = False
    pending_transactions: list[PendingTransaction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics Types
# ---------------------------------------------------------------------------


class RunMetrics(FrozenModel):
    """Summary metrics for a single training or simulation run.

    :param run_id: Identifier for the run these metrics belong to.
    :param total_return: Total return as a decimal (0.10 = 10%).
    :param max_drawdown: Maximum drawdown as a decimal.
    :param volatility: Return volatility.
    :param sharpe_ratio: Risk-adjusted return ratio, or None if not computable.
    :param num_trades: Total number of executed trades.
    :param win_rate: Fraction of profitable trades, or None if no trades.
    """

    run_id: RunId
    total_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float | None
    num_trades: int
    win_rate: float | None


# ---------------------------------------------------------------------------
# Order Types
# ---------------------------------------------------------------------------


class OrderStatus(str, Enum):
    """Status of an order in the system."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"


class OrderRequest(FrozenModel):
    """Request from strategy to place an order.

    :param symbol: Market symbol to trade.
    :param side: Trade direction ("buy" or "sell").
    :param quantity: Number of shares to trade.
    :param order_type: Type of order (only "market" supported initially).
    """

    symbol: Symbol
    side: str
    quantity: float
    order_type: str = "market"


class Order(MutableModel):
    """Order with lifecycle tracking.

    :param order_id: Unique identifier for this order.
    :param symbol: Market symbol being traded.
    :param side: Trade direction ("buy" or "sell").
    :param quantity: Requested quantity.
    :param order_type: Type of order.
    :param status: Current order status.
    :param created_at: When the order was created.
    :param filled_quantity: Quantity filled so far.
    :param average_fill_price: Average price of fills, or None if unfilled.
    """

    order_id: str
    symbol: Symbol
    side: str
    quantity: float
    order_type: str
    status: OrderStatus
    created_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: float | None = None


class Execution(FrozenModel):
    """Record of a completed trade execution.

    :param symbol: Market symbol traded.
    :param side: Trade direction.
    :param quantity: Quantity executed.
    :param price: Execution price.
    :param timestamp: When the execution occurred.
    :param order_id: Identifier of the originating order.
    """

    symbol: Symbol
    side: str
    quantity: float
    price: float
    timestamp: datetime
    order_id: str


# ---------------------------------------------------------------------------
# Configuration Types
# ---------------------------------------------------------------------------


class FetchDataConfig(FrozenModel):
    """Configuration for fetching historical market data.

    :param symbols: List of symbols to fetch.
    :param date_range: Time range to fetch.
    :param granularity: Bar granularity (e.g., "5m", "1h", "1d").
    :param data_source: Data source type (e.g., "yahoo", "local", "csv").
    :param source_params: Provider-specific parameters.
    :param dataset_id: Dataset identifier, auto-generated if None.
    """

    symbols: list[Symbol]
    date_range: DateRange
    granularity: str
    data_source: str
    source_params: dict[str, Any] = Field(default_factory=dict)
    dataset_id: DatasetId | None = None


class GenSynthConfig(FrozenModel):
    """Configuration for generating synthetic market data.

    :param symbols: List of synthetic symbol names.
    :param date_range: Time range to generate.
    :param granularity: Bar granularity.
    :param generator_type: Generator algorithm (e.g., "geometric_brownian").
    :param generator_params: Generator-specific parameters.
    :param random_seed: Seed for reproducibility, or None for random.
    :param dataset_id: Dataset identifier, auto-generated if None.
    """

    symbols: list[Symbol]
    date_range: DateRange
    granularity: str
    generator_type: str
    generator_params: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    dataset_id: DatasetId | None = None


class TrainingConfig(FrozenModel):
    """Configuration for a training/simulation run.

    :param run_id: Run identifier, auto-generated if None.
    :param datasets: List of dataset IDs to use.
    :param strategy_class_path: Fully qualified Python class path for strategy.
    :param strategy_params: Parameters to pass to strategy constructor.
    :param account_starting_balance: Initial account balance.
    :param account_base_currency: Account currency.
    :param clearing_delay_hours: Hours until trades clear.
    :param use_business_days: Whether clearing uses business days.
    :param risk_max_position_size: Maximum position size in currency.
    :param risk_max_leverage: Maximum leverage allowed.
    :param analysis_universe: Symbols available for analysis (None = all).
    :param tradable_universe: Symbols that can be traded (None = all).
    :param log_level: Logging level.
    :param checkpoint_interval: Checkpoint every N time slices (None = disabled).
    :param enable_event_logging: Whether to log structured events.
    """

    run_id: RunId | None = None
    datasets: list[DatasetId] = Field(default_factory=list)
    strategy_class_path: str = ""
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    account_starting_balance: float = 10000.0
    account_base_currency: str = "USD"
    clearing_delay_hours: int = 24
    use_business_days: bool = False
    risk_max_position_size: float | None = None
    risk_max_leverage: float = 1.0
    analysis_universe: list[Symbol] | None = None
    tradable_universe: list[Symbol] | None = None
    log_level: str = "INFO"
    checkpoint_interval: int | None = None
    enable_event_logging: bool = True


# ---------------------------------------------------------------------------
# Training/Run Types
# ---------------------------------------------------------------------------


class TimeSlice(FrozenModel):
    """A single time point in a dataset.

    Supports variable sampling rates - bars dict may contain bars from different
    timestamps (the most recent available bar for each symbol at or before this
    timestamp). Missing bars are simply not included in the dict.

    :param timestamp: The evaluation timestamp.
    :param bars: Most recent bars available at or before this timestamp.
    :param bar_timestamps: Actual timestamp of each bar (may differ from slice).
    """

    timestamp: datetime
    bars: dict[str, NormalizedBar] = Field(default_factory=dict)
    bar_timestamps: dict[str, datetime] = Field(default_factory=dict)


class AnalysisSnapshot(FrozenModel):
    """Market snapshot provided to strategies for decision-making.

    :param timestamp: Current evaluation timestamp.
    :param bars: Available market data bars.
    :param account: Current account state.
    """

    timestamp: datetime
    bars: dict[str, NormalizedBar] = Field(default_factory=dict)
    account: Account


class RunState(MutableModel):
    """State maintained during a training run.

    :param run_id: Identifier for this run.
    :param config: Training configuration.
    :param account: Current account state.
    :param time_slices: Processed time slices.
    :param executions: All executions during the run.
    :param order_requests: All order requests generated.
    :param step_records: Per-step records for analysis.
    """

    run_id: RunId
    config: TrainingConfig
    account: Account
    time_slices: list[TimeSlice] = Field(default_factory=list)
    executions: list[Execution] = Field(default_factory=list)
    order_requests: list[OrderRequest] = Field(default_factory=list)
    step_records: list[dict[str, Any]] = Field(default_factory=list)


class DatasetMetadata(FrozenModel):
    """Metadata about a stored dataset.

    :param dataset_id: Unique identifier for this dataset.
    :param symbols: Symbols included in the dataset.
    :param date_range: Time range covered.
    :param granularity: Bar granularity.
    :param data_source: Source of the data.
    :param source_params: Source-specific parameters used.
    :param created_at: When the dataset was created.
    :param bar_count: Number of bars in the dataset.
    """

    dataset_id: DatasetId
    symbols: list[Symbol]
    date_range: DateRange
    granularity: str
    data_source: str
    source_params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    bar_count: int


class RunArtifacts(FrozenModel):
    """All artifacts from a completed training run.

    :param run_id: Identifier for the run.
    :param config: Training configuration used.
    :param metrics: Computed performance metrics.
    :param run_state: Final run state.
    :param created_at: When the run completed.
    """

    run_id: RunId
    config: TrainingConfig
    metrics: RunMetrics
    run_state: RunState
    created_at: datetime


class InspectRunRequest(FrozenModel):
    """Request parameters for inspect-run command.

    :param run_id: Run to inspect.
    :param output_path: Optional path to export data.
    """

    run_id: RunId
    output_path: str | None = None


# ---------------------------------------------------------------------------
# Utility Types
# ---------------------------------------------------------------------------


class DatasetBundle(MutableModel):
    """Container for multiple datasets providing unified access to bars.

    Supports variable sampling rates per symbol and mixed granularities.
    Bars are stored per-symbol with their own timestamps, allowing different
    symbols to have different resolutions and handle missing data gracefully.

    :param datasets: Metadata for included datasets.
    :param bars_by_symbol: Bars indexed by symbol, sorted by timestamp.
    """

    datasets: dict[str, DatasetMetadata] = Field(default_factory=dict)
    bars_by_symbol: dict[str, list[NormalizedBar]] = Field(default_factory=dict)

    def get_bars_at(
        self,
        timestamp: datetime,
        tolerance: timedelta | None = None,
    ) -> dict[str, NormalizedBar]:
        """Get bars available at or near a specific timestamp.

        For each symbol, returns the most recent bar at or before the timestamp.
        If tolerance is provided, only returns bars within tolerance of timestamp.

        :param timestamp: Target timestamp.
        :param tolerance: Maximum age for bars to include.
        :returns: Dictionary mapping symbol to most recent bar.
        """
        result: dict[str, NormalizedBar] = {}
        for symbol, bars in self.bars_by_symbol.items():
            bar = self._find_bar_at_or_before(bars, timestamp)
            if bar is not None:
                if tolerance is None or (timestamp - bar.timestamp) <= tolerance:
                    result[symbol] = bar
        return result

    def get_latest_bar(
        self,
        symbol: str,
        end_time: datetime,
    ) -> NormalizedBar | None:
        """Get the most recent bar for a symbol at or before end_time.

        :param symbol: Symbol to look up.
        :param end_time: Maximum timestamp.
        :returns: Most recent bar or None if not found.
        """
        bars = self.bars_by_symbol.get(symbol, [])
        return self._find_bar_at_or_before(bars, end_time)

    def get_symbol_history(
        self,
        symbol: str,
        end_time: datetime,
        lookback_periods: int,
    ) -> list[NormalizedBar]:
        """Get historical bars for a symbol (for lookback strategies).

        Returns the last N bars for the symbol at or before end_time.
        Handles variable granularity - returns actual bars, not resampled.

        :param symbol: Symbol to look up.
        :param end_time: Maximum timestamp.
        :param lookback_periods: Number of bars to return.
        :returns: List of bars, oldest first.
        """
        bars = self.bars_by_symbol.get(symbol, [])
        # Find index of first bar after end_time
        end_idx = len(bars)
        for i, bar in enumerate(bars):
            if bar.timestamp > end_time:
                end_idx = i
                break
        start_idx = max(0, end_idx - lookback_periods)
        return bars[start_idx:end_idx]

    def get_all_timestamps(self) -> list[datetime]:
        """Get all unique timestamps across all symbols, sorted chronologically.

        :returns: Sorted list of unique timestamps.
        """
        timestamps: set[datetime] = set()
        for bars in self.bars_by_symbol.values():
            for bar in bars:
                timestamps.add(bar.timestamp)
        return sorted(timestamps)

    def get_symbol_timestamps(self, symbol: str) -> list[datetime]:
        """Get all timestamps for a specific symbol, sorted chronologically.

        :param symbol: Symbol to look up.
        :returns: Sorted list of timestamps.
        """
        bars = self.bars_by_symbol.get(symbol, [])
        return [bar.timestamp for bar in bars]

    @staticmethod
    def _find_bar_at_or_before(
        bars: list[NormalizedBar],
        timestamp: datetime,
    ) -> NormalizedBar | None:
        """Binary search for most recent bar at or before timestamp."""
        if not bars:
            return None
        # Simple linear search for now (can optimize with bisect later)
        result = None
        for bar in bars:
            if bar.timestamp <= timestamp:
                result = bar
            else:
                break
        return result


class Gap(FrozenModel):
    """Represents a gap in market data.

    :param symbol: Symbol with the gap.
    :param start_time: Start of the gap.
    :param end_time: End of the gap.
    :param expected_bars: Number of bars expected in this period.
    """

    symbol: Symbol
    start_time: datetime
    end_time: datetime
    expected_bars: int


class RewardSignal(FrozenModel):
    """Reward signal for reinforcement learning.

    :param reward: Numeric reward value.
    :param timestamp: When the reward was computed.
    :param execution_id: Related execution ID, if any.
    :param context: Additional context for the reward.
    """

    reward: float
    timestamp: datetime
    execution_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class RunProgress(FrozenModel):
    """Current progress of a running training run.

    :param run_id: Identifier for the run.
    :param current_timestamp: Current simulation timestamp.
    :param total_timestamps: Total timestamps to process.
    :param completed_timestamps: Timestamps processed so far.
    :param current_account_equity: Current account equity.
    :param num_executions: Number of executions so far.
    :param elapsed_time_seconds: Wall-clock time elapsed.
    """

    run_id: RunId
    current_timestamp: datetime
    total_timestamps: int
    completed_timestamps: int
    current_account_equity: float
    num_executions: int
    elapsed_time_seconds: float


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Type aliases
    "Symbol",
    "DatasetId",
    "RunId",
    # Base models
    "FrozenModel",
    "MutableModel",
    # Date/Time
    "DateRange",
    # Market data
    "Bar",
    "NormalizedBar",
    # Account
    "Position",
    "PendingTransaction",
    "Account",
    # Metrics
    "RunMetrics",
    # Orders
    "OrderStatus",
    "OrderRequest",
    "Order",
    "Execution",
    # Configuration
    "FetchDataConfig",
    "GenSynthConfig",
    "TrainingConfig",
    # Training/Run
    "TimeSlice",
    "AnalysisSnapshot",
    "RunState",
    "DatasetMetadata",
    "RunArtifacts",
    "InspectRunRequest",
    # Utility
    "DatasetBundle",
    "Gap",
    "RewardSignal",
    "RunProgress",
]
