from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import NewType


Symbol = NewType("Symbol", str)
DatasetId = NewType("DatasetId", str)
RunId = NewType("RunId", str)


@dataclass(slots=True)
class DateRange:
    """Inclusive start, exclusive end range for time-bounded queries."""

    start: datetime
    end: datetime


@dataclass(slots=True)
class Bar:
    """Raw bar of market data for a symbol.

    This is the common structure produced by all data sources before any
    additional normalization or enrichment.
    """

    symbol: Symbol
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class NormalizedBar(Bar):
    """Bar that has been normalized into the engine's canonical format.

    For now this is identical to ``Bar`` but is modeled separately so future
    transformations (currency conversion, corporate actions, etc.) can be
    reflected without changing call sites.
    """

    # Additional normalized fields can be added here later.
    pass


@dataclass(slots=True)
class RunMetrics:
    """Summary metrics for a single training or simulation run.

    These are computed at the end of a run and can be extended over time
    as new metrics are needed.
    """

    run_id: RunId
    total_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float | None
    num_trades: int
    win_rate: float | None


@dataclass(slots=True)
class Position:
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


@dataclass(slots=True)
class PendingTransaction:
    """Tracks a transaction whose results are pending clearing.

    :param transaction_id: Unique identifier for the transaction.
    :param symbol: Market symbol impacted by the transaction.
    :param quantity: Quantity change to apply once the transaction clears (positive for buy, negative for sell).
    :param amount: Cash impact once the transaction clears (positive for credit, negative for debit).
    :param timestamp: Execution timestamp for the transaction.
    :param side: Side of the trade that produced the transaction.
    """

    transaction_id: str
    symbol: Symbol
    quantity: float
    amount: float
    timestamp: datetime
    side: str


@dataclass(slots=True)
class Account:
    """Training account state including balances and pending activity.

    :param account_id: Identifier for the account.
    :param base_currency: Currency used for the account.
    :param cleared_balance: Funds currently available for trading.
    :param pending_balance: Funds that are awaiting clearing.
    :param reserved_balance: Funds reserved for orders that have been requested but not executed.
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
    positions: dict[Symbol, Position] = field(default_factory=dict)
    clearing_delay_hours: int = 24
    use_business_days: bool = False
    pending_transactions: list[PendingTransaction] = field(default_factory=list)
