"""Backtest engine for running strategies against historical data.

This module provides the main orchestration for backtesting trading strategies.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from trading.types import (
    Account,
    AnalysisSnapshot,
    Execution,
    NormalizedBar,
    RunId,
    RunMetrics,
    Symbol,
)

if TYPE_CHECKING:
    from trading.strategies.base import Strategy


class BacktestResult(BaseModel):
    """Results from a backtest run.

    :param run_id: Unique identifier for this run.
    :param metrics: Computed performance metrics.
    :param final_account: Final account state.
    :param executions: All executions during the backtest.
    :param equity_history: List of (timestamp, equity) pairs.
    """

    run_id: RunId
    metrics: RunMetrics
    final_account: Account
    executions: list[Execution] = Field(default_factory=list)
    equity_history: list[tuple[datetime, float]] = Field(default_factory=list)


class Backtest:
    """Backtest engine that runs a strategy against historical data.

    Example usage::

        from trading.training import Backtest
        from trading.strategies.examples import BuyAndHoldStrategy
        from trading.data import YahooDataSource
        from trading.types import DateRange, Symbol
        from datetime import datetime, timezone

        # Fetch data
        source = YahooDataSource()
        bars = list(source.fetch_bars(
            [Symbol("AAPL")],
            DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            "1d"
        ))

        # Run backtest
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})
        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        print(f"Total return: {result.metrics.total_return:.2%}")
        print(f"Max drawdown: {result.metrics.max_drawdown:.2%}")

    :param bars: List of NormalizedBar or bar dicts to backtest against.
    :param strategy: Strategy instance to run.
    :param initial_balance: Starting account balance.
    :param run_id: Optional run ID (auto-generated if not provided).
    """

    def __init__(
        self,
        bars: list[NormalizedBar] | list[dict[str, Any]],
        strategy: Strategy,
        initial_balance: float = 10000.0,
        run_id: str | None = None,
    ) -> None:
        """Initialize backtest engine."""
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.run_id = RunId(run_id or self._generate_run_id())

        # Convert bars to NormalizedBar if needed and organize by timestamp
        self.bars_by_timestamp: dict[datetime, dict[str, NormalizedBar]] = {}
        self._organize_bars(bars)

        # Get sorted timestamps
        self.timestamps = sorted(self.bars_by_timestamp.keys())

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import uuid

        return f"backtest-{uuid.uuid4().hex[:8]}"

    def _organize_bars(self, bars: list[NormalizedBar] | list[dict[str, Any]]) -> None:
        """Organize bars by timestamp for efficient lookup."""
        from trading.types import Bar

        for bar in bars:
            if isinstance(bar, dict):
                # Convert dict to NormalizedBar
                bar_obj = NormalizedBar(
                    symbol=Symbol(bar["symbol"]),
                    timestamp=bar["timestamp"],
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                )
            elif isinstance(bar, Bar) and not isinstance(bar, NormalizedBar):
                # Convert Bar to NormalizedBar
                bar_obj = NormalizedBar(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            else:
                bar_obj = bar

            ts = bar_obj.timestamp
            if ts not in self.bars_by_timestamp:
                self.bars_by_timestamp[ts] = {}

            self.bars_by_timestamp[ts][str(bar_obj.symbol)] = bar_obj

    def run(self) -> BacktestResult:
        """Run the backtest and return results.

        :returns: BacktestResult with metrics and final state.
        """
        from trading._core import compute_run_metrics, execute_orders

        # Initialize account
        account = Account(
            account_id=f"backtest-{self.run_id}",
            base_currency="USD",
            cleared_balance=self.initial_balance,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
            clearing_delay_hours=0,
            use_business_days=False,
            pending_transactions=[],
        )

        # Track state
        all_executions: list[Execution] = []
        equity_history: list[tuple[datetime, float]] = []

        # Notify strategy of start
        self.strategy.on_start()

        # Iterate through each timestamp
        for ts in self.timestamps:
            bars_at_ts = self.bars_by_timestamp[ts]

            # Build snapshot for strategy
            snapshot = AnalysisSnapshot(
                timestamp=ts,
                bars=bars_at_ts,
                account=account,
            )

            # Get strategy decisions
            order_requests = self.strategy.decide(snapshot, account)

            if order_requests:
                # Convert order requests to dicts for Rust
                orders = [
                    {
                        "symbol": str(req.symbol),
                        "side": req.side,
                        "quantity": req.quantity,
                    }
                    for req in order_requests
                ]

                # Convert bars to dicts for Rust
                bars_dict = {
                    symbol: {
                        "close": bar.close,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "volume": bar.volume,
                    }
                    for symbol, bar in bars_at_ts.items()
                }

                # Execute orders
                account_dict = account.model_dump()
                executions = execute_orders(orders, bars_dict, account_dict, ts)

                # Update account from dict
                account = Account.model_validate(account_dict)

                # Record executions
                for i, exec_dict in enumerate(executions):
                    order_id = f"{self.run_id}-{len(all_executions) + i}"
                    all_executions.append(
                        Execution(
                            symbol=Symbol(exec_dict["symbol"]),
                            side=exec_dict["side"],
                            quantity=exec_dict["quantity"],
                            price=exec_dict["price"],
                            timestamp=exec_dict["timestamp"],
                            order_id=order_id,
                        )
                    )

            # Calculate current equity
            equity = self._calculate_equity(account, bars_at_ts)
            equity_history.append((ts, equity))

        # Notify strategy of end
        self.strategy.on_end()

        # Compute metrics
        metrics_dict = compute_run_metrics(
            equity_history,
            self.initial_balance,
            len(all_executions),
        )

        metrics = RunMetrics(
            run_id=self.run_id,
            total_return=metrics_dict["total_return"],
            max_drawdown=metrics_dict["max_drawdown"],
            volatility=metrics_dict["volatility"],
            sharpe_ratio=metrics_dict.get("sharpe_ratio"),
            num_trades=metrics_dict["num_trades"],
            win_rate=metrics_dict.get("win_rate"),
        )

        return BacktestResult(
            run_id=self.run_id,
            metrics=metrics,
            final_account=account,
            executions=all_executions,
            equity_history=equity_history,
        )

    def _calculate_equity(
        self,
        account: Account,
        current_bars: dict[str, NormalizedBar],
    ) -> float:
        """Calculate total equity (cash + positions at market value)."""
        equity = account.cleared_balance + account.pending_balance

        for symbol, position in account.positions.items():
            if symbol in current_bars:
                price = current_bars[symbol].close
                equity += position.quantity * price

        return equity
