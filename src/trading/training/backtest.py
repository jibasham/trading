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
    :param trade_pnls: P&L for each closed position (round-trip trade).
    """

    run_id: RunId
    metrics: RunMetrics
    final_account: Account
    executions: list[Execution] = Field(default_factory=list)
    equity_history: list[tuple[datetime, float]] = Field(default_factory=list)
    trade_pnls: list[float] = Field(default_factory=list)


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
    :param max_position_size: Optional max position size in dollars (None = no limit).
    :param max_leverage: Max leverage allowed (default 1.0 = no leverage).
    :param checkpoint_interval: Steps between checkpoints (None = no checkpointing).
    :param resume: If True, attempt to resume from checkpoint.
    """

    def __init__(
        self,
        bars: list[NormalizedBar] | list[dict[str, Any]],
        strategy: Strategy,
        initial_balance: float = 10000.0,
        run_id: str | None = None,
        max_position_size: float | None = None,
        max_leverage: float = 1.0,
        checkpoint_interval: int | None = None,
        resume: bool = False,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        """Initialize backtest engine."""
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.run_id = RunId(run_id or self._generate_run_id())
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.rejected_orders: list[dict[str, Any]] = []

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
        import json

        from trading._core import (
            apply_risk_constraints,
            checkpoint_exists,
            compute_run_metrics,
            execute_orders,
            load_checkpoint,
            save_checkpoint,
        )

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
        trade_pnls: list[float] = []  # P&L for each closed position

        # Track open positions for P&L calculation (symbol -> list of (quantity, entry_price))
        open_lots: dict[str, list[tuple[float, float]]] = {}

        # Resume from checkpoint if requested
        start_step = 0
        if self.resume and checkpoint_exists(str(self.run_id)):
            checkpoint_data = load_checkpoint(str(self.run_id))
            if checkpoint_data is not None:
                step_index, state_json = checkpoint_data
                state = json.loads(state_json)
                start_step = step_index + 1
                account = Account.model_validate(state["account"])
                all_executions = [
                    Execution.model_validate(e) for e in state["executions"]
                ]
                equity_history = [
                    (datetime.fromisoformat(h["timestamp"]), h["equity"])
                    for h in state["equity_history"]
                ]
                trade_pnls = state["trade_pnls"]
                open_lots = {
                    k: [(lot["qty"], lot["price"]) for lot in v]
                    for k, v in state["open_lots"].items()
                }

        # Notify strategy of start
        self.strategy.on_start()

        # Iterate through each timestamp
        for step_idx, ts in enumerate(self.timestamps):
            # Skip steps before resume point
            if step_idx < start_step:
                continue
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

                # Apply risk constraints
                account_dict = account.model_dump()
                accepted_orders, rejected = apply_risk_constraints(
                    orders,
                    account_dict,
                    bars_dict,
                    self.max_position_size,
                    self.max_leverage,
                )

                # Track rejected orders
                for rejection in rejected:
                    self.rejected_orders.append(
                        {
                            "timestamp": ts,
                            "order": rejection["order"],
                            "reason": rejection["reason"],
                        }
                    )

                if not accepted_orders:
                    # All orders rejected, continue to next timestamp
                    equity = self._calculate_equity(account, bars_at_ts)
                    equity_history.append((ts, equity))
                    continue

                # Execute accepted orders
                executions = execute_orders(
                    accepted_orders,
                    bars_dict,
                    account_dict,
                    ts,
                    self.commission_per_trade if self.commission_per_trade > 0 else None,
                    self.slippage_pct if self.slippage_pct > 0 else None,
                )

                # Update account from dict
                account = Account.model_validate(account_dict)

                # Record executions and track P&L
                for i, exec_dict in enumerate(executions):
                    order_id = f"{self.run_id}-{len(all_executions) + i}"
                    symbol = exec_dict["symbol"]
                    side = exec_dict["side"]
                    quantity = exec_dict["quantity"]
                    price = exec_dict["price"]

                    all_executions.append(
                        Execution(
                            symbol=Symbol(symbol),
                            side=side,
                            quantity=quantity,
                            price=price,
                            timestamp=exec_dict["timestamp"],
                            order_id=order_id,
                            commission=exec_dict.get("commission", 0.0),
                            slippage_pct=exec_dict.get("slippage_pct", 0.0),
                        )
                    )

                    # Track P&L using FIFO matching
                    if side == "buy":
                        # Add to open lots
                        if symbol not in open_lots:
                            open_lots[symbol] = []
                        open_lots[symbol].append((quantity, price))
                    elif side == "sell":
                        # Match against open lots (FIFO)
                        remaining_qty = quantity
                        if symbol in open_lots:
                            while remaining_qty > 0 and open_lots[symbol]:
                                lot_qty, lot_price = open_lots[symbol][0]
                                match_qty = min(remaining_qty, lot_qty)

                                # Calculate P&L for this match
                                pnl = (price - lot_price) * match_qty
                                trade_pnls.append(pnl)

                                remaining_qty -= match_qty
                                if match_qty >= lot_qty:
                                    open_lots[symbol].pop(0)
                                else:
                                    open_lots[symbol][0] = (
                                        lot_qty - match_qty,
                                        lot_price,
                                    )

            # Calculate current equity
            equity = self._calculate_equity(account, bars_at_ts)
            equity_history.append((ts, equity))

            # Save checkpoint if interval reached
            if (
                self.checkpoint_interval
                and (step_idx + 1) % self.checkpoint_interval == 0
            ):
                checkpoint_state = {
                    "account": account.model_dump(),
                    "executions": [e.model_dump() for e in all_executions],
                    "equity_history": [
                        {"timestamp": h[0].isoformat(), "equity": h[1]}
                        for h in equity_history
                    ],
                    "trade_pnls": trade_pnls,
                    "open_lots": {
                        k: [{"qty": lot[0], "price": lot[1]} for lot in v]
                        for k, v in open_lots.items()
                    },
                }
                save_checkpoint(
                    str(self.run_id), step_idx, json.dumps(checkpoint_state)
                )

        # Notify strategy of end
        self.strategy.on_end()

        # Compute metrics with trade P&L data
        metrics_dict = compute_run_metrics(
            equity_history,
            self.initial_balance,
            len(all_executions),
            trade_pnls if trade_pnls else None,
        )

        metrics = RunMetrics(
            run_id=self.run_id,
            total_return=metrics_dict["total_return"],
            max_drawdown=metrics_dict["max_drawdown"],
            volatility=metrics_dict["volatility"],
            sharpe_ratio=metrics_dict.get("sharpe_ratio"),
            sortino_ratio=metrics_dict.get("sortino_ratio"),
            num_trades=metrics_dict["num_trades"],
            win_rate=metrics_dict.get("win_rate"),
            avg_win=metrics_dict.get("avg_win"),
            avg_loss=metrics_dict.get("avg_loss"),
            profit_factor=metrics_dict.get("profit_factor"),
            expectancy=metrics_dict.get("expectancy"),
        )

        return BacktestResult(
            run_id=self.run_id,
            metrics=metrics,
            final_account=account,
            executions=all_executions,
            equity_history=equity_history,
            trade_pnls=trade_pnls,
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
