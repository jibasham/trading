"""Paper trading engine for forward-testing strategies with live data."""

from __future__ import annotations

import json
import signal
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from trading._core import (
    append_paper_order,
    apply_risk_constraints,
    execute_orders,
    load_paper_account,
    paper_account_exists,
    save_paper_account,
)
from trading.paper.quotes import LiveQuoteSource, QuoteSource
from trading.types import (
    Account,
    AnalysisSnapshot,
    Execution,
    NormalizedBar,
    Symbol,
)

if TYPE_CHECKING:
    from trading.strategies.base import Strategy


class PaperTradingConfig(BaseModel):
    """Configuration for paper trading.

    :param account_id: Unique identifier for this paper account.
    :param symbols: List of symbols to trade.
    :param initial_balance: Starting balance if account doesn't exist.
    :param tick_interval: Seconds between strategy evaluations.
    :param max_position_size: Max position size per symbol (optional).
    :param max_leverage: Max leverage allowed.
    :param only_market_hours: Only trade during market hours.
    :param commission_per_trade: Fixed commission per trade.
    :param slippage_pct: Slippage as a percentage (e.g., 0.001 = 0.1%).
    """

    account_id: str
    symbols: list[str]
    initial_balance: float = 10000.0
    tick_interval: float = 30.0
    max_position_size: float | None = None
    max_leverage: float = 1.0
    only_market_hours: bool = True
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.0


class PaperTradingStatus(BaseModel):
    """Current status of paper trading engine.

    :param account_id: Account identifier.
    :param running: Whether engine is currently running.
    :param ticks: Number of ticks processed.
    :param orders_placed: Total orders placed.
    :param last_tick: Timestamp of last tick.
    :param equity: Current account equity.
    :param market_open: Whether market is currently open.
    """

    account_id: str
    running: bool = False
    ticks: int = 0
    orders_placed: int = 0
    last_tick: datetime | None = None
    equity: float = 0.0
    market_open: bool = False


class PaperTradingEngine:
    """Engine for paper trading with live market data.

    Example usage::

        from trading.paper import PaperTradingEngine
        from trading.strategies import BuyAndHoldStrategy

        config = PaperTradingConfig(
            account_id="my_paper_account",
            symbols=["AAPL", "GOOGL"],
            initial_balance=10000.0,
            tick_interval=60.0,  # Check every minute
        )

        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})
        engine = PaperTradingEngine(config, strategy)

        # Run until interrupted
        engine.run()

    :param config: Paper trading configuration.
    :param strategy: Strategy to run.
    :param quote_source: Source for live quotes (defaults to Yahoo Finance).
    """

    def __init__(
        self,
        config: PaperTradingConfig,
        strategy: Strategy,
        quote_source: QuoteSource | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.quote_source = quote_source or LiveQuoteSource()

        self._running = False
        self._ticks = 0
        self._orders_placed = 0

        # Load or create account
        self.account = self._load_or_create_account()

    def _load_or_create_account(self) -> Account:
        """Load existing account or create a new one."""
        if paper_account_exists(self.config.account_id):
            account_json = load_paper_account(self.config.account_id)
            if account_json:
                return Account.model_validate_json(account_json)

        # Create new account
        return Account(
            account_id=self.config.account_id,
            base_currency="USD",
            cleared_balance=self.config.initial_balance,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
            clearing_delay_hours=0,  # Instant clearing for paper trading
            use_business_days=False,
            pending_transactions=[],
        )

    def _save_account(self) -> None:
        """Persist account state to disk."""
        save_paper_account(self.config.account_id, self.account.model_dump_json())

    def _log_order(self, execution: Execution) -> None:
        """Log an order execution."""
        order_data = {
            "timestamp": execution.timestamp.isoformat() if execution.timestamp else datetime.now(timezone.utc).isoformat(),
            "symbol": str(execution.symbol),
            "side": execution.side,
            "quantity": execution.quantity,
            "price": execution.price,
            "order_id": execution.order_id,
        }
        append_paper_order(self.config.account_id, json.dumps(order_data))

    def status(self) -> PaperTradingStatus:
        """Get current engine status.

        :returns: Current status information.
        """
        # Calculate equity
        equity = self.account.cleared_balance + self.account.pending_balance
        # Note: Would need current prices to get accurate equity

        return PaperTradingStatus(
            account_id=self.config.account_id,
            running=self._running,
            ticks=self._ticks,
            orders_placed=self._orders_placed,
            last_tick=datetime.now(timezone.utc) if self._ticks > 0 else None,
            equity=equity,
            market_open=self.quote_source.is_market_open(),
        )

    def tick(self) -> list[Execution]:
        """Execute one tick of the trading loop.

        Fetches current quotes, runs strategy, and executes any orders.

        :returns: List of executions from this tick.
        """
        self._ticks += 1
        now = datetime.now(timezone.utc)

        # Skip if market closed and configured to respect market hours
        if self.config.only_market_hours and not self.quote_source.is_market_open():
            return []

        # Fetch current quotes
        symbols = [Symbol(s) for s in self.config.symbols]
        quotes = self.quote_source.get_quotes(symbols)

        if not quotes:
            return []

        # Build snapshot for strategy
        bars_dict: dict[str, NormalizedBar] = quotes
        snapshot = AnalysisSnapshot(
            timestamp=now,
            bars=bars_dict,
            account=self.account,
        )

        # Get strategy decisions
        order_requests = self.strategy.decide(snapshot, self.account)

        if not order_requests:
            self._save_account()
            return []

        # Convert to dicts for Rust
        orders_for_rust = [
            {
                "symbol": str(req.symbol),
                "side": req.side,
                "quantity": req.quantity,
            }
            for req in order_requests
        ]

        bars_for_rust: dict[str, dict[str, float]] = {}
        for sym, bar in quotes.items():
            bars_for_rust[sym] = {
                "close": float(bar.close),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume),
            }

        # Apply risk constraints
        accepted_orders, rejected_orders = apply_risk_constraints(
            orders_for_rust,
            self.account.model_dump(),
            bars_for_rust,
            self.config.max_position_size,
            self.config.max_leverage,
        )

        if not accepted_orders:
            self._save_account()
            return []

        # Execute orders
        account_dict = self.account.model_dump()
        executions_raw = execute_orders(
            accepted_orders,
            bars_for_rust,
            account_dict,
            now,
            self.config.commission_per_trade if self.config.commission_per_trade > 0 else None,
            self.config.slippage_pct if self.config.slippage_pct > 0 else None,
        )

        # Update account
        self.account = Account.model_validate(account_dict)

        # Record executions
        executions: list[Execution] = []
        for i, exec_dict in enumerate(executions_raw):
            order_id = f"paper-{self.config.account_id}-{self._orders_placed + i}"
            execution = Execution(
                symbol=Symbol(exec_dict["symbol"]),
                side=exec_dict["side"],
                quantity=exec_dict["quantity"],
                price=exec_dict["price"],
                timestamp=exec_dict["timestamp"],
                order_id=order_id,
                commission=exec_dict.get("commission", 0.0),
                slippage_pct=exec_dict.get("slippage_pct", 0.0),
            )
            executions.append(execution)
            self._log_order(execution)

        self._orders_placed += len(executions)

        # Persist account
        self._save_account()

        return executions

    def run(self, max_ticks: int | None = None) -> None:
        """Run the paper trading loop.

        Runs continuously until interrupted (Ctrl+C) or max_ticks reached.

        :param max_ticks: Optional maximum number of ticks to run.
        """
        self._running = True

        # Set up signal handler for graceful shutdown
        def signal_handler(signum: int, frame: Any) -> None:
            print("\n🛑 Stopping paper trading...")
            self._running = False

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            self.strategy.on_start()
            print(f"🚀 Starting paper trading: {self.config.account_id}")
            print(f"   Symbols: {', '.join(self.config.symbols)}")
            print(f"   Balance: ${self.account.cleared_balance:,.2f}")
            print(f"   Tick interval: {self.config.tick_interval}s")
            print("   Press Ctrl+C to stop\n")

            while self._running:
                if max_ticks and self._ticks >= max_ticks:
                    break

                try:
                    executions = self.tick()
                    if executions:
                        for ex in executions:
                            print(
                                f"   ✅ {ex.side.upper()} {ex.quantity:.0f} {ex.symbol} @ ${ex.price:.2f}"
                            )

                    # Log periodic status
                    if self._ticks % 10 == 0:
                        status = self.status()
                        market_status = "🟢 OPEN" if status.market_open else "🔴 CLOSED"
                        print(
                            f"   [Tick {self._ticks}] Market: {market_status} | "
                            f"Orders: {self._orders_placed}"
                        )

                except Exception as e:
                    print(f"   ⚠️ Tick error: {e}")

                if self._running:
                    time.sleep(self.config.tick_interval)

        finally:
            signal.signal(signal.SIGINT, original_handler)
            self.strategy.on_end()
            self._save_account()
            self._running = False
            print("\n✅ Paper trading stopped. Account saved.")
            print(f"   Final balance: ${self.account.cleared_balance:,.2f}")
            print(f"   Orders placed: {self._orders_placed}")

    def run_once(self) -> list[Execution]:
        """Run a single tick (useful for testing).

        :returns: List of executions from the tick.
        """
        self.strategy.on_start()
        try:
            return self.tick()
        finally:
            self.strategy.on_end()
            self._save_account()

