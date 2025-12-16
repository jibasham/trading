"""Tests for the backtest engine and strategies."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.strategies.base import Strategy
from trading.strategies.examples import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
)
from trading.training import Backtest
from trading.types import Account, AnalysisSnapshot, NormalizedBar, Symbol


def create_test_bars(
    symbol: str = "AAPL",
    num_bars: int = 20,
    start_price: float = 100.0,
    price_increment: float = 1.0,
) -> list[dict]:
    """Create test bar data with incrementing prices."""
    base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    bars = []
    price = start_price

    for i in range(num_bars):
        bars.append(
            {
                "symbol": symbol,
                "timestamp": base_time + timedelta(days=i),
                "open": price,
                "high": price + 2,
                "low": price - 1,
                "close": price + price_increment,
                "volume": 1000.0,
            }
        )
        price += price_increment

    return bars


class TestStrategy:
    """Tests for Strategy base class."""

    def test_strategy_is_abstract(self) -> None:
        """Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Strategy()  # type: ignore[abstract]

    def test_strategy_requires_decide_method(self) -> None:
        """Subclasses must implement decide."""

        class IncompleteStrategy(Strategy):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteStrategy()


class TestBuyAndHoldStrategy:
    """Tests for BuyAndHoldStrategy."""

    def test_init_with_defaults(self) -> None:
        """Strategy initializes with default parameters."""
        strategy = BuyAndHoldStrategy()

        assert strategy.target_symbol == "AAPL"
        assert strategy.quantity == 10
        assert strategy.has_bought is False

    def test_init_with_custom_params(self) -> None:
        """Strategy accepts custom parameters."""
        strategy = BuyAndHoldStrategy({"symbol": "GOOGL", "quantity": 5})

        assert strategy.target_symbol == "GOOGL"
        assert strategy.quantity == 5

    def test_decide_buys_once(self) -> None:
        """Strategy buys once when bars are available."""
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        # Create snapshot with bars
        bar = NormalizedBar(
            symbol=Symbol("AAPL"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
        )
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
        )
        snapshot = AnalysisSnapshot(
            timestamp=bar.timestamp,
            bars={"AAPL": bar},
            account=account,
        )

        # First call should return buy order
        orders = strategy.decide(snapshot, account)
        assert len(orders) == 1
        assert orders[0].side == "buy"
        assert orders[0].quantity == 10

        # Second call should return empty (already bought)
        orders = strategy.decide(snapshot, account)
        assert len(orders) == 0


class TestMovingAverageCrossoverStrategy:
    """Tests for MovingAverageCrossoverStrategy."""

    def test_init_with_defaults(self) -> None:
        """Strategy initializes with default parameters."""
        strategy = MovingAverageCrossoverStrategy()

        assert strategy.short_period == 5
        assert strategy.long_period == 20
        assert strategy.quantity == 10

    def test_needs_enough_history(self) -> None:
        """Strategy needs enough price history before trading."""
        strategy = MovingAverageCrossoverStrategy({"short_period": 2, "long_period": 5})

        bar = NormalizedBar(
            symbol=Symbol("AAPL"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
        )
        account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
        )
        snapshot = AnalysisSnapshot(
            timestamp=bar.timestamp,
            bars={"AAPL": bar},
            account=account,
        )

        # First few calls should return no orders (not enough history)
        for _ in range(4):
            orders = strategy.decide(snapshot, account)
            assert len(orders) == 0


class TestBacktest:
    """Tests for Backtest engine."""

    def test_backtest_with_no_trades(self) -> None:
        """Backtest with strategy that makes no trades."""

        class NoOpStrategy(Strategy):
            def decide(self, snapshot, account):
                return []

        bars = create_test_bars(num_bars=10)
        strategy = NoOpStrategy()

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        assert result.metrics.total_return == 0.0
        assert result.metrics.num_trades == 0
        assert result.final_account.cleared_balance == 10000.0
        assert len(result.executions) == 0

    def test_backtest_buy_and_hold(self) -> None:
        """Backtest with buy and hold strategy."""
        bars = create_test_bars(
            symbol="AAPL",
            num_bars=10,
            start_price=100.0,
            price_increment=1.0,
        )

        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        # Should have executed 1 trade
        assert result.metrics.num_trades == 1

        # Should have AAPL position
        assert "AAPL" in result.final_account.positions

        # Should have positive return (prices went up)
        assert result.metrics.total_return > 0

    def test_backtest_tracks_equity_history(self) -> None:
        """Backtest tracks equity at each timestamp."""
        bars = create_test_bars(num_bars=10)
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        # Should have equity point for each bar
        assert len(result.equity_history) == 10

        # Equity should generally increase (prices go up)
        equities = [e for _, e in result.equity_history]
        assert equities[-1] > equities[0]

    def test_backtest_with_multiple_symbols(self) -> None:
        """Backtest with multiple symbols."""
        # Create bars for two symbols
        bars_aapl = create_test_bars(symbol="AAPL", num_bars=10, start_price=100.0)
        bars_googl = create_test_bars(symbol="GOOGL", num_bars=10, start_price=2800.0)
        all_bars = bars_aapl + bars_googl

        # Strategy that only trades AAPL
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=all_bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        # Should only have AAPL position
        assert "AAPL" in result.final_account.positions
        assert "GOOGL" not in result.final_account.positions

    def test_backtest_insufficient_funds(self) -> None:
        """Backtest handles insufficient funds gracefully."""
        bars = create_test_bars(start_price=1000.0)  # Expensive stock

        # Try to buy more than we can afford
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 100})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=1000.0,  # Only $1000
        )
        result = backtest.run()

        # Order should be skipped - no trades
        assert result.metrics.num_trades == 0
        assert result.final_account.cleared_balance == 1000.0

    def test_backtest_custom_run_id(self) -> None:
        """Backtest uses provided run ID."""
        bars = create_test_bars(num_bars=5)
        strategy = BuyAndHoldStrategy()

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
            run_id="my-custom-run",
        )
        result = backtest.run()

        assert result.run_id == "my-custom-run"

    def test_backtest_auto_generates_run_id(self) -> None:
        """Backtest generates run ID if not provided."""
        bars = create_test_bars(num_bars=5)
        strategy = BuyAndHoldStrategy()

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        assert result.run_id.startswith("backtest-")

    def test_backtest_result_contains_executions(self) -> None:
        """Backtest result includes execution details."""
        bars = create_test_bars(num_bars=10)
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        assert len(result.executions) == 1
        exec = result.executions[0]
        assert exec.symbol == "AAPL"
        assert exec.side == "buy"
        assert exec.quantity == 10
        assert exec.price > 0
        assert exec.order_id is not None


class TestBacktestMetrics:
    """Tests for backtest metrics computation."""

    def test_metrics_with_profit(self) -> None:
        """Metrics correctly show profit."""
        # Prices go up 10% each day
        bars = create_test_bars(
            num_bars=10,
            start_price=100.0,
            price_increment=10.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        assert result.metrics.total_return > 0
        assert result.metrics.max_drawdown >= 0

    def test_metrics_with_loss(self) -> None:
        """Metrics correctly show loss."""
        # Prices go down
        bars = create_test_bars(
            num_bars=10,
            start_price=100.0,
            price_increment=-5.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})

        backtest = Backtest(
            bars=bars,
            strategy=strategy,
            initial_balance=10000.0,
        )
        result = backtest.run()

        # Should have a loss
        assert result.metrics.total_return < 0
        # Should have some drawdown
        assert result.metrics.max_drawdown > 0
