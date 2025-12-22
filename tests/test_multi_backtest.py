"""Tests for multi-strategy backtesting."""

from datetime import datetime, timezone

import pytest

from trading.strategies.examples import BuyAndHoldStrategy, MovingAverageCrossoverStrategy
from trading.training import ComparisonResult, MultiBacktest, StrategyConfig, compare_strategies
from trading.types import NormalizedBar, Symbol


@pytest.fixture
def sample_bars() -> list[NormalizedBar]:
    """Create sample bar data for testing."""
    bars = []
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 102, 101, 105, 108, 107, 110, 115, 112, 118]

    for i, price in enumerate(prices):
        bars.append(
            NormalizedBar(
                symbol=Symbol("AAPL"),
                timestamp=base_date.replace(day=i + 1),
                open=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000.0,
            )
        )
    return bars


class TestMultiBacktest:
    """Tests for MultiBacktest class."""

    def test_runs_multiple_strategies(self, sample_bars: list[NormalizedBar]) -> None:
        """Runs all configured strategies."""
        strategies = [
            StrategyConfig("Buy Hold", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})),
            StrategyConfig("MA Cross", MovingAverageCrossoverStrategy({"symbol": "AAPL", "short_period": 2, "long_period": 3})),
        ]

        multi = MultiBacktest(
            bars=sample_bars,
            strategies=strategies,
            initial_balance=10000.0,
        )
        result = multi.run()

        assert len(result.results) == 2
        assert "Buy Hold" in result.results
        assert "MA Cross" in result.results

    def test_ranks_strategies_by_return(self, sample_bars: list[NormalizedBar]) -> None:
        """Ranks strategies by total return."""
        strategies = [
            StrategyConfig("Buy Hold", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})),
            StrategyConfig("MA Cross", MovingAverageCrossoverStrategy({"symbol": "AAPL", "short_period": 2, "long_period": 4})),
        ]

        multi = MultiBacktest(
            bars=sample_bars,
            strategies=strategies,
            initial_balance=10000.0,
        )
        result = multi.run()

        # Rankings should be sorted by return (highest first)
        assert len(result.rankings) == 2
        returns = [r[1] for r in result.rankings]
        assert returns == sorted(returns, reverse=True)

    def test_identifies_best_strategy(self, sample_bars: list[NormalizedBar]) -> None:
        """Identifies the best performing strategy."""
        strategies = [
            StrategyConfig("Strategy A", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})),
            StrategyConfig("Strategy B", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 20})),
        ]

        multi = MultiBacktest(
            bars=sample_bars,
            strategies=strategies,
            initial_balance=10000.0,
        )
        result = multi.run()

        # More shares = higher return for rising prices
        assert result.best_strategy == "Strategy B"
        assert result.worst_strategy == "Strategy A"

    def test_summary_table_format(self, sample_bars: list[NormalizedBar]) -> None:
        """Generates formatted summary table."""
        strategies = [
            StrategyConfig("Buy Hold", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})),
        ]

        multi = MultiBacktest(
            bars=sample_bars,
            strategies=strategies,
            initial_balance=10000.0,
        )
        table = multi.summary_table()

        assert "Buy Hold" in table
        assert "Return" in table
        assert "Drawdown" in table
        assert "Best:" in table


class TestCompareStrategies:
    """Tests for compare_strategies convenience function."""

    def test_convenience_function(self, sample_bars: list[NormalizedBar]) -> None:
        """compare_strategies works with tuple format."""
        strategies = [
            ("Buy Hold", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})),
            ("MA Cross", MovingAverageCrossoverStrategy({"symbol": "AAPL"})),
        ]

        result = compare_strategies(
            bars=sample_bars,
            strategies=strategies,
            initial_balance=10000.0,
        )

        assert isinstance(result, ComparisonResult)
        assert len(result.strategies) == 2


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_stores_all_results(self, sample_bars: list[NormalizedBar]) -> None:
        """Stores results for all strategies."""
        strategies = [
            StrategyConfig("A", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})),
            StrategyConfig("B", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 10})),
            StrategyConfig("C", BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 15})),
        ]

        multi = MultiBacktest(bars=sample_bars, strategies=strategies)
        result = multi.run()

        assert result.strategies == ["A", "B", "C"]
        assert len(result.results) == 3
        assert len(result.rankings) == 3



