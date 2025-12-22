"""Tests for commission and slippage modeling."""

from datetime import datetime, timezone

import pytest

from trading.strategies import BuyAndHoldStrategy
from trading.training.backtest import Backtest
from trading.types import NormalizedBar, Symbol


@pytest.fixture
def sample_bars() -> list[NormalizedBar]:
    """Create sample bars for testing."""
    return [
        NormalizedBar(
            symbol=Symbol("TEST"),
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=10000.0,
        ),
        NormalizedBar(
            symbol=Symbol("TEST"),
            timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=105.0,
            volume=12000.0,
        ),
        NormalizedBar(
            symbol=Symbol("TEST"),
            timestamp=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
            open=105.0,
            high=110.0,
            low=104.0,
            close=110.0,
            volume=15000.0,
        ),
    ]


class TestCommissionModeling:
    """Tests for commission deduction."""

    def test_no_commission_baseline(self, sample_bars: list[NormalizedBar]) -> None:
        """Baseline test with no commission."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            commission_per_trade=0.0,
        )
        result = backtest.run()

        # Should have one execution (buy on first bar)
        assert len(result.executions) == 1
        assert result.executions[0].commission == 0.0

        # Balance after buy: 10000 - 10*100 = 9000
        # Equity at end: 9000 + 10*110 = 10100
        final_equity = result.equity_history[-1][1]
        assert final_equity == pytest.approx(10100.0, rel=0.01)

    def test_commission_deducted_on_buy(self, sample_bars: list[NormalizedBar]) -> None:
        """Commission is deducted from balance on buy."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            commission_per_trade=5.0,  # $5 per trade
        )
        result = backtest.run()

        # Should have one execution
        assert len(result.executions) == 1
        assert result.executions[0].commission == 5.0

        # Balance after buy: 10000 - 10*100 - 5 = 8995
        # Equity at end: 8995 + 10*110 = 10095
        final_equity = result.equity_history[-1][1]
        assert final_equity == pytest.approx(10095.0, rel=0.01)

    def test_commission_recorded_in_execution(
        self, sample_bars: list[NormalizedBar]
    ) -> None:
        """Commission is recorded in the execution object."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            commission_per_trade=7.50,
        )
        result = backtest.run()

        execution = result.executions[0]
        assert execution.commission == 7.50

    def test_commission_prevents_order_if_insufficient_funds(
        self, sample_bars: list[NormalizedBar]
    ) -> None:
        """Order is rejected if commission + value exceeds balance."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 100})  # 100 * 100 = 10000
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,  # Exactly enough for shares
            commission_per_trade=10.0,  # But not for commission
        )
        result = backtest.run()

        # Order should be rejected due to insufficient funds
        assert len(result.executions) == 0


class TestSlippageModeling:
    """Tests for slippage modeling."""

    def test_no_slippage_baseline(self, sample_bars: list[NormalizedBar]) -> None:
        """Baseline test with no slippage."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            slippage_pct=0.0,
        )
        result = backtest.run()

        # Execution price should be exactly the close price
        assert result.executions[0].price == 100.0
        assert result.executions[0].slippage_pct == 0.0

    def test_slippage_increases_buy_price(
        self, sample_bars: list[NormalizedBar]
    ) -> None:
        """Slippage increases the execution price on buys."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            slippage_pct=0.01,  # 1% slippage
        )
        result = backtest.run()

        # Execution price should be 100 * 1.01 = 101
        assert result.executions[0].price == pytest.approx(101.0, rel=0.001)
        assert result.executions[0].slippage_pct == 0.01

    def test_slippage_affects_balance(self, sample_bars: list[NormalizedBar]) -> None:
        """Slippage reduces effective returns."""
        # Use separate strategy instances to avoid state leakage
        strategy_no_slip = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        strategy_slip = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})

        # Without slippage
        backtest_no_slip = Backtest(
            bars=sample_bars,
            strategy=strategy_no_slip,
            initial_balance=10000.0,
            slippage_pct=0.0,
        )
        result_no_slip = backtest_no_slip.run()

        # With 1% slippage
        backtest_slip = Backtest(
            bars=sample_bars,
            strategy=strategy_slip,
            initial_balance=10000.0,
            slippage_pct=0.01,
        )
        result_slip = backtest_slip.run()

        # Slippage should reduce returns
        # No slip: buy at 100, value = 1000
        # With slip: buy at 101, value = 1010
        # Difference in balance after buy = 10
        final_equity_no_slip = result_no_slip.equity_history[-1][1]
        final_equity_slip = result_slip.equity_history[-1][1]
        assert final_equity_slip < final_equity_no_slip

    def test_slippage_recorded_in_execution(
        self, sample_bars: list[NormalizedBar]
    ) -> None:
        """Slippage percentage is recorded in execution."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            slippage_pct=0.005,  # 0.5%
        )
        result = backtest.run()

        execution = result.executions[0]
        assert execution.slippage_pct == 0.005


class TestCombinedCommissionSlippage:
    """Tests for combined commission and slippage."""

    def test_both_commission_and_slippage(
        self, sample_bars: list[NormalizedBar]
    ) -> None:
        """Commission and slippage work together."""
        strategy = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        backtest = Backtest(
            bars=sample_bars,
            strategy=strategy,
            initial_balance=10000.0,
            commission_per_trade=5.0,
            slippage_pct=0.01,  # 1%
        )
        result = backtest.run()

        execution = result.executions[0]
        assert execution.commission == 5.0
        assert execution.slippage_pct == 0.01
        assert execution.price == pytest.approx(101.0, rel=0.001)

        # Balance after buy:
        # Price with slippage: 101
        # Value: 10 * 101 = 1010
        # Commission: 5
        # Total deducted: 1015
        # Remaining balance: 10000 - 1015 = 8985
        # Final equity: 8985 + 10 * 110 = 10085
        final_equity = result.equity_history[-1][1]
        assert final_equity == pytest.approx(10085.0, rel=0.01)

    def test_realistic_costs(self, sample_bars: list[NormalizedBar]) -> None:
        """Test with realistic trading costs."""
        # Use separate strategy instances to avoid state leakage
        strategy_base = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})
        strategy_costs = BuyAndHoldStrategy({"symbol": "TEST", "quantity": 10})

        # Baseline (no costs)
        backtest_base = Backtest(
            bars=sample_bars,
            strategy=strategy_base,
            initial_balance=10000.0,
        )
        result_base = backtest_base.run()

        # Realistic costs: $1 commission, 0.1% slippage
        backtest_costs = Backtest(
            bars=sample_bars,
            strategy=strategy_costs,
            initial_balance=10000.0,
            commission_per_trade=1.0,
            slippage_pct=0.001,
        )
        result_costs = backtest_costs.run()

        # Costs should reduce returns
        final_equity_base = result_base.equity_history[-1][1]
        final_equity_costs = result_costs.equity_history[-1][1]
        assert final_equity_costs < final_equity_base

        # But impact should be relatively small for small trades
        # Base: buy at 100, end at 110 -> +10% on position
        # With costs: buy at 100.1, commission 1 -> slightly less
        cost_impact = final_equity_base - final_equity_costs
        assert cost_impact > 0  # Costs reduce equity
        assert cost_impact < 20  # But not by too much for this trade size

