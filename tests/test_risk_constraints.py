"""Tests for risk constraint functions."""

import pytest

from trading._core import apply_risk_constraints


@pytest.fixture
def sample_account():
    """A sample account with some balance and positions."""
    return {
        "account_id": "test_account",
        "base_currency": "USD",
        "cleared_balance": 10000.0,
        "pending_balance": 0.0,
        "reserved_balance": 0.0,
        "positions": {
            "AAPL": {"quantity": 10.0, "cost_basis": 150.0},
        },
        "clearing_delay_hours": 0,
        "use_business_days": False,
        "pending_transactions": [],
    }


@pytest.fixture
def sample_bars():
    """Sample bar data for testing."""
    return {
        "AAPL": {"open": 150.0, "high": 155.0, "low": 149.0, "close": 152.0, "volume": 1000000},
        "GOOGL": {"open": 140.0, "high": 142.0, "low": 138.0, "close": 141.0, "volume": 500000},
    }


class TestApplyRiskConstraints:
    """Tests for apply_risk_constraints function."""

    def test_accept_valid_buy_order(self, sample_account, sample_bars):
        """Valid buy order with sufficient balance is accepted."""
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 1
        assert len(rejected) == 0
        assert accepted[0]["symbol"] == "GOOGL"

    def test_accept_valid_sell_order(self, sample_account, sample_bars):
        """Valid sell order with sufficient position is accepted."""
        orders = [{"symbol": "AAPL", "side": "sell", "quantity": 5}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_reject_buy_insufficient_balance(self, sample_account, sample_bars):
        """Buy order exceeding balance is rejected."""
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 100}]  # 100 * 141 = 14100 > 10000

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Insufficient balance" in rejected[0]["reason"]

    def test_reject_sell_insufficient_position(self, sample_account, sample_bars):
        """Sell order exceeding position is rejected."""
        orders = [{"symbol": "AAPL", "side": "sell", "quantity": 20}]  # Only have 10

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Insufficient position" in rejected[0]["reason"]

    def test_reject_sell_no_position(self, sample_account, sample_bars):
        """Sell order for unowned symbol is rejected."""
        orders = [{"symbol": "GOOGL", "side": "sell", "quantity": 5}]  # Don't own GOOGL

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Insufficient position" in rejected[0]["reason"]

    def test_reject_order_no_price_data(self, sample_account, sample_bars):
        """Order for symbol without price data is rejected."""
        orders = [{"symbol": "MSFT", "side": "buy", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "No price data" in rejected[0]["reason"]

    def test_reject_invalid_quantity(self, sample_account, sample_bars):
        """Order with zero or negative quantity is rejected."""
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 0}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Invalid quantity" in rejected[0]["reason"]

    def test_reject_unknown_side(self, sample_account, sample_bars):
        """Order with unknown side is rejected."""
        orders = [{"symbol": "AAPL", "side": "hold", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Unknown order side" in rejected[0]["reason"]


class TestMaxPositionSize:
    """Tests for max_position_size constraint."""

    def test_accept_within_max_position_size(self, sample_account, sample_bars):
        """Order within max position size is accepted."""
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 10}]  # 10 * 141 = 1410

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, 2000.0, 1.0  # Max 2000
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_reject_exceeds_max_position_size(self, sample_account, sample_bars):
        """Order exceeding max position size is rejected."""
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 20}]  # 20 * 141 = 2820

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, 2000.0, 1.0  # Max 2000
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "max position size" in rejected[0]["reason"]

    def test_reject_add_to_existing_exceeds_max(self, sample_account, sample_bars):
        """Adding to existing position that would exceed max is rejected."""
        # Already have 10 AAPL @ 152 = 1520
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 5}]  # +5 * 152 = 760, total 2280

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, 2000.0, 1.0  # Max 2000
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "max position size" in rejected[0]["reason"]

    def test_no_limit_when_none(self, sample_account, sample_bars):
        """No position size limit when max_position_size is None."""
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 50}]  # 50 * 141 = 7050

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0  # No limit
        )

        assert len(accepted) == 1
        assert len(rejected) == 0


class TestMaxLeverage:
    """Tests for max_leverage constraint."""

    def test_accept_within_max_leverage(self, sample_account, sample_bars):
        """Order within max leverage is accepted."""
        # Portfolio value: 10000 cash + 10 * 152 = 11520
        # Buying 10 GOOGL adds 1410, total exposure 12930
        # Leverage = 12930 / 10000 = 1.29
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 2.0  # Max 2x leverage
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_reject_exceeds_max_leverage(self, sample_account, sample_bars):
        """Order exceeding max leverage is rejected."""
        # At 0.5x leverage, max buying power = 10000 * 0.5 = 5000
        # Buying 60 GOOGL = 60 * 141 = 8460 > 5000, so rejected
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 60}]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 0.5  # 0.5x leverage (restricted)
        )

        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "max leverage" in rejected[0]["reason"]


class TestMultipleOrders:
    """Tests for handling multiple orders."""

    def test_accept_multiple_valid_orders(self, sample_account, sample_bars):
        """Multiple valid orders are all accepted."""
        orders = [
            {"symbol": "GOOGL", "side": "buy", "quantity": 5},  # 705
            {"symbol": "AAPL", "side": "sell", "quantity": 3},  # Sell from position
        ]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 2
        assert len(rejected) == 0

    def test_cumulative_balance_check(self, sample_account, sample_bars):
        """Second buy order rejected if first uses up balance."""
        orders = [
            {"symbol": "GOOGL", "side": "buy", "quantity": 50},  # 7050
            {"symbol": "AAPL", "side": "buy", "quantity": 30},  # 4560, total 11610 > 10000
        ]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 10.0  # High leverage to not trigger that
        )

        # First order uses 7050, leaving 2950
        # Second order needs 4560 > 2950
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert accepted[0]["quantity"] == 50
        assert "Insufficient balance" in rejected[0]["reason"]

    def test_mixed_accepted_and_rejected(self, sample_account, sample_bars):
        """Mix of valid and invalid orders is handled correctly."""
        orders = [
            {"symbol": "GOOGL", "side": "buy", "quantity": 5},  # Valid
            {"symbol": "MSFT", "side": "buy", "quantity": 10},  # No price data
            {"symbol": "AAPL", "side": "sell", "quantity": 100},  # Insufficient position
        ]

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 1
        assert len(rejected) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_orders_list(self, sample_account, sample_bars):
        """Empty orders list returns empty results."""
        accepted, rejected = apply_risk_constraints(
            [], sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 0
        assert len(rejected) == 0

    def test_empty_positions(self, sample_bars):
        """Account with no positions handles correctly."""
        account = {
            "account_id": "test",
            "cleared_balance": 5000.0,
            "positions": {},
        }
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, account, sample_bars, None, 1.0
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_sell_exact_position(self, sample_account, sample_bars):
        """Selling exact position quantity is accepted."""
        orders = [{"symbol": "AAPL", "side": "sell", "quantity": 10}]  # Exactly what we have

        accepted, rejected = apply_risk_constraints(
            orders, sample_account, sample_bars, None, 1.0
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_buy_exact_balance(self, sample_bars):
        """Buying that uses exact balance is accepted."""
        account = {
            "account_id": "test",
            "cleared_balance": 1410.0,  # Exactly 10 * 141
            "positions": {},
        }
        orders = [{"symbol": "GOOGL", "side": "buy", "quantity": 10}]

        accepted, rejected = apply_risk_constraints(
            orders, account, sample_bars, None, 10.0  # High leverage to not trigger
        )

        assert len(accepted) == 1
        assert len(rejected) == 0

