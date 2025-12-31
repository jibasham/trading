"""Tests for position sizing strategies."""

import pytest

from trading.strategies.sizing import (
    FixedDollarSizer,
    FixedQuantitySizer,
    KellyCriterionSizer,
    PercentOfEquitySizer,
    RiskPercentSizer,
    VolatilityAdjustedSizer,
    get_position_sizer,
)
from trading.types import Account, Position, Symbol


@pytest.fixture
def sample_account() -> Account:
    """Create a sample account for testing."""
    return Account(
        account_id="test",
        base_currency="USD",
        cleared_balance=10000.0,
        pending_balance=0.0,
        reserved_balance=0.0,
        positions={
            "AAPL": Position(symbol=Symbol("AAPL"), quantity=10, cost_basis=150.0),
        },
        clearing_delay_hours=0,
        use_business_days=False,
        pending_transactions=[],
    )


class TestFixedQuantitySizer:
    """Tests for FixedQuantitySizer."""

    def test_returns_fixed_quantity(self, sample_account: Account) -> None:
        """Returns the configured fixed quantity."""
        sizer = FixedQuantitySizer(quantity=25.0)
        qty = sizer.calculate_quantity(sample_account, "AAPL", "buy", 150.0)
        assert qty == 25.0

    def test_scales_with_signal_strength(self, sample_account: Account) -> None:
        """Quantity scales with signal strength."""
        sizer = FixedQuantitySizer(quantity=20.0)
        qty = sizer.calculate_quantity(sample_account, "AAPL", "buy", 150.0, signal_strength=0.5)
        assert qty == 10.0

    def test_ignores_price(self, sample_account: Account) -> None:
        """Fixed quantity ignores price changes."""
        sizer = FixedQuantitySizer(quantity=10.0)
        qty1 = sizer.calculate_quantity(sample_account, "AAPL", "buy", 100.0)
        qty2 = sizer.calculate_quantity(sample_account, "AAPL", "buy", 200.0)
        assert qty1 == qty2 == 10.0


class TestFixedDollarSizer:
    """Tests for FixedDollarSizer."""

    def test_calculates_quantity_from_dollars(self, sample_account: Account) -> None:
        """Calculates quantity based on dollar amount and price."""
        sizer = FixedDollarSizer(dollar_amount=1000.0)
        qty = sizer.calculate_quantity(sample_account, "AAPL", "buy", 100.0)
        assert qty == 10.0  # 1000 / 100

    def test_adjusts_for_different_prices(self, sample_account: Account) -> None:
        """Adjusts quantity based on price."""
        sizer = FixedDollarSizer(dollar_amount=1000.0)
        qty_low = sizer.calculate_quantity(sample_account, "AAPL", "buy", 50.0)
        qty_high = sizer.calculate_quantity(sample_account, "AAPL", "buy", 200.0)
        assert qty_low == 20.0  # 1000 / 50
        assert qty_high == 5.0  # 1000 / 200

    def test_returns_zero_for_zero_price(self, sample_account: Account) -> None:
        """Returns zero quantity for zero price."""
        sizer = FixedDollarSizer(dollar_amount=1000.0)
        qty = sizer.calculate_quantity(sample_account, "AAPL", "buy", 0.0)
        assert qty == 0.0

    def test_scales_with_signal_strength(self, sample_account: Account) -> None:
        """Scales with signal strength."""
        sizer = FixedDollarSizer(dollar_amount=1000.0)
        qty = sizer.calculate_quantity(sample_account, "AAPL", "buy", 100.0, signal_strength=0.5)
        assert qty == 5.0  # 500 / 100


class TestPercentOfEquitySizer:
    """Tests for PercentOfEquitySizer."""

    def test_calculates_from_equity(self, sample_account: Account) -> None:
        """Calculates based on percentage of total equity."""
        # Equity = 10000 cash + 10 * 150 = 11500
        sizer = PercentOfEquitySizer(percent=0.10)  # 10%
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        # 11500 * 0.10 / 100 = 11.5
        assert qty == pytest.approx(11.5)

    def test_includes_positions_in_equity(self, sample_account: Account) -> None:
        """Includes position value in equity calculation."""
        # Without positions, equity would be just cash
        empty_account = Account(
            account_id="test",
            base_currency="USD",
            cleared_balance=10000.0,
            pending_balance=0.0,
            positions={},
        )
        sizer = PercentOfEquitySizer(percent=0.10)
        qty_with_pos = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        qty_without_pos = sizer.calculate_quantity(empty_account, "GOOGL", "buy", 100.0)
        assert qty_with_pos > qty_without_pos


class TestRiskPercentSizer:
    """Tests for RiskPercentSizer."""

    def test_calculates_from_risk_and_stop(self, sample_account: Account) -> None:
        """Calculates based on risk percentage and stop loss."""
        # Equity = 11500, Risk 1% = 115, Stop 2% at $100 = $2 risk per share
        # Quantity = 115 / 2 = 57.5
        sizer = RiskPercentSizer(risk_percent=0.01, default_stop_pct=0.02)
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == pytest.approx(57.5)

    def test_uses_custom_stop_percent(self, sample_account: Account) -> None:
        """Uses custom stop percent when set."""
        sizer = RiskPercentSizer(risk_percent=0.01, default_stop_pct=0.02)
        sizer.set_stop_percent(0.05)  # 5% stop
        # Equity = 11500, Risk 1% = 115, Stop 5% at $100 = $5 risk per share
        # Quantity = 115 / 5 = 23
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == pytest.approx(23.0)

    def test_resets_stop_after_use(self, sample_account: Account) -> None:
        """Resets stop percent to default after use."""
        sizer = RiskPercentSizer(risk_percent=0.01, default_stop_pct=0.02)
        sizer.set_stop_percent(0.05)
        sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        # Second call should use default
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == pytest.approx(57.5)


class TestKellyCriterionSizer:
    """Tests for KellyCriterionSizer."""

    def test_calculates_kelly_fraction(self) -> None:
        """Calculates correct Kelly fraction."""
        # 50% win rate, 2:1 reward/risk
        # Kelly = (2 * 0.5 - 0.5) / 2 = 0.25
        sizer = KellyCriterionSizer(
            win_rate=0.5, avg_win=200.0, avg_loss=100.0, fraction=1.0
        )
        kelly = sizer.calculate_kelly_fraction()
        assert kelly == pytest.approx(0.25)

    def test_half_kelly(self) -> None:
        """Half Kelly reduces position size."""
        sizer = KellyCriterionSizer(
            win_rate=0.5, avg_win=200.0, avg_loss=100.0, fraction=0.5
        )
        kelly = sizer.calculate_kelly_fraction()
        assert kelly == pytest.approx(0.125)

    def test_negative_expectancy_returns_zero(self) -> None:
        """Returns zero for negative expectancy."""
        # 30% win rate, 1:1 reward/risk = negative expectancy
        sizer = KellyCriterionSizer(
            win_rate=0.30, avg_win=100.0, avg_loss=100.0, fraction=1.0
        )
        kelly = sizer.calculate_kelly_fraction()
        assert kelly == 0.0

    def test_caps_at_max_fraction(self) -> None:
        """Caps Kelly fraction at 25%."""
        # Very favorable odds
        sizer = KellyCriterionSizer(
            win_rate=0.9, avg_win=500.0, avg_loss=100.0, fraction=1.0
        )
        kelly = sizer.calculate_kelly_fraction()
        assert kelly <= 0.25


class TestVolatilityAdjustedSizer:
    """Tests for VolatilityAdjustedSizer."""

    def test_baseline_volatility(self, sample_account: Account) -> None:
        """At target volatility, returns base position."""
        sizer = VolatilityAdjustedSizer(base_dollars=1000.0, target_atr_pct=0.02)
        sizer.set_volatility(0.02)  # Matches target
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == 10.0  # 1000 / 100

    def test_low_volatility_increases_size(self, sample_account: Account) -> None:
        """Lower volatility increases position size."""
        sizer = VolatilityAdjustedSizer(base_dollars=1000.0, target_atr_pct=0.02)
        sizer.set_volatility(0.01)  # Half the target
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == 20.0  # 2000 / 100 (doubled)

    def test_high_volatility_decreases_size(self, sample_account: Account) -> None:
        """Higher volatility decreases position size."""
        sizer = VolatilityAdjustedSizer(base_dollars=1000.0, target_atr_pct=0.02)
        sizer.set_volatility(0.04)  # Double the target
        qty = sizer.calculate_quantity(sample_account, "GOOGL", "buy", 100.0)
        assert qty == 5.0  # 500 / 100 (halved)


class TestGetPositionSizer:
    """Tests for get_position_sizer factory function."""

    def test_get_fixed_qty(self) -> None:
        """Gets fixed quantity sizer."""
        sizer = get_position_sizer("fixed_qty", quantity=15.0)
        assert isinstance(sizer, FixedQuantitySizer)
        assert sizer.quantity == 15.0

    def test_get_fixed_dollar(self) -> None:
        """Gets fixed dollar sizer."""
        sizer = get_position_sizer("fixed_dollar", dollar_amount=2000.0)
        assert isinstance(sizer, FixedDollarSizer)
        assert sizer.dollar_amount == 2000.0

    def test_get_percent_equity(self) -> None:
        """Gets percent of equity sizer."""
        sizer = get_position_sizer("percent_equity", percent=0.15)
        assert isinstance(sizer, PercentOfEquitySizer)
        assert sizer.percent == 0.15

    def test_get_risk_percent(self) -> None:
        """Gets risk percent sizer."""
        sizer = get_position_sizer("risk_percent", risk_percent=0.02)
        assert isinstance(sizer, RiskPercentSizer)
        assert sizer.risk_percent == 0.02

    def test_get_kelly(self) -> None:
        """Gets Kelly criterion sizer."""
        sizer = get_position_sizer("kelly", win_rate=0.6, avg_win=150.0, avg_loss=100.0)
        assert isinstance(sizer, KellyCriterionSizer)
        assert sizer.win_rate == 0.6

    def test_get_volatility(self) -> None:
        """Gets volatility adjusted sizer."""
        sizer = get_position_sizer("volatility", base_dollars=1500.0)
        assert isinstance(sizer, VolatilityAdjustedSizer)
        assert sizer.base_dollars == 1500.0

    def test_unknown_sizer_raises(self) -> None:
        """Unknown sizer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sizer"):
            get_position_sizer("invalid_sizer")




