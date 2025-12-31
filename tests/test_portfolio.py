"""Tests for portfolio allocation strategies."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.training.portfolio import (
    AllocationWeights,
    CustomWeightAllocation,
    EqualWeightAllocation,
    InverseVolatilityAllocation,
    MarketCapWeightAllocation,
    MinVarianceAllocation,
    MomentumAllocation,
    get_allocation_strategy,
)
from trading.types import NormalizedBar, Symbol


def generate_bars(
    symbol: str,
    start_date: datetime,
    num_days: int,
    start_price: float = 100.0,
    daily_return: float = 0.001,
    volatility: float = 0.01,
) -> list[NormalizedBar]:
    """Generate synthetic bars with configurable characteristics."""
    import random

    bars = []
    price = start_price

    for i in range(num_days):
        # Add trend and noise
        noise = random.gauss(0, volatility)
        price *= 1 + daily_return + noise
        price = max(price, 1.0)  # Floor at $1

        bars.append(
            NormalizedBar(
                symbol=Symbol(symbol),
                timestamp=start_date + timedelta(days=i),
                open=price * 0.99,
                high=price * 1.01,
                low=price * 0.98,
                close=price,
                volume=10000.0 * (1 + random.random()),
            )
        )

    return bars


class TestAllocationWeights:
    """Tests for AllocationWeights."""

    def test_normalize(self) -> None:
        """Weights normalize to sum to 1."""
        weights = AllocationWeights(weights={"A": 2.0, "B": 3.0, "C": 5.0})
        normalized = weights.normalize()

        assert normalized.weights["A"] == pytest.approx(0.2)
        assert normalized.weights["B"] == pytest.approx(0.3)
        assert normalized.weights["C"] == pytest.approx(0.5)

    def test_normalize_zero_total(self) -> None:
        """Normalizing all-zero weights returns original."""
        weights = AllocationWeights(weights={"A": 0, "B": 0})
        normalized = weights.normalize()

        assert normalized.weights["A"] == 0
        assert normalized.weights["B"] == 0


class TestEqualWeightAllocation:
    """Tests for EqualWeightAllocation."""

    def test_equal_weights(self) -> None:
        """All symbols get equal weight."""
        strategy = EqualWeightAllocation()
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        weights = strategy.compute_weights(symbols, {})

        assert len(weights.weights) == 4
        for w in weights.weights.values():
            assert w == pytest.approx(0.25)

    def test_empty_symbols(self) -> None:
        """Empty symbols list returns empty weights."""
        strategy = EqualWeightAllocation()
        weights = strategy.compute_weights([], {})

        assert len(weights.weights) == 0


class TestMarketCapWeightAllocation:
    """Tests for MarketCapWeightAllocation."""

    def test_higher_volume_higher_weight(self) -> None:
        """Higher market cap (price*volume) gets higher weight."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Higher priced, higher volume = higher market cap
        high_cap = generate_bars("HIGH", start, 30, start_price=200.0)
        # Manually increase volume
        high_cap = [
            NormalizedBar(
                symbol=b.symbol,
                timestamp=b.timestamp,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume * 10,  # 10x volume
            )
            for b in high_cap
        ]

        low_cap = generate_bars("LOW", start, 30, start_price=50.0)

        bars_by_symbol = {"HIGH": high_cap, "LOW": low_cap}
        strategy = MarketCapWeightAllocation(lookback_bars=20)

        weights = strategy.compute_weights(["HIGH", "LOW"], bars_by_symbol)

        # HIGH should have much higher weight
        assert weights.weights["HIGH"] > weights.weights["LOW"]


class TestMomentumAllocation:
    """Tests for MomentumAllocation."""

    def test_positive_momentum_higher_weight(self) -> None:
        """Positive momentum symbols get weight, negative don't."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Strong positive momentum
        up = generate_bars("UP", start, 30, start_price=100.0, daily_return=0.01)
        # Negative momentum
        down = generate_bars("DOWN", start, 30, start_price=100.0, daily_return=-0.01)

        bars_by_symbol = {"UP": up, "DOWN": down}
        strategy = MomentumAllocation(lookback_days=30)

        weights = strategy.compute_weights(["UP", "DOWN"], bars_by_symbol)

        # UP should have weight, DOWN should have 0 (negative momentum)
        assert weights.weights["UP"] > 0
        assert weights.weights["DOWN"] == 0

    def test_top_n_filter(self) -> None:
        """top_n limits allocation to top momentum symbols."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        best = generate_bars("BEST", start, 30, daily_return=0.02)
        mid = generate_bars("MID", start, 30, daily_return=0.01)
        worst = generate_bars("WORST", start, 30, daily_return=0.001)

        bars_by_symbol = {"BEST": best, "MID": mid, "WORST": worst}
        strategy = MomentumAllocation(lookback_days=30, top_n=2)

        weights = strategy.compute_weights(["BEST", "MID", "WORST"], bars_by_symbol)

        # WORST should be excluded
        assert weights.weights.get("WORST", 0) == 0


class TestInverseVolatilityAllocation:
    """Tests for InverseVolatilityAllocation."""

    def test_lower_vol_higher_weight(self) -> None:
        """Lower volatility gets higher weight."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Low volatility
        low_vol = generate_bars(
            "LOW_VOL", start, 60, daily_return=0.001, volatility=0.005
        )
        # High volatility
        high_vol = generate_bars(
            "HIGH_VOL", start, 60, daily_return=0.001, volatility=0.03
        )

        bars_by_symbol = {"LOW_VOL": low_vol, "HIGH_VOL": high_vol}
        strategy = InverseVolatilityAllocation(lookback_days=30)

        weights = strategy.compute_weights(["LOW_VOL", "HIGH_VOL"], bars_by_symbol)

        # Lower vol should get higher weight
        assert weights.weights["LOW_VOL"] > weights.weights["HIGH_VOL"]


class TestMinVarianceAllocation:
    """Tests for MinVarianceAllocation."""

    def test_lower_variance_higher_weight(self) -> None:
        """Lower variance gets higher weight."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        low_var = generate_bars(
            "LOW_VAR", start, 60, daily_return=0.001, volatility=0.005
        )
        high_var = generate_bars(
            "HIGH_VAR", start, 60, daily_return=0.001, volatility=0.03
        )

        bars_by_symbol = {"LOW_VAR": low_var, "HIGH_VAR": high_var}
        strategy = MinVarianceAllocation(lookback_days=30)

        weights = strategy.compute_weights(["LOW_VAR", "HIGH_VAR"], bars_by_symbol)

        assert weights.weights["LOW_VAR"] > weights.weights["HIGH_VAR"]


class TestCustomWeightAllocation:
    """Tests for CustomWeightAllocation."""

    def test_custom_weights(self) -> None:
        """Returns predefined weights."""
        strategy = CustomWeightAllocation({"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.2})
        weights = strategy.compute_weights(["AAPL", "GOOGL", "MSFT"], {})

        assert weights.weights["AAPL"] == pytest.approx(0.5)
        assert weights.weights["GOOGL"] == pytest.approx(0.3)
        assert weights.weights["MSFT"] == pytest.approx(0.2)

    def test_missing_symbol_gets_zero(self) -> None:
        """Symbols not in custom weights get 0."""
        strategy = CustomWeightAllocation({"AAPL": 1.0})
        weights = strategy.compute_weights(["AAPL", "UNKNOWN"], {})

        # After normalization
        assert weights.weights["AAPL"] == pytest.approx(1.0)
        assert weights.weights["UNKNOWN"] == pytest.approx(0.0)


class TestGetAllocationStrategy:
    """Tests for get_allocation_strategy factory."""

    def test_get_equal(self) -> None:
        """Create equal weight strategy."""
        strategy = get_allocation_strategy("equal")
        assert isinstance(strategy, EqualWeightAllocation)

    def test_get_momentum(self) -> None:
        """Create momentum strategy with params."""
        strategy = get_allocation_strategy("momentum", lookback_days=60, top_n=5)
        assert isinstance(strategy, MomentumAllocation)
        assert strategy.lookback_days == 60
        assert strategy.top_n == 5

    def test_get_inverse_vol(self) -> None:
        """Create inverse volatility strategy."""
        strategy = get_allocation_strategy("inverse_vol", lookback_days=30)
        assert isinstance(strategy, InverseVolatilityAllocation)

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy name raises error."""
        with pytest.raises(ValueError, match="Unknown allocation strategy"):
            get_allocation_strategy("unknown")


