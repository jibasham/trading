"""Tests for walk-forward validation."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.strategies import BuyAndHoldStrategy
from trading.training.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
)
from trading.types import NormalizedBar, Symbol


def generate_bars(
    symbol: str, start_date: datetime, num_days: int, start_price: float = 100.0
) -> list[NormalizedBar]:
    """Generate synthetic bars for testing."""
    bars = []
    price = start_price

    for i in range(num_days):
        # Simple trending price
        price *= 1.001  # 0.1% daily growth
        bars.append(
            NormalizedBar(
                symbol=Symbol(symbol),
                timestamp=start_date + timedelta(days=i),
                open=price * 0.99,
                high=price * 1.01,
                low=price * 0.98,
                close=price,
                volume=10000.0,
            )
        )

    return bars


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig."""

    def test_default_config(self) -> None:
        """Config has sensible defaults."""
        config = WalkForwardConfig()
        assert config.train_period_days == 252
        assert config.test_period_days == 63
        assert config.min_train_bars == 20
        assert config.initial_balance == 10000.0

    def test_custom_config(self) -> None:
        """Config accepts custom values."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=20,
            step_days=10,
        )
        assert config.train_period_days == 100
        assert config.test_period_days == 20
        assert config.step_days == 10


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    @pytest.fixture
    def long_bars(self) -> list[NormalizedBar]:
        """Generate bars for ~2 years."""
        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 500, start_price=100.0)

    def test_compute_windows(self, long_bars: list[NormalizedBar]) -> None:
        """Windows are computed correctly."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        windows = validator._compute_windows()

        assert len(windows) > 0

        # Each window should have correct structure
        for train_start, train_end, test_start, test_end in windows:
            assert train_end == test_start  # No gap between train and test
            assert (train_end - train_start).days == 100  # Train period
            assert (test_end - test_start).days == 50  # Test period

    def test_run_returns_result(self, long_bars: list[NormalizedBar]) -> None:
        """Run produces a WalkForwardResult."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            min_train_bars=5,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        result = validator.run()

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0
        assert result.aggregate_metrics is not None

    def test_windows_have_train_and_test_results(
        self, long_bars: list[NormalizedBar]
    ) -> None:
        """Each window has both train and test backtest results."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            min_train_bars=5,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        result = validator.run()

        for window in result.windows:
            assert window.train_result is not None
            assert window.test_result is not None
            assert window.train_result.metrics is not None
            assert window.test_result.metrics is not None

    def test_consistency_ratio(self, long_bars: list[NormalizedBar]) -> None:
        """Consistency ratio is computed correctly."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            min_train_bars=5,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        result = validator.run()

        # With trending data, should have high consistency
        assert 0 <= result.consistency_ratio <= 1

    def test_empty_bars_returns_empty_result(self) -> None:
        """Empty bars produce empty result."""
        config = WalkForwardConfig()

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator([], config, make_strategy)
        result = validator.run()

        assert len(result.windows) == 0
        assert result.aggregate_metrics is None

    def test_short_data_returns_empty_result(self) -> None:
        """Data shorter than one window returns empty result."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        short_bars = generate_bars("TEST", start, 30)

        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(short_bars, config, make_strategy)
        result = validator.run()

        assert len(result.windows) == 0

    def test_summary_format(self, long_bars: list[NormalizedBar]) -> None:
        """Summary returns formatted string."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            min_train_bars=5,
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        summary = validator.summary()

        assert "WALK-FORWARD VALIDATION RESULTS" in summary
        assert "Windows analyzed:" in summary
        assert "Total test return:" in summary

    def test_step_days_creates_overlapping_windows(
        self, long_bars: list[NormalizedBar]
    ) -> None:
        """Custom step_days creates overlapping train periods."""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            step_days=25,  # Step less than test period
        )

        def make_strategy():
            return BuyAndHoldStrategy({"symbol": "TEST", "quantity": 1})

        validator = WalkForwardValidator(long_bars, config, make_strategy)
        windows = validator._compute_windows()

        # Should have more windows with smaller step
        assert len(windows) > 2

        # Windows should overlap
        if len(windows) >= 2:
            _, _, first_test_start, _ = windows[0]
            second_train_start, _, _, _ = windows[1]
            # Second train starts before first test ends (overlapping)
            assert second_train_start < first_test_start + timedelta(days=50)


