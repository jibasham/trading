"""Tests for hyperparameter optimization."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.strategies.base import Strategy
from trading.training.optimization import (
    GridSearchOptimizer,
    OptimizationResult,
    ParameterSpec,
    RandomSearchOptimizer,
)
from trading.types import Account, AnalysisSnapshot, NormalizedBar, OrderRequest, Symbol


def generate_bars(
    symbol: str, start_date: datetime, num_days: int, start_price: float = 100.0
) -> list[NormalizedBar]:
    """Generate synthetic bars for testing."""
    bars = []
    price = start_price

    for i in range(num_days):
        price *= 1.001
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


class ConfigurableStrategy(Strategy):
    """Test strategy with configurable parameters."""

    def __init__(self, config: dict) -> None:
        self.symbol = config.get("symbol", "TEST")
        self.threshold = config.get("threshold", 0.5)
        self.quantity = config.get("quantity", 10)
        self._bought = False

    def decide(
        self, snapshot: AnalysisSnapshot, account: Account
    ) -> list[OrderRequest]:
        """Buy if threshold allows."""
        if self._bought:
            return []

        self._bought = True
        return [
            OrderRequest(
                symbol=Symbol(self.symbol),
                side="buy",
                quantity=self.quantity,
            )
        ]


class TestParameterSpec:
    """Tests for ParameterSpec."""

    def test_grid_spec(self) -> None:
        """Grid search spec with values list."""
        spec = ParameterSpec(name="window", values=[5, 10, 20])
        assert spec.name == "window"
        assert spec.values == [5, 10, 20]

    def test_random_spec(self) -> None:
        """Random search spec with min/max."""
        spec = ParameterSpec(
            name="window", min_val=5, max_val=100, param_type="int"
        )
        assert spec.min_val == 5
        assert spec.max_val == 100
        assert spec.param_type == "int"


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    @pytest.fixture
    def bars(self) -> list[NormalizedBar]:
        """Generate test bars."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 100)

    def test_optimize_returns_result(self, bars: list[NormalizedBar]) -> None:
        """Optimization returns a result."""
        params = [
            ParameterSpec(name="threshold", values=[0.3, 0.5, 0.7]),
            ParameterSpec(name="quantity", values=[5, 10]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            metric="total_return",
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_result is not None

    def test_all_combinations_tested(self, bars: list[NormalizedBar]) -> None:
        """Grid search tests all parameter combinations."""
        params = [
            ParameterSpec(name="threshold", values=[0.3, 0.5, 0.7]),
            ParameterSpec(name="quantity", values=[5, 10]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            metric="total_return",
        )

        result = optimizer.optimize()

        # 3 * 2 = 6 combinations
        assert len(result.all_results) == 6

    def test_best_params_match_metric(self, bars: list[NormalizedBar]) -> None:
        """Best params have the best metric value."""
        params = [
            ParameterSpec(name="quantity", values=[1, 5, 10, 20]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            metric="total_return",
            maximize=True,
        )

        result = optimizer.optimize()

        # Best metric should be the max
        all_metrics = [r[1] for r in result.all_results]
        assert result.best_metric == max(all_metrics)

    def test_minimize_mode(self, bars: list[NormalizedBar]) -> None:
        """Optimizer can minimize a metric."""
        params = [
            ParameterSpec(name="quantity", values=[1, 5, 10]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            metric="max_drawdown",
            maximize=False,  # Minimize drawdown
        )

        result = optimizer.optimize()

        all_metrics = [r[1] for r in result.all_results]
        assert result.best_metric == min(all_metrics)

    def test_summary_format(self, bars: list[NormalizedBar]) -> None:
        """Summary returns formatted string."""
        params = [
            ParameterSpec(name="quantity", values=[5, 10]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
        )

        summary = optimizer.summary()

        assert "GRID SEARCH OPTIMIZATION RESULTS" in summary
        assert "Combinations tested:" in summary
        assert "Best Parameters:" in summary


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    @pytest.fixture
    def bars(self) -> list[NormalizedBar]:
        """Generate test bars."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return generate_bars("TEST", start, 100)

    def test_optimize_returns_result(self, bars: list[NormalizedBar]) -> None:
        """Random search returns a result."""
        params = [
            ParameterSpec(
                name="quantity", min_val=1, max_val=20, param_type="int"
            ),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=10,
            seed=42,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None

    def test_n_iterations_respected(self, bars: list[NormalizedBar]) -> None:
        """Runs specified number of iterations."""
        params = [
            ParameterSpec(
                name="quantity", min_val=1, max_val=20, param_type="int"
            ),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=15,
            seed=42,
        )

        result = optimizer.optimize()

        assert len(result.all_results) == 15

    def test_seed_reproducibility(self, bars: list[NormalizedBar]) -> None:
        """Same seed produces same results."""
        params = [
            ParameterSpec(
                name="quantity", min_val=1, max_val=100, param_type="int"
            ),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        opt1 = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=5,
            seed=123,
        )
        result1 = opt1.optimize()

        opt2 = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=5,
            seed=123,
        )
        result2 = opt2.optimize()

        assert result1.best_params == result2.best_params

    def test_choice_parameters(self, bars: list[NormalizedBar]) -> None:
        """Random search supports choice parameters."""
        params = [
            ParameterSpec(name="quantity", values=[5, 10, 15, 20]),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=10,
            seed=42,
        )

        result = optimizer.optimize()

        # All sampled quantities should be from the list
        for params, _, _ in result.all_results:
            assert params["quantity"] in [5, 10, 15, 20]

    def test_float_parameters(self, bars: list[NormalizedBar]) -> None:
        """Random search samples float parameters."""
        params = [
            ParameterSpec(
                name="threshold", min_val=0.1, max_val=0.9, param_type="float"
            ),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=10,
            seed=42,
        )

        result = optimizer.optimize()

        # All thresholds should be in range
        for params, _, _ in result.all_results:
            assert 0.1 <= params["threshold"] <= 0.9

    def test_summary_format(self, bars: list[NormalizedBar]) -> None:
        """Summary returns formatted string."""
        params = [
            ParameterSpec(
                name="quantity", min_val=1, max_val=20, param_type="int"
            ),
        ]

        def build_strategy(p: dict) -> Strategy:
            return ConfigurableStrategy({"symbol": "TEST", **p})

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=5,
            seed=42,
        )

        summary = optimizer.summary()

        assert "RANDOM SEARCH OPTIMIZATION RESULTS" in summary
        assert "Iterations:" in summary


