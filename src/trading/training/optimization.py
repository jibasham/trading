"""Hyperparameter optimization for trading strategies.

Provides grid search and random search for finding optimal strategy parameters.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from trading.training.backtest import Backtest, BacktestResult
from trading.types import NormalizedBar, RunMetrics

if TYPE_CHECKING:
    from trading.strategies.base import Strategy


class ParameterSpec(BaseModel):
    """Specification for a hyperparameter.

    :param name: Parameter name.
    :param values: List of values for grid search.
    :param min_val: Minimum value for random search.
    :param max_val: Maximum value for random search.
    :param param_type: Type of parameter ('int', 'float', 'choice').
    """

    name: str
    values: list[Any] | None = None  # For grid search / choice
    min_val: float | None = None  # For random search
    max_val: float | None = None  # For random search
    param_type: str = "float"  # 'int', 'float', 'choice'


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization.

    :param best_params: Best parameter combination found.
    :param best_metric: Value of the optimization metric.
    :param best_result: Full backtest result for best params.
    :param all_results: All parameter combinations tested.
    :param metric_name: Name of the metric optimized.
    """

    best_params: dict[str, Any]
    best_metric: float
    best_result: BacktestResult
    all_results: list[tuple[dict[str, Any], float, BacktestResult]] = field(
        default_factory=list
    )
    metric_name: str = "sharpe_ratio"


StrategyBuilder = Callable[[dict[str, Any]], "Strategy"]


class GridSearchOptimizer:
    """Grid search optimizer for strategy hyperparameters.

    Tests all combinations of specified parameter values.

    Example::

        from trading.training import GridSearchOptimizer, ParameterSpec
        from trading.strategies import MovingAverageCrossoverStrategy

        params = [
            ParameterSpec(name="short_window", values=[5, 10, 20]),
            ParameterSpec(name="long_window", values=[20, 50, 100]),
        ]

        def build_strategy(p: dict) -> Strategy:
            config = {"symbol": "AAPL", **p}
            return MovingAverageCrossoverStrategy(config)

        optimizer = GridSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
        )
        result = optimizer.optimize()
        print(f"Best params: {result.best_params}")

    :param bars: Historical bar data.
    :param param_specs: Parameter specifications.
    :param strategy_builder: Function to create strategy from params.
    :param metric: Metric to optimize ('sharpe_ratio', 'total_return', etc).
    :param maximize: Whether to maximize (True) or minimize the metric.
    :param initial_balance: Starting balance for backtests.
    :param commission_per_trade: Commission per trade.
    :param slippage_pct: Slippage percentage.
    """

    def __init__(
        self,
        bars: list[NormalizedBar],
        param_specs: list[ParameterSpec],
        strategy_builder: StrategyBuilder,
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        initial_balance: float = 10000.0,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self.bars = bars
        self.param_specs = param_specs
        self.strategy_builder = strategy_builder
        self.metric = metric
        self.maximize = maximize
        self.initial_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct

    def _get_metric_value(self, metrics: RunMetrics) -> float | None:
        """Extract the target metric from results."""
        return getattr(metrics, self.metric, None)

    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        param_values = []
        param_names = []

        for spec in self.param_specs:
            param_names.append(spec.name)
            if spec.values is not None:
                param_values.append(spec.values)
            else:
                raise ValueError(f"Grid search requires 'values' for param {spec.name}")

        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]

    def optimize(self) -> OptimizationResult:
        """Run grid search optimization.

        :returns: Optimization result with best parameters.
        """
        param_combos = self._generate_param_combinations()

        if not param_combos:
            raise ValueError("No parameter combinations to test")

        all_results: list[tuple[dict[str, Any], float, BacktestResult]] = []
        best_params: dict[str, Any] = {}
        best_metric: float = float("-inf") if self.maximize else float("inf")
        best_result: BacktestResult | None = None

        for params in param_combos:
            try:
                strategy = self.strategy_builder(params)
                backtest = Backtest(
                    bars=self.bars,
                    strategy=strategy,
                    initial_balance=self.initial_balance,
                    commission_per_trade=self.commission_per_trade,
                    slippage_pct=self.slippage_pct,
                )
                result = backtest.run()

                metric_val = self._get_metric_value(result.metrics)
                if metric_val is None:
                    continue

                all_results.append((params, metric_val, result))

                is_better = (
                    metric_val > best_metric
                    if self.maximize
                    else metric_val < best_metric
                )
                if is_better:
                    best_params = params
                    best_metric = metric_val
                    best_result = result

            except Exception:
                # Skip invalid parameter combinations
                continue

        if best_result is None:
            raise ValueError("No valid results from optimization")

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            best_result=best_result,
            all_results=all_results,
            metric_name=self.metric,
        )

    def summary(self) -> str:
        """Run optimization and return formatted summary."""
        result = self.optimize()

        lines = [
            "=" * 60,
            "GRID SEARCH OPTIMIZATION RESULTS",
            "=" * 60,
            f"Combinations tested: {len(result.all_results)}",
            f"Optimized metric: {result.metric_name}",
            f"Best {result.metric_name}: {result.best_metric:.4f}",
            "",
            "Best Parameters:",
        ]

        for name, value in result.best_params.items():
            lines.append(f"  {name}: {value}")

        lines.extend([
            "",
            "Top 5 Combinations:",
            "-" * 60,
        ])

        # Sort by metric
        sorted_results = sorted(
            result.all_results,
            key=lambda x: x[1],
            reverse=self.maximize,
        )

        for params, metric, _ in sorted_results[:5]:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"  {metric:.4f}: {param_str}")

        return "\n".join(lines)


class RandomSearchOptimizer:
    """Random search optimizer for strategy hyperparameters.

    Samples random parameter combinations from specified ranges.

    Example::

        from trading.training import RandomSearchOptimizer, ParameterSpec

        params = [
            ParameterSpec(name="short_window", min_val=5, max_val=30, param_type="int"),
            ParameterSpec(name="long_window", min_val=20, max_val=200, param_type="int"),
        ]

        optimizer = RandomSearchOptimizer(
            bars=bars,
            param_specs=params,
            strategy_builder=build_strategy,
            n_iterations=100,
        )
        result = optimizer.optimize()

    :param bars: Historical bar data.
    :param param_specs: Parameter specifications with min/max ranges.
    :param strategy_builder: Function to create strategy from params.
    :param n_iterations: Number of random combinations to try.
    :param metric: Metric to optimize.
    :param maximize: Whether to maximize the metric.
    :param seed: Random seed for reproducibility.
    :param initial_balance: Starting balance for backtests.
    :param commission_per_trade: Commission per trade.
    :param slippage_pct: Slippage percentage.
    """

    def __init__(
        self,
        bars: list[NormalizedBar],
        param_specs: list[ParameterSpec],
        strategy_builder: StrategyBuilder,
        n_iterations: int = 100,
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        seed: int | None = None,
        initial_balance: float = 10000.0,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self.bars = bars
        self.param_specs = param_specs
        self.strategy_builder = strategy_builder
        self.n_iterations = n_iterations
        self.metric = metric
        self.maximize = maximize
        self.initial_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct

        if seed is not None:
            random.seed(seed)

    def _get_metric_value(self, metrics: RunMetrics) -> float | None:
        """Extract the target metric from results."""
        return getattr(metrics, self.metric, None)

    def _sample_params(self) -> dict[str, Any]:
        """Sample a random parameter combination."""
        params = {}

        for spec in self.param_specs:
            if spec.values is not None:
                # Choice from list
                params[spec.name] = random.choice(spec.values)
            elif spec.min_val is not None and spec.max_val is not None:
                if spec.param_type == "int":
                    params[spec.name] = random.randint(
                        int(spec.min_val), int(spec.max_val)
                    )
                else:
                    params[spec.name] = random.uniform(spec.min_val, spec.max_val)
            else:
                raise ValueError(
                    f"Param {spec.name} needs either 'values' or 'min_val/max_val'"
                )

        return params

    def optimize(self) -> OptimizationResult:
        """Run random search optimization.

        :returns: Optimization result with best parameters.
        """
        all_results: list[tuple[dict[str, Any], float, BacktestResult]] = []
        best_params: dict[str, Any] = {}
        best_metric: float = float("-inf") if self.maximize else float("inf")
        best_result: BacktestResult | None = None

        for _ in range(self.n_iterations):
            params = self._sample_params()

            try:
                strategy = self.strategy_builder(params)
                backtest = Backtest(
                    bars=self.bars,
                    strategy=strategy,
                    initial_balance=self.initial_balance,
                    commission_per_trade=self.commission_per_trade,
                    slippage_pct=self.slippage_pct,
                )
                result = backtest.run()

                metric_val = self._get_metric_value(result.metrics)
                if metric_val is None:
                    continue

                all_results.append((params, metric_val, result))

                is_better = (
                    metric_val > best_metric
                    if self.maximize
                    else metric_val < best_metric
                )
                if is_better:
                    best_params = params
                    best_metric = metric_val
                    best_result = result

            except Exception:
                continue

        if best_result is None:
            raise ValueError("No valid results from optimization")

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            best_result=best_result,
            all_results=all_results,
            metric_name=self.metric,
        )

    def summary(self) -> str:
        """Run optimization and return formatted summary."""
        result = self.optimize()

        lines = [
            "=" * 60,
            "RANDOM SEARCH OPTIMIZATION RESULTS",
            "=" * 60,
            f"Iterations: {self.n_iterations}",
            f"Valid results: {len(result.all_results)}",
            f"Optimized metric: {result.metric_name}",
            f"Best {result.metric_name}: {result.best_metric:.4f}",
            "",
            "Best Parameters:",
        ]

        for name, value in result.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")

        return "\n".join(lines)


