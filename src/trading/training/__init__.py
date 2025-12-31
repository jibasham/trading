"""Training and backtesting module."""

from trading.training.backtest import Backtest, BacktestResult
from trading.training.multi_backtest import (
    ComparisonResult,
    MultiBacktest,
    StrategyConfig,
    compare_strategies,
)
from trading.training.optimization import (
    GridSearchOptimizer,
    OptimizationResult,
    ParameterSpec,
    RandomSearchOptimizer,
)
from trading.training.portfolio import (
    AllocationStrategy,
    AllocationWeights,
    CustomWeightAllocation,
    EqualWeightAllocation,
    InverseVolatilityAllocation,
    MarketCapWeightAllocation,
    MinVarianceAllocation,
    MomentumAllocation,
    get_allocation_strategy,
)
from trading.training.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
    WindowResult,
)

__all__ = [
    # Backtest
    "Backtest",
    "BacktestResult",
    # Multi-backtest
    "ComparisonResult",
    "MultiBacktest",
    "StrategyConfig",
    "compare_strategies",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardValidator",
    "WindowResult",
    # Optimization
    "GridSearchOptimizer",
    "OptimizationResult",
    "ParameterSpec",
    "RandomSearchOptimizer",
    # Portfolio
    "AllocationStrategy",
    "AllocationWeights",
    "CustomWeightAllocation",
    "EqualWeightAllocation",
    "InverseVolatilityAllocation",
    "MarketCapWeightAllocation",
    "MinVarianceAllocation",
    "MomentumAllocation",
    "get_allocation_strategy",
]
