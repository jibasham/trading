"""Strategy module for trading bot implementations."""

from trading.strategies.base import Strategy
from trading.strategies.examples import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    RandomStrategy,
    RSIStrategy,
)
from trading.strategies.sizing import (
    FixedDollarSizer,
    FixedQuantitySizer,
    KellyCriterionSizer,
    PercentOfEquitySizer,
    PositionSizer,
    RiskPercentSizer,
    VolatilityAdjustedSizer,
    get_position_sizer,
)

__all__ = [
    # Base
    "Strategy",
    # Examples
    "BuyAndHoldStrategy",
    "MeanReversionStrategy",
    "MovingAverageCrossoverStrategy",
    "RandomStrategy",
    "RSIStrategy",
    # Position Sizing
    "PositionSizer",
    "FixedQuantitySizer",
    "FixedDollarSizer",
    "PercentOfEquitySizer",
    "RiskPercentSizer",
    "KellyCriterionSizer",
    "VolatilityAdjustedSizer",
    "get_position_sizer",
]
