"""Reinforcement learning module for trading strategies.

Provides Gymnasium-compatible environments and utilities for training
RL agents on trading tasks.
"""

from trading.rl.env import TradingEnv, TradingEnvConfig
from trading.rl.features import (
    FeatureExtractor,
    OHLCVFeatures,
    TechnicalFeatures,
    AccountFeatures,
    CombinedFeatures,
)
from trading.rl.rewards import (
    RewardFunction,
    SimpleReturnReward,
    RiskAdjustedReward,
    SharpeReward,
    DrawdownPenaltyReward,
)
from trading.rl.wrappers import RLStrategy

__all__ = [
    # Environment
    "TradingEnv",
    "TradingEnvConfig",
    # Features
    "FeatureExtractor",
    "OHLCVFeatures",
    "TechnicalFeatures",
    "AccountFeatures",
    "CombinedFeatures",
    # Rewards
    "RewardFunction",
    "SimpleReturnReward",
    "RiskAdjustedReward",
    "SharpeReward",
    "DrawdownPenaltyReward",
    # Wrappers
    "RLStrategy",
]


