"""Feature extractors for RL observations.

Transforms raw market data into normalized feature vectors suitable
for neural network inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from trading.types import Account, NormalizedBar


class FeatureExtractor(ABC):
    """Base class for feature extractors.

    Feature extractors transform raw market data and account state
    into normalized numpy arrays suitable for RL observations.
    """

    @abstractmethod
    def extract(
        self,
        bars: list["NormalizedBar"],
        current_idx: int,
        account: "Account",
    ) -> NDArray[np.float32]:
        """Extract features from current state.

        :param bars: Full list of bars in the episode.
        :param current_idx: Index of current bar.
        :param account: Current account state.
        :returns: Feature vector as float32 numpy array.
        """
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of features this extractor produces."""
        pass


class OHLCVFeatures(FeatureExtractor):
    """Extract normalized OHLCV features.

    Normalizes prices relative to current close and volume relative
    to a rolling average.

    :param lookback: Number of past bars to include.
    :param volume_lookback: Bars for volume normalization.
    """

    def __init__(self, lookback: int = 10, volume_lookback: int = 20) -> None:
        self.lookback = lookback
        self.volume_lookback = volume_lookback

    @property
    def num_features(self) -> int:
        # For each lookback bar: open, high, low, close, volume (5 features)
        return self.lookback * 5

    def extract(
        self,
        bars: list["NormalizedBar"],
        current_idx: int,
        account: "Account",
    ) -> NDArray[np.float32]:
        """Extract normalized OHLCV features."""
        features = []

        # Get current close for price normalization
        current_bar = bars[current_idx]
        current_close = current_bar.close

        # Compute average volume for normalization
        vol_start = max(0, current_idx - self.volume_lookback)
        vol_bars = bars[vol_start : current_idx + 1]
        avg_volume = np.mean([b.volume for b in vol_bars]) if vol_bars else 1.0
        avg_volume = max(avg_volume, 1.0)  # Avoid division by zero

        # Extract lookback bars
        for i in range(self.lookback):
            bar_idx = current_idx - (self.lookback - 1 - i)
            if bar_idx < 0 or bar_idx >= len(bars):
                # Pad with zeros for missing bars
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                bar = bars[bar_idx]
                # Normalize prices relative to current close (returns-like)
                features.extend([
                    (bar.open / current_close) - 1.0,
                    (bar.high / current_close) - 1.0,
                    (bar.low / current_close) - 1.0,
                    (bar.close / current_close) - 1.0,
                    (bar.volume / avg_volume) - 1.0,  # Relative to average
                ])

        return np.array(features, dtype=np.float32)


class TechnicalFeatures(FeatureExtractor):
    """Extract technical indicator features.

    Includes RSI, moving averages, and momentum indicators.

    :param rsi_period: Period for RSI calculation.
    :param short_ma: Short moving average period.
    :param long_ma: Long moving average period.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        short_ma: int = 10,
        long_ma: int = 50,
    ) -> None:
        self.rsi_period = rsi_period
        self.short_ma = short_ma
        self.long_ma = long_ma

    @property
    def num_features(self) -> int:
        # RSI, short MA ratio, long MA ratio, MA crossover, momentum
        return 5

    def _compute_rsi(self, bars: list["NormalizedBar"], idx: int) -> float:
        """Compute RSI at given index."""
        if idx < self.rsi_period:
            return 0.5  # Neutral

        gains = []
        losses = []

        for i in range(idx - self.rsi_period + 1, idx + 1):
            if i > 0:
                change = bars[i].close - bars[i - 1].close
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            return 1.0  # Max RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to [0, 1]

    def _compute_ma(self, bars: list["NormalizedBar"], idx: int, period: int) -> float:
        """Compute moving average."""
        start = max(0, idx - period + 1)
        ma_bars = bars[start : idx + 1]
        if not ma_bars:
            return bars[idx].close
        return np.mean([b.close for b in ma_bars])

    def extract(
        self,
        bars: list["NormalizedBar"],
        current_idx: int,
        account: "Account",
    ) -> NDArray[np.float32]:
        """Extract technical features."""
        current_bar = bars[current_idx]
        current_close = current_bar.close

        # RSI (normalized to [-1, 1])
        rsi = self._compute_rsi(bars, current_idx)
        rsi_normalized = (rsi - 0.5) * 2  # Map [0,1] to [-1,1]

        # Moving averages relative to current price
        short_ma = self._compute_ma(bars, current_idx, self.short_ma)
        long_ma = self._compute_ma(bars, current_idx, self.long_ma)

        short_ma_ratio = (current_close / short_ma) - 1.0 if short_ma > 0 else 0.0
        long_ma_ratio = (current_close / long_ma) - 1.0 if long_ma > 0 else 0.0

        # MA crossover signal (positive = short above long)
        ma_crossover = (short_ma / long_ma) - 1.0 if long_ma > 0 else 0.0

        # Momentum (return over short period)
        momentum_idx = max(0, current_idx - self.short_ma)
        momentum = (current_close / bars[momentum_idx].close) - 1.0

        return np.array(
            [rsi_normalized, short_ma_ratio, long_ma_ratio, ma_crossover, momentum],
            dtype=np.float32,
        )


class AccountFeatures(FeatureExtractor):
    """Extract account state features.

    Includes position info, cash ratio, and unrealized P&L.

    :param symbol: Symbol to track position for.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def num_features(self) -> int:
        # Position ratio, cash ratio, unrealized PnL ratio
        return 3

    def extract(
        self,
        bars: list["NormalizedBar"],
        current_idx: int,
        account: "Account",
    ) -> NDArray[np.float32]:
        """Extract account features."""
        current_bar = bars[current_idx]
        current_price = current_bar.close

        # Get position
        position = account.positions.get(self.symbol)
        position_qty = position.quantity if position else 0.0
        cost_basis = position.cost_basis if position else 0.0

        # Calculate values
        position_value = position_qty * current_price
        cash = account.cleared_balance
        total_equity = cash + position_value

        # Avoid division by zero
        if total_equity <= 0:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Position as fraction of portfolio (0 = no position, 1 = all in)
        position_ratio = position_value / total_equity

        # Cash as fraction of portfolio
        cash_ratio = cash / total_equity

        # Unrealized P&L ratio (relative to cost)
        if position_qty > 0 and cost_basis > 0:
            unrealized_pnl = (current_price - cost_basis) / cost_basis
        else:
            unrealized_pnl = 0.0

        return np.array(
            [position_ratio, cash_ratio, unrealized_pnl],
            dtype=np.float32,
        )


class CombinedFeatures(FeatureExtractor):
    """Combine multiple feature extractors.

    :param extractors: List of feature extractors to combine.
    """

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        self.extractors = extractors

    @property
    def num_features(self) -> int:
        return sum(e.num_features for e in self.extractors)

    def extract(
        self,
        bars: list["NormalizedBar"],
        current_idx: int,
        account: "Account",
    ) -> NDArray[np.float32]:
        """Extract and concatenate features from all extractors."""
        all_features = []
        for extractor in self.extractors:
            features = extractor.extract(bars, current_idx, account)
            all_features.append(features)
        return np.concatenate(all_features)


