"""Position sizing strategies for risk management.

This module provides various position sizing algorithms that determine
how much capital to allocate to each trade based on account state and risk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading.types import Account, OrderRequest


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies.

    Position sizers calculate the appropriate quantity for a trade based on
    account state, current price, and risk parameters.
    """

    @abstractmethod
    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate the position size for a trade.

        :param account: Current account state.
        :param symbol: Symbol to trade.
        :param side: "buy" or "sell".
        :param price: Current price of the asset.
        :param signal_strength: Optional signal strength (0-1) to scale position.
        :returns: Quantity to trade (may be 0 if trade should be skipped).
        """
        pass

    def adjust_order(
        self,
        order: OrderRequest,
        account: Account,
        current_price: float,
        signal_strength: float = 1.0,
    ) -> OrderRequest:
        """Adjust an order's quantity based on position sizing.

        :param order: Original order request.
        :param account: Current account state.
        :param current_price: Current price of the asset.
        :param signal_strength: Optional signal strength.
        :returns: Adjusted order with new quantity.
        """
        from trading.types import OrderRequest as OR

        new_qty = self.calculate_quantity(
            account=account,
            symbol=str(order.symbol),
            side=order.side,
            price=current_price,
            signal_strength=signal_strength,
        )

        return OR(
            symbol=order.symbol,
            side=order.side,
            quantity=new_qty,
            order_id=order.order_id,
        )


class FixedQuantitySizer(PositionSizer):
    """Always trade a fixed quantity.

    :param quantity: Fixed quantity to trade.
    """

    def __init__(self, quantity: float = 10.0) -> None:
        self.quantity = quantity

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Return fixed quantity regardless of account state."""
        return self.quantity * signal_strength


class FixedDollarSizer(PositionSizer):
    """Trade a fixed dollar amount per position.

    :param dollar_amount: Dollar amount to allocate per trade.
    """

    def __init__(self, dollar_amount: float = 1000.0) -> None:
        self.dollar_amount = dollar_amount

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate quantity based on fixed dollar amount."""
        if price <= 0:
            return 0.0
        return (self.dollar_amount * signal_strength) / price


class PercentOfEquitySizer(PositionSizer):
    """Trade a percentage of current account equity.

    :param percent: Percentage of equity to allocate (e.g., 0.10 = 10%).
    """

    def __init__(self, percent: float = 0.10) -> None:
        self.percent = percent

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate quantity based on percentage of equity."""
        if price <= 0:
            return 0.0

        # Calculate total equity (cash + positions)
        equity = account.cleared_balance + account.pending_balance
        for sym, pos in account.positions.items():
            # Use cost basis as estimate if we don't have current price
            equity += pos.quantity * pos.cost_basis

        dollar_amount = equity * self.percent * signal_strength
        return dollar_amount / price


class RiskPercentSizer(PositionSizer):
    """Size positions based on risk per trade (requires stop loss).

    This implements the popular "risk 1%" or "risk 2%" approach where you
    determine position size based on how much you're willing to lose if
    your stop loss is hit.

    :param risk_percent: Percentage of equity to risk per trade (e.g., 0.01 = 1%).
    :param default_stop_pct: Default stop loss percentage if not specified.
    """

    def __init__(
        self,
        risk_percent: float = 0.01,
        default_stop_pct: float = 0.02,
    ) -> None:
        self.risk_percent = risk_percent
        self.default_stop_pct = default_stop_pct
        self._stop_pct: float | None = None

    def set_stop_percent(self, stop_pct: float) -> None:
        """Set the stop loss percentage for the next trade."""
        self._stop_pct = stop_pct

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate quantity based on risk percentage.

        Position size = (Equity × Risk%) / (Price × Stop%)
        """
        if price <= 0:
            return 0.0

        # Calculate total equity
        equity = account.cleared_balance + account.pending_balance
        for pos in account.positions.values():
            equity += pos.quantity * pos.cost_basis

        stop_pct = self._stop_pct if self._stop_pct else self.default_stop_pct
        self._stop_pct = None  # Reset for next trade

        if stop_pct <= 0:
            return 0.0

        # Risk per share = price × stop_pct
        risk_per_share = price * stop_pct

        # Maximum dollar risk = equity × risk_percent
        max_risk = equity * self.risk_percent * signal_strength

        # Quantity = max_risk / risk_per_share
        return max_risk / risk_per_share


class KellyCriterionSizer(PositionSizer):
    """Size positions using Kelly Criterion for optimal growth.

    Kelly formula: f* = (bp - q) / b
    where:
    - b = odds received on the bet (profit ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)
    - f* = fraction of bankroll to bet

    :param win_rate: Historical win rate (0-1).
    :param avg_win: Average winning trade amount.
    :param avg_loss: Average losing trade amount.
    :param fraction: Kelly fraction to use (0.5 = half Kelly, more conservative).
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win: float = 100.0,
        avg_loss: float = 100.0,
        fraction: float = 0.5,
    ) -> None:
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction

    def update_stats(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> None:
        """Update the Kelly parameters with new statistics."""
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def calculate_kelly_fraction(self) -> float:
        """Calculate the optimal Kelly fraction."""
        if self.avg_loss <= 0:
            return 0.0

        # b = profit ratio (avg_win / avg_loss)
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p

        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b

        # Apply fraction and cap at reasonable maximum
        return max(0.0, min(kelly * self.fraction, 0.25))

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate quantity using Kelly Criterion."""
        if price <= 0:
            return 0.0

        # Calculate total equity
        equity = account.cleared_balance + account.pending_balance
        for pos in account.positions.values():
            equity += pos.quantity * pos.cost_basis

        kelly_fraction = self.calculate_kelly_fraction()
        dollar_amount = equity * kelly_fraction * signal_strength

        return dollar_amount / price


class VolatilityAdjustedSizer(PositionSizer):
    """Adjust position size based on volatility (ATR-based).

    Lower volatility = larger position, higher volatility = smaller position.
    This helps maintain consistent risk across different market conditions.

    :param base_dollars: Base dollar amount for average volatility.
    :param target_atr_pct: Target ATR percentage for position sizing.
    """

    def __init__(
        self,
        base_dollars: float = 1000.0,
        target_atr_pct: float = 0.02,
    ) -> None:
        self.base_dollars = base_dollars
        self.target_atr_pct = target_atr_pct
        self._current_atr_pct: float | None = None

    def set_volatility(self, atr_pct: float) -> None:
        """Set the current ATR percentage for the symbol."""
        self._current_atr_pct = atr_pct

    def calculate_quantity(
        self,
        account: Account,
        symbol: str,
        side: str,
        price: float,
        signal_strength: float = 1.0,
    ) -> float:
        """Calculate volatility-adjusted quantity."""
        if price <= 0:
            return 0.0

        atr_pct = self._current_atr_pct or self.target_atr_pct
        self._current_atr_pct = None  # Reset

        if atr_pct <= 0:
            atr_pct = self.target_atr_pct

        # Scale position inversely with volatility
        volatility_adjustment = self.target_atr_pct / atr_pct
        adjusted_dollars = self.base_dollars * volatility_adjustment * signal_strength

        return adjusted_dollars / price


# Convenience function to get a sizer by name
def get_position_sizer(name: str, **kwargs: Any) -> PositionSizer:
    """Get a position sizer by name.

    :param name: Sizer name (fixed_qty, fixed_dollar, percent_equity, risk_percent, kelly, volatility).
    :param kwargs: Parameters to pass to the sizer.
    :returns: PositionSizer instance.
    """
    sizers = {
        "fixed_qty": FixedQuantitySizer,
        "fixed_dollar": FixedDollarSizer,
        "percent_equity": PercentOfEquitySizer,
        "risk_percent": RiskPercentSizer,
        "kelly": KellyCriterionSizer,
        "volatility": VolatilityAdjustedSizer,
    }

    if name not in sizers:
        raise ValueError(f"Unknown sizer: {name}. Available: {list(sizers.keys())}")

    return sizers[name](**kwargs)

