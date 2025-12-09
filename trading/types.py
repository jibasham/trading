from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import NewType


Symbol = NewType("Symbol", str)


@dataclass(slots=True)
class Bar:
    """Raw bar of market data for a symbol.

    This is the common structure produced by all data sources before any
    additional normalization or enrichment.
    """

    symbol: Symbol
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class NormalizedBar(Bar):
    """Bar that has been normalized into the engine's canonical format.

    For now this is identical to ``Bar`` but is modeled separately so future
    transformations (currency conversion, corporate actions, etc.) can be
    reflected without changing call sites.
    """

    # Additional normalized fields can be added here later.
    pass
