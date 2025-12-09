from datetime import datetime, timezone

from trading.types import Bar, NormalizedBar, Symbol


def test_bar_creation_and_attributes() -> None:
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    bar = Bar(symbol=Symbol("QQQ"), timestamp=ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=1234.0)

    assert bar.symbol == Symbol("QQQ")
    assert bar.timestamp == ts
    assert bar.open == 1.0
    assert bar.high == 2.0
    assert bar.low == 0.5
    assert bar.close == 1.5
    assert bar.volume == 1234.0


def test_normalized_bar_is_subclass_of_bar() -> None:
    ts = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
    nbar = NormalizedBar(symbol=Symbol("QQQ"), timestamp=ts, open=10.0, high=11.0, low=9.5, close=10.5, volume=10_000.0)

    assert isinstance(nbar, Bar)
    assert nbar.symbol == Symbol("QQQ")
    assert nbar.timestamp == ts
