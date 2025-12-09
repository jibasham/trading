from datetime import datetime, timezone

from trading.types import Bar, DateRange, DatasetId, NormalizedBar, RunId, RunMetrics, Symbol


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


def test_identifier_newtypes_wrap_strings() -> None:
    ds_id = DatasetId("local:qqq_5m_2020_2024")
    run_id = RunId("run-001")

    assert isinstance(ds_id, str)
    assert isinstance(run_id, str)
    assert ds_id == "local:qqq_5m_2020_2024"
    assert run_id == "run-001"


def test_date_range_holds_start_and_end() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    dr = DateRange(start=start, end=end)

    assert dr.start == start
    assert dr.end == end


def test_run_metrics_dataclass_fields() -> None:
    metrics = RunMetrics(
        run_id=RunId("run-123"),
        total_return=0.10,
        max_drawdown=0.05,
        volatility=0.02,
        sharpe_ratio=1.5,
        num_trades=42,
        win_rate=0.6,
    )

    assert metrics.run_id == RunId("run-123")
    assert metrics.total_return == 0.10
    assert metrics.max_drawdown == 0.05
    assert metrics.volatility == 0.02
    assert metrics.sharpe_ratio == 1.5
    assert metrics.num_trades == 42
    assert metrics.win_rate == 0.6
