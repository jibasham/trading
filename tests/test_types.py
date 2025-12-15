"""Tests for core type definitions."""

from datetime import datetime, timezone

from trading.types import (Account, Bar, DatasetBundle, DatasetId,
                           DatasetMetadata, DateRange, Execution,
                           FetchDataConfig, Gap, GenSynthConfig, NormalizedBar,
                           Order, OrderRequest, OrderStatus,
                           PendingTransaction, Position, RewardSignal, RunId,
                           RunMetrics, RunProgress, Symbol, TimeSlice,
                           TrainingConfig)

# ---------------------------------------------------------------------------
# Basic Type Tests
# ---------------------------------------------------------------------------


def test_bar_creation_and_attributes() -> None:
    """Bar should store all OHLCV fields correctly."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    bar = Bar(
        symbol=Symbol("QQQ"),
        timestamp=ts,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=1234.0,
    )

    assert bar.symbol == Symbol("QQQ")
    assert bar.timestamp == ts
    assert bar.open == 1.0
    assert bar.high == 2.0
    assert bar.low == 0.5
    assert bar.close == 1.5
    assert bar.volume == 1234.0


def test_normalized_bar_is_subclass_of_bar() -> None:
    """NormalizedBar should inherit from Bar."""
    ts = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
    nbar = NormalizedBar(
        symbol=Symbol("QQQ"),
        timestamp=ts,
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        volume=10_000.0,
    )

    assert isinstance(nbar, Bar)
    assert nbar.symbol == Symbol("QQQ")
    assert nbar.timestamp == ts


def test_identifier_newtypes_wrap_strings() -> None:
    """NewType identifiers should behave as strings."""
    ds_id = DatasetId("local:qqq_5m_2020_2024")
    run_id = RunId("run-001")

    assert isinstance(ds_id, str)
    assert isinstance(run_id, str)
    assert ds_id == "local:qqq_5m_2020_2024"
    assert run_id == "run-001"


def test_date_range_holds_start_and_end() -> None:
    """DateRange should store start and end timestamps."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    dr = DateRange(start=start, end=end)

    assert dr.start == start
    assert dr.end == end


def test_run_metrics_dataclass_fields() -> None:
    """RunMetrics should store all metric fields."""
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


# ---------------------------------------------------------------------------
# Position and Account Tests
# ---------------------------------------------------------------------------


def test_position_with_pending_quantity() -> None:
    """Position should track pending quantity."""
    pos = Position(
        symbol=Symbol("AAPL"),
        quantity=100.0,
        cost_basis=150.0,
        pending_quantity=25.0,
    )

    assert pos.symbol == Symbol("AAPL")
    assert pos.quantity == 100.0
    assert pos.cost_basis == 150.0
    assert pos.pending_quantity == 25.0


def test_account_defaults() -> None:
    """Account should have sensible defaults."""
    account = Account(
        account_id="test-account",
        base_currency="USD",
        cleared_balance=10000.0,
        pending_balance=0.0,
    )

    assert account.account_id == "test-account"
    assert account.reserved_balance == 0.0
    assert account.positions == {}
    assert account.clearing_delay_hours == 24
    assert account.use_business_days is False
    assert account.pending_transactions == []


def test_pending_transaction_fields() -> None:
    """PendingTransaction should store all transaction details."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    txn = PendingTransaction(
        transaction_id="txn-001",
        symbol=Symbol("AAPL"),
        quantity=10.0,
        amount=-1500.0,
        timestamp=ts,
        side="buy",
    )

    assert txn.transaction_id == "txn-001"
    assert txn.symbol == Symbol("AAPL")
    assert txn.quantity == 10.0
    assert txn.amount == -1500.0
    assert txn.timestamp == ts
    assert txn.side == "buy"


# ---------------------------------------------------------------------------
# Order Tests
# ---------------------------------------------------------------------------


def test_order_status_enum() -> None:
    """OrderStatus should have expected values."""
    assert OrderStatus.PENDING == "pending"
    assert OrderStatus.PARTIALLY_FILLED == "partially_filled"
    assert OrderStatus.FILLED == "filled"
    assert OrderStatus.CANCELED == "canceled"


def test_order_request_defaults() -> None:
    """OrderRequest should default to market order."""
    req = OrderRequest(symbol=Symbol("AAPL"), side="buy", quantity=10.0)

    assert req.symbol == Symbol("AAPL")
    assert req.side == "buy"
    assert req.quantity == 10.0
    assert req.order_type == "market"


def test_order_lifecycle() -> None:
    """Order should track fill status."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    order = Order(
        order_id="order-001",
        symbol=Symbol("AAPL"),
        side="buy",
        quantity=100.0,
        order_type="market",
        status=OrderStatus.PENDING,
        created_at=ts,
    )

    assert order.filled_quantity == 0.0
    assert order.average_fill_price is None

    # Simulate fill (mutable model)
    order.status = OrderStatus.FILLED
    order.filled_quantity = 100.0
    order.average_fill_price = 150.0

    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 100.0
    assert order.average_fill_price == 150.0


def test_execution_record() -> None:
    """Execution should record trade details."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    execution = Execution(
        symbol=Symbol("AAPL"),
        side="buy",
        quantity=100.0,
        price=150.0,
        timestamp=ts,
        order_id="order-001",
    )

    assert execution.symbol == Symbol("AAPL")
    assert execution.price == 150.0
    assert execution.order_id == "order-001"


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


def test_fetch_data_config() -> None:
    """FetchDataConfig should store all fetch parameters."""
    date_range = DateRange(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    config = FetchDataConfig(
        symbols=[Symbol("QQQ"), Symbol("SPY")],
        date_range=date_range,
        granularity="5m",
        data_source="yahoo",
    )

    assert len(config.symbols) == 2
    assert config.granularity == "5m"
    assert config.data_source == "yahoo"
    assert config.dataset_id is None


def test_gen_synth_config() -> None:
    """GenSynthConfig should store generation parameters."""
    date_range = DateRange(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    config = GenSynthConfig(
        symbols=[Symbol("SYNTH1")],
        date_range=date_range,
        granularity="5m",
        generator_type="geometric_brownian",
        generator_params={"initial_price": 100.0, "volatility": 0.02},
        random_seed=42,
    )

    assert config.generator_type == "geometric_brownian"
    assert config.random_seed == 42
    assert config.generator_params["initial_price"] == 100.0


def test_training_config_defaults() -> None:
    """TrainingConfig should have sensible defaults."""
    config = TrainingConfig()

    assert config.account_starting_balance == 10000.0
    assert config.account_base_currency == "USD"
    assert config.clearing_delay_hours == 24
    assert config.risk_max_leverage == 1.0
    assert config.log_level == "INFO"
    assert config.enable_event_logging is True


# ---------------------------------------------------------------------------
# Training/Run Types Tests
# ---------------------------------------------------------------------------


def test_time_slice_with_bars() -> None:
    """TimeSlice should hold bars at a timestamp."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    bar = NormalizedBar(
        symbol=Symbol("AAPL"),
        timestamp=ts,
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1000.0,
    )
    slice_ = TimeSlice(timestamp=ts, bars={"AAPL": bar})

    assert slice_.timestamp == ts
    assert "AAPL" in slice_.bars
    assert slice_.bars["AAPL"].close == 150.5


def test_dataset_metadata() -> None:
    """DatasetMetadata should store dataset information."""
    date_range = DateRange(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    metadata = DatasetMetadata(
        dataset_id=DatasetId("test-dataset"),
        symbols=[Symbol("QQQ")],
        date_range=date_range,
        granularity="5m",
        data_source="yahoo",
        created_at=datetime.now(timezone.utc),
        bar_count=1000,
    )

    assert metadata.dataset_id == DatasetId("test-dataset")
    assert metadata.bar_count == 1000


# ---------------------------------------------------------------------------
# Utility Types Tests
# ---------------------------------------------------------------------------


def test_gap_model() -> None:
    """Gap should represent data gaps."""
    gap = Gap(
        symbol=Symbol("AAPL"),
        start_time=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        expected_bars=12,
    )

    assert gap.symbol == Symbol("AAPL")
    assert gap.expected_bars == 12


def test_reward_signal() -> None:
    """RewardSignal should store RL reward information."""
    signal = RewardSignal(
        reward=0.5,
        timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        execution_id="exec-001",
        context={"profit": 100.0},
    )

    assert signal.reward == 0.5
    assert signal.execution_id == "exec-001"


def test_run_progress() -> None:
    """RunProgress should track training progress."""
    progress = RunProgress(
        run_id=RunId("run-001"),
        current_timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        total_timestamps=1000,
        completed_timestamps=500,
        current_account_equity=10500.0,
        num_executions=25,
        elapsed_time_seconds=60.0,
    )

    assert progress.completed_timestamps == 500
    assert progress.current_account_equity == 10500.0


# ---------------------------------------------------------------------------
# Pydantic Serialization Tests
# ---------------------------------------------------------------------------


def test_bar_serialization() -> None:
    """Bar should serialize to dict via model_dump()."""
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    bar = Bar(
        symbol=Symbol("QQQ"),
        timestamp=ts,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=1234.0,
    )
    data = bar.model_dump()

    assert data["symbol"] == "QQQ"
    assert data["open"] == 1.0
    assert data["timestamp"] == ts


def test_account_serialization() -> None:
    """Account should serialize to dict via model_dump()."""
    account = Account(
        account_id="test-account",
        base_currency="USD",
        cleared_balance=10000.0,
        pending_balance=500.0,
    )
    data = account.model_dump()

    assert data["account_id"] == "test-account"
    assert data["cleared_balance"] == 10000.0
    assert data["positions"] == {}


# ---------------------------------------------------------------------------
# DatasetBundle Tests
# ---------------------------------------------------------------------------


def test_dataset_bundle_get_bars_at() -> None:
    """DatasetBundle should find bars at or before a timestamp."""
    ts1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)

    bar1 = NormalizedBar(
        symbol=Symbol("AAPL"),
        timestamp=ts1,
        open=150.0,
        high=151.0,
        low=149.0,
        close=150.5,
        volume=1000.0,
    )
    bar2 = NormalizedBar(
        symbol=Symbol("AAPL"),
        timestamp=ts2,
        open=150.5,
        high=152.0,
        low=150.0,
        close=151.0,
        volume=1200.0,
    )

    bundle = DatasetBundle(bars_by_symbol={"AAPL": [bar1, bar2]})

    # Query at ts2 should return bar2
    result = bundle.get_bars_at(ts2)
    assert "AAPL" in result
    assert result["AAPL"].close == 151.0

    # Query between ts1 and ts2 should return bar1
    query_ts = datetime(2024, 1, 1, 10, 3, tzinfo=timezone.utc)
    result = bundle.get_bars_at(query_ts)
    assert result["AAPL"].close == 150.5


def test_dataset_bundle_get_symbol_history() -> None:
    """DatasetBundle should return lookback history."""
    bars = [
        NormalizedBar(
            symbol=Symbol("AAPL"),
            timestamp=datetime(2024, 1, 1, 10, i, tzinfo=timezone.utc),
            open=150.0 + i,
            high=151.0 + i,
            low=149.0 + i,
            close=150.5 + i,
            volume=1000.0,
        )
        for i in range(10)
    ]

    bundle = DatasetBundle(bars_by_symbol={"AAPL": bars})

    # Get last 3 bars before timestamp 10:05
    end_time = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
    history = bundle.get_symbol_history("AAPL", end_time, 3)

    assert len(history) == 3
    # Should be bars at 10:03, 10:04, 10:05
    assert history[0].timestamp.minute == 3
    assert history[2].timestamp.minute == 5
