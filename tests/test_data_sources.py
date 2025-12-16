"""Tests for data source implementations."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading.data.sources import (
    CSVDataSource,
    DataSource,
    LocalDataSource,
    YahooDataSource,
    resolve_data_source,
)
from trading.exceptions import DataSourceError
from trading.types import DateRange, FetchDataConfig, Symbol


@pytest.fixture
def date_range() -> DateRange:
    """Create a test date range."""
    return DateRange(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


@pytest.fixture
def symbols() -> list[Symbol]:
    """Create test symbols."""
    return [Symbol("AAPL"), Symbol("GOOGL")]


class TestDataSourceProtocol:
    """Tests for the DataSource abstract base class."""

    def test_datasource_is_abstract(self) -> None:
        """DataSource cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            DataSource()  # type: ignore[abstract]

    def test_subclass_must_implement_fetch_bars(self) -> None:
        """Subclasses must implement fetch_bars."""

        class IncompleteSource(DataSource):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSource()


class TestYahooDataSource:
    """Tests for YahooDataSource."""

    def test_init_with_defaults(self) -> None:
        """YahooDataSource initializes with default parameters."""
        source = YahooDataSource()

        assert source.timeout == 30
        assert source.progress is False

    def test_init_with_custom_params(self) -> None:
        """YahooDataSource accepts custom parameters."""
        source = YahooDataSource({"timeout": 60, "progress": True})

        assert source.timeout == 60
        assert source.progress is True

    def test_unsupported_granularity_raises_error(
        self, symbols: list[Symbol], date_range: DateRange
    ) -> None:
        """Unsupported granularity should raise DataSourceError."""
        source = YahooDataSource()

        with pytest.raises(DataSourceError, match="Unsupported granularity"):
            list(source.fetch_bars(symbols, date_range, "invalid"))

    def test_fetch_bars_returns_bars(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
    ) -> None:
        """fetch_bars should return Bar objects from yfinance data."""
        import pandas as pd

        # Create mock DataFrame with OHLCV data
        mock_df = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [148.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 10:00:00", tz="UTC")]),
        )

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import sys

            mock_yf = sys.modules["yfinance"]
            mock_yf.Ticker.return_value = mock_ticker

            source = YahooDataSource()
            bars = list(source.fetch_bars([Symbol("AAPL")], date_range, "5m"))

        assert len(bars) == 1
        assert bars[0].symbol == "AAPL"
        assert bars[0].open == 150.0
        assert bars[0].close == 153.0
        assert bars[0].volume == 1000000

    def test_fetch_bars_empty_data_continues(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
    ) -> None:
        """Empty data for a symbol should not raise error."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import sys

            mock_yf = sys.modules["yfinance"]
            mock_yf.Ticker.return_value = mock_ticker

            source = YahooDataSource()
            bars = list(source.fetch_bars(symbols, date_range, "5m"))

        assert bars == []


class TestLocalDataSource:
    """Tests for LocalDataSource."""

    def test_init_requires_dataset_id(self) -> None:
        """LocalDataSource requires dataset_id in source_params."""
        with pytest.raises(DataSourceError, match="requires 'dataset_id'"):
            LocalDataSource()

        with pytest.raises(DataSourceError, match="requires 'dataset_id'"):
            LocalDataSource({})

    def test_init_with_dataset_id(self) -> None:
        """LocalDataSource initializes with dataset_id."""
        source = LocalDataSource({"dataset_id": "test-dataset"})

        assert source.dataset_id == "test-dataset"

    def test_fetch_bars_from_storage(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        date_range: DateRange,
    ) -> None:
        """fetch_bars should load data from local storage."""
        from trading._core import store_dataset

        # Patch HOME for storage
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create test data
        test_bars = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]
        metadata = json.dumps({"dataset_id": "test-local", "bar_count": 1})

        store_dataset(test_bars, "test-local", metadata)

        # Load using LocalDataSource
        source = LocalDataSource({"dataset_id": "test-local"})
        bars = list(source.fetch_bars([], date_range, "5m"))

        assert len(bars) == 1
        assert bars[0].symbol == "AAPL"
        assert bars[0].close == 153.0

    def test_fetch_bars_filters_by_symbol(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        date_range: DateRange,
    ) -> None:
        """fetch_bars should filter by symbols when specified."""
        from trading._core import store_dataset

        monkeypatch.setenv("HOME", str(tmp_path))

        test_bars = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {
                "symbol": "GOOGL",
                "timestamp": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "open": 2800.0,
                "high": 2850.0,
                "low": 2780.0,
                "close": 2830.0,
                "volume": 500.0,
            },
        ]
        metadata = json.dumps({"dataset_id": "test-filter", "bar_count": 2})

        store_dataset(test_bars, "test-filter", metadata)

        source = LocalDataSource({"dataset_id": "test-filter"})
        bars = list(source.fetch_bars([Symbol("AAPL")], date_range, "5m"))

        assert len(bars) == 1
        assert bars[0].symbol == "AAPL"

    def test_fetch_bars_filters_by_date_range(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fetch_bars should filter by date range."""
        from trading._core import store_dataset

        monkeypatch.setenv("HOME", str(tmp_path))

        test_bars = [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {
                "symbol": "AAPL",
                "timestamp": datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc),
                "open": 160.0,
                "high": 165.0,
                "low": 158.0,
                "close": 163.0,
                "volume": 1100.0,
            },
        ]
        metadata = json.dumps({"dataset_id": "test-date", "bar_count": 2})

        store_dataset(test_bars, "test-date", metadata)

        # Only get bars from Jan 1
        date_range = DateRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        source = LocalDataSource({"dataset_id": "test-date"})
        bars = list(source.fetch_bars([], date_range, "5m"))

        assert len(bars) == 1
        assert bars[0].timestamp.day == 1


class TestCSVDataSource:
    """Tests for CSVDataSource."""

    def test_init_requires_file_path(self) -> None:
        """CSVDataSource requires file_path in source_params."""
        with pytest.raises(DataSourceError, match="requires 'file_path'"):
            CSVDataSource()

        with pytest.raises(DataSourceError, match="requires 'file_path'"):
            CSVDataSource({})

    def test_init_with_defaults(self) -> None:
        """CSVDataSource initializes with default column names."""
        source = CSVDataSource({"file_path": "test.csv"})

        assert source.symbol_col == "symbol"
        assert source.timestamp_col == "timestamp"
        assert source.open_col == "open"
        assert source.delimiter == ","

    def test_init_with_custom_columns(self) -> None:
        """CSVDataSource accepts custom column mappings."""
        source = CSVDataSource(
            {
                "file_path": "test.csv",
                "symbol_col": "ticker",
                "timestamp_col": "datetime",
                "delimiter": ";",
            }
        )

        assert source.symbol_col == "ticker"
        assert source.timestamp_col == "datetime"
        assert source.delimiter == ";"

    def test_fetch_bars_from_csv(self, date_range: DateRange) -> None:
        """fetch_bars should read data from CSV file."""
        csv_content = """symbol,timestamp,open,high,low,close,volume
AAPL,2024-01-01T10:00:00+00:00,150.0,155.0,148.0,153.0,1000
AAPL,2024-01-01T10:05:00+00:00,153.0,156.0,152.0,155.0,1200
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            source = CSVDataSource({"file_path": csv_path})
            bars = list(source.fetch_bars([], date_range, "5m"))

            assert len(bars) == 2
            assert bars[0].symbol == "AAPL"
            assert bars[0].open == 150.0
            assert bars[1].close == 155.0
        finally:
            Path(csv_path).unlink()

    def test_fetch_bars_filters_by_symbol(self, date_range: DateRange) -> None:
        """fetch_bars should filter by symbols."""
        csv_content = """symbol,timestamp,open,high,low,close,volume
AAPL,2024-01-01T10:00:00+00:00,150.0,155.0,148.0,153.0,1000
GOOGL,2024-01-01T10:00:00+00:00,2800.0,2850.0,2780.0,2830.0,500
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            source = CSVDataSource({"file_path": csv_path})
            bars = list(source.fetch_bars([Symbol("GOOGL")], date_range, "5m"))

            assert len(bars) == 1
            assert bars[0].symbol == "GOOGL"
        finally:
            Path(csv_path).unlink()

    def test_fetch_bars_with_custom_timestamp_format(
        self, date_range: DateRange
    ) -> None:
        """fetch_bars should parse custom timestamp formats."""
        csv_content = """symbol,timestamp,open,high,low,close,volume
AAPL,2024-01-01 10:00:00,150.0,155.0,148.0,153.0,1000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            source = CSVDataSource(
                {
                    "file_path": csv_path,
                    "timestamp_format": "%Y-%m-%d %H:%M:%S",
                }
            )
            bars = list(source.fetch_bars([], date_range, "5m"))

            assert len(bars) == 1
            assert bars[0].timestamp.tzinfo is not None
        finally:
            Path(csv_path).unlink()

    def test_fetch_bars_file_not_found(self, date_range: DateRange) -> None:
        """fetch_bars should raise error if file not found."""
        source = CSVDataSource({"file_path": "/nonexistent/file.csv"})

        with pytest.raises(DataSourceError, match="not found"):
            list(source.fetch_bars([], date_range, "5m"))

    def test_fetch_bars_filters_by_date_range(self) -> None:
        """fetch_bars should filter by date range."""
        csv_content = """symbol,timestamp,open,high,low,close,volume
AAPL,2024-01-01T10:00:00+00:00,150.0,155.0,148.0,153.0,1000
AAPL,2024-01-05T10:00:00+00:00,160.0,165.0,158.0,163.0,1100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            date_range = DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )

            source = CSVDataSource({"file_path": csv_path})
            bars = list(source.fetch_bars([], date_range, "5m"))

            assert len(bars) == 1
            assert bars[0].timestamp.day == 1
        finally:
            Path(csv_path).unlink()


class TestResolveDataSource:
    """Tests for resolve_data_source function."""

    def test_resolve_yahoo(self) -> None:
        """resolve_data_source returns YahooDataSource for 'yahoo'."""
        config = FetchDataConfig(
            symbols=[Symbol("AAPL")],
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            granularity="5m",
            data_source="yahoo",
            source_params={},
        )

        source = resolve_data_source(config)

        assert isinstance(source, YahooDataSource)

    def test_resolve_local(self) -> None:
        """resolve_data_source returns LocalDataSource for 'local'."""
        config = FetchDataConfig(
            symbols=[Symbol("AAPL")],
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            granularity="5m",
            data_source="local",
            source_params={"dataset_id": "test"},
        )

        source = resolve_data_source(config)

        assert isinstance(source, LocalDataSource)

    def test_resolve_csv(self) -> None:
        """resolve_data_source returns CSVDataSource for 'csv'."""
        config = FetchDataConfig(
            symbols=[Symbol("AAPL")],
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            granularity="5m",
            data_source="csv",
            source_params={"file_path": "test.csv"},
        )

        source = resolve_data_source(config)

        assert isinstance(source, CSVDataSource)

    def test_resolve_case_insensitive(self) -> None:
        """resolve_data_source should be case-insensitive."""
        config = FetchDataConfig(
            symbols=[Symbol("AAPL")],
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            granularity="5m",
            data_source="YAHOO",
            source_params={},
        )

        source = resolve_data_source(config)

        assert isinstance(source, YahooDataSource)

    def test_resolve_unknown_raises_error(self) -> None:
        """resolve_data_source raises error for unknown source types."""
        config = FetchDataConfig(
            symbols=[Symbol("AAPL")],
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            granularity="5m",
            data_source="unknown",
            source_params={},
        )

        with pytest.raises(DataSourceError, match="Unrecognized"):
            resolve_data_source(config)
