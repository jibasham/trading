"""Data source implementations for fetching market data.

This module provides a protocol-based interface for data sources and concrete
implementations for Yahoo Finance, local storage, and CSV files.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from trading.exceptions import DataSourceError
from trading.types import Bar, DateRange, Symbol

if TYPE_CHECKING:
    from trading.types import FetchDataConfig


class DataSource(ABC):
    """Abstract base class for data sources.

    All data source implementations must inherit from this class and implement
    the `fetch_bars` method.
    """

    @abstractmethod
    def fetch_bars(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
        granularity: str,
    ) -> Iterator[Bar]:
        """Fetch bar data for the given symbols and time range.

        :param symbols: List of symbols to fetch.
        :param date_range: Time range to fetch (inclusive start, exclusive end).
        :param granularity: Bar granularity (e.g., "1m", "5m", "1h", "1d").
        :returns: Iterator of Bar objects in chronological order.
        :raises DataSourceError: If fetching fails.
        """
        ...


class YahooDataSource(DataSource):
    """Data source that fetches data from Yahoo Finance via yfinance.

    :param source_params: Optional parameters for configuring the source.
        - timeout: Request timeout in seconds (default: 30)
        - progress: Whether to show download progress (default: False)
    """

    # Map our granularity format to yfinance interval format
    GRANULARITY_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "90m": "90m",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }

    def __init__(self, source_params: dict[str, Any] | None = None) -> None:
        """Initialize Yahoo data source.

        :param source_params: Optional configuration parameters.
        """
        self.params = source_params or {}
        self.timeout = self.params.get("timeout", 30)
        self.progress = self.params.get("progress", False)

    def fetch_bars(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
        granularity: str,
    ) -> Iterator[Bar]:
        """Fetch bar data from Yahoo Finance.

        :param symbols: List of symbols to fetch.
        :param date_range: Time range to fetch.
        :param granularity: Bar granularity.
        :returns: Iterator of Bar objects.
        :raises DataSourceError: If fetching fails.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataSourceError(
                "yfinance is not installed. Install it with: pip install yfinance"
            ) from e

        # Convert granularity to yfinance format
        interval = self.GRANULARITY_MAP.get(granularity)
        if interval is None:
            raise DataSourceError(
                f"Unsupported granularity '{granularity}'. "
                f"Supported: {list(self.GRANULARITY_MAP.keys())}"
            )

        # yfinance uses strings for dates
        start_str = date_range.start.strftime("%Y-%m-%d")
        end_str = date_range.end.strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                ticker = yf.Ticker(str(symbol))
                df = ticker.history(
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    timeout=self.timeout,
                )

                if df.empty:
                    # No data for this symbol, log warning but continue
                    continue

                # Iterate over DataFrame rows and yield Bar objects
                for timestamp, row in df.iterrows():
                    # yfinance returns timezone-aware timestamps
                    ts = timestamp.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    yield Bar(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )

            except Exception as e:
                raise DataSourceError(
                    f"Failed to fetch data for symbol '{symbol}': {e}"
                ) from e


class LocalDataSource(DataSource):
    """Data source that loads previously stored datasets from local storage.

    Reads from the Parquet files in ~/.trading/datasets/{dataset_id}/.

    :param source_params: Required parameters:
        - dataset_id: The ID of the dataset to load.
    """

    def __init__(self, source_params: dict[str, Any] | None = None) -> None:
        """Initialize local data source.

        :param source_params: Configuration with dataset_id.
        :raises DataSourceError: If dataset_id is not provided.
        """
        self.params = source_params or {}
        self.dataset_id = self.params.get("dataset_id")
        if not self.dataset_id:
            raise DataSourceError(
                "LocalDataSource requires 'dataset_id' in source_params"
            )

    def fetch_bars(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
        granularity: str,
    ) -> Iterator[Bar]:
        """Load bar data from local storage.

        :param symbols: List of symbols to filter (empty = all symbols).
        :param date_range: Time range to filter.
        :param granularity: Ignored for local source (uses stored granularity).
        :returns: Iterator of Bar objects.
        :raises DataSourceError: If loading fails.
        """
        try:
            from trading._core import load_dataset
        except ImportError as e:
            raise DataSourceError(
                "Rust core module not available for local storage"
            ) from e

        try:
            bars = load_dataset(self.dataset_id)
        except Exception as e:
            raise DataSourceError(
                f"Failed to load dataset '{self.dataset_id}': {e}"
            ) from e

        # Convert symbols to set for fast lookup
        symbol_set = set(str(s) for s in symbols) if symbols else None

        for bar_dict in bars:
            bar_symbol = bar_dict["symbol"]

            # Filter by symbol if specified
            if symbol_set and bar_symbol not in symbol_set:
                continue

            # Filter by date range
            bar_ts = bar_dict["timestamp"]
            if bar_ts < date_range.start or bar_ts >= date_range.end:
                continue

            yield Bar(
                symbol=Symbol(bar_symbol),
                timestamp=bar_ts,
                open=bar_dict["open"],
                high=bar_dict["high"],
                low=bar_dict["low"],
                close=bar_dict["close"],
                volume=bar_dict["volume"],
            )


class CSVDataSource(DataSource):
    """Data source that reads bar data from CSV files.

    Expected CSV format (default columns):
    - symbol: Stock symbol
    - timestamp: ISO format datetime string
    - open, high, low, close: Prices
    - volume: Trading volume

    :param source_params: Required parameters:
        - file_path: Path to the CSV file.
        Optional parameters:
        - symbol_col: Column name for symbol (default: "symbol")
        - timestamp_col: Column name for timestamp (default: "timestamp")
        - open_col: Column name for open price (default: "open")
        - high_col: Column name for high price (default: "high")
        - low_col: Column name for low price (default: "low")
        - close_col: Column name for close price (default: "close")
        - volume_col: Column name for volume (default: "volume")
        - delimiter: CSV delimiter (default: ",")
        - timestamp_format: strptime format for timestamps (default: ISO format)
    """

    def __init__(self, source_params: dict[str, Any] | None = None) -> None:
        """Initialize CSV data source.

        :param source_params: Configuration with file_path and optional column mappings.
        :raises DataSourceError: If file_path is not provided.
        """
        self.params = source_params or {}
        self.file_path = self.params.get("file_path")
        if not self.file_path:
            raise DataSourceError("CSVDataSource requires 'file_path' in source_params")

        # Column name mappings with defaults
        self.symbol_col = self.params.get("symbol_col", "symbol")
        self.timestamp_col = self.params.get("timestamp_col", "timestamp")
        self.open_col = self.params.get("open_col", "open")
        self.high_col = self.params.get("high_col", "high")
        self.low_col = self.params.get("low_col", "low")
        self.close_col = self.params.get("close_col", "close")
        self.volume_col = self.params.get("volume_col", "volume")
        self.delimiter = self.params.get("delimiter", ",")
        self.timestamp_format = self.params.get("timestamp_format")

    def fetch_bars(
        self,
        symbols: list[Symbol],
        date_range: DateRange,
        granularity: str,
    ) -> Iterator[Bar]:
        """Read bar data from CSV file.

        :param symbols: List of symbols to filter (empty = all symbols).
        :param date_range: Time range to filter.
        :param granularity: Ignored for CSV source.
        :returns: Iterator of Bar objects.
        :raises DataSourceError: If reading fails.
        """
        path = Path(self.file_path)
        if not path.exists():
            raise DataSourceError(f"CSV file not found: {self.file_path}")

        # Convert symbols to set for fast lookup
        symbol_set = set(str(s) for s in symbols) if symbols else None

        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)

                for row in reader:
                    # Extract symbol
                    bar_symbol = row.get(self.symbol_col)
                    if not bar_symbol:
                        continue  # Skip rows without symbol

                    # Filter by symbol if specified
                    if symbol_set and bar_symbol not in symbol_set:
                        continue

                    # Parse timestamp
                    ts_str = row.get(self.timestamp_col)
                    if not ts_str:
                        continue

                    try:
                        if self.timestamp_format:
                            ts = datetime.strptime(ts_str, self.timestamp_format)
                        else:
                            # Try ISO format
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                        # Ensure timezone-aware
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)

                    except ValueError as e:
                        raise DataSourceError(
                            f"Failed to parse timestamp '{ts_str}': {e}"
                        ) from e

                    # Filter by date range
                    if ts < date_range.start or ts >= date_range.end:
                        continue

                    # Extract price/volume fields
                    try:
                        yield Bar(
                            symbol=Symbol(bar_symbol),
                            timestamp=ts,
                            open=float(row[self.open_col]),
                            high=float(row[self.high_col]),
                            low=float(row[self.low_col]),
                            close=float(row[self.close_col]),
                            volume=float(row[self.volume_col]),
                        )
                    except (KeyError, ValueError) as e:
                        raise DataSourceError(f"Failed to parse row {row}: {e}") from e

        except csv.Error as e:
            raise DataSourceError(f"CSV parsing error: {e}") from e
        except OSError as e:
            raise DataSourceError(f"Failed to read CSV file: {e}") from e


def resolve_data_source(config: FetchDataConfig) -> DataSource:
    """Construct a data source from configuration.

    :param config: FetchDataConfig with data_source and source_params.
    :returns: DataSource instance for the specified type.
    :raises DataSourceError: If data_source type is unrecognized.
    """
    source_type = config.data_source.lower()

    if source_type == "yahoo":
        return YahooDataSource(config.source_params)
    elif source_type == "local":
        return LocalDataSource(config.source_params)
    elif source_type == "csv":
        return CSVDataSource(config.source_params)
    else:
        raise DataSourceError(
            f"Unrecognized data source type: '{config.data_source}'. "
            f"Supported types: yahoo, local, csv"
        )
