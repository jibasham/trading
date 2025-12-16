"""Tests for Rust data storage functions."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading._core import (StorageError, dataset_exists, list_datasets,
                           load_dataset, read_dataset_metadata, store_dataset)


@pytest.fixture
def temp_trading_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary trading directory and patch HOME."""
    # Create the .trading/datasets structure in tmp_path
    trading_dir = tmp_path / ".trading"
    datasets_dir = trading_dir / "datasets"
    datasets_dir.mkdir(parents=True)

    # Patch HOME environment variable so Rust uses tmp_path as home
    monkeypatch.setenv("HOME", str(tmp_path))

    yield trading_dir

    # Cleanup is automatic with tmp_path


@pytest.fixture
def sample_bars() -> list:
    """Create sample bars for testing."""
    base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    return [
        {
            "symbol": "AAPL",
            "timestamp": base_time,
            "open": 150.0,
            "high": 155.0,
            "low": 148.0,
            "close": 153.0,
            "volume": 1000.0,
        },
        {
            "symbol": "AAPL",
            "timestamp": datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc),
            "open": 153.0,
            "high": 156.0,
            "low": 152.0,
            "close": 155.0,
            "volume": 1200.0,
        },
        {
            "symbol": "GOOGL",
            "timestamp": base_time,
            "open": 2800.0,
            "high": 2850.0,
            "low": 2780.0,
            "close": 2830.0,
            "volume": 500.0,
        },
    ]


@pytest.fixture
def sample_metadata() -> str:
    """Create sample metadata JSON for testing."""
    return json.dumps(
        {
            "dataset_id": "test-dataset",
            "symbols": ["AAPL", "GOOGL"],
            "date_range": {
                "start": "2024-01-01T10:00:00Z",
                "end": "2024-01-01T11:00:00Z",
            },
            "granularity": "5m",
            "bar_count": 3,
            "source": "test",
        }
    )


class TestStoreDataset:
    """Tests for store_dataset function."""

    def test_store_creates_parquet_file(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Storing a dataset should create a parquet file."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        parquet_path = temp_trading_dir / "datasets" / "test-dataset" / "bars.parquet"
        assert parquet_path.exists()

    def test_store_creates_metadata_file(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Storing a dataset should create a metadata.json file."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        metadata_path = temp_trading_dir / "datasets" / "test-dataset" / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["dataset_id"] == "test-dataset"
        assert metadata["bar_count"] == 3

    def test_store_empty_bars(
        self, temp_trading_dir: Path, sample_metadata: str
    ) -> None:
        """Storing empty bars should create valid files."""
        store_dataset([], "empty-dataset", sample_metadata)

        parquet_path = temp_trading_dir / "datasets" / "empty-dataset" / "bars.parquet"
        metadata_path = (
            temp_trading_dir / "datasets" / "empty-dataset" / "metadata.json"
        )

        assert parquet_path.exists()
        assert metadata_path.exists()


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_returns_all_bars(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Loading a dataset should return all stored bars."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        loaded_bars = load_dataset("test-dataset")

        assert len(loaded_bars) == 3

    def test_load_preserves_bar_data(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Loaded bars should have the same data as stored bars."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        loaded_bars = load_dataset("test-dataset")

        # Find the first AAPL bar
        aapl_bar = next(b for b in loaded_bars if b["symbol"] == "AAPL")
        assert aapl_bar["open"] == 150.0
        assert aapl_bar["high"] == 155.0
        assert aapl_bar["low"] == 148.0
        assert aapl_bar["close"] == 153.0
        assert aapl_bar["volume"] == 1000.0

    def test_load_preserves_timestamps(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Loaded bars should have timezone-aware timestamps."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        loaded_bars = load_dataset("test-dataset")

        for bar in loaded_bars:
            assert bar["timestamp"].tzinfo is not None

    def test_load_nonexistent_raises_error(self, temp_trading_dir: Path) -> None:
        """Loading a nonexistent dataset should raise StorageError."""
        with pytest.raises(StorageError, match="not found"):
            load_dataset("nonexistent-dataset")


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_list_empty_when_no_datasets(self, temp_trading_dir: Path) -> None:
        """Listing datasets should return empty list when none exist."""
        datasets = list_datasets()

        assert datasets == []

    def test_list_returns_stored_datasets(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Listing datasets should return all stored datasets."""
        store_dataset(sample_bars, "dataset-1", sample_metadata)
        store_dataset(sample_bars, "dataset-2", sample_metadata)

        datasets = list_datasets()

        assert len(datasets) == 2
        assert "dataset-1" in datasets
        assert "dataset-2" in datasets

    def test_list_ignores_incomplete_datasets(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Listing should ignore directories without both parquet and metadata."""
        store_dataset(sample_bars, "complete-dataset", sample_metadata)

        # Create incomplete dataset (only directory)
        incomplete_dir = temp_trading_dir / "datasets" / "incomplete-dataset"
        incomplete_dir.mkdir()

        datasets = list_datasets()

        assert datasets == ["complete-dataset"]


class TestDatasetExists:
    """Tests for dataset_exists function."""

    def test_exists_returns_false_when_not_found(self, temp_trading_dir: Path) -> None:
        """dataset_exists should return False for nonexistent datasets."""
        assert dataset_exists("nonexistent") is False

    def test_exists_returns_true_when_found(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """dataset_exists should return True for existing datasets."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        assert dataset_exists("test-dataset") is True


class TestReadDatasetMetadata:
    """Tests for read_dataset_metadata function."""

    def test_read_returns_metadata_json(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Reading metadata should return the stored JSON."""
        store_dataset(sample_bars, "test-dataset", sample_metadata)

        metadata_json = read_dataset_metadata("test-dataset")
        metadata = json.loads(metadata_json)

        assert metadata["dataset_id"] == "test-dataset"
        assert metadata["bar_count"] == 3

    def test_read_nonexistent_raises_error(self, temp_trading_dir: Path) -> None:
        """Reading metadata for nonexistent dataset should raise StorageError."""
        with pytest.raises(StorageError, match="not found"):
            read_dataset_metadata("nonexistent-dataset")


class TestRoundTrip:
    """Tests for complete store/load round-trip."""

    def test_roundtrip_preserves_multiple_symbols(
        self, temp_trading_dir: Path, sample_bars: list, sample_metadata: str
    ) -> None:
        """Round-trip should preserve data for multiple symbols."""
        store_dataset(sample_bars, "multi-symbol", sample_metadata)

        loaded_bars = load_dataset("multi-symbol")

        symbols = {b["symbol"] for b in loaded_bars}
        assert symbols == {"AAPL", "GOOGL"}

    def test_roundtrip_with_large_dataset(
        self, temp_trading_dir: Path, sample_metadata: str
    ) -> None:
        """Round-trip should work with larger datasets."""
        from datetime import timedelta

        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        large_bars = [
            {
                "symbol": "AAPL",
                "timestamp": base_time + timedelta(minutes=i * 5),
                "open": 150.0 + i,
                "high": 155.0 + i,
                "low": 148.0 + i,
                "close": 153.0 + i,
                "volume": 1000.0 + i * 100,
            }
            for i in range(100)
        ]

        store_dataset(large_bars, "large-dataset", sample_metadata)
        loaded_bars = load_dataset("large-dataset")

        assert len(loaded_bars) == 100
