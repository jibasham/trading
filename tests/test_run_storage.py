"""Tests for run storage functions."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from trading._core import (
    StorageError,
    delete_run,
    list_runs,
    load_run_results,
    run_exists,
    store_run_results,
)


@pytest.fixture
def mock_runs_dir(tmp_path: Path):
    """Mock the runs directory to a temporary path."""
    runs_dir = tmp_path / ".trading" / "runs"
    runs_dir.mkdir(parents=True)

    # Patch the home directory
    with patch("os.path.expanduser", return_value=str(tmp_path)):
        # Also need to patch dirs::home_dir in Rust, but since we can't,
        # we'll work around by using environment variables
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(tmp_path)
        try:
            yield runs_dir
        finally:
            if old_home:
                os.environ["HOME"] = old_home
            else:
                del os.environ["HOME"]


@pytest.fixture
def sample_run_data():
    """Sample data for a run."""
    config = {
        "run_id": "test_run_001",
        "datasets": ["dataset_1"],
        "strategy_class_path": "trading.strategies.examples.BuyAndHoldStrategy",
        "strategy_params": {"symbol": "SPY", "quantity": 10},
        "account_starting_balance": 10000.0,
    }

    metrics = {
        "run_id": "test_run_001",
        "total_return": 0.15,
        "max_drawdown": 0.05,
        "volatility": 0.02,
        "sharpe_ratio": 1.5,
        "num_trades": 5,
        "win_rate": None,
    }

    executions = [
        {
            "symbol": "SPY",
            "side": "buy",
            "quantity": 10,
            "price": 400.0,
            "timestamp": "2023-01-15T10:30:00+00:00",
            "order_id": "order_001",
        }
    ]

    equity_history = [
        {"timestamp": "2023-01-01T00:00:00+00:00", "equity": 10000.0},
        {"timestamp": "2023-01-15T00:00:00+00:00", "equity": 11500.0},
    ]

    return {
        "config_json": json.dumps(config),
        "metrics_json": json.dumps(metrics),
        "executions_json": json.dumps(executions),
        "equity_json": json.dumps(equity_history),
        "final_equity": 11500.0,
        "num_trades": 5,
    }


class TestStoreRunResults:
    """Tests for store_run_results function."""

    def test_store_creates_directory(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """store_run_results creates the run directory."""
        run_id = "test_store_run"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        run_dir = mock_runs_dir / run_id
        assert run_dir.exists()

    def test_store_creates_config_file(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """store_run_results creates config.json."""
        run_id = "test_store_config"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        config_path = mock_runs_dir / run_id / "config.json"
        assert config_path.exists()
        content = json.loads(config_path.read_text())
        assert content["run_id"] == "test_run_001"

    def test_store_creates_metrics_file(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """store_run_results creates metrics.json."""
        run_id = "test_store_metrics"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        metrics_path = mock_runs_dir / run_id / "metrics.json"
        assert metrics_path.exists()
        content = json.loads(metrics_path.read_text())
        assert content["total_return"] == 0.15

    def test_store_creates_summary_file(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """store_run_results creates summary.json."""
        run_id = "test_store_summary"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        summary_path = mock_runs_dir / run_id / "summary.json"
        assert summary_path.exists()
        content = json.loads(summary_path.read_text())
        assert content["run_id"] == run_id
        assert content["final_equity"] == 11500.0
        assert content["num_trades"] == 5


class TestLoadRunResults:
    """Tests for load_run_results function."""

    def test_load_returns_all_components(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """load_run_results returns all stored components."""
        run_id = "test_load_all"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        result = load_run_results(run_id)

        assert "config" in result
        assert "metrics" in result
        assert "executions" in result
        assert "equity_history" in result
        assert "summary" in result

    def test_load_preserves_config_data(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """load_run_results preserves config data."""
        run_id = "test_load_config"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        result = load_run_results(run_id)

        assert result["config"]["strategy_class_path"] == "trading.strategies.examples.BuyAndHoldStrategy"
        assert result["config"]["account_starting_balance"] == 10000.0

    def test_load_nonexistent_raises(self, mock_runs_dir: Path) -> None:
        """load_run_results raises StorageError for missing run."""
        with pytest.raises(StorageError, match="Run not found"):
            load_run_results("nonexistent_run")


class TestListRuns:
    """Tests for list_runs function."""

    def test_list_empty_when_no_runs(self, mock_runs_dir: Path) -> None:
        """list_runs returns empty list when no runs exist."""
        result = list_runs()
        assert result == []

    def test_list_returns_run_ids(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """list_runs returns stored run IDs."""
        run_ids = ["run_a", "run_b", "run_c"]
        for run_id in run_ids:
            store_run_results(
                run_id,
                sample_run_data["config_json"],
                sample_run_data["metrics_json"],
                sample_run_data["executions_json"],
                sample_run_data["equity_json"],
                sample_run_data["final_equity"],
                sample_run_data["num_trades"],
            )

        result = list_runs()

        assert len(result) == 3
        for run_id in run_ids:
            assert run_id in result


class TestRunExists:
    """Tests for run_exists function."""

    def test_exists_returns_true_for_valid_run(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """run_exists returns True for existing run."""
        run_id = "existing_run"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        assert run_exists(run_id) is True

    def test_exists_returns_false_for_missing_run(self, mock_runs_dir: Path) -> None:
        """run_exists returns False for non-existent run."""
        assert run_exists("nonexistent_run") is False


class TestDeleteRun:
    """Tests for delete_run function."""

    def test_delete_removes_run(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """delete_run removes the run directory."""
        run_id = "run_to_delete"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        assert run_exists(run_id) is True

        delete_run(run_id)

        assert run_exists(run_id) is False

    def test_delete_nonexistent_raises(self, mock_runs_dir: Path) -> None:
        """delete_run raises StorageError for missing run."""
        with pytest.raises(StorageError, match="Run not found"):
            delete_run("nonexistent_run")


class TestRoundTrip:
    """Tests for store/load round trip."""

    def test_roundtrip_preserves_data(
        self, mock_runs_dir: Path, sample_run_data: dict
    ) -> None:
        """Data is preserved through store/load cycle."""
        run_id = "roundtrip_test"
        store_run_results(
            run_id,
            sample_run_data["config_json"],
            sample_run_data["metrics_json"],
            sample_run_data["executions_json"],
            sample_run_data["equity_json"],
            sample_run_data["final_equity"],
            sample_run_data["num_trades"],
        )

        result = load_run_results(run_id)

        # Verify config
        assert result["config"]["datasets"] == ["dataset_1"]

        # Verify metrics
        assert result["metrics"]["total_return"] == 0.15
        assert result["metrics"]["sharpe_ratio"] == 1.5

        # Verify executions
        assert len(result["executions"]) == 1
        assert result["executions"][0]["symbol"] == "SPY"

        # Verify equity history
        assert len(result["equity_history"]) == 2
        assert result["equity_history"][0]["equity"] == 10000.0

        # Verify summary
        assert result["summary"]["final_equity"] == 11500.0

