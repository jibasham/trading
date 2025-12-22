"""Tests for checkpointing functionality."""

import json
import os
from pathlib import Path

import pytest

from trading._core import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def mock_runs_dir(tmp_path: Path):
    """Mock the runs directory to a temporary path."""
    runs_dir = tmp_path / ".trading" / "runs"
    runs_dir.mkdir(parents=True)

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)
    try:
        yield runs_dir
    finally:
        if old_home:
            os.environ["HOME"] = old_home
        else:
            del os.environ["HOME"]


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_creates_checkpoint_file(self, mock_runs_dir: Path) -> None:
        """save_checkpoint creates the checkpoint file."""
        run_id = "test_run"
        step_index = 10
        state = {"account": {"balance": 10000}, "step": 10}

        path = save_checkpoint(run_id, step_index, json.dumps(state))

        assert Path(path).exists()
        assert "checkpoint_00000010.json" in path

    def test_creates_latest_pointer(self, mock_runs_dir: Path) -> None:
        """save_checkpoint creates the latest pointer file."""
        run_id = "test_run"

        save_checkpoint(run_id, 5, json.dumps({"step": 5}))

        latest_file = mock_runs_dir / run_id / "checkpoints" / "latest"
        assert latest_file.exists()
        assert latest_file.read_text() == "5"

    def test_updates_latest_pointer(self, mock_runs_dir: Path) -> None:
        """save_checkpoint updates the latest pointer."""
        run_id = "test_run"

        save_checkpoint(run_id, 5, json.dumps({"step": 5}))
        save_checkpoint(run_id, 10, json.dumps({"step": 10}))

        latest_file = mock_runs_dir / run_id / "checkpoints" / "latest"
        assert latest_file.read_text() == "10"

    def test_preserves_state_content(self, mock_runs_dir: Path) -> None:
        """save_checkpoint preserves the state content."""
        run_id = "test_run"
        state = {"account": {"balance": 5000}, "positions": ["AAPL"]}

        path = save_checkpoint(run_id, 15, json.dumps(state))

        content = json.loads(Path(path).read_text())
        assert content["account"]["balance"] == 5000
        assert content["positions"] == ["AAPL"]


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_returns_none_when_no_checkpoint(self, mock_runs_dir: Path) -> None:
        """load_checkpoint returns None when no checkpoint exists."""
        result = load_checkpoint("nonexistent_run")
        assert result is None

    def test_loads_latest_checkpoint(self, mock_runs_dir: Path) -> None:
        """load_checkpoint loads the latest checkpoint."""
        run_id = "test_run"
        state = {"account": {"balance": 8000}, "step": 20}

        save_checkpoint(run_id, 10, json.dumps({"step": 10}))
        save_checkpoint(run_id, 20, json.dumps(state))

        step_index, state_json = load_checkpoint(run_id)

        assert step_index == 20
        loaded_state = json.loads(state_json)
        assert loaded_state["account"]["balance"] == 8000

    def test_returns_correct_step_index(self, mock_runs_dir: Path) -> None:
        """load_checkpoint returns the correct step index."""
        run_id = "test_run"

        save_checkpoint(run_id, 42, json.dumps({"step": 42}))

        step_index, _ = load_checkpoint(run_id)
        assert step_index == 42


class TestCheckpointExists:
    """Tests for checkpoint_exists function."""

    def test_returns_false_when_no_checkpoint(self, mock_runs_dir: Path) -> None:
        """checkpoint_exists returns False when no checkpoint exists."""
        assert checkpoint_exists("nonexistent_run") is False

    def test_returns_true_when_checkpoint_exists(self, mock_runs_dir: Path) -> None:
        """checkpoint_exists returns True when checkpoint exists."""
        run_id = "test_run"
        save_checkpoint(run_id, 5, json.dumps({"step": 5}))

        assert checkpoint_exists(run_id) is True


class TestCheckpointRoundTrip:
    """Tests for checkpoint save/load round trip."""

    def test_roundtrip_preserves_complex_state(self, mock_runs_dir: Path) -> None:
        """Complex state is preserved through save/load cycle."""
        run_id = "test_run"
        state = {
            "account": {
                "balance": 12345.67,
                "positions": {"AAPL": 100, "GOOGL": 50},
            },
            "executions": [
                {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 150.0},
            ],
            "equity_history": [
                {"timestamp": "2023-01-01T00:00:00+00:00", "equity": 10000.0},
                {"timestamp": "2023-01-02T00:00:00+00:00", "equity": 10500.0},
            ],
            "trade_pnls": [500.0, -200.0, 300.0],
            "open_lots": {"AAPL": [{"qty": 50, "price": 145.0}]},
        }

        save_checkpoint(run_id, 100, json.dumps(state))
        step_index, loaded_json = load_checkpoint(run_id)
        loaded_state = json.loads(loaded_json)

        assert step_index == 100
        assert loaded_state["account"]["balance"] == 12345.67
        assert loaded_state["executions"][0]["symbol"] == "AAPL"
        assert len(loaded_state["equity_history"]) == 2
        assert loaded_state["trade_pnls"] == [500.0, -200.0, 300.0]
        assert loaded_state["open_lots"]["AAPL"][0]["qty"] == 50

