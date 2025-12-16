"""Tests for command configuration loaders."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from trading.commands.fetch_data import (
    _generate_dataset_id,
    _parse_datetime,
    load_fetch_data_config,
)
from trading.commands.gen_synth import load_gen_synth_config
from trading.commands.run_training import (
    _validate_strategy_class_path,
    load_training_config,
)
from trading.exceptions import ConfigError
from trading.types import DatasetId, DateRange, Symbol


class TestParseDatetime:
    """Tests for datetime parsing utility."""

    def test_parse_iso_format(self) -> None:
        """Parse ISO format datetime string."""
        result = _parse_datetime("2024-01-15T10:30:00Z")
        assert result == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_parse_simple_date(self) -> None:
        """Parse simple YYYY-MM-DD format."""
        result = _parse_datetime("2024-01-15")
        assert result == datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_datetime_object(self) -> None:
        """Pass through datetime objects."""
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = _parse_datetime(dt)
        assert result == dt

    def test_parse_naive_datetime_adds_utc(self) -> None:
        """Naive datetime gets UTC timezone."""
        dt = datetime(2024, 1, 15)
        result = _parse_datetime(dt)
        assert result.tzinfo == timezone.utc

    def test_parse_invalid_format_raises(self) -> None:
        """Invalid format raises ConfigError."""
        with pytest.raises(ConfigError, match="Invalid datetime format"):
            _parse_datetime("not-a-date")


class TestGenerateDatasetId:
    """Tests for dataset ID generation."""

    def test_basic_generation(self) -> None:
        """Generate ID from basic parameters."""
        symbols = [Symbol("SPY"), Symbol("QQQ")]
        date_range = DateRange(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        result = _generate_dataset_id(symbols, date_range, "1d", "yahoo")
        assert result.startswith("yahoo_")
        assert "1d" in result
        assert "20230101" in result
        assert "20240101" in result

    def test_many_symbols_uses_hash(self) -> None:
        """Many symbols uses hash instead of full names."""
        symbols = [Symbol(f"SYM{i}") for i in range(20)]
        date_range = DateRange(
            start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        result = _generate_dataset_id(symbols, date_range, "1d", "yahoo")
        assert "20syms_" in result


class TestLoadFetchDataConfig:
    """Tests for fetch-data config loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Load a valid configuration file."""
        config = {
            "symbols": ["SPY", "QQQ"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "data_source": "yahoo",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_fetch_data_config(config_file)

        assert result.symbols == [Symbol("SPY"), Symbol("QQQ")]
        assert result.granularity == "1d"
        assert result.data_source == "yahoo"
        assert result.dataset_id is not None

    def test_load_with_explicit_dataset_id(self, tmp_path: Path) -> None:
        """Config with explicit dataset_id uses it."""
        config = {
            "symbols": ["SPY"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "data_source": "yahoo",
            "dataset_id": "my_custom_id",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_fetch_data_config(config_file)
        assert result.dataset_id == "my_custom_id"

    def test_load_with_source_params(self, tmp_path: Path) -> None:
        """Config with source_params parses them."""
        config = {
            "symbols": ["SPY"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "data_source": "csv",
            "source_params": {"file_path": "/path/to/data.csv"},
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_fetch_data_config(config_file)
        assert result.source_params == {"file_path": "/path/to/data.csv"}

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """Missing required field raises ConfigError."""
        config = {
            "symbols": ["SPY"],
            # Missing date_range, granularity, data_source
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Missing required field"):
            load_fetch_data_config(config_file)

    def test_invalid_granularity_raises(self, tmp_path: Path) -> None:
        """Invalid granularity raises ConfigError."""
        config = {
            "symbols": ["SPY"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "invalid",
            "data_source": "yahoo",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Invalid granularity"):
            load_fetch_data_config(config_file)

    def test_invalid_data_source_raises(self, tmp_path: Path) -> None:
        """Invalid data_source raises ConfigError."""
        config = {
            "symbols": ["SPY"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "data_source": "unknown",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Invalid data_source"):
            load_fetch_data_config(config_file)

    def test_start_after_end_raises(self, tmp_path: Path) -> None:
        """Start date after end date raises ConfigError."""
        config = {
            "symbols": ["SPY"],
            "date_range": {"start": "2024-01-01", "end": "2023-01-01"},
            "granularity": "1d",
            "data_source": "yahoo",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="must be before"):
            load_fetch_data_config(config_file)

    def test_file_not_found_raises(self) -> None:
        """Non-existent file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_fetch_data_config("/nonexistent/path.yaml")

    def test_empty_symbols_raises(self, tmp_path: Path) -> None:
        """Empty symbols list raises ConfigError."""
        config = {
            "symbols": [],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "data_source": "yahoo",
        }
        config_file = tmp_path / "fetch.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="non-empty list"):
            load_fetch_data_config(config_file)


class TestLoadGenSynthConfig:
    """Tests for gen-synth config loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Load a valid gen-synth configuration."""
        config = {
            "symbols": ["SYNTH_A", "SYNTH_B"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "generator_type": "geometric_brownian",
            "generator_params": {
                "initial_price": 100.0,
                "drift": 0.0001,
                "volatility": 0.02,
            },
            "random_seed": 42,
        }
        config_file = tmp_path / "synth.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_gen_synth_config(config_file)

        assert result.symbols == [Symbol("SYNTH_A"), Symbol("SYNTH_B")]
        assert result.generator_type == "geometric_brownian"
        assert result.random_seed == 42
        assert result.generator_params["initial_price"] == 100.0

    def test_load_without_seed(self, tmp_path: Path) -> None:
        """Config without random_seed is valid."""
        config = {
            "symbols": ["SYNTH"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "generator_type": "mean_reverting",
        }
        config_file = tmp_path / "synth.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_gen_synth_config(config_file)
        assert result.random_seed is None

    def test_invalid_generator_type_raises(self, tmp_path: Path) -> None:
        """Invalid generator_type raises ConfigError."""
        config = {
            "symbols": ["SYNTH"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "generator_type": "unknown_generator",
        }
        config_file = tmp_path / "synth.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Invalid generator_type"):
            load_gen_synth_config(config_file)

    def test_invalid_generator_params_raises(self, tmp_path: Path) -> None:
        """Invalid generator params raise ConfigError."""
        config = {
            "symbols": ["SYNTH"],
            "date_range": {"start": "2023-01-01", "end": "2024-01-01"},
            "granularity": "1d",
            "generator_type": "geometric_brownian",
            "generator_params": {"initial_price": -100.0},  # Negative price
        }
        config_file = tmp_path / "synth.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="initial_price"):
            load_gen_synth_config(config_file)


class TestValidateStrategyClassPath:
    """Tests for strategy class path validation."""

    def test_valid_strategy_class(self) -> None:
        """Valid strategy class passes validation."""
        # This should not raise
        _validate_strategy_class_path("trading.strategies.examples.BuyAndHoldStrategy")

    def test_invalid_module_raises(self) -> None:
        """Non-existent module raises ConfigError."""
        with pytest.raises(ConfigError, match="Cannot import"):
            _validate_strategy_class_path("nonexistent.module.Strategy")

    def test_invalid_class_raises(self) -> None:
        """Non-existent class raises ConfigError."""
        with pytest.raises(ConfigError, match="has no attribute"):
            _validate_strategy_class_path("trading.strategies.examples.NonExistentStrategy")

    def test_non_strategy_raises(self) -> None:
        """Class that isn't a Strategy raises ConfigError."""
        with pytest.raises(ConfigError, match="not a Strategy subclass"):
            _validate_strategy_class_path("trading.types.Symbol")

    def test_no_dot_raises(self) -> None:
        """Class path without dot raises ConfigError."""
        with pytest.raises(ConfigError, match="Invalid strategy class_path"):
            _validate_strategy_class_path("InvalidPath")


class TestLoadTrainingConfig:
    """Tests for training config loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Load a valid training configuration."""
        config = {
            "datasets": ["dataset_1", "dataset_2"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
                "params": {"symbol": "SPY", "quantity": 10},
            },
            "account": {
                "starting_balance": 10000.0,
                "base_currency": "USD",
                "clearing_delay_hours": 24,
            },
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)

        assert result.datasets == [DatasetId("dataset_1"), DatasetId("dataset_2")]
        assert result.strategy_class_path == "trading.strategies.examples.BuyAndHoldStrategy"
        assert result.strategy_params == {"symbol": "SPY", "quantity": 10}
        assert result.account_starting_balance == 10000.0
        assert result.run_id is not None

    def test_load_with_explicit_run_id(self, tmp_path: Path) -> None:
        """Config with explicit run_id uses it."""
        config = {
            "run_id": "my-custom-run",
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)
        assert result.run_id == "my-custom-run"

    def test_load_with_risk_config(self, tmp_path: Path) -> None:
        """Config with risk settings parses them."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
            "risk": {"max_position_size": 5000.0, "max_leverage": 2.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)
        assert result.risk_max_position_size == 5000.0
        assert result.risk_max_leverage == 2.0

    def test_load_with_universe_config(self, tmp_path: Path) -> None:
        """Config with universe settings parses them."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
            "universe": {"analysis": ["SPY", "QQQ"], "tradable": ["SPY"]},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)
        assert result.analysis_universe == [Symbol("SPY"), Symbol("QQQ")]
        assert result.tradable_universe == [Symbol("SPY")]

    def test_load_with_checkpoint_config(self, tmp_path: Path) -> None:
        """Config with checkpoint settings parses them."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
            "checkpoint": {"interval": 100},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)
        assert result.checkpoint_interval == 100

    def test_missing_datasets_raises(self, tmp_path: Path) -> None:
        """Missing datasets raises ConfigError."""
        config = {
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Missing required field: datasets"):
            load_training_config(config_file)

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        """Invalid strategy class path raises ConfigError."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "nonexistent.Strategy",
            },
            "account": {"starting_balance": 10000.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="Cannot import"):
            load_training_config(config_file)

    def test_negative_balance_raises(self, tmp_path: Path) -> None:
        """Negative starting balance raises ConfigError."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": -1000.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="must be positive"):
            load_training_config(config_file)

    def test_defaults_applied(self, tmp_path: Path) -> None:
        """Default values are applied for optional fields."""
        config = {
            "datasets": ["dataset_1"],
            "strategy": {
                "class_path": "trading.strategies.examples.BuyAndHoldStrategy",
            },
            "account": {"starting_balance": 10000.0},
        }
        config_file = tmp_path / "training.yaml"
        config_file.write_text(yaml.dump(config))

        result = load_training_config(config_file)

        # Check defaults
        assert result.account_base_currency == "USD"
        assert result.clearing_delay_hours == 24
        assert result.use_business_days is False
        assert result.risk_max_leverage == 1.0
        assert result.log_level == "INFO"
        assert result.enable_event_logging is True
        assert result.checkpoint_interval is None

