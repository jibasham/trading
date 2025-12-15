"""Tests for trading exception hierarchy."""

import pytest

from trading.exceptions import (AccountError, ConfigError, DataSourceError,
                                DataValidationError, StorageError,
                                StrategyError, TradingError)


def test_trading_error_is_base_exception() -> None:
    """TradingError should be catchable as Exception."""
    with pytest.raises(Exception):
        raise TradingError("test error")


def test_config_error_inherits_from_trading_error() -> None:
    """ConfigError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise ConfigError("invalid config")


def test_data_source_error_inherits_from_trading_error() -> None:
    """DataSourceError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise DataSourceError("data source failed")


def test_data_validation_error_inherits_from_trading_error() -> None:
    """DataValidationError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise DataValidationError("validation failed")


def test_storage_error_inherits_from_trading_error() -> None:
    """StorageError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise StorageError("storage failed")


def test_strategy_error_inherits_from_trading_error() -> None:
    """StrategyError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise StrategyError("strategy failed")


def test_account_error_inherits_from_trading_error() -> None:
    """AccountError should be catchable as TradingError."""
    with pytest.raises(TradingError):
        raise AccountError("account error")


def test_exception_messages_preserved() -> None:
    """Exception messages should be accessible via str()."""
    msg = "detailed error message"
    err = ConfigError(msg)
    assert str(err) == msg

