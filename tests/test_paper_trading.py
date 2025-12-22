"""Tests for paper trading functionality."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading._core import (
    append_paper_order,
    delete_paper_account,
    list_paper_accounts,
    load_paper_account,
    load_paper_orders,
    paper_account_exists,
    save_paper_account,
)
from trading.paper.engine import PaperTradingConfig, PaperTradingEngine
from trading.paper.quotes import MockQuoteSource
from trading.strategies import BuyAndHoldStrategy
from trading.types import Account, NormalizedBar, Symbol


@pytest.fixture
def mock_home(tmp_path: Path):
    """Mock the home directory to a temporary path."""
    paper_dir = tmp_path / ".trading" / "paper"
    paper_dir.mkdir(parents=True)

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if old_home:
            os.environ["HOME"] = old_home
        else:
            del os.environ["HOME"]


class TestPaperAccountPersistence:
    """Tests for Rust paper account functions."""

    def test_save_and_load_account(self, mock_home: Path) -> None:
        """Save and load account roundtrip."""
        account = Account(
            account_id="test_account",
            base_currency="USD",
            cleared_balance=15000.0,
            pending_balance=500.0,
            reserved_balance=0.0,
            positions={},
        )

        save_paper_account("test_account", account.model_dump_json())

        loaded_json = load_paper_account("test_account")
        assert loaded_json is not None

        loaded = Account.model_validate_json(loaded_json)
        assert loaded.account_id == "test_account"
        assert loaded.cleared_balance == 15000.0
        assert loaded.pending_balance == 500.0

    def test_account_exists(self, mock_home: Path) -> None:
        """paper_account_exists returns correct status."""
        assert paper_account_exists("nonexistent") is False

        save_paper_account("exists", json.dumps({"test": True}))
        assert paper_account_exists("exists") is True

    def test_list_accounts(self, mock_home: Path) -> None:
        """list_paper_accounts returns all accounts."""
        save_paper_account("account_a", json.dumps({"id": "a"}))
        save_paper_account("account_b", json.dumps({"id": "b"}))
        save_paper_account("account_c", json.dumps({"id": "c"}))

        accounts = list_paper_accounts()
        assert len(accounts) == 3
        assert "account_a" in accounts
        assert "account_b" in accounts
        assert "account_c" in accounts

    def test_delete_account(self, mock_home: Path) -> None:
        """delete_paper_account removes the account."""
        save_paper_account("to_delete", json.dumps({"test": True}))
        assert paper_account_exists("to_delete") is True

        delete_paper_account("to_delete")
        assert paper_account_exists("to_delete") is False

    def test_load_nonexistent_returns_none(self, mock_home: Path) -> None:
        """load_paper_account returns None for missing account."""
        result = load_paper_account("does_not_exist")
        assert result is None


class TestPaperOrderLog:
    """Tests for order logging."""

    def test_append_and_load_orders(self, mock_home: Path) -> None:
        """Orders can be appended and loaded."""
        order1 = {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}
        order2 = {"symbol": "AAPL", "side": "sell", "quantity": 5, "price": 155.0}

        append_paper_order("test_orders", json.dumps(order1))
        append_paper_order("test_orders", json.dumps(order2))

        orders = load_paper_orders("test_orders")
        assert len(orders) == 2

        loaded1 = json.loads(orders[0])
        assert loaded1["symbol"] == "AAPL"
        assert loaded1["side"] == "buy"

        loaded2 = json.loads(orders[1])
        assert loaded2["side"] == "sell"

    def test_load_empty_orders(self, mock_home: Path) -> None:
        """load_paper_orders returns empty list for no orders."""
        orders = load_paper_orders("no_orders")
        assert orders == []


class TestPaperTradingEngine:
    """Tests for PaperTradingEngine."""

    @pytest.fixture
    def sample_quotes(self) -> dict[str, NormalizedBar]:
        """Create sample quotes."""
        now = datetime.now(timezone.utc)
        return {
            "AAPL": NormalizedBar(
                symbol=Symbol("AAPL"),
                timestamp=now,
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000.0,
            ),
        }

    def test_creates_new_account(self, mock_home: Path, sample_quotes: dict) -> None:
        """Engine creates a new account if none exists."""
        config = PaperTradingConfig(
            account_id="new_account",
            symbols=["AAPL"],
            initial_balance=25000.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=True)

        engine = PaperTradingEngine(config, strategy, quote_source)

        assert engine.account.cleared_balance == 25000.0
        assert engine.account.account_id == "new_account"

    def test_loads_existing_account(self, mock_home: Path, sample_quotes: dict) -> None:
        """Engine loads existing account."""
        # Create an account first
        existing = Account(
            account_id="existing",
            base_currency="USD",
            cleared_balance=50000.0,
            pending_balance=0.0,
            reserved_balance=0.0,
            positions={},
        )
        save_paper_account("existing", existing.model_dump_json())

        config = PaperTradingConfig(
            account_id="existing",
            symbols=["AAPL"],
            initial_balance=10000.0,  # Should be ignored
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=True)

        engine = PaperTradingEngine(config, strategy, quote_source)

        # Should load existing balance, not initial
        assert engine.account.cleared_balance == 50000.0

    def test_tick_executes_strategy(self, mock_home: Path, sample_quotes: dict) -> None:
        """Tick executes strategy and creates orders."""
        config = PaperTradingConfig(
            account_id="tick_test",
            symbols=["AAPL"],
            initial_balance=10000.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=True)

        engine = PaperTradingEngine(config, strategy, quote_source)
        executions = engine.run_once()

        # BuyAndHold should buy on first tick
        assert len(executions) == 1
        assert executions[0].symbol == Symbol("AAPL")
        assert executions[0].side == "buy"
        assert executions[0].quantity == 5

    def test_tick_respects_market_hours(
        self, mock_home: Path, sample_quotes: dict
    ) -> None:
        """Tick skips execution when market is closed."""
        config = PaperTradingConfig(
            account_id="hours_test",
            symbols=["AAPL"],
            only_market_hours=True,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=False)

        engine = PaperTradingEngine(config, strategy, quote_source)
        engine.strategy.on_start()
        executions = engine.tick()
        engine.strategy.on_end()

        # No executions when market closed
        assert len(executions) == 0

    def test_account_persisted_after_tick(
        self, mock_home: Path, sample_quotes: dict
    ) -> None:
        """Account is saved after each tick."""
        config = PaperTradingConfig(
            account_id="persist_test",
            symbols=["AAPL"],
            initial_balance=10000.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=True)

        engine = PaperTradingEngine(config, strategy, quote_source)
        engine.run_once()

        # Load from disk
        loaded_json = load_paper_account("persist_test")
        assert loaded_json is not None
        loaded = Account.model_validate_json(loaded_json)

        # Should have position after buy
        assert "AAPL" in loaded.positions
        assert loaded.positions["AAPL"].quantity == 5

    def test_orders_logged(self, mock_home: Path, sample_quotes: dict) -> None:
        """Orders are logged to order history."""
        config = PaperTradingConfig(
            account_id="log_test",
            symbols=["AAPL"],
            initial_balance=10000.0,
        )
        strategy = BuyAndHoldStrategy({"symbol": "AAPL", "quantity": 5})
        quote_source = MockQuoteSource(sample_quotes, market_open=True)

        engine = PaperTradingEngine(config, strategy, quote_source)
        engine.run_once()

        # Check order log
        orders = load_paper_orders("log_test")
        assert len(orders) == 1

        order = json.loads(orders[0])
        assert order["symbol"] == "AAPL"
        assert order["side"] == "buy"
        assert order["quantity"] == 5


class TestMockQuoteSource:
    """Tests for MockQuoteSource."""

    def test_returns_configured_quotes(self) -> None:
        """Returns pre-configured quotes."""
        now = datetime.now(timezone.utc)
        quotes = {
            "AAPL": NormalizedBar(
                symbol=Symbol("AAPL"),
                timestamp=now,
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=1000.0,
            ),
        }

        source = MockQuoteSource(quotes)
        result = source.get_quotes([Symbol("AAPL")])

        assert "AAPL" in result
        assert result["AAPL"].close == 102.0

    def test_market_open_configurable(self) -> None:
        """Market status is configurable."""
        source = MockQuoteSource(market_open=True)
        assert source.is_market_open() is True

        source.set_market_open(False)
        assert source.is_market_open() is False


