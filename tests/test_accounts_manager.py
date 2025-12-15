"""Tests for Rust account management functions."""

from datetime import datetime, timedelta, timezone

import pytest

from trading._core import (AccountError, calculate_account_equity,
                           is_business_day, next_business_day,
                           process_clearing, release_reservation,
                           reserve_funds)
from trading.types import Account, PendingTransaction, Position, Symbol


def _account_dict() -> dict:
    """Create a test account as a dictionary for Rust functions."""
    base_account = Account(
        account_id="acct-1",
        base_currency="USD",
        cleared_balance=1000.0,
        pending_balance=200.0,
        positions={
            "ABC": Position(
                symbol=Symbol("ABC"),
                quantity=5.0,
                cost_basis=10.0,
                pending_quantity=5.0,
            )
        },
        clearing_delay_hours=0,
        use_business_days=False,
        pending_transactions=[
            PendingTransaction(
                transaction_id="txn-1",
                symbol=Symbol("ABC"),
                quantity=5.0,
                amount=-500.0,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
                side="buy",
            )
        ],
    )
    return base_account.model_dump()


def test_calculate_equity_with_mark_to_market() -> None:
    """Equity should include cleared balance, pending balance, and position value."""
    account = _account_dict()
    current_prices = {"ABC": 12.0}

    equity = calculate_account_equity(account, current_prices)

    assert equity == pytest.approx(1000.0 + 200.0 + 5.0 * 12.0)


def test_process_clearing_moves_pending_transaction() -> None:
    """Processing clearing should settle pending transactions."""
    account = _account_dict()
    current_timestamp = datetime.now(timezone.utc)

    process_clearing(account, current_timestamp)

    assert account["pending_transactions"] == []
    assert account["cleared_balance"] == pytest.approx(1000.0 - 500.0)
    assert account["pending_balance"] == pytest.approx(200.0 + 500.0)
    position = account["positions"]["ABC"]
    assert position["quantity"] == pytest.approx(10.0)
    assert position["pending_quantity"] == pytest.approx(0.0)


def test_reserve_and_release_funds_updates_reserved_balance() -> None:
    """Reserve and release should update reserved_balance correctly."""
    account = _account_dict()

    reserve_funds(account, 100.0)
    assert account["reserved_balance"] == pytest.approx(100.0)

    release_reservation(account, 50.0)
    assert account["reserved_balance"] == pytest.approx(50.0)


def test_reserve_funds_fails_when_insufficient_available() -> None:
    """Reserving more than available should raise AccountError."""
    account = _account_dict()

    with pytest.raises(AccountError):
        reserve_funds(account, 2_000.0)


def test_business_day_helpers() -> None:
    """Business day helpers should correctly identify weekdays."""
    saturday = datetime(2025, 1, 4, tzinfo=timezone.utc)
    monday = datetime(2025, 1, 6, tzinfo=timezone.utc)

    assert not is_business_day(saturday)
    assert is_business_day(monday)

    next_business = next_business_day(saturday, 1)
    assert next_business.date() == monday.date()
