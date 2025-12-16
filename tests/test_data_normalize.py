"""Tests for Rust data normalization functions."""

from datetime import datetime, timedelta, timezone

import pytest

from trading._core import (DataValidationError, detect_data_gaps,
                           fill_data_gaps, normalize_bars, validate_bars)


class TestNormalizeBars:
    """Tests for normalize_bars function."""

    def test_normalize_valid_bars(self) -> None:
        """Valid bars should be normalized without changes."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = normalize_bars(bars)

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["open"] == 150.0
        assert result[0]["high"] == 155.0
        assert result[0]["low"] == 148.0
        assert result[0]["close"] == 153.0
        assert result[0]["volume"] == 1000.0

    def test_normalize_adds_utc_to_naive_timestamp(self) -> None:
        """Naive timestamps should get UTC timezone added."""
        ts_naive = datetime(2024, 1, 1, 10, 0)  # No timezone
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts_naive,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = normalize_bars(bars)

        assert len(result) == 1
        # Timestamp should now have timezone
        assert result[0]["timestamp"].tzinfo is not None

    def test_normalize_preserves_existing_timezone(self) -> None:
        """Bars with existing timezone should keep it."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = normalize_bars(bars)

        assert result[0]["timestamp"].tzinfo == timezone.utc

    def test_normalize_multiple_bars(self) -> None:
        """Multiple bars should all be normalized."""
        ts1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts1,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {
                "symbol": "AAPL",
                "timestamp": ts2,
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        result = normalize_bars(bars)

        assert len(result) == 2
        assert result[0]["close"] == 153.0
        assert result[1]["close"] == 155.0

    def test_normalize_rejects_negative_price(self) -> None:
        """Negative prices should raise DataValidationError."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": -150.0,  # Negative!
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        with pytest.raises(DataValidationError):
            normalize_bars(bars)

    def test_normalize_rejects_zero_price(self) -> None:
        """Zero prices should raise DataValidationError."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 0.0,  # Zero!
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        with pytest.raises(DataValidationError):
            normalize_bars(bars)

    def test_normalize_rejects_negative_volume(self) -> None:
        """Negative volume should raise DataValidationError."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": -1000.0,  # Negative!
            }
        ]

        with pytest.raises(DataValidationError):
            normalize_bars(bars)

    def test_normalize_allows_zero_volume(self) -> None:
        """Zero volume should be allowed."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 0.0,  # Zero volume is OK
            }
        ]

        result = normalize_bars(bars)

        assert len(result) == 1
        assert result[0]["volume"] == 0.0

    def test_normalize_rejects_missing_field(self) -> None:
        """Missing required fields should raise DataValidationError."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                # Missing high, low, close, volume
            }
        ]

        with pytest.raises(DataValidationError):
            normalize_bars(bars)


class TestValidateBars:
    """Tests for validate_bars function."""

    def test_validate_valid_bars(self) -> None:
        """Valid bars should pass validation."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 1

    def test_validate_skips_invalid_ohlc_high_lt_low(self) -> None:
        """Bars with high < low should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 145.0,  # high < low!
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0  # Bar should be skipped

    def test_validate_skips_invalid_ohlc_high_lt_open(self) -> None:
        """Bars with high < open should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 160.0,  # open > high!
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0  # Bar should be skipped

    def test_validate_skips_invalid_ohlc_low_gt_close(self) -> None:
        """Bars with low > close should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 154.0,  # low > close!
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0  # Bar should be skipped

    def test_validate_skips_nan_values(self) -> None:
        """Bars with NaN values should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": float("nan"),  # NaN!
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0  # Bar should be skipped

    def test_validate_skips_inf_values(self) -> None:
        """Bars with infinity values should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": float("inf"),  # Infinity!
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0  # Bar should be skipped

    def test_validate_skips_out_of_order_timestamps(self) -> None:
        """Bars with out-of-order timestamps for same symbol should be skipped."""
        ts1 = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)  # Earlier!
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts1,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {
                "symbol": "AAPL",
                "timestamp": ts2,  # Out of order!
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        result = validate_bars(bars)

        assert len(result) == 1  # Second bar should be skipped
        assert result[0]["timestamp"] == ts1

    def test_validate_allows_different_symbols_same_timestamp(self) -> None:
        """Different symbols can have the same timestamp."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {
                "symbol": "GOOGL",
                "timestamp": ts,  # Same timestamp, different symbol
                "open": 2800.0,
                "high": 2850.0,
                "low": 2780.0,
                "close": 2830.0,
                "volume": 500.0,
            },
        ]

        result = validate_bars(bars)

        assert len(result) == 2  # Both bars should be valid

    def test_validate_mixed_valid_and_invalid(self) -> None:
        """Only valid bars should be returned from mixed input."""
        ts1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 10, 5, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 1, 10, 10, tzinfo=timezone.utc)
        bars = [
            {  # Valid
                "symbol": "AAPL",
                "timestamp": ts1,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {  # Invalid: high < low
                "symbol": "AAPL",
                "timestamp": ts2,
                "open": 150.0,
                "high": 145.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            },
            {  # Valid
                "symbol": "AAPL",
                "timestamp": ts3,
                "open": 153.0,
                "high": 158.0,
                "low": 152.0,
                "close": 156.0,
                "volume": 1100.0,
            },
        ]

        result = validate_bars(bars)

        assert len(result) == 2  # Middle bar should be skipped
        assert result[0]["timestamp"] == ts1
        assert result[1]["timestamp"] == ts3

    def test_validate_skips_bars_missing_symbol(self) -> None:
        """Bars without symbol should be skipped."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                # No symbol!
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        result = validate_bars(bars)

        assert len(result) == 0

    def test_validate_empty_input(self) -> None:
        """Empty input should return empty output."""
        result = validate_bars([])

        assert len(result) == 0


class TestDetectDataGaps:
    """Tests for detect_data_gaps function."""

    def test_no_gaps_in_continuous_data(self) -> None:
        """Continuous data should have no gaps."""
        # 5-minute bars with no gaps
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": base_time + timedelta(minutes=i * 5),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
            for i in range(5)
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)  # 300 seconds = 5 minutes

        assert len(gaps) == 0

    def test_detects_single_gap(self) -> None:
        """Should detect a gap between bars."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
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
                "timestamp": base_time + timedelta(minutes=20),  # 20 min gap
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)  # 300 seconds = 5 minutes

        assert len(gaps) == 1
        assert gaps[0]["symbol"] == "AAPL"
        assert gaps[0]["expected_bars"] >= 1

    def test_detects_multiple_gaps(self) -> None:
        """Should detect multiple gaps in data."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
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
                "timestamp": base_time + timedelta(minutes=15),  # 15 min gap
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
            {
                "symbol": "AAPL",
                "timestamp": base_time + timedelta(minutes=35),  # Another 20 min gap
                "open": 155.0,
                "high": 158.0,
                "low": 154.0,
                "close": 157.0,
                "volume": 1100.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)

        assert len(gaps) == 2

    def test_separate_gaps_per_symbol(self) -> None:
        """Gaps should be tracked separately per symbol."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            # AAPL - continuous
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
                "timestamp": base_time + timedelta(minutes=5),
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
            # GOOGL - has gap
            {
                "symbol": "GOOGL",
                "timestamp": base_time,
                "open": 2800.0,
                "high": 2850.0,
                "low": 2780.0,
                "close": 2830.0,
                "volume": 500.0,
            },
            {
                "symbol": "GOOGL",
                "timestamp": base_time + timedelta(minutes=20),  # Gap
                "open": 2830.0,
                "high": 2870.0,
                "low": 2820.0,
                "close": 2860.0,
                "volume": 600.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)

        assert len(gaps) == 1
        assert gaps[0]["symbol"] == "GOOGL"

    def test_empty_bars_returns_no_gaps(self) -> None:
        """Empty input should return no gaps."""
        gaps = detect_data_gaps([], 300, 1.5)

        assert len(gaps) == 0

    def test_single_bar_returns_no_gaps(self) -> None:
        """Single bar should return no gaps."""
        ts = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": ts,
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)

        assert len(gaps) == 0


class TestFillDataGaps:
    """Tests for fill_data_gaps function."""

    def test_fills_gap_with_forward_fill(self) -> None:
        """Gaps should be filled with forward-filled prices."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
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
                "timestamp": base_time + timedelta(minutes=15),
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)
        filled = fill_data_gaps(bars, gaps, 300)

        # Should have original 2 bars + fill bars
        assert len(filled) > 2
        # Original bars should still be present
        assert any(b["timestamp"] == base_time for b in filled)

    def test_filled_bars_have_zero_volume(self) -> None:
        """Filled bars should have volume = 0."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
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
                "timestamp": base_time + timedelta(minutes=15),
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)
        filled = fill_data_gaps(bars, gaps, 300)

        # Find filled bars (volume = 0)
        fill_bars = [b for b in filled if b["volume"] == 0.0]
        assert len(fill_bars) > 0

    def test_filled_bars_use_last_close_price(self) -> None:
        """Filled bars should use the last close price."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
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
                "timestamp": base_time + timedelta(minutes=15),
                "open": 153.0,
                "high": 156.0,
                "low": 152.0,
                "close": 155.0,
                "volume": 1200.0,
            },
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)
        filled = fill_data_gaps(bars, gaps, 300)

        # Find filled bars
        fill_bars = [b for b in filled if b["volume"] == 0.0]
        for fb in fill_bars:
            # Forward fill uses 153.0 (close of last bar before gap)
            assert fb["open"] == fb["high"] == fb["low"] == fb["close"]

    def test_no_gaps_returns_original_bars(self) -> None:
        """When no gaps, should return original bars unchanged."""
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            {
                "symbol": "AAPL",
                "timestamp": base_time + timedelta(minutes=i * 5),
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 1000.0,
            }
            for i in range(3)
        ]

        gaps = detect_data_gaps(bars, 300, 1.5)
        filled = fill_data_gaps(bars, gaps, 300)

        assert len(filled) == len(bars)

    def test_empty_bars_with_empty_gaps(self) -> None:
        """Empty input should return empty output."""
        filled = fill_data_gaps([], [], 300)

        assert len(filled) == 0
