# Data Normalization Implementation Checklist

**Ticket**: Data Normalization Functions (Rust)
**Target Files**: `rust/src/lib.rs`, `tests/test_data_normalize.py`
**Reference**: `docs/cli-and-implementation.md` Section 4.3 (normalize_bars) and Section 5.3 (normalize_synth_bars)

## Overview

Implement data normalization functions in Rust for processing raw market data bars into the engine's canonical format. These functions handle timezone normalization, field validation, data cleaning, and gap detection/filling.

---

## Phase 1: Rust Implementation

### normalize_bars function
Per Section 4.3 of cli-and-implementation.md:

- [x] **[CRITICAL]** Implement `normalize_bars` function that accepts Python iterator of Bar dicts
- [x] Ensure timestamp has timezone info (default to UTC if missing)
- [x] Ensure all price fields (open, high, low, close) are positive floats
- [x] Ensure volume is non-negative float
- [x] Return iterator of NormalizedBar dicts
- [x] Add to module exports

### validate_bars function  
Per Section 4.3 of cli-and-implementation.md:

- [x] **[CRITICAL]** Implement `validate_bars` function
- [x] Validate high >= low and high >= open/close and low <= open/close
- [x] Validate timestamps are strictly increasing per symbol
- [x] Validate all numeric fields are finite (not NaN, not inf)
- [x] Skip invalid bars with warning (don't raise exceptions)
- [x] Add to module exports

### Gap detection and filling

- [x] **[CRITICAL]** Implement `detect_data_gaps` function
- [x] Track timestamps per symbol
- [x] Detect gaps larger than threshold (granularity * multiplier)
- [x] Return list of Gap dicts with symbol, start_time, end_time, expected_bars
- [x] **[CRITICAL]** Implement `fill_data_gaps` function
- [x] Forward fill prices from last known bar
- [x] Set volume to 0 for filled bars
- [x] Add both functions to module exports

### Utility functions

- [x] Implement `ensure_timezone` helper for timestamp handling
- [x] Implement `is_finite` helper for numeric validation
- [x] Implement `validate_ohlc` helper for price relationship checks
- [x] Implement `extract_positive_float` and `extract_non_negative_float` helpers

---

## Phase 2: Python Tests

### normalize_bars tests (10 tests)
- [x] Create `tests/test_data_normalize.py`
- [x] Test normalize_bars with valid data
- [x] Test normalize_bars with missing timezone (should add UTC)
- [x] Test normalize_bars with invalid prices (should handle gracefully)

### validate_bars tests (10 tests)
- [x] Test validate_bars with valid OHLC relationships
- [x] Test validate_bars with invalid OHLC (high < low) - should skip
- [x] Test validate_bars with out-of-order timestamps - should skip
- [x] Test validate_bars with NaN values - should skip

### Gap detection tests (6 tests)
- [x] Test no gaps in continuous data
- [x] Test detects single gap
- [x] Test detects multiple gaps
- [x] Test separate gaps per symbol
- [x] Test empty bars returns no gaps
- [x] Test single bar returns no gaps

### Gap filling tests (5 tests)
- [x] Test fills gap with forward fill
- [x] Test filled bars have zero volume
- [x] Test filled bars use last close price
- [x] Test no gaps returns original bars
- [x] Test empty bars with empty gaps

---

## Phase 3: Validation

- [x] Run `cargo clippy --all-targets -- -D warnings`
- [x] Run `maturin develop`
- [x] Run `pytest tests/test_data_normalize.py -v` (31 tests passed)
- [x] Run `pytest tests/ -v` (all 72 tests passed)
- [x] Run `ruff check src/ tests/` (all checks passed)
- [x] Update documentation

---

## Completion Criteria

- [x] All normalization functions implemented in Rust
- [x] All gap detection/filling functions implemented in Rust
- [x] All functions exposed via PyO3
- [x] Comprehensive Python tests pass (31 new tests)
- [x] No clippy warnings
- [x] No ruff linting errors

---

## Progress Notes

**Phase 1 Complete (2024-12-15):**
- Implemented `normalize_bars` function in Rust
  - Ensures timestamps have UTC timezone if missing
  - Validates all price fields are positive
  - Validates volume is non-negative
- Implemented `validate_bars` function in Rust
  - Validates OHLC price relationships
  - Validates timestamps are strictly increasing per symbol
  - Validates all numeric fields are finite
  - Skips invalid bars rather than raising exceptions
- Added `DataValidationError` exception to Rust module
- Added helper functions: `ensure_timezone`, `is_finite`, `validate_ohlc`, `extract_positive_float`, `extract_non_negative_float`, `extract_float_or_skip`

**Phase 1b Complete (2024-12-16):**
- Implemented `detect_data_gaps` function in Rust
  - Tracks timestamps per symbol
  - Detects gaps based on granularity and multiplier threshold
  - Returns Gap dicts compatible with Python Gap type
- Implemented `fill_data_gaps` function in Rust
  - Forward fills prices from last known close
  - Sets volume to 0 for filled bars
  - Maintains original bars in output

**Phase 2 Complete (2024-12-16):**
- Created comprehensive test suite with 31 tests
- Tests cover:
  - Valid bar normalization (10 tests)
  - OHLC validation (10 tests)
  - Gap detection (6 tests)
  - Gap filling (5 tests)

**Phase 3 Complete (2024-12-16):**
- All 72 tests pass
- No clippy warnings
- No ruff linting errors
- Documentation updated

