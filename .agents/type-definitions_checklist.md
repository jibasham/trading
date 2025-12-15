# Type Definitions Implementation Checklist

**Ticket**: Type Definitions Migration and Expansion
**Target Files**: `src/trading/types.py`, `src/trading/exceptions.py`, `pyproject.toml`, `tests/test_types.py`
**Reference**: `docs/cli-and-implementation.md` Section 2 (Core Type Definitions) and Section 3 (Exception Definitions)

## Overview

Migrate existing dataclass-based types to Pydantic BaseModel and add all missing types specified in the implementation document. Update tests accordingly.

---

## Phase 1: Dependencies and Setup

- [x] **[CRITICAL]** Add `pydantic>=2.0.0` to `pyproject.toml` dependencies
- [x] Install updated dependencies

---

## Phase 2: Exception Definitions

Per Section 3 of cli-and-implementation.md, add missing exceptions to `src/trading/exceptions.py`:

- [x] Add `ConfigError` exception class
- [x] Add `DataSourceError` exception class  
- [x] Add `DataValidationError` exception class (named to avoid conflict with pydantic's ValidationError)
- [x] Add `StorageError` exception class
- [x] Add `StrategyError` exception class
- [x] Update `__all__` export list
- [x] Add tests for new exceptions in `tests/test_exceptions.py`

---

## Phase 3: Migrate Existing Types to Pydantic

Migrate existing dataclasses to Pydantic BaseModel while maintaining backward compatibility with existing tests:

- [x] **[CRITICAL]** Migrate `DateRange` to Pydantic BaseModel
- [x] **[CRITICAL]** Migrate `Bar` to Pydantic BaseModel
- [x] **[CRITICAL]** Migrate `NormalizedBar` to Pydantic BaseModel  
- [x] **[CRITICAL]** Migrate `RunMetrics` to Pydantic BaseModel
- [x] **[CRITICAL]** Migrate `Position` to Pydantic BaseModel
- [x] **[CRITICAL]** Migrate `PendingTransaction` to Pydantic BaseModel
- [x] **[CRITICAL]** Migrate `Account` to Pydantic BaseModel
- [x] Update `tests/test_types.py` for Pydantic models
- [x] Update `tests/test_accounts_manager.py` to use `.model_dump()` instead of `asdict()`

---

## Phase 4: Add New Config Types

Per Section 2 of cli-and-implementation.md:

- [x] Add `FetchDataConfig` model
- [x] Add `GenSynthConfig` model
- [x] Add `TrainingConfig` model
- [x] Add tests for config types

---

## Phase 5: Add Order/Execution Types

- [x] Add `OrderStatus` enum
- [x] Add `OrderRequest` model
- [x] Add `Order` model (with lifecycle tracking)
- [x] Add `Execution` model
- [x] Add tests for order/execution types

---

## Phase 6: Add Training/Run Types

- [x] Add `TimeSlice` model
- [x] Add `AnalysisSnapshot` model
- [x] Add `RunState` model
- [x] Add `DatasetMetadata` model
- [x] Add `RunArtifacts` model
- [x] Add `InspectRunRequest` model
- [x] Add tests for training/run types

---

## Phase 7: Add Utility Types

- [x] Add `DatasetBundle` model (with method stubs)
- [x] Add `Gap` model
- [x] Add `RewardSignal` model
- [x] Add `RunProgress` model
- [x] Add tests for utility types

---

## Phase 8: Validation and Documentation

- [x] Run `pytest` - all tests must pass (41 passed)
- [x] Run `ruff check` - no linting errors
- [x] Update `docs/cli-and-implementation.md` to mark type definitions as complete
- [x] Update `__all__` exports in `types.py`

---

## Completion Criteria

- [x] All types from Section 2 of cli-and-implementation.md are implemented
- [x] All exceptions from Section 3 are implemented
- [x] All existing tests pass
- [x] New tests cover all new types
- [x] No ruff linting errors
- [x] Documentation updated to reflect completion

---

## Progress Notes

**Phase 1 Complete (2024-12-15):**
- Added pydantic>=2.0.0 and ruff>=0.8.0 to pyproject.toml
- Installed dependencies successfully

**Phase 2 Complete (2024-12-15):**
- Added ConfigError, DataSourceError, DataValidationError, StorageError, StrategyError
- Created test_exceptions.py with 8 tests

**Phases 3-7 Complete (2024-12-15):**
- Migrated all existing types from dataclass to Pydantic BaseModel
- Added FrozenModel and MutableModel base classes
- Added all missing types: FetchDataConfig, GenSynthConfig, TrainingConfig, OrderStatus, OrderRequest, Order, Execution, TimeSlice, AnalysisSnapshot, RunState, DatasetMetadata, RunArtifacts, InspectRunRequest, DatasetBundle, Gap, RewardSignal, RunProgress
- Updated test_accounts_manager.py to use model_dump() instead of asdict()
- Added comprehensive tests for all new types
- All 41 tests pass
- No ruff linting errors

