# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PLAN.md` documenting SDK modernization roadmap
- `CHANGELOG.md` for tracking changes

### Changed

#### Phase 1: Linting (ruff)
- Expanded ruff linting from ~5% to 100% of codebase
- Updated `ruff.toml` to include all Python files
- Added `D105` (magic method docstrings) to ignore list
- Fixed 38 lint issues across 22 files:
  - Import sorting (I001) - 18 files auto-fixed
  - Unused variables (F841) - `schema_version_conversion.py`
  - Unused loop variables (B007) - `metrics.py`
  - Undefined forward references (F821) - `syncify.py`
  - Blind exception assertions (B017) - 3 test files
  - Duplicate function definitions (F811) - `test_endpoint_retry.py`

#### Phase 2: Type Checking (mypy)
- Configured mypy to pass with 0 errors (down from 463)
- Updated `pyproject.toml` with comprehensive mypy configuration:
  - Added `pyarrow`, `websockets`, `pandas` to `ignore_missing_imports`
  - Added per-module overrides for complex patterns
  - Disabled overly strict warnings (`warn_return_any`, `warn_unreachable`)
- Added `# type: ignore[assignment]` to Pydantic discriminated union fields
- Removed `continue-on-error: true` from CI typecheck step
- Changed Taskfile to use `uv run mypy` for proper venv access

#### Phase 3: Refactor Large Modules (PR 3.1)
- Split `data_types.py` (968 lines) into domain-specific modules under `eyepop/data/types/`:
  - `enums.py` - All enumeration types and constants
  - `common.py` - Shared base types (Point2d, Box, Contour, Mask)
  - `prediction.py` - Prediction-related types
  - `dataset.py` - Dataset CRUD types
  - `asset.py` - Asset types
  - `model.py` - Model training and export types
  - `events.py` - Change event types
  - `workflow.py` - Argo workflow types
  - `vlm.py` - VLM/inference types
  - `__init__.py` - Re-exports all types for backward compatibility
- Original `data_types.py` now re-exports from `eyepop.data.types` for backward compatibility

### Fixed
- `test_endpoint_retry.py` - Removed duplicate test function, fixed mock URL
- Test assertions now use `KeyError` instead of broad `Exception`
