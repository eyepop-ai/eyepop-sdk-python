# EyePop Python SDK Modernization Plan

This document outlines a comprehensive plan to update the EyePop Python SDK to use modern tools and best practices. Each phase is designed to be a separate PR with testable outcomes.

## Current State Analysis

| Area | Current State | Issues |
|------|---------------|--------|
| **Linting** | ruff configured but only for `eyepop/compute/*` and `tests/test_compute_*.py` | 95%+ of codebase excluded |
| **Type Checking** | mypy configured with `\|\| true` in Taskfile | 212 errors ignored |
| **Code Size** | `data_endpoint.py` (1214 lines), `data_syncify.py` (820 lines) | Large files hard to maintain |
| **Sync Wrapper** | Thread-based with manual event loop | Verbose, type-unsafe, 820 lines of boilerplate |
| **Tests** | 107 tests with pytest | No coverage reporting in CI |
| **CI/CD** | Basic lint + typecheck (continue-on-error) + test | No matrix testing, no coverage |
| **Docs** | mkdocs deps defined but no build task | No automated doc generation |

---

## Phase 1: Expand Linting Coverage

**Goal**: Enable ruff linting across the entire codebase and fix all issues.

### PR 1.1: Configure ruff for full codebase

**Changes**:
- Update `ruff.toml` to include all Python files (remove `exclude = ["*"]`)
- Keep current rule set: `E4, E7, E9, F, B, D, I001`
- Add per-file ignores for legacy patterns as needed

**Files to modify**:
- `ruff.toml`

**Test**: `uvx ruff check .` passes with no errors

**Estimated scope**: ~50-100 auto-fixable issues

---

### PR 1.2: Fix ruff errors in `eyepop/worker/`

**Changes**:
- Fix import sorting (I001)
- Fix unused imports (F401)
- Fix bugbear issues (B)
- Apply `ruff check --fix` where safe

**Files to modify**:
- `eyepop/worker/*.py`

**Test**: `uvx ruff check eyepop/worker/` passes

---

### PR 1.3: Fix ruff errors in `eyepop/data/`

**Changes**:
- Same as PR 1.2 for data module
- Special attention to `data_endpoint.py` and `data_types.py`

**Files to modify**:
- `eyepop/data/*.py`
- `eyepop/data/arrow/*.py`

**Test**: `uvx ruff check eyepop/data/` passes

---

### PR 1.4: Fix ruff errors in remaining files

**Changes**:
- Fix issues in root `eyepop/` files
- Update Taskfile to remove `|| true` from lint task

**Files to modify**:
- `eyepop/*.py`
- `Taskfile.yml`

**Test**: `task lint:all` passes, CI lint step enforced

---

## Phase 2: Fix Type Checking

**Goal**: Achieve zero mypy errors and enforce in CI.

### PR 2.1: Fix `import-not-found` errors

**Changes**:
- Add `py.typed` marker (already exists)
- Configure mypy to handle `_version.py` generated file
- Add missing type stubs or ignore specific third-party imports

**Files to modify**:
- `pyproject.toml` (mypy config)

**Test**: `uvx mypy eyepop` has no `import-not-found` errors

---

### PR 2.2: Fix `no-any-return` errors in syncify modules

**Changes**:
- Add proper return type annotations to `run_coro_thread_save()`
- Use `TypeVar` for generic return types in sync wrappers
- Fix ~100 `no-any-return` errors in `data_syncify.py` and `worker_syncify.py`

**Files to modify**:
- `eyepop/syncify.py`
- `eyepop/data/data_syncify.py`
- `eyepop/worker/worker_syncify.py`

**Test**: `uvx mypy eyepop/*syncify*.py` passes

---

### PR 2.3: Fix remaining mypy errors

**Changes**:
- Fix `arg-type`, `assignment`, `return-value` errors
- Add type annotations where missing
- Use `cast()` or `# type: ignore[code]` sparingly with comments

**Files to modify**:
- Various files based on mypy output

**Test**: `uvx mypy eyepop` passes with zero errors

---

### PR 2.4: Enforce mypy in CI

**Changes**:
- Remove `continue-on-error: true` from typecheck step
- Remove `|| true` from Taskfile typecheck task

**Files to modify**:
- `.github/workflows/checks.yml`
- `Taskfile.yml`

**Test**: CI fails on type errors

---

## Phase 3: Refactor Large Modules

**Goal**: Break down large files into smaller, focused modules.

### PR 3.1: Split `data_types.py` into domain modules

**Changes**:
- Create `eyepop/data/types/` package
- Split into:
  - `types/dataset.py` - Dataset, DatasetCreate, DatasetUpdate, etc.
  - `types/asset.py` - Asset, AssetImport, AssetStatus, etc.
  - `types/model.py` - Model, ModelCreate, ModelAlias, etc.
  - `types/prediction.py` - Prediction, Box, Point2d, etc.
  - `types/workflow.py` - Workflow types
  - `types/vlm.py` - VLM ability types
  - `types/__init__.py` - Re-export all for backward compatibility

**Files to modify**:
- `eyepop/data/data_types.py` → split into `eyepop/data/types/*.py`

**Test**: All existing tests pass, imports work unchanged

---

### PR 3.2: Split `data_endpoint.py` by domain

**Changes**:
- Create mixin classes for different API domains:
  - `DatasetMixin` - dataset CRUD operations
  - `AssetMixin` - asset operations
  - `ModelMixin` - model operations
  - `WorkflowMixin` - workflow operations
  - `VlmMixin` - VLM ability operations
- `DataEndpoint` composes these mixins

**Files to modify**:
- `eyepop/data/data_endpoint.py` → split into `eyepop/data/endpoints/*.py`

**Test**: All existing tests pass

---

### PR 3.3: Generate `data_syncify.py` automatically

**Changes**:
- Create a code generator script that:
  - Reads `DataEndpoint` method signatures
  - Generates corresponding sync wrapper methods
- Reduces 820 lines of boilerplate to ~50 lines of generator code
- Add `scripts/generate_sync_wrappers.py`

**Files to modify**:
- Add `scripts/generate_sync_wrappers.py`
- Regenerate `eyepop/data/data_syncify.py`
- Add generation task to `Taskfile.yml`

**Test**: Generated file matches expected output, all tests pass

---

## Phase 4: Improve Testing Infrastructure

**Goal**: Add coverage reporting and improve test organization.

### PR 4.1: Add test coverage reporting

**Changes**:
- Configure pytest-cov in `pyproject.toml`
- Add coverage reporting to CI
- Add coverage badge to README

**Files to modify**:
- `pyproject.toml`
- `.github/workflows/checks.yml`
- `README.md`

**Test**: CI generates coverage report

---

### PR 4.2: Add integration test separation

**Changes**:
- Mark integration tests with `@pytest.mark.integration`
- Add pytest markers configuration
- Separate CI jobs for unit vs integration tests
- Integration tests require `EYEPOP_` env vars

**Files to modify**:
- `pyproject.toml`
- `tests/conftest.py`
- `.github/workflows/checks.yml`

**Test**: `pytest -m "not integration"` runs only unit tests

---

### PR 4.3: Add Python version matrix testing

**Changes**:
- Test against Python 3.12 and 3.13
- Use matrix strategy in GitHub Actions

**Files to modify**:
- `.github/workflows/checks.yml`

**Test**: CI runs tests on multiple Python versions

---

## Phase 5: Modernize Async/Sync Pattern

**Goal**: Simplify the sync wrapper pattern using modern approaches.

### PR 5.1: Evaluate `asyncio.run()` vs thread-based approach

**Research PR** - No code changes, just analysis document.

**Questions to answer**:
- Can we use `asyncio.run()` for simpler sync wrappers?
- What are the implications for nested event loops?
- Should we adopt `anyio` for backend flexibility?
- Benchmark current vs proposed approaches

**Deliverable**: `docs/adr/001-sync-wrapper-pattern.md`

---

### PR 5.2: Implement simplified sync wrapper (if approved)

**Changes** (based on PR 5.1 findings):
- Option A: Use `asyncio.run()` with proper loop handling
- Option B: Use `anyio.from_thread.run()` for better compatibility
- Option C: Keep current pattern but improve typing

**Files to modify**:
- `eyepop/syncify.py`
- `eyepop/data/data_syncify.py`
- `eyepop/worker/worker_syncify.py`

**Test**: All sync tests pass, no performance regression

---

## Phase 6: Documentation and Developer Experience

**Goal**: Improve documentation and developer onboarding.

### PR 6.1: Add mkdocs build task and configuration

**Changes**:
- Add `task docs` and `task docs:serve` commands
- Configure mkdocs with API reference generation
- Add basic documentation structure

**Files to modify**:
- `Taskfile.yml`
- `mkdocs.yml` (create)
- `docs/` directory structure

**Test**: `task docs` generates documentation site

---

### PR 6.2: Add CONTRIBUTING.md

**Changes**:
- Document development setup
- Document PR process
- Document testing requirements
- Document release process

**Files to create**:
- `CONTRIBUTING.md`

---

### PR 6.3: Add pre-commit hooks

**Changes**:
- Configure pre-commit with ruff, mypy
- Add `.pre-commit-config.yaml`
- Document in CONTRIBUTING.md

**Files to create**:
- `.pre-commit-config.yaml`

**Test**: `pre-commit run --all-files` passes

---

## Phase 7: Dependency and Build Improvements

**Goal**: Modernize dependency management and build process.

### PR 7.1: Audit and tighten dependency versions

**Changes**:
- Review dependency version ranges
- Tighten ranges where appropriate
- Document why wide ranges are needed (if any)
- Ensure `uv.lock` is committed and up-to-date

**Files to modify**:
- `pyproject.toml`
- `uv.lock`

**Test**: `uv sync` works, all tests pass

---

### PR 7.2: Add dependabot configuration

**Changes**:
- Configure dependabot for Python dependencies
- Set up weekly update schedule
- Configure auto-merge for patch updates

**Files to create**:
- `.github/dependabot.yml`

---

## Execution Order and Dependencies

```
Phase 1 (Linting)
├── PR 1.1 → PR 1.2 → PR 1.3 → PR 1.4
│
Phase 2 (Types) - can start after PR 1.4
├── PR 2.1 → PR 2.2 → PR 2.3 → PR 2.4
│
Phase 3 (Refactor) - can start after PR 2.4
├── PR 3.1 → PR 3.2 → PR 3.3
│
Phase 4 (Testing) - can start in parallel with Phase 1
├── PR 4.1 (parallel)
├── PR 4.2 (after 4.1)
├── PR 4.3 (parallel with 4.2)
│
Phase 5 (Async) - after Phase 2 and 3
├── PR 5.1 → PR 5.2
│
Phase 6 (Docs) - can start in parallel
├── PR 6.1 (parallel)
├── PR 6.2 (parallel)
├── PR 6.3 (after Phase 1)
│
Phase 7 (Deps) - can start anytime
├── PR 7.1 (parallel)
├── PR 7.2 (parallel)
```

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Ruff coverage | ~5% of files | 100% |
| Mypy errors | 212 | 0 |
| Largest file | 1214 lines | <500 lines |
| Test coverage | Unknown | >80% reported |
| CI strictness | Warnings ignored | All checks enforced |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes in refactors | High | Comprehensive test suite, careful re-exports |
| Type annotation changes break users | Medium | Use `@overload` for backward compatibility |
| Performance regression in sync wrapper | Medium | Benchmark before/after |
| Large PRs hard to review | Medium | Keep PRs focused, max 500 lines changed |

## Timeline Estimate

- **Phase 1**: 1-2 weeks (4 PRs)
- **Phase 2**: 2-3 weeks (4 PRs)
- **Phase 3**: 2-3 weeks (3 PRs)
- **Phase 4**: 1 week (3 PRs, parallel)
- **Phase 5**: 1-2 weeks (2 PRs)
- **Phase 6**: 1 week (3 PRs, parallel)
- **Phase 7**: 1 week (2 PRs, parallel)

**Total**: ~6-10 weeks with parallelization
