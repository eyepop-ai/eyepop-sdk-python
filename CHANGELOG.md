# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed (BREAKING)
- `secret_key` parameter and `EYEPOP_SECRET_KEY` environment variable. Construction now raises `ValueError` if either is set. Use `api_key` / `EYEPOP_API_KEY` and create deployments via the EyePop dashboard or CLI for named, persistent sessions.
- Legacy named-pop config path (`/pops/<pop_id>/config`). The SDK now talks only to the compute API session surface (`/v1/sessions`).
- `auto_start` parameter on `workerEndpoint()` / `sync_worker()` / `async_worker()`. Sessions are always provisioned through the compute API.
- `should_use_compute_api()` helper. The compute API is the only supported path.

### Deprecated
- `pop_id` parameter. Pop selection now happens at deployment creation time (dashboard / CLI). Passing `pop_id` other than `"transient"` emits a deprecation warning.

### Changed
- `eyepop_url` defaults to `https://compute.eyepop.ai` (no fallback to `https://api.eyepop.ai`).

## [3.15.2] - 2026-04-27

### Added
- `mediaCacheSeconds` field on `InferenceComponent` to retain the last N seconds of live streams for replay and debugging.

## [3.15.1] - 2026-04-03

### Added
- `pipeline_image` and `pipeline_version` parameters on `workerEndpoint()` for custom worker Docker images (also configurable via `EYEPOP_PIPELINE_IMAGE` and `EYEPOP_PIPELINE_VERSION` environment variables)
- `videoChunkLengthSeconds` and `videoChunkOverlap` fields on `InferenceComponent` for chunked video processing

### Fixed
- InferJob polling now uses GET instead of POST for checking job status
- Removed unnecessary `Content-Type: application/json` header from compute API session creation (aiohttp sets it automatically)
- Fixed `EvaluateJob` potentially unbound `result` variable
- Improved null-safety guards and type annotations in worker endpoint

### Changed
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
