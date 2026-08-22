# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Frame-level `depth` prediction member (`Depth` type) as produced by depth estimation abilities (e.g. `eyepop.depth.*`): base64 little-endian float32 map with the source frame's aspect ratio, sky pixels as `+Infinity`. New `eyepop.DepthMap` decodes it lazily to a numpy array with sky mask, finite min/max, and proportional source-coordinate sampling; `EyePopPlot.depth()` overlays it as a turbo heatmap. `pop_demo.py` gains a `depth` example and summarizes depth/mask binaries instead of dumping base64; the webui2 viewer renders depth via `Render2d.renderDepth()` where available. numpy is now a declared dependency (it was already required transitively).

### Changed
- `filter_prediction_top_k` preserves all prediction members (timestamp, depth, embeddings, details, motions, ...) instead of dropping everything but objects/classes/texts/meshs/keyPoints.
- Model artifact variant support on the Data API (OPA-75): `upload_model_artifact()` accepts `exported_by` and a `variant` attribute dict (list values expand to the cartesian product, registering one binary for multiple variants); `export_model_urls()` / `export_model_artifacts()` accept a single-combination `variant` for exact-match selection with default-variant fallback; `ModelExport` exposes `variant`; new `Quantization` and `TargetRuntime` enums carry the well-known variant values.

### Deprecated
- `device_name` on `export_model_urls()` / `export_model_artifacts()` — use `variant={"qualcomm_device_name": ...}` instead.
- Scheduled sessions smoke workflow for validating transient SDK inference against production with optional Slack status alerts and selectable SDK package versions.
- `session_name` support on worker session creation, also configurable via `EYEPOP_SESSION_NAME`.
- `pop` support on worker session creation so transient compute sessions can be scheduled before starting a worker pipeline.

### Fixed
- Transient sessions started with a `pop` now wait for the compute API to finish creating the pipeline before reporting an ownership failure. Previously the SDK checked pipeline ownership on the initial session response and raised immediately, so a session created a moment before its pipeline row landed (common right after a compute API deploy) failed spuriously. The client-visible "did not return an owned pipeline" error is preserved for sessions that genuinely never receive a pipeline.
- Worker connections without a `session_uuid` no longer adopt an existing persistent session. The compute API session list is now filtered by the new `persistent` flag so ephemeral connections always pick (or create) an ephemeral session, and persistent sessions are only reachable when their UUID is passed explicitly. (AWSU-166)

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
