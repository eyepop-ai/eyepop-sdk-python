# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PointCloud` gains the rest of `DepthMap`'s shape: `from_prediction` returns every cloud in a prediction, nested objects included - a list rather than a single value, because a depth map is frame level while a cloud belongs to one object's mask - plus `placed_points`, the (N, 3) array with the NaN holes dropped, and `bounds`, the per-axis counterpart to `finite_min`/`finite_max`.
- `EyePopWorldPlot` scatters everything in a prediction that carries world coordinates into a 3D axes, in metres - key points, outlines, contours and mask point clouds alike, so a pop that produces no masks still has something to show. One colour and legend entry per carrier, via `labelled_world_points`. The point budget is shared across series so adding objects thins the scatter rather than multiplying it, but sparse series are exempt: a 17 point skeleton and a 40,000 point cloud are both series, and an evenly shared budget would thin the skeleton away to save a fraction of the cloud. The box takes the data's own proportions, since an unequal box would misrepresent the geometry the coordinates exist to measure. Z is drawn into the scene and Y up the page rather than in component order, since depth is what reads as distance from the viewer in either frame; the coordinates themselves are never altered. The vertical axis is flipped so a camera frame scene stands the right way up: Y grows downwards there, so drawn as-is a figure's feet sit above its head. That is the default because a source supplying no extrinsics, or identity ones, gets the camera frame; `invert_y=False` suits coordinates already in a Z-up world frame. `pop_demo.py --visualize-world` renders the results this way.
- `pop_demo.py --translate-to-world` adds world coordinates to any of its example pops, composed or flat: it names a depth ability and walks the component tree - nested `forward` targets included - opting in every component that can carry them. Hidden components are skipped, since their predictions never reach the response, while their targets are still walked. `--camera-hfov-degrees` or `--camera-intrinsics` supply the source calibration, and without one the demo says which guess the worker is falling back to. `--output` now summarizes `mask.world` instead of dumping megabytes of base64, and reports how many points came back placed - which is what separates a calibration that worked from one that quietly did not.
- World coordinates on predictions. `worldX`, `worldY` and `worldZ` (metres) now deserialise on key points, outline points and contour points, including cutouts - they are members of `Point2d`, so every carrier the worker enriches inherits them. Prediction v2 only, and a point the worker could not place carries none of the three rather than a zero or a NaN, so test for `None`. `PredictedKeyPoint.z` is untouched and independent: `z` is model-relative depth in whatever convention the model uses, `worldZ` is metres.
- `Mask.world`, the dense per-object point cloud, with `eyepop.PointCloud` to decode it: one xyz triple per mask pixel as a numpy `(height, width, 3)` float32 array indexed exactly like the bitmap. `PointCloud.at(i, j)` samples by mask pixel and `at_source(x, y)` by source coordinate, since the mask spans the object's bounding box. NaN is the wire's omission sentinel here and is preserved, which is why this does not share `DepthMap`'s validator - that one rejects any NaN and so would reject every valid cloud.
- `Depth.semantic` says what a depth map's values mean: `canonical_metric`, `metric`, `relative` or `unknown`. Always present in prediction v2, so an absent member means a worker that predates the field rather than a map that declined to say. Only the two metric flavours can be back-projected.
- Camera calibration per source: a `camera` argument on `upload`, `upload_stream`, `upload_group`, `upload_stream_group`, `load_from`, `load_from_group` and `load_asset`. `Camera` carries normalized intrinsics (`fx`, `fy`, `cx`, `cy` as fractions of the frame rather than pixels, so one calibration survives a resolution change), the OpenCV distortion coefficients, and a camera-to-world pose as a unit quaternion plus a translation in metres. Without one the worker falls back to an assumed 60 degree field of view, which is a development scaffold: for canonical metric depth the guess cancels out of X and Y and survives only in Z, so lateral measurements stay exact while every distance along the optical axis is wrong by however wrong the guess was.
- `Camera` also accepts an `hfovDegrees` shorthand for an operator who knows their lens's field of view but not its calibration matrix. Exactly one of `intrinsics` and `hfovDegrees` is required; both is rejected rather than resolved by precedence, since two descriptions of one lens that disagree have no right answer. Calibrations are validated locally, so a bad one is a `ValidationError` rather than a 400 halfway through an upload.
- `Pop.defaults`, the source level parameters a Pop sets once for every source it processes - `camera`, `roi`, `fps` and the motion detection group - each overridable per source and merged per field, so a source giving its own roi but no camera keeps its roi and takes the default camera.
- A Pop can request world coordinates: `depthMapAbility` / `depthMapAbilityUuid` names the ability whose depth map the worker back-projects through, and `translateToWorld` on a component selects whose predictions get translated. Use a metric depth ability; a `relative` one is accepted and silently yields no coordinates, because relative depth is scale- and shift-invariant and a cloud recovered from it would be distorted rather than merely unscaled. `translateToWorld` only means something on a component that runs its own inference, which is what gives it an id for the worker to select on.
- Frame-level `depth` prediction member (`Depth` type) as produced by depth estimation abilities (e.g. `eyepop.depth.*`): base64 little-endian float32 map with the source frame's aspect ratio, sky pixels as `+Infinity`. New `eyepop.DepthMap` decodes it lazily to a numpy array with sky mask, finite min/max, and proportional source-coordinate sampling; `EyePopPlot.depth()` overlays it as a turbo heatmap. `pop_demo.py` gains a `depth` example and summarizes depth/mask binaries instead of dumping base64; the webui2 viewer renders depth via `Render2d.renderDepth()` where available. numpy is now a declared dependency (it was already required transitively).

### Fixed
- Predictions larger than the HTTP read buffer (64kb) no longer fail with `ValueError: Chunk too big`. Worker result lines are now accumulated without a size limit instead of relying on `aiohttp`'s `readline()`, which any prediction carrying a depth map exceeds (~1mb of base64 per frame).

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
