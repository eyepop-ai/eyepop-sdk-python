---
name: eyepop-sdk
description: EyePop Python SDK - usage patterns, testing examples, and composable pops
---

# EyePop Python SDK

## Repo

| Path | Branch | Language |
|------|--------|----------|
| `~/Code/eyepop/eyepop-sdk-python` | `main` | Python 3.12+ |

Package: `eyepop` on PyPI (current: ~3.11.x)

## Authentication

The SDK uses `EYEPOP_API_KEY` for authentication. Keys are stored in vault files:

| Environment | Vault File | API Base |
|-------------|-----------|----------|
| Staging | `~/Code/eyepop/.vault/.env.staging` | `https://api.staging.eyepop.xyz` |
| Production | `~/Code/eyepop/.vault/.env.production` | `https://web-api.eyepop.ai` |

```bash
# Load staging credentials (must also set EYEPOP_URL for staging)
set -a && source ~/Code/eyepop/.vault/.env.staging && set +a
export EYEPOP_URL=https://compute.staging.eyepop.xyz

# Load production credentials (EYEPOP_URL defaults to production, no override needed)
set -a && source ~/Code/eyepop/.vault/.env.production && set +a
```

**Important:** The vault files use plain assignments (no `export`), so use `set -a` to auto-export all sourced variables.

The SDK reads `EYEPOP_API_KEY` from the environment automatically. Other env vars:

| Variable | Required | Description |
|----------|----------|-------------|
| `EYEPOP_API_KEY` | Yes | API key from dashboard.eyepop.ai |
| `EYEPOP_POP_ID` | No | Pop Id from dashboard. Defaults to `"transient"` |
| `EYEPOP_ACCOUNT_ID` | For Data API | Account UUID for `EyePopSdk.dataEndpoint()` |
| `EYEPOP_URL` | For staging | Compute API URL. Defaults to `https://compute.eyepop.ai` (production). Set to `https://compute.staging.eyepop.xyz` for staging. |

**Important:** `EYEPOP_SECRET_KEY` is deprecated and should NOT be used in any code or documentation.

See `eyepop-auth` skill for full authentication details.

## SDK Entry Points

### WorkerEndpoint (inference)

```python
from eyepop import EyePopSdk

# Sync
with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload('image.jpg').predict()

# Async
async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
    job = await endpoint.upload('image.jpg')
    result = await job.predict()
```

### DataEndpoint (datasets, VLM, evaluation)

```python
from eyepop import EyePopSdk

async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
    datasets = await endpoint.list_datasets()
```

## Composable Pops

Use `Pop` objects with `endpoint.set_pop(pop)` to configure inference pipelines at runtime.

### Key Types

```python
from eyepop.worker.worker_types import (
    Pop,                      # Top-level pipeline definition
    InferenceComponent,       # ML model component
    TrackingComponent,        # Object tracking
    ContourFinderComponent,   # Segmentation contour extraction
    ComponentFinderComponent, # Connected components
    ForwardComponent,         # Route between stages
    CropForward,              # Crop detections -> sub-components
    FullForward,              # Full image -> sub-components
    ComponentParams,          # Runtime parameters (prompts, ROI)
)
```

### Important: Use `ability=`, NOT `model=`

The `model=` and `modelUuid=` fields on `InferenceComponent` are **deprecated**. Always use:
- `ability='eyepop.<name>:latest'` (by alias)
- `abilityUuid='<uuid>'` (by UUID)

### Common Pop Patterns

```python
# Simple detection
Pop(components=[
    InferenceComponent(ability='eyepop.person:latest', categoryName="person")
])

# Chained: detect -> crop -> sub-model
Pop(components=[
    InferenceComponent(
        ability='eyepop.vehicle:latest',
        forward=CropForward(targets=[
            InferenceComponent(ability='eyepop.vehicle.licence-plate:latest')
        ])
    )
])

# VLM with prompts
Pop(components=[
    InferenceComponent(
        id=1,
        ability='eyepop.localize-objects:latest',
        params={"prompts": [{"prompt": "person"}]},
    )
])
```

### `set_pop()` Only Accepts `Pop` Objects

```python
# CORRECT
await endpoint.set_pop(Pop(components=[...]))

# WRONG - string form is removed
await endpoint.set_pop('ep_infer model=...')  # TypeError
```

## Data API Types

```python
from eyepop.data.data_types import (
    InferRequest,      # VLM inference request
    EvaluateRequest,   # Batch dataset evaluation
    TranscodeMode,     # Image transcode options
    Dataset,           # Dataset model
    Asset,             # Asset model
)
```

## Examples Directory

Located at `~/Code/eyepop/eyepop-sdk-python/examples/`

### Working Examples

| File | Purpose | Run Command |
|------|---------|-------------|
| `pop_demo.py` | Composable pops (main demo) | `python pop_demo.py --pop person -l example.jpg -o` |
| `caption_demo.py` | VLM caption generation | `python caption_demo.py -c eyepop.vlm.preview:latest -l example.jpg` |
| `infer_demo.py` | VLM inference via data API | `python infer_demo.py -m qwen3-instruct -p "describe this" -a <uuid>` |
| `upload_image_timing.py` | Upload benchmarks (4 strategies) | `python upload_image_timing.py example.jpg 10` |
| `upload_streaming.py` | Stream upload | `python upload_streaming.py example.jpg` |
| `upload_video.py` | Video upload | `python upload_video.py video.mp4` |
| `load_video_from_http.py` | Video from URL | Standalone (hardcoded URL) |
| `live_rtmp_stream.py` | RTSP stream | Standalone (hardcoded URL) |
| `visualize_on_image.py` | Matplotlib visualization | `python visualize_on_image.py example.jpg` |
| `visualize_with_webui2.py` | Desktop webui2 visualization | `python visualize_with_webui2.py example.jpg` |
| `workflow_cli.py` | Workflow management CLI | `python workflow_cli.py list-workflows` |
| `auth_session.py` | OAuth2 device code flow | `python auth_session.py -a <account-uuid>` |
| `download_video.py` | Download video asset | `python download_video.py output.mp4 -a <uuid>` |
| `import_dataset.py` | Dataset import + auto-annotate | `python import_dataset.py ./assets/` |

### Test Data

Test images and videos are at `~/Code/eyepop/test_data/`:

| Directory | Contents |
|-----------|----------|
| `images/` | 161 test images (people, kittens, misc scenes) |
| `video/` | Test videos |
| `kagel_dataset/` | Kaggle dataset |

### Testing an Example

The examples directory has its own venv at `.venv/`. Use `.venv/bin/python` directly.

**Production** (default — SDK defaults to `compute.eyepop.ai`, no `EYEPOP_URL` needed):

```bash
cd ~/Code/eyepop/eyepop-sdk-python/examples
set -a && source ~/Code/eyepop/.vault/.env.production && set +a
.venv/bin/python pop_demo.py --pop person -l ~/Code/eyepop/test_data/images/Hackers_02.jpg -o
```

**Staging** (must set `EYEPOP_URL`):

```bash
cd ~/Code/eyepop/eyepop-sdk-python/examples
set -a && source ~/Code/eyepop/.vault/.env.staging && set +a
export EYEPOP_URL=https://compute.staging.eyepop.xyz
.venv/bin/python pop_demo.py --pop person -l ~/Code/eyepop/test_data/images/Hackers_02.jpg -o
```

## Development

```bash
cd ~/Code/eyepop/eyepop-sdk-python

# Install in dev mode
pip install -e ".[dev]"

# Run tests
uv run python -m pytest tests/

# Lint (changed files vs main)
task lint

# Lint all
task lint:all

# Type check
uvx basedpyright
```

## Test PyPI Packages

**Every push to a PR** automatically publishes a dev package to [test.pypi.org](https://test.pypi.org/project/eyepop/). The version is derived from `setuptools_scm` (e.g., `3.11.1.dev4`). You can also trigger a manual publish via `workflow_dispatch` with a custom version.

The workflow is defined in `.github/workflows/test-pypi-publish.yml`.

### Installing a Test PyPI Package

```bash
# Create a fresh venv
uv venv /tmp/eyepop-test-pypi --python 3.12

# Install the test PyPI version (need --index-strategy for uv to find it)
VIRTUAL_ENV=/tmp/eyepop-test-pypi uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  eyepop==<VERSION>

# Install example dependencies
VIRTUAL_ENV=/tmp/eyepop-test-pypi uv pip install webui2 pybars3 python-dotenv pillow
```

### Running Examples Against Test PyPI Package

```bash
# Load staging credentials and run
export $(grep -v '^#' ~/Code/eyepop/.vault/.env.staging | xargs)
export EYEPOP_URL="https://compute.staging.eyepop.xyz"

/tmp/eyepop-test-pypi/bin/python ~/Code/eyepop/eyepop-sdk-python/examples/pop_demo.py \
  -p person \
  -u "https://picsum.photos/id/237/200/300" \
  -o
```

### Finding the Published Version

Check the test-pypi-publish CI job logs for the line: `Uploading eyepop-<VERSION>-py3-none-any.whl`

Or browse: https://test.pypi.org/project/eyepop/#history

## Key Source Files

| Purpose | Path |
|---------|------|
| SDK entry point | `eyepop/eyepopsdk.py` |
| Worker endpoint | `eyepop/worker/worker_endpoint.py` |
| Worker types (Pop, components) | `eyepop/worker/worker_types.py` |
| Data endpoint | `eyepop/data/data_endpoint.py` |
| Data types (InferRequest, etc.) | `eyepop/data/data_types.py` |
| Visualization | `eyepop/visualize.py` |
| Package exports | `eyepop/__init__.py` |

## Related Skills

| Skill | Description |
|-------|-------------|
| `eyepop-auth` | Authentication, API keys, JWT tokens, vault files |
| `eyepop-compute-api` | Compute API that manages worker sessions |
| `eyepop-worker` | Worker runtime (Go API + C pipeline) |
| `eyepop-vlm` | VLM API details |
| `eyepop-dataset-api` | Dataset API service |
