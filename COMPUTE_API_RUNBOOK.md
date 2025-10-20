## Using Compute API with the Python SDK

### Overview

The EyePop Python SDK supports two backend APIs:
- **Worker API (v1)**: Traditional endpoint-based system for named pops using `EYEPOP_SECRET_KEY`
- **Compute API (v2)**: Session-based compute platform for transient pops using `EYEPOP_API_KEY`

The SDK automatically uses the Compute API when:
1. `EYEPOP_API_KEY` is provided (instead of `EYEPOP_SECRET_KEY`)
2. `pop_id` is set to `"transient"` (the default)
3. `EYEPOP_URL` points to a compute endpoint (optional, auto-detected)

### 1. Environment Configuration

Set your environment variables for Compute API:

```bash
export EYEPOP_API_KEY="your-compute-api-key"
export EYEPOP_URL="https://compute.staging.eyepop.xyz"  # Optional
export EYEPOP_POP_ID="transient"  # Optional (this is the default)
```

Or use a `.env` file:

```env
EYEPOP_API_KEY=your-compute-api-key
EYEPOP_URL=https://compute.staging.eyepop.xyz
EYEPOP_POP_ID=transient
```

### 2. Using the Python SDK

The SDK usage is identical to the Worker API - the Compute API integration is transparent:

#### Basic Image Processing

```python
import os
from eyepop import EyePopSdk

# Using environment variables
async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
    result = await (await endpoint.upload('example.jpg')).predict()
    print(result)

# Or explicitly passing api_key
async with EyePopSdk.workerEndpoint(
    api_key=os.getenv("EYEPOP_API_KEY"),
    eyepop_url="https://compute.staging.eyepop.xyz",
    is_async=True
) as endpoint:
    result = await (await endpoint.upload('example.jpg')).predict()
    print(result)
```

#### Synchronous Mode

```python
from eyepop import EyePopSdk

with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload('example.jpg').predict()
    print(result)
```

#### Processing from URLs

```python
from eyepop import EyePopSdk

async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
    job = await endpoint.load_from('https://example.com/image.jpg')
    result = await job.predict()
    print(result)
```

#### Custom Pop Configuration

```python
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent

# Define your custom pop
custom_pop = Pop(components=[
    InferenceComponent(
        model='eyepop.person:latest',
        categoryName="person"
    )
])

async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
    await endpoint.set_pop(custom_pop)
    result = await (await endpoint.upload('example.jpg')).predict()
    print(result)
```

### 3. How It Works

#### Session Lifecycle

1. **Session Creation**: When `connect()` is called, the SDK:
   - Creates a Compute API session via `POST /v1/sessions`
   - Receives a JWT access token and session endpoint
   - Waits for the session to become ready (polling `/health`)

2. **Pipeline Creation**: After session is ready:
   - Creates a pipeline via `POST /pipelines`
   - Configures with your Pop definition

3. **Job Execution**: Your uploads/load operations work normally:
   - Uses the session-specific endpoint
   - Authenticated with session JWT token

4. **Cleanup**: When `disconnect()` is called:
   - Deletes the pipeline
   - Session remains active (configurable TTL)

#### Token Refresh

JWT tokens expire after a configurable period. The SDK automatically:
- Detects 401 responses
- Refreshes the token via `POST /v1/sessions/{session_uuid}/token`
- Retries the failed request

### 4. Working with Compute API Directly

While the SDK handles everything, you can also use the Compute API directly:

#### Create a Session

```bash
curl --location --request POST 'https://compute.staging.eyepop.xyz/v1/sessions' \
  --header 'Content-Type: application/json' \
  --header 'Accept: application/json' \
  --header 'Authorization: Bearer $EYEPOP_API_KEY'
```

#### Get Session Info

```bash
curl --location 'https://compute.staging.eyepop.xyz/v1/sessions/${SESSION_UUID}' \
  --header 'Accept: application/json' \
  --header 'Authorization: Bearer $EYEPOP_API_KEY'
```

#### Refresh Token

```bash
curl --location --request POST 'https://compute.staging.eyepop.xyz/v1/sessions/${SESSION_UUID}/token' \
  --header 'Accept: application/json' \
  --header 'Authorization: Bearer $EYEPOP_API_KEY'
```

#### Delete Session

```bash
curl --location --request DELETE 'https://compute.staging.eyepop.xyz/v1/sessions/${SESSION_UUID}' \
  --header 'Authorization: Bearer $EYEPOP_API_KEY'
```

### 5. Configuration Options

Override default settings via environment variables:

```bash
# Session health check timeout (default: 60 seconds)
export EYEPOP_COMPUTE_SESSION_TIMEOUT=120

# Health check poll interval (default: 2 seconds)
export EYEPOP_COMPUTE_SESSION_INTERVAL=5

# Default compute URL (default: https://compute.staging.eyepop.xyz)
export EYEPOP_COMPUTE_DEFAULT_COMPUTE_URL=https://compute.prod.eyepop.xyz
```

Or in Python:

```python
from eyepop.settings import settings

# View current settings
print(f"Session timeout: {settings.session_timeout}s")
print(f"Poll interval: {settings.session_interval}s")
```

### 6. Error Handling

The SDK uses custom exceptions for Compute API errors:

```python
from eyepop import EyePopSdk
from eyepop.exceptions import (
    ComputeSessionException,
    ComputeTokenException,
    ComputeHealthCheckException
)

try:
    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        result = await (await endpoint.upload('example.jpg')).predict()
        print(result)
except ComputeSessionException as e:
    print(f"Session creation failed: {e}")
    print(f"Session UUID: {e.session_uuid}")
except ComputeTokenException as e:
    print(f"Token operation failed: {e}")
except ComputeHealthCheckException as e:
    print(f"Health check failed: {e}")
    print(f"Last status: {e.last_status}")
```

### 7. Debugging

Enable debug logging to see Compute API interactions:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('eyepop.compute').setLevel(logging.DEBUG)
logging.getLogger('eyepop.requests').setLevel(logging.DEBUG)
```

You'll see logs like:

```
DEBUG:eyepop.compute:Fetching sessions from: https://compute.staging.eyepop.xyz/v1/sessions
DEBUG:eyepop.compute:Session endpoint: https://session-abc123.compute.eyepop.xyz
DEBUG:eyepop.compute:Access token expires in: 3600s
DEBUG:eyepop.requests:Fetching compute API session
DEBUG:eyepop.requests:Compute session ready: https://session-abc123.compute.eyepop.xyz
```

### 8. Migration from Worker API

To migrate from Worker API to Compute API:

**Before (Worker API):**
```python
from eyepop import EyePopSdk
import os

async with EyePopSdk.workerEndpoint(
    secret_key=os.getenv("EYEPOP_SECRET_KEY"),
    pop_id="my-named-pop",
    is_async=True
) as endpoint:
    result = await (await endpoint.upload('example.jpg')).predict()
```

**After (Compute API):**
```python
from eyepop import EyePopSdk
import os

async with EyePopSdk.workerEndpoint(
    api_key=os.getenv("EYEPOP_API_KEY"),  # Changed
    pop_id="transient",  # Changed
    eyepop_url="https://compute.staging.eyepop.xyz",  # Added
    is_async=True
) as endpoint:
    result = await (await endpoint.upload('example.jpg')).predict()
```

**Key Differences:**
- Use `EYEPOP_API_KEY` instead of `EYEPOP_SECRET_KEY`
- Set `pop_id="transient"` (Compute API only supports transient mode)
- Point to compute URL (or set via `EYEPOP_URL`)
- SDK handles session creation and token refresh automatically
