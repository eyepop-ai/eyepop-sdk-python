---
description: Credentials and environment variables
icon: gear
---

# Configuration

The SDK reads your API key from `EYEPOP_API_KEY`. Create one in the [dashboard](https://dashboard.eyepop.ai).

```shell
export EYEPOP_API_KEY=eyp_...
```

You can pass it directly instead, though the environment variable keeps it out of your source:

```python
endpoint = EyePopSdk.sync_worker(api_key="eyp_...", pop=pop)
```

If your backend already holds a short-lived token, pass it as `access_token=` instead of an API key. In pipeline mode the token is sent to the pipeline API as given. On a worker session (`is_pipeline_mode=False`) a token targets `https://api.eyepop.ai`, the legacy worker API. On a worker session the Compute API requires an API key — pointing `eyepop_url` at it with only a token raises `Compute API endpoint (https://compute.eyepop.ai) requires EYEPOP_API_KEY` — so `session_uuid` and `EYEPOP_SESSION_UUID` are ignored on the token path, and a persistent Deployment needs `api_key`. In pipeline mode `session_uuid` raises `ValueError`.

### Optional variables

| Variable | Description |
| --- | --- |
| `EYEPOP_PIPELINE_MODE` | Worker endpoints run on the pipeline API by default. Set to `false` to run on a worker session instead, which the three variables below and the sections on persistent Deployments and local mode need. A named Pop, a session UUID or a secret key raises `ValueError` in pipeline mode. |
| `EYEPOP_SECRET_KEY` | Worker session only (`is_pipeline_mode=False`): secret key authentication, which works with named Pops. In pipeline mode a secret key alone raises `KeyError`. `EYEPOP_API_KEY` does not, and has to be unset — the SDK reads both, and a named Pop with an API key present raises `EYEPOP_API_KEY can only be used with transient pops`. |
| `EYEPOP_POP_ID` | Worker session only: named Pop to run. Defaults to `transient`. A named Pop raises `ValueError` in pipeline mode. |
| `EYEPOP_SESSION_UUID` | Worker session only: attach to a persistent Deployment instead of creating a transient session. Raises `ValueError` in pipeline mode. |
| `EYEPOP_ACCOUNT_UUID` | The account to work in. Required for some Data API calls, and names the account a worker's compute session runs for when the API key does not (same as `account_id=`). `EYEPOP_ACCOUNT_ID` is still read when this is unset, with a `DeprecationWarning`. |
| `EYEPOP_URL` | Override the API base URL. |
| `EYEPOP_PIPELINE_WAIT_TIMEOUT` | Pipeline mode: how many seconds a refused socket (HTTP 503 or 429 from a pipeline API at capacity) is retried, waiting the `Retry-After` it names plus jitter, before `PipelineOverloadedException`. Defaults to 600; `0` does not wait. The first wait is logged at `WARNING`. A script or test suite that should fail fast sets `0` or a few seconds. |
| `EYEPOP_LOG_LEVEL` | Log verbosity for the `eyepop` logger tree. Unset, the SDK installs no handler of its own and its records fall through to your application's logging configuration. `eyepop.requests` stays at `WARNING` unless the level is `DEBUG`; `LOG_LEVEL` is accepted as a fallback. |

### Transient and persistent sessions

A worker session is deprecated: opening one warns with a `DeprecationWarning`, and pipeline mode replaces it. This section and the next describe it, so they pass `is_pipeline_mode=False` or set `EYEPOP_PIPELINE_MODE=false`. In the default pipeline mode there is no session: the endpoint opens a socket to the pipeline API and sets your `pop` on it.

With no session UUID and a `pop` — the flow every example here uses — the SDK creates a new **transient** session each time it connects, and deletes the pipeline it created on exit. It does not delete the session itself. This is the right default for building and testing.

With no `pop`, the SDK reuses an existing non-persistent session when one is available, and leaves both it and its pipeline untouched on exit.

To run against a persistent Deployment, pass its session UUID. The Pop is fixed when the Deployment is created, so you do not pass one:

```python
with EyePopSdk.sync_worker(session_uuid="<your-session-uuid>", is_pipeline_mode=False) as endpoint:
    result = endpoint.upload("photo.jpg").predict()
```

### Local mode

An [on-premise instance](https://docs.eyepop.ai/developer-documentation/cli/on-premise) serves the EyePop runtime on your own machine. **Local mode** points the SDK at the instance on `http://127.0.0.1:8080` instead of the cloud and sends no account credentials — the instance is already registered to your account, and reaching the loopback port is what authorizes the client.

The instance serves one Pop, chosen when it is set up with `eyepop instance init` and changed with `eyepop instance set`, so you do not pass one from the SDK. This needs `eyepop` 3.25.2 or later.

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker(is_local_mode=True, is_pipeline_mode=False) as endpoint:
    result = endpoint.upload("image.jpg").predict()
    print(result)
```

`EYEPOP_LOCAL_MODE=true` in the environment selects it without the argument. The snippets here pass `is_pipeline_mode=False` (or set `EYEPOP_PIPELINE_MODE=false`) to reach the instance, on port 8080 by default; in the default pipeline mode local mode means a local pipeline API on `http://127.0.0.1:9002` instead, not an instance. Pass `eyepop_url` for an instance on another port or host:

```python
endpoint = EyePopSdk.sync_worker(
    is_local_mode=True,
    is_pipeline_mode=False,
    eyepop_url="http://127.0.0.1:9090",
)
```

The endpoint runs on the instance's own pipeline and leaves it running when it disconnects. Reuse one connected endpoint for many images rather than connecting for each.

### Next steps

* [Running Inference](inference.md) — process files, streams, URLs, and video
* [Composable Pops](composable-pops.md) — chain models into a pipeline
* [On-Premise Instances](https://docs.eyepop.ai/developer-documentation/cli/on-premise) — set up an instance with the CLI to run local mode against
