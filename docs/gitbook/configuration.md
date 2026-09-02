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
endpoint = EyePopSdk.sync_worker(api_key="eyp_...")
```

If your backend already holds a short-lived token, pass it as `access_token=` instead of an API key.

### Optional variables

| Variable | Description |
| --- | --- |
| `EYEPOP_SESSION_UUID` | Attach to a persistent Deployment instead of creating a transient session. |
| `EYEPOP_ACCOUNT_ID` | Required for some Data API calls. |
| `EYEPOP_URL` | Override the API base URL. |
| `EYEPOP_LOG_LEVEL` | Log verbosity for the `eyepop` logger. |

### Transient and persistent sessions

With no session UUID the SDK creates a **transient** session on connect and releases it on exit — the right default for building and testing.

To run against a persistent Deployment, pass its session UUID. The Pop is fixed when the Deployment is created, so you do not pass one:

```python
with EyePopSdk.sync_worker(session_uuid="<your-session-uuid>") as endpoint:
    result = endpoint.upload("photo.jpg").predict()
```

### Next steps

* [Running Inference](inference.md) — process files, streams, URLs, and video
* [Composable Pops](composable-pops.md) — chain models into a pipeline
