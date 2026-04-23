# EyePop SDK Examples

Runnable scripts demonstrating common SDK workflows.

## Setup

```shell
cd examples
python3 -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
```

To run against your working copy of the SDK instead of PyPI:

```shell
pip install -e ..
```

## Credentials

Get an API key at [dashboard.eyepop.ai](https://dashboard.eyepop.ai), then either:

```shell
export EYEPOP_API_KEY=...
```

…or copy `../.env.example` to `.env` and fill it in (all examples call `load_dotenv()`).

## Running

```shell
python pop_demo.py --pop person --output --local-path ./example.jpg
```

Every example with CLI flags accepts `-h` / `--help`.

## What's here

| Script | Purpose |
|---|---|
| `pop_demo.py` | Compose multi-stage Pops (vehicles, faces, hands, SAM, VLM, tracking). The big one. |
| `infer_demo.py` | Run VLM inference and dataset evaluation via the Data API. |
| `caption_demo.py` | Generate captions via VLM/LLM. |
| `upload_video.py` / `download_video.py` | Upload a local video; download a (trimmed) video asset. |
| `upload_streaming.py` | Upload a binary stream with MIME type. |
| `upload_image_timing.py` | Measure per-image upload + predict latency. |
| `load_video_from_http.py` | Process a video by URL. |
| `live_rtmp_stream.py` | Process a live RTMP stream. |
| `visualize_on_image.py` | Overlay predictions on an image with matplotlib. |
| `visualize_with_webui2.py` | Interactive web viewer for predictions. |
| `import_dataset.py` | Import local assets into a dataset. |
| `auth_session.py` | Browser-based OAuth session. |
| `workflow_cli.py` | Argo workflow CLI helper. |
