"""WorkerJob.predict() hands back predictions and skips the events between them.

The pipeline writes an event line as ``{"event": "<type>", "source_id": ..., "info": ...}``
(eyepop-pipeline META-DATA.md, Event document), and an instance session forwards it as
is. New event types are added without a version bump (``warning`` since worker v4.19.11),
so a client must skip one it does not know. Only an error ends the stream.
"""
import asyncio
from typing import Any

import pytest

from eyepop.worker.worker_jobs import WorkerJob, WorkerSource


def job_with(messages: list[dict[str, Any] | None]) -> WorkerJob:
    job = WorkerJob(session=None, backend=None, source=WorkerSource(), on_ready=None)  # type: ignore[arg-type]
    for message in messages:
        job._queue.put_nowait(message)  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]
    return job


async def all_predictions(job: WorkerJob) -> list[dict[str, Any]]:
    predictions = []
    while (prediction := await asyncio.wait_for(job.predict(), timeout=5)) is not None:
        predictions.append(prediction)
    return predictions


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    {"event": "warning", "source_id": "s", "message": "no keyframe for 12 s",
     "info": {"code": "media_cache_gop_too_long", "level": "warn", "gap_ms": 12000, "threshold_ms": 10000}},
    {"event": "usage", "source_id": "s", "info": {"milli_compute_units": 3, "usage": {"video_frames_decoded": 10}}},
    {"event": "start", "source_id": "s"},
    {"event": "eos", "source_id": "s"},
    {"event": "a_type_no_sdk_knows_yet", "source_id": "s", "info": {"anything": [1, 2]}},
    {"event": {"type": "prepared", "source_id": "s"}},
    {"event": {"type": "recording", "mediaId": "v2.opaque"}},
])
async def test_an_event_between_predictions_is_skipped(event: dict[str, Any]):
    first, second = {"seconds": 0.0, "objects": []}, {"seconds": 1.0, "objects": []}
    job = job_with([first, event, second, None])

    assert await all_predictions(job) == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    {"event": "error", "source_id": "s", "message": "cannot open the camera"},
    {"event": {"type": "error", "source_id": "s", "message": "cannot open the camera"}},
])
async def test_an_error_event_still_raises(error: dict[str, Any]):
    job = job_with([{"seconds": 0.0}, error, None])

    assert await job.predict() == {"seconds": 0.0}
    with pytest.raises(ValueError, match="Error in source s: cannot open the camera"):
        await job.predict()
