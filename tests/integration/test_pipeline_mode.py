"""Pipeline mode against a real pipeline API (SDK-24): eyepop-instance spec 019 T051's claims through WorkerEndpoint.

The compute API at EYEPOP_URL names the pipeline API in its /v1/configs. The
live url is read by the pipeline API, so it may be a cluster-internal address.
"""
import asyncio
import os
from importlib import resources
from pathlib import Path

import pytest

import tests
from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

TEST_IMAGE = str(resources.files(tests) / "test.jpg")
VIDEO = Path(__file__).parents[2] / "examples" / "temporal_context" / "media" / "office_walk_by.mp4"
LIVE_URL = os.getenv("EYEPOP_TEST_LIVE_URL", "rtsp://stream-sim.stream-sim:8554/cam-01")
MULTI_IMAGE_ABILITY = os.getenv("EYEPOP_MULTI_IMAGE_ABILITY", "eyepop.vlm.image:latest")
PERSON = Pop(components=[InferenceComponent(ability="eyepop.person:latest")])
# test.jpg holds at least 10 people
MIN_PERSONS = 10

pytestmark = pytest.mark.skipif(not os.getenv("EYEPOP_API_KEY"), reason="EYEPOP_API_KEY environment variable not set")


def persons(result: dict | None) -> int:
    assert result is not None
    return sum(1 for o in result.get("objects") or [] if o.get("classLabel") == "person")


async def all_of(job) -> list[dict]:
    results = []
    while (result := await job.predict()) is not None:
        results.append(result)
    return results


async def test_one_image():
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=PERSON) as endpoint:
        [result] = await all_of(await endpoint.upload(TEST_IMAGE))
    assert persons(result) >= MIN_PERSONS


async def test_a_combined_batch_on_one_socket():
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=PERSON) as endpoint:
        jobs = [await endpoint.upload(TEST_IMAGE) for _ in range(5)]
        results = await asyncio.gather(*(all_of(job) for job in jobs))
    assert [len(r) for r in results] == [1] * 5
    assert all(persons(r[0]) >= MIN_PERSONS for r in results)


async def test_a_group_is_one_inference():
    pop = Pop(components=[InferenceComponent(
        ability=MULTI_IMAGE_ABILITY, params={"prompt": "Describe these images together in one sentence."})])
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=pop) as endpoint:
        results = await all_of(await endpoint.upload_group([TEST_IMAGE, TEST_IMAGE]))
    assert len(results) == 1


async def test_a_chunked_video_predicts_every_frame_to_its_end():
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=PERSON) as endpoint:
        with VIDEO.open("rb") as video:
            results = await all_of(await endpoint.upload_stream(video, "video/mp4"))
    assert len(results) > 10
    assert any(persons(r) > 0 for r in results)


async def test_a_live_url_keeps_predicting():
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=PERSON) as endpoint:
        job = await endpoint.load_from(LIVE_URL)
        results = [await asyncio.wait_for(job.predict(), 60) for _ in range(50)]
        await job.cancel()
    assert all(r is not None for r in results)
