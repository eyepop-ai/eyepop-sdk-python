import os
from importlib import resources

import pytest

import tests
from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

TEST_IMAGE = str(resources.files(tests) / 'test.jpg')
PUBLIC_TEST_IMAGE_URL = "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg"

# A multi-image-capable VLM caption ability. Override via env if the default is
# not multi-image-capable in the target environment.
MULTI_IMAGE_ABILITY = os.getenv("EYEPOP_MULTI_IMAGE_ABILITY", "eyepop.vlm.image:latest")
MULTI_IMAGE_PROMPT = os.getenv(
    "EYEPOP_MULTI_IMAGE_PROMPT", "Describe these images together in one sentence.")


def requires_api_key():
    return pytest.mark.skipif(
        not os.getenv("EYEPOP_API_KEY"),
        reason="EYEPOP_API_KEY environment variable not set",
    )


def caption_pop() -> Pop:
    return Pop(components=[
        InferenceComponent(ability=MULTI_IMAGE_ABILITY, params={"prompt": MULTI_IMAGE_PROMPT})
    ])


@requires_api_key()
def test_upload_group_single_inference_unit():
    # A group of images is one inference unit: exactly one prediction for the
    # whole group, not one per image.
    with EyePopSdk.sync_worker(pop_id="transient") as endpoint:
        endpoint.set_pop(caption_pop())

        job = endpoint.upload_group([TEST_IMAGE, TEST_IMAGE, TEST_IMAGE])
        result = job.predict()

        assert result is not None
        assert job.predict() is None


@requires_api_key()
@pytest.mark.asyncio
async def test_upload_group_single_inference_unit_async():
    async with EyePopSdk.async_worker(pop_id="transient") as endpoint:
        await endpoint.set_pop(caption_pop())

        job = await endpoint.upload_group([TEST_IMAGE, TEST_IMAGE])
        result = await job.predict()

        assert result is not None
        assert await job.predict() is None


@requires_api_key()
def test_upload_stream_group_single_inference_unit():
    with EyePopSdk.sync_worker(pop_id="transient") as endpoint:
        endpoint.set_pop(caption_pop())

        with open(TEST_IMAGE, 'rb') as a, open(TEST_IMAGE, 'rb') as b:
            job = endpoint.upload_stream_group([a, b], mime_types=['image/jpeg', 'image/jpeg'])
            result = job.predict()

        assert result is not None
        assert job.predict() is None


@requires_api_key()
def test_load_from_group_single_inference_unit():
    with EyePopSdk.sync_worker(pop_id="transient") as endpoint:
        endpoint.set_pop(caption_pop())

        job = endpoint.load_from_group([PUBLIC_TEST_IMAGE_URL, PUBLIC_TEST_IMAGE_URL])
        result = job.predict()

        assert result is not None
        assert job.predict() is None
