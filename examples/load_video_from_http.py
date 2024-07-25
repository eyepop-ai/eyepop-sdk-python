import asyncio
import logging
import time
from pathlib import Path

from eyepop import EyePopSdk

source_path = Path(__file__).resolve()
source_dir = source_path.parent
example_url_1 = 'https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4'

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


def load_video_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)
            if result['seconds'] >= 10.0:
                job.cancel()


async def async_load_video_from_url(url: str):
    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        job = await endpoint.load_from(url)
        while result := await job.predict():
            print(result)
            if result['seconds'] >= 10.0:
                await job.cancel()


t1 = time.time()
asyncio.run(async_load_video_from_url(example_url_1))
t2 = time.time()
print("1x video async: ", t2 - t1)

t1 = time.time()
load_video_from_url(example_url_1)
t2 = time.time()
print("1x video sync: ", t2 - t1)

