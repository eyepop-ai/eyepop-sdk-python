import asyncio
import logging
import time
from pathlib import Path

from eyepop.eyepopsdk import EyePopSdk

source_path = Path(__file__).resolve()
source_dir = source_path.parent
example_url_1 = 'https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4'

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


async def async_load_from_video(url: str):
    async with EyePopSdk.connect(is_async=True) as endpoint:
        job = await endpoint.load_from(url)
        while result := await job.predict():
            print(result)


def sync_load_from_video(url: str):
    with EyePopSdk.connect() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)


t1 = time.time()
sync_load_from_video(example_url_1)
t2 = time.time()
print("1x video sync: ", t2 - t1)

t1 = time.time()
asyncio.run(async_load_from_video(example_url_1))
t2 = time.time()
print("1x video async: ", t2 - t1)
