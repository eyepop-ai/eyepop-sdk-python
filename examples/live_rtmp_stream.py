import asyncio
import logging
import time
from pathlib import Path

from eyepop import EyePopSdk
from eyepop import Job

source_path = Path(__file__).resolve()
source_dir = source_path.parent
example_url_1 = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


async def async_load_from_rtmp(url: str):
    async def on_ready(job: Job):
        print('async_load_from_rtmp on_ready')
        while result := await job.predict():
            print(result)

    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        await endpoint.load_from(url, on_ready)


t1 = time.time()
asyncio.run(async_load_from_rtmp(example_url_1))
t2 = time.time()
print("1x video async: ", t2 - t1)
