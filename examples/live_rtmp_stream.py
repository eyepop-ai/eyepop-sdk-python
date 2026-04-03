import asyncio
import time
from pathlib import Path

from eyepop import EyePopSdk, Job
from eyepop.worker.worker_jobs import WorkerJob

source_path = Path(__file__).resolve()
source_dir = source_path.parent
example_url_1 = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'


async def async_load_from_rtmp(url: str):
    async def on_ready(job: WorkerJob):
        print('async_load_from_rtmp on_ready')
        while result := await job.predict():
            print(result)

    async with EyePopSdk.async_worker() as endpoint:
        await endpoint.load_from(url, on_ready=on_ready)


t1 = time.time()
asyncio.run(async_load_from_rtmp(example_url_1))
t2 = time.time()
print("1x video async: ", t2 - t1)
