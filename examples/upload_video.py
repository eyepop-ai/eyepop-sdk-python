import asyncio
import logging
import sys
import time
from pathlib import Path

from eyepop import EyePopSdk
from eyepop.worker.worker_types import VideoMode

example_file_1 = sys.argv[1]

is_streaming = len(sys.argv) > 2 and sys.argv[2].startswith("s")

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


def upload_video(location: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.upload(
            location=location,
            video_mode=VideoMode.STREAM if is_streaming else None,
        )
        while result := job.predict():
            print(result)



t1 = time.time()
upload_video(example_file_1)
t2 = time.time()
print("1x video async: ", t2 - t1, " (streaming, no buffer)" if is_streaming else "")


