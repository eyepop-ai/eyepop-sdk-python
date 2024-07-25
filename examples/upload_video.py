import asyncio
import logging
import time
from pathlib import Path

from eyepop import EyePopSdk

example_file_1 = '/home/torsten/Downloads/remux_pet_owner.mp4.mp4'

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


def upload_video(location: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.upload(location)
        while result := job.predict():
            print(result)



t1 = time.time()
upload_video(example_file_1)
t2 = time.time()
print("1x video async: ", t2 - t1)


