import asyncio
import logging
import time
from pathlib import Path

from eyepop.eyepopsdk import EyePopSdk

source_path = Path(__file__).resolve()
source_dir = source_path.parent

example_image_path = f'{source_dir}/example.jpg'
example_image_paths = []
for i in range(100):
    example_image_paths.append(example_image_path)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


async def async_upload_photo(file_path: str):
    async with EyePopSdk.connect(is_async=True) as endpoint:
        job = await endpoint.upload(file_path)
        result = await job.predict()
        print(result)


def sync_upload_photo(file_path: str):
    with EyePopSdk.connect() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)


async def async_upload_photos(file_paths: list[str]):
    async with EyePopSdk.connect(is_async=True) as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(await endpoint.upload(file_path))
        for job in jobs:
            await job.predict()


def sync_upload_photos(file_paths: list[str]):
    with EyePopSdk.connect() as endpoint:
        for file_path in file_paths:
            endpoint.upload(file_path).predict()


t1 = time.time()
sync_upload_photo(example_image_path)
t2 = time.time()
print("1x photo sync: ", t2 - t1)

t1 = time.time()
asyncio.run(async_upload_photo(example_image_path))
t2 = time.time()
print("1x photo async: ", t2 - t1)

t1 = time.time()
sync_upload_photos(example_image_paths)
t2 = time.time()
print(len(example_image_paths), "x photo sync: ", t2 - t1)

t1 = time.time()
asyncio.run(async_upload_photos(example_image_paths))
t2 = time.time()
print(len(example_image_paths), "x photo async: ", t2 - t1)
