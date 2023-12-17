import asyncio
import logging
import time
from pathlib import Path

from eyepop.eyepopsdk import EyePopSdk
from eyepop.jobs import Job

source_path = Path(__file__).resolve()
source_dir = source_path.parent

example_image_path = f'{source_dir}/example.jpg'
example_image_paths = [example_image_path] * 100

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


def upload_photo(file_path: str):
    with EyePopSdk.connect() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)


async def async_upload_photo(file_path: str):
    async def on_ready(job: Job):
        print('on_ready', job.job_type, job.location)
        print(await job.predict())

    async with EyePopSdk.connect(is_async=True) as endpoint:
        await endpoint.upload(file_path, on_ready)


def upload_photos_sequentially(file_paths: list[str]):
    with EyePopSdk.connect() as endpoint:
        for file_path in file_paths:
            endpoint.upload(file_path).predict()


def upload_photos(file_paths: list[str]):
    with EyePopSdk.connect() as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(endpoint.upload(file_path))
        for job in jobs:
            job.predict()


async def async_upload_photos(file_paths: list[str]):
    async def on_ready(job: Job):
        await job.predict()

    async with EyePopSdk.connect(is_async=True) as endpoint:
        for file_path in file_paths:
            await endpoint.upload(file_path, on_ready)


t1 = time.time()
upload_photo(example_image_path)
t2 = time.time()
print("1x photo sync: ", t2 - t1)

t1 = time.time()
asyncio.run(async_upload_photo(example_image_path))
t2 = time.time()
print("1x photo async: ", t2 - t1)

t1 = time.time()
upload_photos_sequentially(example_image_paths)
t2 = time.time()
print(len(example_image_paths), "x photo sync: ", t2 - t1)

t1 = time.time()
upload_photos(example_image_paths)
t2 = time.time()
print(len(example_image_paths), "x photo sync: ", t2 - t1)

t1 = time.time()
asyncio.run(async_upload_photos(example_image_paths))
t2 = time.time()
print(len(example_image_paths), "x photo async: ", t2 - t1)
