import sys
import asyncio
import logging
import time

from eyepop import EyePopSdk
from eyepop import Job


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger('eyepop').setLevel(level=logging.DEBUG)
logging.getLogger('eyepop.metrics').setLevel(level=logging.DEBUG)


def upload_photos_sequentially(file_paths: list[str]):
    '''
    Sequential processing of batch uploads - simple but slowest option.
    '''
    with EyePopSdk.endpoint() as endpoint:
        for file_path in file_paths:
            job = endpoint.upload(file_path)
            while job.predict() is not None:
                pass


def upload_photos(file_paths: list[str]):
    '''
    Parallel processing of batch uploads - fast but limited by memory
    '''
    with EyePopSdk.endpoint() as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(endpoint.upload(file_path))
        for job in jobs:
            while job.predict() is not None:
                pass


async def async_upload_photos(file_paths: list[str]):
    '''
    Async processing of batch uploads - fast and memory efficient
    '''
    sem = asyncio.Semaphore(0)

    async def on_ready(job: Job):
        nonlocal sem
        try:
            while await job.predict() is not None:
                pass
        except Exception as e:
            logging.exception(e)
        finally:
            sem.release()

    async with EyePopSdk.endpoint(is_async=True, job_queue_length=512) as endpoint:
        n = 0
        for file_path in file_paths:
            await endpoint.upload(file_path, on_ready=on_ready)
            n += 1
        for i in range(n):
            await sem.acquire()


example_image_path = sys.argv[1]
example_image_paths = [example_image_path] * int(sys.argv[2])

t1 = time.time()
upload_photos_sequentially(example_image_paths)
t2 = time.time()
print("%d x photo sync took %.3f seconds\n\n" % (len(example_image_paths), (t2 - t1)))

t1 = time.time()
upload_photos(example_image_paths)
t2 = time.time()
print("%d x photo sync took %.3f seconds\n\n" % (len(example_image_paths), (t2 - t1)))

t1 = time.time()
asyncio.run(async_upload_photos(example_image_paths))
t2 = time.time()
print("%d x photo async took %.3f seconds\n\n" % (len(example_image_paths), (t2 - t1)))
