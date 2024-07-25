import sys
import asyncio
import logging
import time

from eyepop import EyePopSdk
from eyepop import Job
from eyepop.worker.worker_endpoint import WorkerEndpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger('eyepop').setLevel(level=logging.DEBUG)
logging.getLogger('eyepop.metrics').setLevel(level=logging.DEBUG)


async def async_load_from_photos(endpoint: WorkerEndpoint, urls: list[str]):
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

    n = 0
    for url in urls:
        await endpoint.load_from(url, on_ready=on_ready)
        n += 1
    for i in range(n):
        await sem.acquire()


async def run_test_async(urls: list[str]):
    t0 = time.time()
    async with EyePopSdk.workerEndpoint(is_async=True, pop_id='transient', job_queue_length=10124) as endpoint:
        await endpoint.set_pop_comp('ep_infer model=eyepop-person:EPPersonB1_Person_TorchScriptCpu_float32')
        t1 = time.time()
        await async_load_from_photos(endpoint, urls)
    t2 = time.time()

    print("%d x photo async took %.3f seconds after %.3f seconds connect time\n\n" % (len(example_image_urls), (t2 - t1), (t1 - t0)))

example_image_url = sys.argv[1]
example_image_urls = [example_image_url] * int(sys.argv[2])

asyncio.run(run_test_async(example_image_urls))
