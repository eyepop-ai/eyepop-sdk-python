"""Runs a worker endpoint in pipeline mode: the same calls as with a worker session, on the pipeline API.

Run it with EYEPOP_API_KEY set and EYEPOP_URL naming the compute API, for instance
`python examples/pipeline_mode.py examples/example.jpg examples/bus.jpg`.
"""
import asyncio
import sys

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PERSON = Pop(components=[InferenceComponent(ability='eyepop.person:latest')])


async def main(paths: list[str]):
    async with EyePopSdk.async_worker(is_pipeline_mode=True, pop=PERSON) as endpoint:
        # Every job goes out at once on the one socket; the pipeline runs them in order.
        jobs = [await endpoint.upload(path) for path in paths]
        for path, job in zip(paths, jobs, strict=True):
            while (result := await job.predict()) is not None:
                print(path, len(result.get('objects') or []), 'objects')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:] or ['examples/example.jpg']))
