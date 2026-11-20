import mimetypes
import sys
import time

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PERSON = Pop(components=[InferenceComponent(ability='eyepop.person:latest')])

example_file_1 = sys.argv[1]


def upload_streaming(location: str):
    with EyePopSdk.sync_worker(pop=PERSON) as endpoint:
        with open(location, 'rb') as f:
            job = endpoint.upload_stream(stream=f, mime_type=mimetypes.guess_type(location)[0])
            while result := job.predict():
                print(result)



t1 = time.time()
upload_streaming(example_file_1)
t2 = time.time()
print("1x streaming file: ", t2 - t1)


