import logging
import mimetypes
import sys
import time
from pathlib import Path

from eyepop import EyePopSdk
from eyepop.logging import configure_logging

example_file_1 = sys.argv[1]

# Configure logging at DEBUG level
configure_logging(level='DEBUG')


def upload_streaming(location: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        with open(location, 'rb') as f:
            job = endpoint.upload_stream(stream=f, mime_type=mimetypes.guess_type(location)[0])
            while result := job.predict():
                print(result)



t1 = time.time()
upload_streaming(example_file_1)
t2 = time.time()
print("1x streaming file: ", t2 - t1)


