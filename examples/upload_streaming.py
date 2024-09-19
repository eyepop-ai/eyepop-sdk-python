import logging
import mimetypes
import sys
import time
from pathlib import Path

from eyepop import EyePopSdk

example_file_1 = sys.argv[1]

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)


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


