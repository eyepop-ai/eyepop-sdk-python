# EyePop.ai Python SDK
The EyePop.ai Python SDK provides convenient access to the EyePop.ai's inference API from applications written in the 
Python language. 

## Requirements 
* Python 3.8+

## Install
```shell
pip install eyepop
```

## Configuration
The EyePop SDK needs to be configured with the __Pop Id__ and your __Secret Api Key__.

```python
import os
from eyepop import EyePopSdk

endpoint = EyePopSdk.workerEndpoint(
    # This is the default and can be omitted
    pop_id=os.environ.get('EYEPOP_POP_ID'),
    # This is the default and can be omitted
    secret_key=os.environ.get('EYEPOP_SECRET_KEY'),
)

endpoint.connect()
# do work ....
endpoint.disconnect()
```
While you can provide a secret_key keyword argument, we recommend using python-dotenv to add EYEPOP_SECRET_KEY="My API Key" 
to your .env file so that your API Key is not stored in source control. By default, the SDK will read the following environment variables:
* `EYEPOP_POP_ID`: The Pop Id to use as an endpoint. You can copy'n paste this string from your EyePop Dashboard in the Pop -> Settings section.
* `EYEPOP_SECRET_KEY`: Your Secret Api Key. You can create Api Keys in the profile section of youe EyePop dashboard.
* `EYEPOP_URL`: (Optional) URL of the EyePop API service, if you want to use any other endpoint than production `http://api.eyepop.ai`  
## Usage Examples
  
### Uploading and processing one single image

```python
from eyepop import EyePopSdk


def upload_photo(file_path: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)


upload_photo('examples/example.jpg')
```
1. `EyePopSdk.workerEndpoint()` returns a local endpoint object, that will authenticate with the Api Key found in 
EYEPOP_SECRET_KEY and load the worker configuration for the Pop identified by EYEPOP_POP_ID. 
2. The usage of `with ... endpoint:` will automatically manage the runtime context, connect to the worker upon entering
the context and releasing all underlying resources upon exiting the context. Alternatively your code can call 
endpoint.connect() before any job is submitted and endpoint.disconnect() to release all resources.
2. `endpoint.upload('examples/example.jpg')` initiates the upload to the local file to the worker service. The image will
be queued and processed immediately when the worker becomes available.
3. `predict()` waits for the first prediction result as reports it as a dict. In case of a single image, there will be 
one single prediction result and subsequent calls to predict() will return None. If the uploaded file is a video
e.g. 'video/mp4' or image container format e.g. 'image/gif', subsequent calls to predict() will return a prediction 
for each individual frame and None when the entire file has been processed.

Note: since v0.19.0 `EyePopSdk.workerEndpoint()` was introduced and replaces `EyePopSdk.endpoint()` which is now deprecated. 
Support for `EyePopSdk.endpoint()` will be removed in v1.0.0.

To upload a binary stream, i.e. a file-like object, you can use the method `upload_stream()` and pass the file-like 
object and the mime-type:

```python
from eyepop import EyePopSdk


def upload_photo_from_stream(file_path: str, mime_type: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        with open(file_path, 'rb') as file:
            result = endpoint.upload_stream(file, mime_type).predict()
            print(result)


upload_photo_from_stream('examples/example.jpg', 'image/jpeg')
```

### Visualizing Results
The EyePop SDK includes helper classes to to visualize bounding boxes `matplotlib.pyplot`.

```python
from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk

with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload('examples/example.jpg').predict()
with Image.open('examples/example.jpg') as image:
    plt.imshow(image)
plot = EyePopSdk.plot(plt.gca())
plot.prediction(result)
plt.show()
```
Depending on the environment, you might need to install an interactive backend, e.g. with `pip3 install pyqt5`. 
EyePop's Python Sdk does not include visualization helpers for any other prediction types than object bounding boxes.
Check out [visualize_with_webui2.py](examples/visualize_with_webui2.py) for an example how to use the comprehensive visualization support provided by the EyePop Node Sdk.

### Uploading and processing batches of images
For batches of images, instead of waiting for each result `predict()` _before_ submitting the next job, you can queue 
all jobs first, let them process in parallel and collect the results later. This avoids the sequential accumulation of 
the HTTP roundtrip time.

```python
from eyepop import EyePopSdk


def upload_photos(file_paths: list[str]):
    with EyePopSdk.workerEndpoint() as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(endpoint.upload(file_path))
        for job in jobs:
            print(job.predict())


upload_photos(['examples/example.jpg'] * 100)
```
### Asynchronous uploading and processing of images
The above _synchronous_ way is great for individual images or reasonable sized batches. If your batch size is 'large'
this can cause memory and performance issues. Consider that `endpoint.upload()` is a very fast, local operation. 
In fact, it creates and schedules a task that will execute the 'slow' IO operations in the background. Consequently, 
when your code calls `enpoint.upload()` 1,000,000 times it will cause a background task list with ~ 1,000,000 entries. 
And the example code above will only start clearing out this list by receiving the result via `predict()` after the 
entire list was submitted.

For high throughput applications, consider using the `async` variant which supports a callback parameter `on_ready`. 
Within the callback, your code can process the results asynchronously and clearing the task list as soon as the results 
are available.

```python
import asyncio
from eyepop import EyePopSdk
from eyepop import Job


async def async_upload_photos(file_paths: list[str]):
    async def on_ready(job: Job):
        print(await job.predict())

    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        for file_path in file_paths:
            await endpoint.upload(file_path, on_ready)


asyncio.run(async_upload_photos(['examples/example.jpg'] * 100000000))
```
### Loading images from URLs
Alternatively to uploading files, you can also submit a publicly accessible URL for processing. This works for both,
synchronous and asynchronous mode. Supported protocols are:
* HTTP(s) URLs with response Content-Type image/* or video/*   
* RTSP (live streaming)
* RTMP (live streaming)

```python
from eyepop import EyePopSdk


def load_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        result = endpoint.load_from(url).predict()
        print(result)


load_from_url('https://farm2.staticflickr.com/1080/1301049949_532835a8b5_z.jpg')
```
### Processing Videos 
You can process videos via upload or public URLs. This example shows how to process all video frames of a file 
retrieved from a public URL. This works for both, synchronous and asynchronous mode.

```python
from eyepop import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```
### Canceling Jobs
Any job that has been queued or is in-progress can be cancelled. E.g. stop the video processing after
predictions have been processed for 10 seconds duration of the video.

```python
from eyepop import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)
            if result['seconds'] >= 10.0:
                job.cancel()


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```
## Other Usage Options
#### Auto start workers
By default, `EyePopSdk.workerEndpoint().connect()` will start a worker if none is running yet. To disable this behavior 
create an endpoint with `EyePopSdk.endpoint(auto_start=False)`.
#### Stop pending jobs
By default, `EyePopSdk.workerEndpoint().connect()` will cancel all currently running or queued jobs on the worker. 
It is assumed that the caller _takes full control_ of that worker. To disable this behavior create an endpoint with 
`EyePopSdk.workerEndpoint(stop_jobs=False)`.

## Data endpoint (experimental)
To support managing your own datasets and control model optimization v0.19.0 introduces `EyePopSdk.dataEndpoint()`,
an experimental pre-release which is subject to change. An officially supported version will be released with v1.0.0


