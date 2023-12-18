# EyePop.ai Python SDK
The EyePop.ai Python SDK provides convenient access to the EyePop.ai's inference API from applications written in the 
Python language. 

## Requirements 
* Python 3.8+

## Install
```shell
pip install eyepop-sdk-python
```

## Configuration
The EyePop SDK needs to be configured with the __Pop Id__ and your __Secret Api Key__. The SDK will read the following 
environment variables:
* `EYEPOP_POP_ID`: The Pop Id to use as an endpoint. You can copy'n paste this string from your EyePop Dashboard in the Pop -> Settings section.
* `EYEPOP_SECRET_KEY`: Your Secret Api Key. You can create Api Keys in the profile section of youe EyePop dashboard.
* `EYEPOP_URL`: (Optional) URL of the EyePop API service, if you want to use any other endpoint than production `http://api.eyepop.ai`  

## Usage Examples
After the installation step a setting the environment variables you can execute the following examples.
### Uploading and processing one single image

```python
from eyepop.eyepopsdk import EyePopSdk


def upload_photo(file_path: str):
    with EyePopSdk.endpoint() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)


upload_photo('examples/examples.jpg')
```
1. `EyePopSdk.endpoint()` returns a local endpoint object, that will authenticate with the Api Key found in 
`EYEPOP_SECRET_KEY` and load the worker configuration for the Pop identified by `EYEPOP_POP_ID`. 
By default, `with ... endpoint:` will initiate the connection and start a worker if the Pop was *paused*.
2. `endpoint.upload('examples/examples.jpg')` initiates the upload to the local file to the worker service. The image will
be queued and processed immediately when the worker becomes available.
3. `predict()` waits for the first prediction result as reports it as a dict. In case of a single image, there will be 
one single prediction result and subsequent calls to `predict()` will return `None`. If the uploaded file is a video
e.g. 'video/mp4' or image container format e.g. 'image/gif', subsequent calls to `predict()` will return a prediction 
for each individual frame and `None` when the entire file has been processed. 
### Uploading and processing batches of images
For batches of images, instead of waiting for each result `predict()` _before_ submitting the next job, you can queue 
all jobs first, let them process in parallel and collect the results later. This avoids the sequential accumulation of 
the HTTP roundtrip time.

```python
from eyepop.eyepopsdk import EyePopSdk


def upload_photos(file_paths: list[str]):
    with EyePopSdk.endpoint() as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(endpoint.upload(file_path))
        for job in jobs:
            print(job.predict())


upload_photos(['examples/examples.jpg'] * 100)
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
from eyepop.eyepopsdk import EyePopSdk
from eyepop.jobs import Job


async def async_upload_photos(file_paths: list[str]):
    async def on_ready(job: Job):
        print(await job.predict())

    async with EyePopSdk.endpoint(is_async=True) as endpoint:
        for file_path in file_paths:
            await endpoint.upload(file_path, on_ready)


asyncio.run(async_upload_photos(['examples/examples.jpg'] * 100000000))
```
### Loading images from URLs
Alternatively to uploading files, you can also submit a publicly accessible URL for processing. This works for both,
synchronous and asynchronous mode. Supported protocols are:
* HTTP(s) URLs with response Content-Type image/* or video/*   
* RTSP (live streaming)
* RTMP (live streaming)

```python
from eyepop.eyepopsdk import EyePopSdk


def load_from_url(url: str):
    with EyePopSdk.endpoint() as endpoint:
        result = endpoint.load_from(url).predict()
        print(result)


load_from_url('https://farm2.staticflickr.com/1080/1301049949_532835a8b5_z.jpg')
```
### Processing Videos 
You can process videos via upload or public URLs. This example shows how to process all video frames of a file 
retrieved from a public URL. This works for both, synchronous and asynchronous mode.

```python
from eyepop.eyepopsdk import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.endpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```
### Canceling Jobs
Any job that is queued up or in ready and has not been completed can be cancelled. E.g. stop the video processing after
prediction have been processed for 10 seconds of video.

```python
from eyepop.eyepopsdk import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.endpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)
            if result['seconds'] >= 10.0:
                job.cancel()


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```

## Other Usage Options
### Pop and Authentication Configuration
Instead of setting environment variables, you can pass Pop Id and Secret Key in code. Parameters to `connect()` take 
precedence over environment variables. This enables you to establish multiple connections to different Pop workers,
potentially using different credentials in the same process.

```python
from eyepop.eyepopsdk import EyePopSdk

EyePopSdk.endpoint(pop_id='...', secret_key='')
```
### Initialization Behaviour
#### Auto start workers
By default, `EyePopSdk.endpoint().connect()` will start a worker if none is running. To avoid this behavior create an 
endpoint with `EyePopSdk.endpoint(auto_start=False)`.
#### Stop pending jobs
By default, `EyePopSdk.endpoint().connect()` will cancel all current or queued jobs at the assigned worker. 
It is assumed that the caller _takes control_ of that worker. To avoid this behavior create an endpoint with 
`EyePopSdk.endpoint(stop_jobs=False)`.



