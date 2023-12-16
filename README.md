# EyePop.ai Python SDK
The EyePop.ai Python SDK provides convenient access to the EyePop.ai's inference API from applications written in the 
Python language. 

## Requirements 
* Python 3.6+ (PyPy supported)

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

## Base Usage
After the installation step a setting the environment variables you can execute the follwoing examples.
### Uploading and processing one single image
```python
from eyepop.eyepopsdk import EyePopSdk

def upload_one(file_path:str):
    with EyePopSdk.connect() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)

upload_one('examples/examples.jpg')
```
1. `EyePopSdk.connect()` authenticates with the Api Key found in `EYEPOP_SECRET_KEY` and loads the worker configuration 
for the Pop identified by `EYEPOP_POP_ID`. By default, this call will start a worker if the Pop was *paused*.
2. `endpoint.upload('examples/examples.jpg')` initiates the upload to the local file to the worker service. The image will
be queued and processed immediately when the worker becomes available.
3. `predict()` waits for the first prediction result as reports it as a dict. In case of a single image, there will be 
one single prediction result and subsequent calls to `predict()` will return `None`. If the uploaded file is a video
or container format (e.g. `image/gif`), subsequent calls to `predict()` will return a prediction for each individual frame. 
### Asynchronous uploading and processing of images
The above _synchronous_ example is concise and great for individual images or sequential uploads that don't need to be 
optimized for throughput. For high throughput applications, consider using the `async` variant which will addresses the 
HTTP transport latency better and will lead to a better utilization of your worker(s).
```python
import asyncio
from eyepop.eyepopsdk import EyePopSdk

async def upload_many(file_paths: list[str]):
    async with EyePopSdk.connect(is_async=True) as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.push(await endpoint.upload(file_path))
        for job in jobs:
            result = await job.predict()            
            print(result)

asyncio.run(upload_many([...]))
```
### Loading images from URLs
Instead of uploading files, you can also submit a publically accessible URL for processing.
```python
from eyepop.eyepopsdk import EyePopSdk

def load_from_url(url:str):
    with EyePopSdk.connect() as endpoint:
        result = endpoint.load_from(url).predict()
        print(result)

load_from_url('https://farm2.staticflickr.com/1080/1301049949_532835a8b5_z.jpg')
```
### Processing Videos 
You can process videos via upload or public URLs. This example shows how to process all frames via a public URL. 
Both the synchronous as well as the asynchronous version works for this use case. As for images, consider the 
asynchronous version if you are processing later set's if data and throughput is important. 
```python
from eyepop.eyepopsdk import EyePopSdk

def load_video_from_url(url:str):
    with EyePopSdk.connect() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)

load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```
## Other Usage Options
### Pop and Authentication Configuration
Instead of setting environment variables, you can pass Pop Id and Secret Key in code. Parameters to `connect()` take 
precedence over environment variables. This enables you to establish multipe connections to different Pop workers,
potentially using different credentials in the same process.
```python
from eyepop.eyepopsdk import EyePopSdk

EyePopSdk.connect(pop_id='...', secret_key='')
```
### Initialization Behaviour
#### Auto start workers
By default, `EyePopSdk.connect()` will start a worker if none is running. You can change this behavior by passing 
`EyePopSdk.connect(auto_start=False)`.
#### Stop pending jobs
By default, `EyePopSdk.connect()` will cancel all current or queued jobs at the assigned worker. It is assumed that the 
caller _takes control_ of that worker. To avoid this behavior pass `EyePopSdk.connect(stop_jobs=False)`.



