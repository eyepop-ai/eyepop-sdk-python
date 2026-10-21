import asyncio
import pathlib

from eyepop.pipeline import STOP, Ability, Model, Pop, batch, image, pipeline, run, stream, video

IMAGE = pathlib.Path(__file__).parent / "example.jpg"
IMAGE_URLS = [
    "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg",
    "https://farm1.staticflickr.com/22/29604617_a60740ce78_z.jpg",
]
VIDEO = "clip.mp4"
RTSP_URL = "rtsp://camera.local/stream"
MODEL_UUID = "your-model-uuid"

person = Ability("eyepop.person")
pose = person.crop(Ability("eyepop.person.2d-body-points"))


# a job: the decorated function gets each frame, and STOP ends it
@stream(person, RTSP_URL)
async def watch(frame):
    print(frame.timestamp, len(frame.objects))
    if frame.timestamp and frame.timestamp > 30e9:
        return STOP


async def main():
    # EYEPOP_PIPELINE_URL and EYEPOP_API_KEY, from the environment or .env
    prediction = await image(person, IMAGE)
    for obj in prediction.objects:
        print(obj.label, obj.confidence, obj.x, obj.y, obj.width, obj.height)

    print(await image(pose, IMAGE))
    print(await image(Model(MODEL_UUID), IMAGE))
    print(await image(Pop({"components": [{"type": "inference", "ability": "eyepop.person"}]}), IMAGE))

    async for result in batch(person, IMAGE_URLS):
        print(result.source, result.error or result.prediction)

    async for frame in video(person, VIDEO):
        print(frame.timestamp, len(frame.objects))

    await run(watch)

    # a pipeline to keep: change its pop while the camera runs, then queue a clip behind it
    async with pipeline(person) as p:
        async for frame in p.stream(RTSP_URL):
            if frame.objects:
                await p.set_pop(pose)
                break
        async for frame in p.video(VIDEO, mode="preempt"):
            print(frame.timestamp, len(frame.objects))
        print(await p.graph("dot"))


asyncio.run(main())
