import asyncio
import contextlib
import pathlib

from eyepop.pipeline import Ability, EyePop, Model, Pop

IMAGE = pathlib.Path(__file__).parent / "example.jpg"
IMAGE_URLS = [
    "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg",
    "https://farm1.staticflickr.com/22/29604617_a60740ce78_z.jpg",
]
VIDEO = "clip.mp4"
RTSP_URL = "rtsp://camera.local/stream"
MODEL_UUID = "your-model-uuid"


async def main():
    # EYEPOP_PIPELINE_URL and EYEPOP_API_KEY, from the environment or .env
    async with EyePop.init() as ep:
        person = ep.use(Ability("eyepop.person"))

        prediction = await person.image(IMAGE)
        for obj in prediction.objects:
            print(obj.label, obj.confidence, obj.x, obj.y, obj.width, obj.height)

        pose = ep.use(Ability("eyepop.person").crop(Ability("eyepop.person.2d-body-points")))
        print(await pose.image(IMAGE))
        print(await ep.use(Model(MODEL_UUID)).image(IMAGE))
        print(await ep.use(Pop({"components": [{"type": "inference", "ability": "eyepop.person"}]})).image(IMAGE))

        async for result in person.batch(IMAGE_URLS):
            print(result.source, result.error or result.prediction)

        async for frame in person.video(VIDEO):
            print(frame.timestamp, len(frame.objects))

        # a live stream never ends on its own: watch it for 30 seconds
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(30):
                async for frame in person.stream(RTSP_URL):
                    print(frame.timestamp, len(frame.objects))


asyncio.run(main())
