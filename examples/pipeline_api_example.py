import asyncio
import contextlib
import os
import pathlib

from eyepop.pipeline import EyePopClient

IMAGE = pathlib.Path(__file__).parent / "example.jpg"
IMAGE_URLS = [
    "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg",
    "https://farm1.staticflickr.com/22/29604617_a60740ce78_z.jpg",
]
VIDEO = pathlib.Path("clip.mp4")
RTSP_URL = "rtsp://camera.local/stream"
MODEL_UUID = "your-model-uuid"

# a pop: find people, then find each person's body points in their crop
POP = {
    "components": [
        {
            "type": "inference",
            "ability": "eyepop.person:latest",
            "forward": {
                "operator": {"type": "crop"},
                "targets": [{"type": "inference", "ability": "eyepop.person.2d-body-points:latest"}],
            },
        }
    ]
}


async def main():
    async with EyePopClient(os.environ["EYEPOP_PIPELINE_URL"], api_key=os.environ["EYEPOP_API_KEY"]) as client:
        print(await client.image(IMAGE, ability="eyepop.person"))
        print(await client.image(IMAGE, pop=POP))
        print(await client.image(IMAGE, model=MODEL_UUID))

        batch = [{"id": str(i), "url": url} for i, url in enumerate(IMAGE_URLS)]
        async for message in client.socket(batch, ability="eyepop.person"):
            print(message)

        video = [{"id": "video", "upload": {"contentType": "video/mp4"}, "file": VIDEO}]
        async for message in client.socket(video, ability="eyepop.person"):
            print(message)

        # a live stream never ends on its own: watch it for 30 seconds
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(30):
                async for message in client.socket([{"id": "camera", "url": RTSP_URL}], ability="eyepop.person"):
                    print(message)


asyncio.run(main())
