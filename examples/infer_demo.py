import argparse
import asyncio
import json
import logging
import os

from eyepop import EyePopSdk
from eyepop.data.data_types import InferRequest, TranscodeMode

script_dir = os.path.dirname(__file__)

log = logging.getLogger('eyepop.example')

parser = argparse.ArgumentParser(
                    prog='Vlm examples',
                    description='Demonstrates how to run a VLM infer request on image or video assets',
                    epilog='.')
parser.add_argument('-m', '--model', required=True, type=str, help="run this model (named by worker release)", choices=('smol', 'qwen3-instruct'))
parser.add_argument('-a', '--asset-uuid', required=True, type=str, help="run the inference on an asset by its Uuid")
parser.add_argument('-p', '--prompt', type=str, required=True, help="use this prompt to generate the caption")
parser.add_argument('-s', '--start-timestamp', type=int, required=False, default=None, help="Start timestamp in nano seconds for video")
parser.add_argument('-e', '--end-timestamp', type=int, required=False, default=None, help="End timestamp in nano seconds for video")


main_args = parser.parse_args()

async def main(args):
    infer_request=InferRequest(
        worker_release=args.model,
        text_prompt=args.prompt,
    )
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        asset = await endpoint.get_asset(args.asset_uuid)
        if asset.mime_type.startswith('image'):
            job = await endpoint.infer_asset(
                asset_uuid=args.asset_uuid,
                infer_request=infer_request,
                transcode_mode=TranscodeMode.image_cover_1024,
            )
        if asset.mime_type.startswith('video'):
            job = await endpoint.infer_asset(
                asset_uuid=args.asset_uuid,
                infer_request=infer_request,
                transcode_mode=TranscodeMode.video_original_size,
                start_timestamp=args.start_timestamp,
                end_timestamp=args.end_timestamp,
            )
        else:
            log.error(f"Unsupported mime type: {asset.mime_type}")
            exit(1)
        while result := await job.predict():
            print(json.dumps(result, indent=2))

asyncio.run(main(main_args))