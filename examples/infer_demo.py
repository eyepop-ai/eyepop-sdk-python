import argparse
import asyncio
import json
import logging
import os

from eyepop import EyePopSdk
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_jobs import InferJob
from eyepop.data.data_types import InferRequest, TranscodeMode, Asset, AssetInclusionMode, EvaluateRequest

script_dir = os.path.dirname(__file__)

logging.getLogger('eyepop.requests').setLevel(logging.DEBUG)
log = logging.getLogger('eyepop.example')

parser = argparse.ArgumentParser(
                    prog='Vlm examples',
                    description='Demonstrates how to run a VLM infer request on image or video assets',
                    epilog='.')
parser.add_argument('-m', '--model', required=True, type=str, help="run this model (named by worker release)", choices=('smol', 'qwen3-instruct'))
parser.add_argument('-d', '--dataset-uuid', required=False, type=str, default=None, help="run the inference on an asset by its Uuid")
parser.add_argument('-a', '--asset-uuid', required=False, type=str, default=None, help="run the inference on an asset by its Uuid")
parser.add_argument('-p', '--prompt', type=str, required=True, help="use this prompt to generate the caption")
parser.add_argument('-s', '--start-timestamp', type=int, required=False, default=None, help="Start timestamp in nano seconds for video")
parser.add_argument('-e', '--end-timestamp', type=int, required=False, default=None, help="End timestamp in nano seconds for video")


main_args = parser.parse_args()

async def infer(
        asset: Asset,
        infer_request: InferRequest,
        start_timestamp: int | None,
        end_timestamp: int | None,
        endpoint: DataEndpoint
) -> InferJob | None:
    if asset.mime_type.startswith('image'):
        return await endpoint.infer_asset(
            asset_uuid=asset.uuid,
            infer_request=infer_request,
            transcode_mode=TranscodeMode.image_cover_1024,
        )
    if asset.mime_type.startswith('video'):
        return await endpoint.infer_asset(
            asset_uuid=asset.uuid,
            infer_request=infer_request,
            transcode_mode=TranscodeMode.video_original_size,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
    else:
        log.error(f"Unsupported mime type: {asset.mime_type}")
        return None

async def main(args):
    if args.asset_uuid is None and args.dataset_uuid is None:
        print("Need either --asset-uuid or --dataset-uuid")
        exit(1)

    if args.asset_uuid is not None and args.dataset_uuid is not None:
        print("Only one of --asset-uuid or --dataset-uuid is supported")
        exit(1)

    infer_request=InferRequest(
        worker_release=args.model,
        text_prompt=args.prompt,
    )

    async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=4) as endpoint:
        if args.asset_uuid is not None:
            asset = await endpoint.get_asset(args.asset_uuid)
            job = await infer(
                asset=asset,
                infer_request=infer_request,
                start_timestamp=args.start_timestamp,
                end_timestamp=args.end_timestamp,
                endpoint=endpoint
            )
            if job is not None:
                while result := await job.predict():
                    print(f"result for {asset.uuid}:", json.dumps(result, indent=2))
        else:
            evaluate_request = EvaluateRequest(
                dataset_uuid=args.dataset_uuid,
                infer=infer_request,
            )
            job = await endpoint.evaluate_dataset(
                evaluate_request=evaluate_request,
            )
            print(f"result for {args.dataset_uuid}:", (await job.response).model_dump_json(indent=2))

asyncio.run(main(main_args))