import argparse
import json
import logging
import os
import sys

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent, FullForward

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.INFO)

log = logging.getLogger('eyepop.example')

parser = argparse.ArgumentParser(
                    prog='Pop examples',
                    description='Demonstrates the caption generation via VLM/LLM',
                    epilog='.')
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file or directory")
parser.add_argument('-a', '--asset-uuid', required=False, type=str, default=False, help="run the inference on an asset by its Uuid")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")
parser.add_argument('-c', '--caption-ability', type=str, default="eyepop.vlm.preview:latest", help="run this ability to generate captions", choices=["eyepop.vlm.preview:latest", "eyepop.image-caption.preview.qwen2:latest"])
parser.add_argument('-r', '--role', type=str, default=None, required=False, help="use this role to generate the caption")
parser.add_argument('-p', '--prompt', type=str, default=None, required=False, help="use this prompt to generate the caption")
parser.add_argument('-q', '--question', type=str, default=None, required=False, help="apply this question to the caption")


args = parser.parse_args()

if not args.local_path and not args.url and not args.asset_uuid:
    print("Need something to run inference on; pass either --url or --local-path or --asset-uuid")
    parser.print_help()
    sys.exit(1)

forward = None
if args.question is not None:
    forward = FullForward(
        targets=[InferenceComponent(
            id=2,
            categoryName='answers',
            model='eyepop.question-answer.preview:latest',
            params={
                "prompt": args.question
            }
        )]
    )

caption_params = dict()
if args.role is not None:
    caption_params['role'] = args.role
if args.prompt is not None:
    caption_params['prompt'] = args.prompt

caption_pop = Pop(components=[
    InferenceComponent(
        id=1,
        categoryName='captions',
        ability=args.caption_ability,
        forward=forward,
        params=caption_params,
    )
])

params = None

with EyePopSdk.workerEndpoint() as endpoint:
    endpoint.set_pop(caption_pop)
    if args.local_path:
        if not os.path.exists(args.local_path):
            log.warning(f"local path {args.local_path} does not exist")
            sys.exit(1)
        if os.path.isfile(args.local_path):
            local_files = [args.local_path]
        else:
            local_files = []
            for f in os.listdir(args.local_path):
                local_file = os.path.join(args.local_path, f)
                if os.path.isfile(local_file):
                    local_files.append(local_file)

        for local_file in local_files:
            log.info("uploading %s", local_file)
            job = endpoint.upload(local_file, params=params)
            while result := job.predict():
                print(json.dumps(result, indent=2))
    elif args.url:
        job = endpoint.load_from(args.url, params=params)
        while result := job.predict():
            print(json.dumps(result, indent=2))
    elif args.asset_uuid:
        job = endpoint.load_asset(args.asset_uuid, params=params)
        while result := job.predict():
            print(json.dumps(result, indent=2))

