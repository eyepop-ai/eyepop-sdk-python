import argparse
import ast
import base64
import json
import logging
import os
import sys
from io import BytesIO
from typing import BinaryIO

from webui import webui
from pybars import Compiler

import requests
from PIL import Image

from eyepop import EyePopSdk
from eyepop.data.data_types import TranscodeMode
from eyepop.worker.worker_types import Pop, InferenceComponent, PopForward, PopForwardOperator, ForwardOperatorType, \
    PopCrop, ContourFinderComponent, ContourType, CropForward, FullForward, ComponentParams

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.INFO)

# params = {'prompt': 'what entities are in the picture?'}

caption_pop = Pop(components=[
    InferenceComponent(
        categoryName='VLM_caption',
        ability="eyepop.vlm.preview:latest",
        forward=FullForward(
            targets=[InferenceComponent(
                categoryName='Q_A',
                model='eyepop.question-answer.preview:latest',
            )]
        )
    )
])

parser = argparse.ArgumentParser(
                    prog='Pop examples',
                    description='Demonstrates the caption generation via VLM/LLM',
                    epilog='.')
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file")
parser.add_argument('-a', '--asset-uuid', required=False, type=str, default=False, help="run the inference on an asset by its Uuid")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")


args = parser.parse_args()

if not args.local_path and not args.url and not args.asset_uuid:
    print("Need something to run inference on; pass either --url or --local-path or --asset-uuid")
    parser.print_help()
    sys.exit(1)

with EyePopSdk.workerEndpoint() as endpoint:
    endpoint.set_pop(caption_pop)
    if args.local_path:
        job = endpoint.upload(args.local_path)
        while result := job.predict():
           visualize_result = result
           if args.output:
                logging.getLogger('eyepop.example').info(json.dumps(result, indent=2))
        if args.visualize:
            image = Image.open(args.local_path)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            example_image_src = f"data:image/png;base64, {base64.b64encode(buffer.getvalue()).decode()}"
    elif args.url:
        job = endpoint.load_from(args.url)
        while result := job.predict():
            logging.getLogger('eyepop.example').info(json.dumps(result, indent=2))
            with requests.get(args.url) as response:
                image = Image.open(BytesIO(response.content))
            example_image_src = args.url
    elif args.asset_uuid:
        job = endpoint.load_asset(args.asset_uuid)
        while result := job.predict():
            logging.getLogger('eyepop.example').info(json.dumps(result, indent=2))

