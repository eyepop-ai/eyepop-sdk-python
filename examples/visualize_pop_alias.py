import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO

from webui import webui
from pybars import Compiler

import requests
from PIL import Image

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent, PopForward, PopForwardOperator, ForwardOperatorType

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

pop_examples = {

    "person": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person")
    ]),

    "2d-body-points": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(type=ForwardOperatorType.CROP, maxItems=128),
            targets=[InferenceComponent(model='eyepop.person.2d-body-points:latest', categoryName="2d-body-points")]
        ))
    ]),

    "face": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(type=ForwardOperatorType.CROP, maxItems=128, boxPadding=0.5),
            targets=[InferenceComponent(model='eyepop.person.face.long-range:latest', hidden=True, forward=PopForward(
                operator=PopForwardOperator(type=ForwardOperatorType.CROP, boxPadding=0.5, orientationTargetAngle=-90.0),
                targets=[InferenceComponent(model='eyepop.person.face-mesh:latest', categoryName="3d-face-mesh")]
            ))]
        ))
    ]),

    "hands": Pop(components=[
        InferenceComponent(model='eyepop.palm:latest', hidden=True, forward=PopForward(
            operator=PopForwardOperator(type=ForwardOperatorType.CROP, orientationTargetAngle=-90.0),
            targets=[InferenceComponent(model='eyepop.person.3d-hand-points:latest', categoryName="3d-hand-points")]
        ))
    ]),

    "3d-body-points": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(type=ForwardOperatorType.CROP, boxPadding=0.5),
            targets=[InferenceComponent(model='eyepop.person.pose:latest', hidden=True, forward=PopForward(
                operator=PopForwardOperator(type=ForwardOperatorType.CROP, boxPadding=0.5, orientationTargetAngle=-90.0),
                targets=[InferenceComponent(model='eyepop.person.3d-body-points.heavy:latest', categoryName="3d-body-points")]
            ))]
        ))
    ]),

    "text": Pop(components=[
        InferenceComponent(model='eyepop.text:latest', categoryName="text", forward=PopForward(
            operator=PopForwardOperator(type=ForwardOperatorType.CROP),
            targets=[InferenceComponent(model='eyepop.text.recognize.square:latest')]
        ))
    ]),
}

parser = argparse.ArgumentParser(
                    prog='Pop examples',
                    description='Demonstrates the composition of a Pop',
                    epilog='.')
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")
parser.add_argument('-p', '--pop', required=True, type=str, help="run this pop", choices=list(pop_examples.keys()))

args = parser.parse_args()

if not args.local_path and not args.url:
    parser.print_help()
    sys.exit(1)

with EyePopSdk.workerEndpoint() as endpoint:
    endpoint.set_pop(pop_examples[args.pop])
    if args.local_path:
        result = endpoint.upload(args.local_path).predict()
        image = Image.open(args.local_path)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        example_image_src = f"data:image/png;base64, {base64.b64encode(buffer.getvalue()).decode()}"
    elif args.url:
        result = endpoint.load_from(args.url).predict()
        with requests.get(args.url) as response:
            image = Image.open(response.raw)
        example_image_src = args.url

    logging.getLogger('eyepop.example').info(json.dumps(result, indent=2))

    with open(os.path.join(script_dir, 'viewer.html')) as file:
        compiler = Compiler()
        html_template = compiler.compile(file.read())

    preview = html_template({
        'image_src': example_image_src,
        'result_json': json.dumps(result)
    })
    window = webui.window()
    window.set_root_folder('.')
    window.show(preview, webui.browser.chrome)
    webui.wait()

