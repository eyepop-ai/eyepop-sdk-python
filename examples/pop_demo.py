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
from eyepop.worker.worker_types import Pop, InferenceComponent, PopForward, PopForwardOperator, ForwardOperatorType, \
    PopCrop

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

pop_examples = {

    "person": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person")
    ]),

    "2d-body-points": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(
                type=ForwardOperatorType.CROP,
                crop=PopCrop(maxItems=128)
            ),
            targets=[InferenceComponent(model='eyepop.person.2d-body-points:latest', categoryName="2d-body-points")]
        ))
    ]),

    "faces": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(
                type=ForwardOperatorType.CROP,
                crop=PopCrop(maxItems=128)
            ),
            targets=[InferenceComponent(model='eyepop.person.face.short-range:latest', categoreyNmae="2d-face-points", forward=PopForward(
                operator=PopForwardOperator(
                    type=ForwardOperatorType.CROP,
                    crop=PopCrop(boxPadding=0.5, orientationTargetAngle=-90.0)
                ),
                targets=[InferenceComponent(model='eyepop.person.face-mesh:latest', categoryName="3d-face-mesh")]
            ))]
        ))
    ]),

    "hands": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(
                type=ForwardOperatorType.CROP,
                crop=PopCrop(maxItems=128, boxPadding=0.25)),
            targets=[InferenceComponent(model='eyepop.person.palm:latest', forward=PopForward(
                        operator=PopForwardOperator(
                            type=ForwardOperatorType.CROP,
                            includeClasses=["hand circumference"],
                            crop=PopCrop(orientationTargetAngle=-90.0)
                        ),
                        targets=[InferenceComponent(model='eyepop.person.3d-hand-points:latest', categoryName="3d-hand-points")]
            ))]
        ))
    ]),

    "3d-body-points": Pop(components=[
        InferenceComponent(model='eyepop.person:latest', categoryName="person", forward=PopForward(
            operator=PopForwardOperator(
                type=ForwardOperatorType.CROP,
                crop=PopCrop(boxPadding=0.5)
            ),
            targets=[InferenceComponent(model='eyepop.person.pose:latest', hidden=True, forward=PopForward(
                operator=PopForwardOperator(
                    type=ForwardOperatorType.CROP,
                    crop=PopCrop(boxPadding=0.5, orientationTargetAngle=-90.0)
                ),
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
parser.add_argument('-p', '--pop', required=False, type=str, help="run this pop", choices=list(pop_examples.keys()))
parser.add_argument('-m', '--model-uuid', required=False, type=str, help="run this model by uuid")
parser.add_argument('-v', '--visualize', required=False, help="show rendered output", default=False, action="store_true")
parser.add_argument('-o', '--output', required=False, help="print results to stdout", default=False, action="store_true")


args = parser.parse_args()

if not args.local_path and not args.url:
    parser.print_help()
    sys.exit(1)

with EyePopSdk.workerEndpoint() as endpoint:
    if args.pop:
        endpoint.set_pop(pop_examples[args.pop])
    elif args.model_uuid:
        endpoint.set_pop(pop_examples[args.pop])
    else:
        raise ValueError("pop or model required")
            
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
            visualize_result = result
            if args.output:
                logging.getLogger('eyepop.example').info(json.dumps(result, indent=2))
        if args.visualize:
            with requests.get(args.url) as response:
                image = Image.open(BytesIO(response.content))
            example_image_src = args.url

    if args.visualize:
        with open(os.path.join(script_dir, 'viewer.html')) as file:
            compiler = Compiler()
            html_template = compiler.compile(file.read())

        preview = html_template({
            'image_src': example_image_src,
            'result_json': json.dumps(visualize_result)
        })
        window = webui.window()
        window.set_root_folder('.')
        window.show(preview, webui.browser.chrome)
        webui.wait()

