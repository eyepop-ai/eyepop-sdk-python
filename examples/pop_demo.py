import argparse
import ast
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
    PopCrop, ContourFinderComponent, ContourType, CropForward, FullForward

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

pop_examples = {
    "person": Pop(components=[
        InferenceComponent(
            model='eyepop.person:latest',
            categoryName="person"
        )
    ]),

    "2d-body-points": Pop(components=[
        InferenceComponent(
            model='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    model='eyepop.person.2d-body-points:latest',
                    categoryName="2d-body-points",
                    confidenceThreshold=0.25
                )]
        ))
    ]),

    "faces": Pop(components=[
        InferenceComponent(
            model='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    model='eyepop.person.face.short-range:latest',
                    categoryName="2d-face-points",
                    forward=CropForward(
                        boxPadding=1.5,
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            model='eyepop.person.face-mesh:latest',
                            categoryName="3d-face-mesh"
                        )]
                    )
                )]
            )
        )
    ]),

    "hands": Pop(components=[
        InferenceComponent(
            model='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                boxPadding=0.25,
                targets=[InferenceComponent(
                    model='eyepop.person.palm:latest',
                    forward=CropForward(
                        includeClasses=["hand circumference"],
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            model='eyepop.person.3d-hand-points:latest',
                            categoryName="3d-hand-points"
                        )]
                    )
                )]
            )
        )
    ]),

    "3d-body-points": Pop(components=[
        InferenceComponent(
            model='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                boxPadding=0.5,
                targets=[InferenceComponent(
                    model='eyepop.person.pose:latest',
                    hidden=True,
                    forward=CropForward(
                        boxPadding=0.5,
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            model='eyepop.person.3d-body-points.heavy:latest',
                            categoryName="3d-body-points",
                            confidenceThreshold=0.25
                        )]
                    )
                )]
            )
        )
    ]),

    "text": Pop(components=[
        InferenceComponent(
            model='eyepop.text:latest',
            categoryName="text",
            forward=CropForward(
                targets=[InferenceComponent(
                    model='eyepop.text.recognize.square:latest'
                )]
            )
        )
    ]),

    "sam1": Pop(components=[
        InferenceComponent(
            model='eyepop.sam.small:latest',
            forward=FullForward(
                targets=[ContourFinderComponent(
                    contourType=ContourType.POLYGON,
                    areaThreshold=0.005
                )]
            )
        )
    ]),

    "sam2": Pop(components=[
        InferenceComponent(
            model="eyepop.sam2.encoder.tiny:latest",
            hidden=True,
            forward=FullForward(
                targets=[InferenceComponent(
                    model='eyepop.sam2.decoder:latest',
                    forward=FullForward(
                        targets=[ContourFinderComponent(
                            contourType=ContourType.POLYGON,
                            areaThreshold=0.005
                        )]
                    )
                )]
            )
        )
    ]),
}

def list_of_points(arg: str) -> list[dict[str, any]]:
    points = []
    points_as_tuples = ast.literal_eval(f'[{arg}]')
    for tuple in points_as_tuples:
        points.append({
            "x": tuple[0],
            "y": tuple[1]
        })
    return points


def list_of_boxes(arg: str) -> list[dict[str, any]]:
    boxes = []
    boxes_as_tuples = ast.literal_eval(f'[{arg}]')
    for tuple in boxes_as_tuples:
        boxes.append({
            "topLeft": {
                "x": tuple[0],
                "y": tuple[1]
            },
            "bottomRight": {
                "x": tuple[2],
                "y": tuple[3]
            }
        })
    return boxes

parser = argparse.ArgumentParser(
                    prog='Pop examples',
                    description='Demonstrates the composition of a Pop',
                    epilog='.')
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")
parser.add_argument('-p', '--pop', required=False, type=str, help="run this pop", choices=list(pop_examples.keys()))
parser.add_argument('-m', '--model-uuid', required=False, type=str, help="run this model by uuid")
parser.add_argument('-ms1', '--model-uuid-sam1', required=False, type=str, help="run this model by uuid and compose with SAM1 (EfficientSAM) and Contour Finder")
parser.add_argument('-ms2', '--model-uuid-sam2', required=False, type=str, help="run this model by uuid and compose with SAM2 and Contour Finder")
parser.add_argument('-po', '--points', required=False, type=list_of_points, help="List of POIs as coordinates like (x1, y1), (x2, y2) in the original image coordinate system")
parser.add_argument('-bo', '--boxes', required=False, type=list_of_boxes, help="List of POIs as boxes like (left1, top1, right1, bottom1), (left1, top1, right1, bottom1) in the original image coordinate system")
parser.add_argument('-v', '--visualize', required=False, help="show rendered output", default=False, action="store_true")
parser.add_argument('-o', '--output', required=False, help="print results to stdout", default=False, action="store_true")


args = parser.parse_args()

if not args.local_path and not args.url:
    print("Need something to run inference on; pass either --url or --local-path")
    parser.print_help()
    sys.exit(1)

if not args.pop and not args.model_uuid and not args.model_uuid_sam1 and not args.model_uuid_sam2:
    print("Need something do do, pass either --pop or --model-uuid or --model-uuid-sam1 or --model-uuid-sam2")
    parser.print_help()
    sys.exit(1)

with EyePopSdk.workerEndpoint() as endpoint:
    if args.pop:
        endpoint.set_pop(pop_examples[args.pop])
    elif args.model_uuid:
        endpoint.set_pop(Pop(components=[
            InferenceComponent(
                modelUuid=args.model_uuid
            )
        ]))
    elif args.model_uuid_sam1:
        endpoint.set_pop(Pop(components=[
            InferenceComponent(
                modelUuid=args.model_uuid_sam1,
                forward=CropForward(
                    targets=[InferenceComponent(
                        model='eyepop.sam.small:latest',
                        forward=FullForward(
                            targets=[ContourFinderComponent(
                                contourType=ContourType.POLYGON,
                                areaThreshold=0.005
                            )]
                        )
                    )]
                )
            )
        ]))
    elif args.model_uuid_sam2:
        endpoint.set_pop(Pop(components=[
            InferenceComponent(
                model="eyepop.sam2.encoder.tiny:latest",
                hidden=True,
                forward=FullForward(
                    targets=[InferenceComponent(
                        modelUuid=args.model_uuid_sam2,
                        forward=CropForward(
                            targets=[InferenceComponent(
                                model='eyepop.sam2.decoder:latest',
                                forward=FullForward(
                                    targets=[ContourFinderComponent(
                                        contourType=ContourType.POLYGON,
                                        areaThreshold=0.005
                                    )]
                                )
                            )]
                        )
                    )]
                )
            )
        ]))
    else:
        raise ValueError("pop or model required")

    params = None
    if args.points:
        params = {
          "roi": {
              "points": args.points
          }
        }
    elif args.boxes:
        params = {
            "roi": {
                "boxes": args.boxes
            }
        }

    if args.local_path:
        job = endpoint.upload(args.local_path, params=params)
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
        job = endpoint.load_from(args.url, params=params)
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

