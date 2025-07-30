import argparse
import ast
import asyncio
import base64
import json
import logging

import os
import sys
from io import BytesIO
from typing import Any

from webui import webui
from pybars import Compiler

from PIL import Image

from eyepop import EyePopSdk, Job
from eyepop.data.data_types import TranscodeMode
from eyepop.worker.worker_types import Pop, InferenceComponent, \
    ContourFinderComponent, ContourType, CropForward, FullForward, ComponentParams, TracingComponent

script_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.INFO)

log = logging.getLogger('eyepop.example')

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
            confidenceThreshold=0.7,
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    model='eyepop.text.recognize.landscape:latest',
                    confidenceThreshold=0.1
                )]
            )
        )
    ]),

    "sam1": Pop(components=[
        InferenceComponent(
            model='eyepop.sam.small:latest',
            id=1,
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
            id=1,
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
    "image-contents": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.image-contents:latest',
        )
    ]),
    "localize-objects": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
        )
    ]),
    "localize-objects-plus": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
            params={
                "prompts": [{"prompt": "person"}]
            },
            forward=CropForward(
                targets=[InferenceComponent(
                    model='eyepop.image-contents:latest',
                    params={
                        "prompts": [{"prompt": "hair color blond"},{"prompt": "hair color brown"}]
                    }
                )],
            )
        )
    ]),
    "localize-objects-t4": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
            params={
                "prompts": [{"prompt": "person"}]
            },
            forward=CropForward(
                targets=[InferenceComponent(
                    model='eyepop.image-contents-t4:latest',
                    params={
                        "prompts": [{"prompt": "shirt color?"}]
                    }
                )],
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
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file, or all files on a directory")
parser.add_argument('-a', '--asset-uuid', required=False, type=str, default=False, help="run the inference on an asset by its Uuid")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")
parser.add_argument('-p', '--pop', required=False, type=str, help="run this pop", choices=list(pop_examples.keys()))
parser.add_argument('-m', '--model-uuid', required=False, type=str, action="append", help="run this model(s) by uuid")
parser.add_argument('-ma', '--model-alias', required=False, type=str, action="append", help="run this model(s) by its tagged alias")
parser.add_argument('-ms1', '--model-uuid-sam1', required=False, type=str, help="run this model by uuid and compose with SAM1 (EfficientSAM) and Contour Finder")
parser.add_argument('-ms2', '--model-uuid-sam2', required=False, type=str, help="run this model by uuid and compose with SAM2 and Contour Finder")
parser.add_argument('-po', '--points', required=False, type=list_of_points, help="List of POIs as coordinates like (x1, y1), (x2, y2) in the original image coordinate system")
parser.add_argument('-bo', '--boxes', required=False, type=list_of_boxes, help="List of POIs as boxes like (left1, top1, right1, bottom1), (left1, top1, right1, bottom1) in the original image coordinate system")
parser.add_argument('-sp', '--single-prompt', required=False, type=str, help="Single prompt to pass as parameter")
parser.add_argument('-pr', '--prompt', required=False, type=str, help="Prompt to pass as parameter", action="append")
parser.add_argument('-v', '--visualize', required=False, help="show rendered output", default=False, action="store_true")
parser.add_argument('-o', '--output', required=False, help="print results to stdout", default=False, action="store_true")
parser.add_argument('-ds', '--dataset-uuid', required=False, type=str, help="Ingest all assets into a dataset uuid", default=None)
parser.add_argument('-tk', '--top-k', required=False, type=int, help="For --model-uuid and -model-alias apply this top-k filter", default=None)
parser.add_argument('-ct', '--confidence-threshold', required=False, type=float, help="For --model-uuid and -model-alias apply this confidence threshold filter", default=None)


main_args = parser.parse_args()

if not main_args.local_path and not main_args.url and not main_args.asset_uuid:
    print("Need something to run inference on; pass either --url or --local-path or --asset-uuid")
    parser.print_help()
    sys.exit(1)

if not main_args.pop and not main_args.model_uuid and not main_args.model_alias and not main_args.model_uuid_sam1 and not main_args.model_uuid_sam2:
    print("Need something do do, pass either --pop or --model-uuid or --model-alias or --model-uuid-sam1 or --model-uuid-sam2")
    parser.print_help()
    sys.exit(1)

if main_args.pop:
    pop = pop_examples[main_args.pop]
elif main_args.model_uuid:
    pop = Pop(components=[
        InferenceComponent(
            id=i+1,
            abilityUuid=uuid
        ) for i, uuid in enumerate(main_args.model_uuid)
    ])
    if main_args.top_k is not None:
        for c in pop.components:
            c.topK = main_args.top_k
    if main_args.confidence_threshold is not None:
        for c in pop.components:
            c.confidenceThreshold = main_args.confidence_threshold
elif main_args.model_alias:
    pop = Pop(components=[
        InferenceComponent(
            id=i+1,
            ability=alias
        ) for i, alias in enumerate(main_args.model_alias)
    ])
    if main_args.top_k is not None:
        for c in pop.components:
            c.topK = main_args.top_k
    if main_args.confidence_threshold is not None:
        for c in pop.components:
            c.confidenceThreshold = main_args.confidence_threshold
elif main_args.model_uuid_sam1:
    pop = Pop(components=[
        InferenceComponent(
            modelUuid=main_args.model_uuid_sam1,
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
    ])
elif main_args.model_uuid_sam2:
    pop = Pop(components=[
        InferenceComponent(
            model="eyepop.sam2.encoder.tiny:latest",
            hidden=True,
            forward=FullForward(
                targets=[InferenceComponent(
                    modelUuid=main_args.model_uuid_sam2,
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
    ])
else:
    raise ValueError("pop or model required")

params = None
if main_args.points:
    params = [
        ComponentParams(componentId=1, values={
          "roi": {
              "points": main_args.points
          }
        })
    ]
elif main_args.boxes:
    params = [
        ComponentParams(componentId=1, values={
            "roi": {
                "boxes": main_args.boxes
            }
        })
    ]
elif main_args.prompt is not None and len(main_args.prompt) > 0:
    params = [
        ComponentParams(componentId=1, values={
            "prompts": [{"prompt": p} for p in main_args.prompt]
        })
    ]
elif main_args.single_prompt is not None:
    params = [
        ComponentParams(componentId=1, values={
            "prompt": main_args.single_prompt
        })
    ]

async def main(args) -> (dict[str, Any] | None, str | None):
    visualize_result = None
    example_image_src = None
    async with EyePopSdk.workerEndpoint(dataset_uuid=args.dataset_uuid, is_async=True) as endpoint:
        await endpoint.set_pop(pop)
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
            jobs = []
            async def on_ready(job: Job, path: str):
                while result := await job.predict():
                   if args.output:
                        print(path, json.dumps(result, indent=2))
                return (result, path)
            for local_file in local_files:
                job = await endpoint.upload(local_file, params=params)
                jobs.append(on_ready(job, local_file))
            results = await asyncio.gather(*jobs)
            if args.visualize and len(results) > 0:
                visualize_result = results[0][0]
                visualize_path = results[0][1]
                image = Image.open(visualize_path)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                example_image_src = f"data:image/png;base64, {base64.b64encode(buffer.getvalue()).decode()}"
        elif args.url:
            job = await endpoint.load_from(args.url, params=params)
            while result := await job.predict():
                visualize_result = result
                if args.output:
                    log.info(json.dumps(result, indent=2))
            if args.visualize:
                example_image_src = args.url
        elif args.asset_uuid:
            job = await endpoint.load_asset(args.asset_uuid, params=params)
            while result := await job.predict():
                visualize_result = result
                if args.output:
                    print(json.dumps(result, indent=2))
            if args.visualize:
                async with EyePopSdk.dataEndpoint(is_async=True) as dataEndpoint:
                    buffer = await dataEndpoint.download_asset(
                        args.asset_uuid,
                        transcode_mode=TranscodeMode.image_original_size
                    ).read()
                    example_image_src = f"data:image/jpeg;base64, {base64.b64encode(buffer).decode()}"
    return visualize_result, example_image_src

visualize_result, example_image_src = asyncio.run(main(main_args))
if main_args.visualize:
    with open(os.path.join(script_dir, 'viewer.html')) as file:
        compiler = Compiler()
        html_template = compiler.compile(file.read())

    preview = html_template({
        'image_src': example_image_src,
        'result_json': json.dumps(visualize_result)
    })
    window = webui.window()
    window.set_root_folder('.')
    window.show(preview)
    webui.wait()



