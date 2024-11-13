import json
import logging
import sys

import matplotlib.pyplot as plt
from PIL import Image

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent, InferenceType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

with EyePopSdk.workerEndpoint(pop_id='transient') as endpoint:
    endpoint.set_pop(Pop(
        components=[InferenceComponent(
            inferenceTypes=[InferenceType.OBJECT_DETECTION],
            model='test1.dev.eyepop.ai:latest'
        )]
    ))
    result = endpoint.upload(example_image_path).predict()
    print(json.dumps(result))
    with Image.open(example_image_path) as image:
        plt.imshow(image)
    plot = EyePopSdk.plot(plt.gca())
    plot.prediction(result)
    plt.show()


