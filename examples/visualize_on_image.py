import json
import sys

import matplotlib.pyplot as plt
from PIL import Image

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PERSON = Pop(components=[InferenceComponent(ability='eyepop.person:latest')])

example_image_path = sys.argv[1]

with EyePopSdk.sync_worker(pop=PERSON) as endpoint:
    result = endpoint.upload(example_image_path).predict()
    print(json.dumps(result))
    with Image.open(example_image_path) as image:
        plt.imshow(image)
    plot = EyePopSdk.plot(plt.gca())
    plot.prediction(result)
    plt.show()


