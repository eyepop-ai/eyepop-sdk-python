import json
import logging
import sys

import matplotlib.pyplot as plt
from PIL import Image

from eyepop import EyePopSdk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

with EyePopSdk.workerEndpoint(pop_id='transient') as endpoint:
    modelRef = {
        'id': 'my-yolo-v7',
        'folderUrl': 'https://s3.amazonaws.com/models.eyepop.ai/releases/yolov7/1.0.1/models/YOLOv7/COCO/Latest/TensorFlowLite/float32/'
    }
    endpoint.set_pop_comp(f'ep_infer model=my-yolo-v7', [modelRef])
    result = endpoint.upload(example_image_path).predict()
    print(json.dumps(result))
    with Image.open(example_image_path) as image:
        plt.imshow(image)
    plot = EyePopSdk.plot(plt.gca())
    plot.prediction(result)
    plt.show()


