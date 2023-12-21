import logging
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

from eyepop.eyepopsdk import EyePopSdk
from eyepop.visualize import EyePopPlot

source_path = Path(__file__).resolve()
source_dir = source_path.parent

example_image_path = f'{source_dir}/example.jpg'

logging.basicConfig(level=logging.INFO)
logging.getLogger('eyepop').setLevel(level=logging.DEBUG)

with EyePopSdk.endpoint() as endpoint:
    result = endpoint.upload(example_image_path).predict()
    with Image.open(example_image_path) as image:
        plt.imshow(image)
        plot = EyePopPlot(plt.gca())
        for obj in result['objects']:
            plot.object(obj)
    plt.show()


