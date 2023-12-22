import logging
import sys
from PIL import Image

import matplotlib.pyplot as plt

from eyepop.eyepopsdk import EyePopSdk
from eyepop.visualize import EyePopPlot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

with EyePopSdk.endpoint() as endpoint:
    result = endpoint.upload(example_image_path).predict()
    with Image.open(example_image_path) as image:
        plt.imshow(image)
        plot = EyePopPlot(plt.gca())
        for obj in result['objects']:
            plot.object(obj)
    plt.show()


