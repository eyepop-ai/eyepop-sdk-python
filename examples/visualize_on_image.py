import logging
import sys
from PIL import Image

import matplotlib.pyplot as plt

from eyepop import EyePopSdk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

with EyePopSdk.endpoint() as endpoint:
    result = endpoint.upload(example_image_path).predict()
with Image.open(example_image_path) as image:
    plt.imshow(image)
plot = EyePopSdk.plot(plt.gca())
plot.prediction(result)
plt.show()


