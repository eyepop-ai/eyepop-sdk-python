import logging
import sys
from PIL import Image
import json

import matplotlib.pyplot as plt

from eyepop import EyePopSdk
from eyepop.logging import configure_logging, get_logging_config

# Configure logging: INFO level with DEBUG for requests
config = get_logging_config(level='INFO')
config['loggers']['eyepop.requests']['level'] = 'DEBUG'
configure_logging(config=config)

example_image_path = sys.argv[1]

with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload(example_image_path).predict()
    print(json.dumps(result))
    with Image.open(example_image_path) as image:
        plt.imshow(image)
    plot = EyePopSdk.plot(plt.gca())
    plot.prediction(result)
    plt.show()


