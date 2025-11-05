import json
import logging
import sys
import os
from pybars import Compiler
from webui import webui

from eyepop import EyePopSdk
from eyepop.logging import configure_logging, get_logging_config

script_dir = os.path.dirname(__file__)

# Configure logging: INFO level with DEBUG for requests
config = get_logging_config(level='INFO')
config['loggers']['eyepop.requests']['level'] = 'DEBUG'
configure_logging(config=config)

example_image_path = sys.argv[1]

with open(os.path.join(script_dir, 'viewer.html')) as file:
    compiler = Compiler()
    html_template = compiler.compile(file.read())

with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload(example_image_path).predict()
    preview = html_template({
        'image_src': example_image_path,
        'result_json': json.dumps(result)
    })
    print(preview)
    window = webui.window()
    window.set_root_folder('.')
    window.show(preview, webui.browser.chrome)
    webui.wait()


