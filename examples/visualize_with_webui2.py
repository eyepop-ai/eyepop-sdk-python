import json
import os
import sys

from pybars import Compiler
from webui import webui

from eyepop import EyePopSdk

script_dir = os.path.dirname(__file__)

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


