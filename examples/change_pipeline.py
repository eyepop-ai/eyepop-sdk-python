import argparse

from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)


def list_models():
    with EyePopSdk.endpoint() as endpoint:
        result = endpoint.list_models()
    #print(json.dumps(result, indent=4))

def change_infer_string():
    ep_infer_string = \
        "ep_infer id=1   category-name=person   model=eyepop-person:EPPersonB1_Person_TorchScriptCuda_float32   threshold=0.8   ! ep_infer id=2   tracing=deepsort   model=legacy:reid-mobilenetv2_x1_4_ImageNet_TensorFlowLite_int8    secondary-to-id=1   secondary-for-class-ids=<0>  ! ep_infer id=3    category-name=common-objects   model=eyepop-vehicle:EPVehicleB1_Vehicle_TorchScriptCuda_float32   threshold=0.8  ! ep_infer id=4   tracing=deepsort   secondary-to-id=3"
    with EyePopSdk.endpoint() as endpoint:
        endpoint.change_pop_comp(ep_infer_string)

def check_infer_string():
    with EyePopSdk.endpoint() as endpoint:
        popComp = endpoint.pop_comp()
    print(popComp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    check_infer_string()