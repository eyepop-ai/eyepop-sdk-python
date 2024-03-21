import argparse

from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk
import logging
import json

def main(args):
    debug = args.debug
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG if debug else logging.WARNING)

    with EyePopSdk.endpoint() as endpoint:
        # get manifest
        manifest = endpoint.get_manifest()
        if debug:
            # save manifest to manifest_before.json
            with open('manifest_before.json', 'w') as f:
                f.write(json.dumps(manifest, indent=4))
        for model in manifest:
            # change epvehicle to 1.0.2
            if model['authority'] == 'eyepop-vehicle':
                model_name = 'epvehicle/'
                start_idx = model['manifest'].find(model_name) + len(model_name)
                end_idx = model['manifest'].find('/', start_idx)
                model['manifest'] = model['manifest'][:start_idx] + '1.0.2' + model['manifest'][end_idx:]
            # change parseq to 1.0.2
            if model['authority'] == 'PARSeq':
                model_name = 'PARSeq/'
                start_idx = model['manifest'].find(model_name) + len(model_name)
                end_idx = model['manifest'].find('/', start_idx)
                model['manifest'] = model['manifest'][:start_idx] + '1.0.2' + model['manifest'][end_idx:]
        # add eptext version 1.0.3
        manifest.append(
            {
                "authority": "eyepop-text",
                "manifest": "https://s3.amazonaws.com/models.eyepop.ai/releases/eptext/1.0.3/manifest.json"
            }
        )
        if debug:
            # save manifest to manifest_after.json
            with open('manifest_after.json', 'w') as f:
                f.write(json.dumps(manifest, indent=4))
        # set manifest
        endpoint.set_manifest(manifest)
        list_models = endpoint.list_models()
        if debug:
            with open('list_models.json', 'w') as f:
                f.write(json.dumps(list_models, indent=4))
        # purge/load epvehicle model
        endpoint.load_model(
            {
                "dataset": "Vehicle",
                "format": "TorchScriptCuda",
                "model_id": "eyepop-vehicle:EPVehicleB1",
                "type": "float32",
                "version": "Latest"
            },
            True
        )
        # purge/load parseq model
        endpoint.load_model(
            {
                "dataset": "TextDataset",
                "format": "TorchScriptCuda",
                "model_id": "PARSeq:PARSeq",
                "type": "float32",
                "version": "Latest"
            },
            True
        )
        # purge/load eptext model
        endpoint.load_model(
            {
                "dataset": "Text",
                "format": "TorchScriptCuda",
                "model_id": "eyepop-text:EPTextB1",
                "type": "float32",
                "version": "Latest"
            },
            False
        )
        # set pop comp as vehicle -> text -> parseq
        ep_infer_string = ""
        ep_infer_string += "ep_infer id=1 model=eyepop-vehicle:EPVehicleB1_Vehicle_TorchScriptCuda_float32 threshold=0.7 category-name=common-objects "
        ep_infer_string += "! ep_infer id=2 model=eyepop-text:EPTextB1_Text_TorchScriptCuda_float32 threshold=0.7 category-name=common-objects secondary-to-id=1 "
        ep_infer_string += "! ep_infer id=3 model=PARSeq:PARSeq_TextDataset_TorchScriptCuda_float32 threshold=0.1 secondary-to-id=2"
        endpoint.set_pop_comp(ep_infer_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)