import argparse
import asyncio
import json
import logging
import os
import sys

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

logging.getLogger('eyepop.requests').setLevel(logging.INFO)
log = logging.getLogger('eyepop.example')

parser = argparse.ArgumentParser(
    prog='Multi-image group demo',
    description='Send several images as a single image group (one inference unit) '
                'to a multi-image-capable VLM ability and print the one result.',
    epilog='.')
parser.add_argument('-l', '--local-paths', nargs='+', default=None,
                    help='two or more local image files to send as one group')
parser.add_argument('-u', '--urls', nargs='+', default=None,
                    help='two or more remote image URLs to send as one group')
parser.add_argument('-a', '--ability', type=str, default='eyepop.vlm.image:latest',
                    help='a multi-image-capable ability')
parser.add_argument('-p', '--prompt', type=str,
                    default='Describe these images together in one sentence.',
                    help='prompt for the VLM ability')

args = parser.parse_args()

if not args.local_paths and not args.urls:
    print('Pass --local-paths or --urls with two or more images.')
    parser.print_help()
    sys.exit(1)

group_pop = Pop(components=[
    InferenceComponent(ability=args.ability, params={'prompt': args.prompt})
])


async def main():
    async with EyePopSdk.async_worker() as endpoint:
        await endpoint.set_pop(group_pop)

        if args.local_paths:
            for path in args.local_paths:
                if not os.path.isfile(path):
                    log.warning(f'local path {path} does not exist')
                    sys.exit(1)
            job = await endpoint.upload_group(args.local_paths)
        else:
            job = await endpoint.load_from_group(args.urls)

        # A group is one inference unit: expect a single result.
        result = await job.predict()
        print(json.dumps(result, indent=2))


asyncio.run(main())
