import argparse
import asyncio
import logging

import aiofiles

from eyepop import EyePopSdk
from eyepop.data.data_types import AssetUrlType, TranscodeMode

log = logging.getLogger('eyepop.example')

parser = argparse.ArgumentParser(
                    prog='download video asset',
                    description='Downloading a video asset, optionally trimmed to a certain length',
                    epilog='.')
parser.add_argument('output_path',)
parser.add_argument('-a', '--asset-uuid', required=True, help="export this asset uuid", default=None, type=str)
parser.add_argument('-s', '--start', required=False, type=float, help="start at this second", default=None)
parser.add_argument('-e', '--end', required=False, type=float, help="end at this second", default=None)
main_args = parser.parse_args()

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        result = await endpoint.download_asset(
            asset_uuid=main_args.asset_uuid,
            transcode_mode=TranscodeMode.video_original_size,
            start_timestamp=int(main_args.start * 1000 * 1000 * 1000) if main_args.start is not None else None,
            end_timestamp=int(main_args.end * 1000 * 1000 * 1000) if main_args.end is not None else None,
            url_type=AssetUrlType.gcs,
        )
        log.info("asset %s is stored in the Cloud at %s", main_args.asset_uuid, result.url)
        stream = await endpoint.download_asset(
                asset_uuid=main_args.asset_uuid,
                transcode_mode=TranscodeMode.video_original_size,
                start_timestamp=int(main_args.start * 1000 * 1000 * 1000) if main_args.start is not None else None,
                end_timestamp=int(main_args.end * 1000 * 1000 * 1000) if main_args.end is not None else None,
        )
        async with aiofiles.open(main_args.output_path, mode='wb') as out_file:
            while True:
                buffer = await stream.read(65536)
                if not buffer:
                    break
                await out_file.write(buffer)
        log.info("asset %s downloaded locally to %s", main_args.asset_uuid, main_args.output_path)

asyncio.run(main())
