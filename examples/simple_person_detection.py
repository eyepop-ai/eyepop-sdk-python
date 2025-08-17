#!/usr/bin/env python3

import asyncio
import json
import os

# Set compute API configuration
os.environ["EYEPOP_URL"] = "https://compute.staging.eyepop.xyz"  # Compute API URL
os.environ["EYEPOP_SECRET_KEY"] = "eyp_a2ae589ceb13fe2ed58c89cfab7aa16419c27bcc2855cdf6d7e2a6647726a7ac4da198e31329e7e5ea809b45c67be6109e40846785f908965d024b140fc3f93e"
os.environ["EYEPOP_USER_UUID"] = "ff92bf8c460f11ef8a820a359ae0bb9d"
os.environ["EYEPOP_DATA_API"] = "https://dataset-api.staging.eyepop.xyz"  # Data API for model resolution

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent

# Define the person detection pop
person_pop = Pop(components=[
    InferenceComponent(
        modelUuid='0683202a1bf0788c8000f098f0a00829',  # Using UUID directly instead of alias
        categoryName="person"
    )
])

async def detect_persons():
    # Connect to EyePop endpoint - uses compute API to get endpoint URL
    async with EyePopSdk.workerEndpoint(
        is_async=True,
        pop_id="transient"  # Use transient mode
    ) as endpoint:
        # Set the pop configuration
        await endpoint.set_pop(person_pop)
        
        # Upload and process the example image
        job = await endpoint.upload('examples/example.jpg')
        
        # Get prediction results
        while result := await job.predict():
            # Print the results
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(detect_persons())