#!/usr/bin/env python3

import asyncio
import json
import os
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent

# Define the person detection pop
person_pop = Pop(components=[
    InferenceComponent(
        model='eyepop.person:latest',
        categoryName="person"
    )
])

async def detect_persons():
    # Connect to EyePop endpoint
    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
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