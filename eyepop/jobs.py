import asyncio
import json
import logging
import mimetypes
from enum import Enum
from typing import Coroutine

from aiohttp import ClientSession

log = logging.getLogger('eyepop')


class Job:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=128)

    async def predict(self):
        queue = self.queue
        if queue is None:
            return None
        else:
            prediction = await queue.get()
            if prediction is None:
                self.queue = None
            elif isinstance(prediction, Exception):
                raise prediction
            return prediction


class UploadJob(Job):
    def __init__(self, file_path: str, pipeline_base_url: str, authorization_header: str, session: ClientSession):
        self.session = session
        mime_types = mimetypes.guess_type(file_path)
        if len(mime_types) > 0:
            self.mime_type = mime_types[0]
        else:
            self.mime_type = 'application/octet-stream'
        self.file_path = file_path
        self.target_url = f'{pipeline_base_url}/source?mode=queue&processing=sync'
        self.headers = {
            'Content-Type': self.mime_type,
            'Accept': 'application/jsonl',
            'Authorization': authorization_header
        }
        super().__init__()

    async def execute(self):
        queue = self.queue
        try:
            with open(self.file_path, 'rb') as file:
                response = await self.session.post(self.target_url, headers=self.headers, data=file)
                while line := await response.content.readline():
                    prediction = json.loads(line)
                    await queue.put(prediction)
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(None)


class LoadFromJob(Job):
    def __init__(self, url: str, pipeline_base_url: str, authorization_header: str, session: ClientSession):
        self.session = session
        self.url = url
        self.target_url = f'{pipeline_base_url}/source?mode=queue&processing=sync'
        self.headers = {
            'Accept': 'application/jsonl',
            'Authorization': authorization_header
        }
        super().__init__()

    async def execute(self):
        queue = self.queue
        body = {
            "sourceType": "URL",
            "url": self.url
        }
        try:
            response = await self.session.patch(self.target_url, headers=self.headers, json=body)
            while line := await response.content.readline():
                prediction = json.loads(line)
                await queue.put(prediction)
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(None)

