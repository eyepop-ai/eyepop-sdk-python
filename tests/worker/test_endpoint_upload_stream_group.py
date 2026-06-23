import io
import json
import time

import pytest
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointUploadStreamGroup(BaseEndpointTest):
    test_source_id = 'test_stream_group_source_id'

    def _setup_auth_and_pop(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            return CallbackResult(status=200, body=json.dumps({'pop': Pop(components=[]).model_dump()}))

        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}', callback=get_pop)

    @staticmethod
    def _streams(n: int):
        return [io.BytesIO(f'fake-image-{i}'.encode()) for i in range(n)]

    @aioresponses()
    def test_sync_upload_stream_group_ok(self, mock: aioresponses):
        self._setup_auth_and_pop(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000
            upload_called = 0

            def upload(url, **kwargs) -> CallbackResult:
                nonlocal upload_called
                upload_called += 1
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                return CallbackResult(status=200, body=json.dumps(
                    {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': test_timestamp}))

            mock.post(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync&version=2',
                callback=upload)

            job = endpoint.upload_stream_group(self._streams(2), mime_types=['image/jpeg', 'image/png'])
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertIsNone(job.predict())
            self.assertEqual(upload_called, 1)

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_upload_stream_group_ok_no_mime(self, mock: aioresponses):
        self._setup_auth_and_pop(mock)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000
            upload_called = 0

            def upload(url, **kwargs) -> CallbackResult:
                nonlocal upload_called
                upload_called += 1
                return CallbackResult(status=200, body=json.dumps(
                    {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': test_timestamp}))

            mock.post(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync&version=2',
                callback=upload)

            job = await endpoint.upload_stream_group(self._streams(3))
            result = await job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertIsNone(await job.predict())
            self.assertEqual(upload_called, 1)

    @aioresponses()
    @pytest.mark.asyncio
    async def test_mime_types_length_mismatch_rejected(self, mock: aioresponses):
        self._setup_auth_and_pop(mock)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            with self.assertRaises(ValueError):
                await endpoint.upload_stream_group(self._streams(2), mime_types=['image/jpeg'])
