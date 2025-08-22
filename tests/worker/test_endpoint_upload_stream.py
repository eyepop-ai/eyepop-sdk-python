import json
import time
from importlib import resources

import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.worker.base_endpoint_test import BaseEndpointTest
import tests

class TestEndpointUploadStream(BaseEndpointTest):
    test_source_id = 'test_source_id'
    test_file = resources.files(tests) / 'test.jpg'
    test_content_type = 'image/jpeg'


    @aioresponses()
    def test_sync_upload_ok(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                      pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000

            upload_called = 0

            def upload(url, **kwargs) -> CallbackResult:
                nonlocal test_timestamp
                nonlocal upload_called
                upload_called += 1
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                elif kwargs['headers']['Content-Type'] != self.test_content_type:
                    return CallbackResult(status=40, reason='unsupported content type')
                else:
                    return CallbackResult(status=200,
                                          body=json.dumps(
                                              {'source_id': self.test_source_id, 'seconds': 0,
                                               'system_timestamp': test_timestamp}))

            mock.post(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                       callback=upload)

            with open(self.test_file, 'rb') as file:
                job = endpoint.upload_stream(file, 'image/jpeg')
                result = job.predict()

            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(job.predict())

            self.assertEqual(upload_called, 1)


    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_upload_ok(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))
        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))
        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                    callback=get_pop_comp)
        async with EyePopSdk.workerEndpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                            pop_id=self.test_eyepop_pop_id, is_async=True) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000

            upload_called = 0

            def upload(url, **kwargs) -> CallbackResult:
                nonlocal test_timestamp
                nonlocal upload_called
                upload_called += 1
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                elif kwargs['headers']['Content-Type'] != self.test_content_type:
                    return CallbackResult(status=40, reason='unsupported content type')
                else:
                    return CallbackResult(status=200,
                                          body=json.dumps(
                                              {'source_id': self.test_source_id, 'seconds': 0,
                                               'system_timestamp': test_timestamp}))

            mock.post(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                       callback=upload)

            with open(self.test_file, 'rb') as file:
                job = await endpoint.upload_stream(file, 'image/jpeg')
                result = await job.predict()

            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(await job.predict())

            self.assertEqual(upload_called, 1)

