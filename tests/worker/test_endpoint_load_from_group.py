import json
import time

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from eyepop import EyePopSdk
from eyepop.worker.worker_types import DEFAULT_PREDICTION_VERSION, Pop
from tests.worker.base_endpoint_test import BaseEndpointTest


class TestEndpointLoadFromGroup(BaseEndpointTest):
    test_source_id = 'test_source_id'
    test_urls = ['http://example-media.test/a.png', 'http://example-media.test/b.png']

    def _setup_auth_and_pop(self, mock: aioresponses):
        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        def get_pop(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            return CallbackResult(status=200, body=json.dumps({'pop': Pop(components=[]).model_dump()}))

        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}', callback=get_pop)

    def _expected_group_body(self) -> str:
        return json.dumps({
            'sourceType': 'GROUP',
            'sources': [{'sourceType': 'URL', 'url': url} for url in self.test_urls],
            'version': DEFAULT_PREDICTION_VERSION,
        })

    @aioresponses()
    def test_sync_load_from_group_ok(self, mock: aioresponses):
        self._setup_auth_and_pop(mock)
        with EyePopSdk.sync_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000

            def load(url, **kwargs) -> CallbackResult:
                if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                    return CallbackResult(status=401, reason='test auth token expired')
                return CallbackResult(status=200, body=json.dumps(
                    {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': test_timestamp}))

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                       callback=load)

            job = endpoint.load_from_group(self.test_urls)
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertIsNone(job.predict())

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=self._expected_group_body(),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    @aioresponses()
    @pytest.mark.asyncio
    async def test_async_load_from_group_ok(self, mock: aioresponses):
        self._setup_auth_and_pop(mock)
        async with EyePopSdk.async_worker(
                eyepop_url=self.test_eyepop_url,
                secret_key=self.test_eyepop_secret_key,
                pop_id=self.test_eyepop_pop_id,
        ) as endpoint:
            self.assertBaseMock(mock)
            test_timestamp = time.time() * 1000 * 1000 * 1000

            def load(url, **kwargs) -> CallbackResult:
                return CallbackResult(status=200, body=json.dumps(
                    {'source_id': self.test_source_id, 'seconds': 0, 'system_timestamp': test_timestamp}))

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                       callback=load)

            job = await endpoint.load_from_group(self.test_urls)
            result = await job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertIsNone(await job.predict())

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=self._expected_group_body(),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
