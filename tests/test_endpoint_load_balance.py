import json
import time
import aiohttp
import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.base_endpoint_test import BaseEndpointTest


class TestEndpointLoadBalance(BaseEndpointTest):
    test_source_id = 'test_source_id'
    test_url = 'http://examle-media.test/test.png'

    @aioresponses()
    def test_sync_load_one(self, mock: aioresponses):
        self._prepare_mock(mock, 1)
        with EyePopSdk.endpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock, status='active_prod')
            test_timestamp = time.time() * 1000 * 1000 * 1000

            self.test_access_token = '... a new access token ....'
            self._prepare_mock(mock, 1)

            def loadFrom(url, **kwargs) -> CallbackResult:
                nonlocal test_timestamp
                if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                    return CallbackResult(status=200,
                                          body=json.dumps(
                                              {'source_id': self.test_source_id, 'seconds': 0,
                                               'system_timestamp': test_timestamp}))
                else:
                    return CallbackResult(status=401, reason='test auth token expired')

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-0/source?mode=queue&processing=sync',
                       callback=loadFrom, repeat=True)

            job = endpoint.load_from(self.test_url)
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(job.predict())

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-0/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=json.dumps({'sourceType': 'URL', 'url': self.test_url}),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=60))

    @aioresponses()
    def test_sync_load_two(self, mock: aioresponses):
        self._prepare_mock(mock, 2)
        with EyePopSdk.endpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
                                pop_id=self.test_eyepop_pop_id) as endpoint:
            self.assertBaseMock(mock, status='active_prod')
            test_timestamp = time.time() * 1000 * 1000 * 1000

            self.test_access_token = '... a new access token ....'
            self._prepare_mock(mock, 2)

            def loadFrom(url, **kwargs) -> CallbackResult:
                nonlocal test_timestamp
                if kwargs['headers']['Authorization'] == f'Bearer {self.test_access_token}':
                    return CallbackResult(status=200,
                                          body=json.dumps(
                                              {'source_id': self.test_source_id, 'seconds': 0,
                                               'system_timestamp': test_timestamp}))
                else:
                    return CallbackResult(status=401, reason='test auth token expired')

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-0/source?mode=queue&processing=sync',
                       callback=loadFrom, repeat=True)

            mock.patch(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-1/source?mode=queue&processing=sync',
                       callback=loadFrom, repeat=True)

            job = endpoint.load_from(self.test_url)
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(job.predict())

            job = endpoint.load_from(self.test_url)
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(job.predict())

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-0/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=json.dumps({'sourceType': 'URL', 'url': self.test_url}),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=60))

            mock.assert_called_with(
                f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-1/source?mode=queue&processing=sync',
                method='PATCH',
                headers={
                    'Accept': 'application/jsonl',
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.test_access_token}'
                },
                data=json.dumps({'sourceType': 'URL', 'url': self.test_url}),
                timeout=aiohttp.ClientTimeout(total=None, sock_read=60))

    def _prepare_mock(self, mock, num_endpoints):
        self.setup_base_mock(mock, status='active_prod', num_endpoints=num_endpoints)
        mock.post(f'{self.test_eyepop_url}/authentication/token',
                  status=200,
                  body=json.dumps({
                      'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token
                  }),
                  repeat=True)

        # automatic call to get pop comp and store
        def get_pop_comp(url, **kwargs) -> CallbackResult:
            if kwargs['headers']['Authorization'] != f'Bearer {self.test_access_token}':
                return CallbackResult(status=401, reason='test auth token expired')
            else:
                return CallbackResult(status=200, body=json.dumps({'inferPipeline': self.test_pop_comp}))

        mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}-0',
                 callback=get_pop_comp)
