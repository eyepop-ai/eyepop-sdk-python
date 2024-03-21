import json
import time
import aiohttp
import pytest
from aioresponses import aioresponses, CallbackResult

from eyepop import EyePopSdk
from tests.base_endpoint_test import BaseEndpointTest


class TestEndpointPopComp(BaseEndpointTest):

    @aioresponses()
    def test_sync_get_pop_comp(self, mock: aioresponses):
        test_pop_comp = "ep_infer id=1 model=model_name threshold=0.7"

        self.setup_base_mock(mock)
        mock.post(f'{self.test_eyepop_url}/authentication/token', status=200, body=json.dumps(
            {'expires_in': 1000 * 1000, 'token_type': 'Bearer', 'access_token': self.test_access_token}))

        with EyePopSdk.endpoint(eyepop_url=self.test_eyepop_url, secret_key=self.test_eyepop_secret_key,
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

            mock.get(f'{self.test_worker_url}/pipelines/{self.test_pipeline_id}',
                       callback=upload)

            job = endpoint.upload(self.test_file)
            result = job.predict()
            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], self.test_source_id)
            self.assertEqual(result['seconds'], 0)
            self.assertEqual(result['system_timestamp'], test_timestamp)
            self.assertIsNone(job.predict())

            self.assertEqual(upload_called, 1)
